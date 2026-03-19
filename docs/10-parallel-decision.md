# 张量并行方式的决定机制

## 问题背景

在多 GPU 张量并行推理中，一个关键问题是：**哪一层使用列并行，哪一层使用行并行，是如何决定的？**

答案是：**并行方式在模型代码定义时就已经硬编码确定**，加载权重时根据层的类型自动匹配对应的权重切分策略。

---

## 1. 并行方式的确定时机

### 1.1 模型定义时硬编码

在 `nanovllm/models/qwen3.py` 中，每一层使用什么并行方式是**写死在代码里**的：

```python
class Qwen3Attention(nn.Module):
    def __init__(self, ...):
        # 硬编码：QKV 投影使用列并行
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        
        # 硬编码：输出投影使用行并行
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )


class Qwen3MLP(nn.Module):
    def __init__(self, ...):
        # 硬编码：gate/up 投影使用列并行（合并）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        
        # 硬编码：down 投影使用行并行
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
```

**关键点**：
- 使用 `QKVParallelLinear` → 自动就是列并行
- 使用 `RowParallelLinear` → 自动就是行并行
- 使用 `MergedColumnParallelLinear` → 自动就是合并列并行

### 1.2 为什么不能随意选择

并行方式的选择受以下因素约束：

| 约束因素 | 说明 |
|----------|------|
| **算法结构** | 某些操作（如 Attention）天然是列并行的 |
| **数据依赖** | 层间连接需要匹配输入输出状态 |
| **通信效率** | 最小化不必要的通信 |
| **内存布局** | 权重和激活的存储方式 |

**错误示例**：如果 QKV 投影使用行并行
```python
# 错误：QKV 使用行并行（输入切分）
q_0 = W_q @ x_0   # GPU 0 计算部分输入
q_1 = W_q @ x_1   # GPU 1 计算部分输入
q = q_0 + q_1     # AllReduce 得到完整 Q ← 通信开销巨大！

# 问题：
# 1. 每次计算 QKV 都需要 AllReduce
# 2. Attention 计算需要完整 QKV，无法分片进行
```

---

## 2. 权重加载机制

### 2.1 HuggingFace 原始权重

HuggingFace 的权重是完整的（单卡格式）：

```python
# HuggingFace 权重文件（完整权重）
{
    "model.layers.0.self_attn.q_proj.weight": torch.Size([4096, 4096]),
    "model.layers.0.self_attn.k_proj.weight": torch.Size([4096, 4096]),
    "model.layers.0.self_attn.v_proj.weight": torch.Size([4096, 4096]),
    "model.layers.0.self_attn.o_proj.weight": torch.Size([4096, 4096]),
    "model.layers.0.mlp.gate_proj.weight": torch.Size([11008, 4096]),
    "model.layers.0.mlp.up_proj.weight": torch.Size([11008, 4096]),
    "model.layers.0.mlp.down_proj.weight": torch.Size([4096, 11008]),
}
```

### 2.2 权重映射关系

nano-vLLM 通过 `packed_modules_mapping` 知道哪些权重需要合并加载：

```python
# nanovllm/models/qwen3.py
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),      # q_proj → qkv_proj 的 q 部分
        "k_proj": ("qkv_proj", "k"),      # k_proj → qkv_proj 的 k 部分
        "v_proj": ("qkv_proj", "v"),      # v_proj → qkv_proj 的 v 部分
        "gate_proj": ("gate_up_proj", 0), # gate_proj → gate_up_proj 的第 0 部分
        "up_proj": ("gate_up_proj", 1),   # up_proj → gate_up_proj 的第 1 部分
    }
```

### 2.3 weight_loader 自动切分

每个并行线性层类都有自己的 `weight_loader` 方法，知道如何切分权重：

```python
# ColumnParallelLinear 的 weight_loader（按输出维度切分）
def weight_loader(self, param, loaded_weight):
    param_data = param.data
    shard_size = param_data.size(self.tp_dim)  # tp_dim=0
    start_idx = self.tp_rank * shard_size
    # 从完整权重中切分出当前 GPU 负责的部分（按行切分）
    loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
    param_data.copy_(loaded_weight)


# RowParallelLinear 的 weight_loader（按输入维度切分）
def weight_loader(self, param, loaded_weight):
    param_data = param.data
    shard_size = param_data.size(self.tp_dim)  # tp_dim=1
    start_idx = self.tp_rank * shard_size
    # 从完整权重中切分出当前 GPU 负责的部分（按列切分）
    loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
    param_data.copy_(loaded_weight)


# QKVParallelLinear 的 weight_loader（按输出维度切分 + QKV 偏移）
def weight_loader(self, param, loaded_weight, loaded_shard_id):
    if loaded_shard_id == "q":
        shard_size = self.num_heads * self.head_size
        shard_offset = 0
    elif loaded_shard_id == "k":
        shard_size = self.num_kv_heads * self.head_size
        shard_offset = self.num_heads * self.head_size
    else:  # "v"
        shard_size = self.num_kv_heads * self.head_size
        shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
    
    param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
    param_data.copy_(loaded_weight)
```

---

## 3. 多卡加载流程

### 3.1 运行时信息获取

每个线性层在初始化时自动获取 TP 信息：

```python
class LinearBase(nn.Module):
    def __init__(self, ...):
        self.tp_rank = dist.get_rank()        # 自动获取当前 GPU rank
        self.tp_size = dist.get_world_size()  # 自动获取 GPU 数量
        self.tp_dim = tp_dim                  # 切分维度（0=输出，1=输入）
```

### 3.2 2-GPU 加载示例

```
GPU 0 (rank=0)                          GPU 1 (rank=1)
─────────────────────────────────────────────────────────────
QKV 投影 (列并行)
  Q: [16*128, 4096] (前 16 个头)           Q: [16*128, 4096] (后 16 个头)
  K: [4*128, 4096] (前 4 个头)             K: [4*128, 4096] (后 4 个头)
  V: [4*128, 4096] (前 4 个头)             V: [4*128, 4096] (后 4 个头)

O 投影 (行并行)
  W: [4096, 16*128] (前 16 个头列)          W: [4096, 16*128] (后 16 个头列)

GateUp 投影 (列并行)
  gate: [5504, 4096] (前 5504 行)          gate: [5504, 4096] (后 5504 行)
  up: [5504, 4096] (前 5504 行)            up: [5504, 4096] (后 5504 行)

Down 投影 (行并行)
  W: [4096, 5504] (前 5504 列)             W: [4096, 5504] (后 5504 列)
```

### 3.3 权重切分示意

```
Q 权重 [32*128, 4096] (完整)
┌─────────────────────┐
│     GPU 0           │  [16*128, 4096]  前 16 个头
├─────────────────────┤
│     GPU 1           │  [16*128, 4096]  后 16 个头
└─────────────────────┘
         ↑ 按行切分 (dim=0)

O 权重 [4096, 32*128] (完整)
┌───────────┬───────────┐
│   GPU 0   │   GPU 1   │
│ [4096,    │ [4096,    │
│  16*128]  │  16*128]  │
└───────────┴───────────┘
    ↑ 按列切分 (dim=1)

gate 权重 [11008, 4096] (完整)
┌─────────────────────┐
│     GPU 0           │  [5504, 4096]  前 5504 行
├─────────────────────┤
│     GPU 1           │  [5504, 4096]  后 5504 行
└─────────────────────┘
         ↑ 按行切分 (dim=0)
```

---

## 4. 层间并行方式匹配

### 4.1 Qwen3 的并行模式序列

```
输入 (完整 hidden_states)
    ↓
┌─────────────────────────────────────┐
│ Qwen3DecoderLayer                   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Attention Block             │   │
│  │  Input: 完整                │   │
│  │  ↓                          │   │
│  │  QKV: 列并行 (无需通信)     │   │
│  │  ↓                          │   │
│  │  Attention: 分片计算        │   │
│  │  ↓                          │   │
│  │  O_proj: 行并行 (AllReduce) │   │
│  │  Output: 完整               │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ MLP Block                   │   │
│  │  Input: 完整                │   │
│  │  ↓                          │   │
│  │  GateUp: 列并行 (无需通信)  │   │
│  │  ↓                          │   │
│  │  SwiGLU: 分片计算           │   │
│  │  ↓                          │   │
│  │  Down: 行并行 (AllReduce)   │   │
│  │  Output: 完整               │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
输出 (完整 hidden_states)
```

### 4.2 并行方式配对原则

| 层类型 | 输入状态 | 输出状态 | 并行方式 | 通信 | 原因 |
|--------|----------|----------|----------|------|------|
| QKV 投影 | 完整 hidden | 部分 QKV | 列并行 | 无 | Attention 可分片计算 |
| Attention | 部分 QKV | 部分 O | 分片计算 | 无 | 各 GPU 独立计算 |
| O 投影 | 部分 O | 完整 hidden | 行并行 | AllReduce | 需要完整输出给残差 |
| GateUp | 完整 hidden | 部分 gate_up | 列并行 | 无 | SwiGLU 可分片计算 |
| SwiGLU | 部分 gate_up | 部分 output | 分片计算 | 无 | 逐元素操作可分片 |
| Down | 部分 output | 完整 hidden | 行并行 | AllReduce | 需要完整输出给残差 |

**关键观察**：
1. 列并行输出 → 分片计算 → 行并行输入（完美匹配）
2. 行并行输出（完整） → 下一层列并行输入（完整）（完美匹配）
3. 残差连接需要完整 hidden_states，所以输出层必须是行并行

---

## 5. 设计原则总结

### 5.1 决策树

```
1. 这一层的输出是否需要完整数据给下一层？
   ├─ 是（如残差连接、LM 头） → 行并行（AllReduce 后输出完整）
   └─ 否 → 继续判断

2. 这一层的计算是否可以在分片数据上独立进行？
   ├─ 是（如 Attention、SwiGLU） → 列并行（无需通信）
   └─ 否 → 需要特殊设计

3. 这一层是否是输出层（LM 头）？
   ├─ 是 → 词汇并行 + Gather（需要完整词汇分布）
   └─ 否 → 回到 1
```

### 5.2 核心机制

| 机制 | 说明 |
|------|------|
| **模型代码硬编码** | 使用什么并行类（`QKVParallelLinear` 等）在模型定义时确定 |
| **weight_loader 自动切分** | 每个类有自己的 `weight_loader`，知道如何切分权重 |
| **运行时自动获取 TP 信息** | `dist.get_rank()` 和 `dist.get_world_size()` 自动获取 |
| **层间匹配** | 列并行→分片计算→行并行，形成完整的数据流 |

### 5.3 修改并行方式

如果需要修改并行方式，需要：

1. **修改模型定义**：更换并行类（如 `RowParallelLinear` → `ColumnParallelLinear`）
2. **修改 weight_loader**：确保权重切分逻辑正确
3. **修改层间连接**：确保输入输出状态匹配
4. **修改通信逻辑**：确保 AllReduce/Gather 使用正确

---

## 6. 总结

| 问题 | 答案 |
|------|------|
| 并行方式何时确定？ | 模型代码定义时硬编码 |
| 多卡如何知道怎么切分？ | 每层的 `weight_loader` 根据 `tp_rank` 和 `tp_size` 自动计算 |
| 需要用户指定吗？ | 不需要，模型代码已定义好 |
| 可以修改并行方式吗？ | 可以，但需要修改模型代码并重新设计权重加载逻辑 |
| 为什么不能随意选择？ | 受算法结构、数据依赖、通信效率约束 |

**核心设计思想**：
1. **模型结构决定并行方式**：不同的层有不同的计算特点
2. **解耦模型和权重加载**：通过 `weight_loader` 抽象切分逻辑
3. **自动化运行时配置**：通过分布式环境自动获取 TP 信息
4. **最小化通信开销**：能分片计算的操作尽量避免通信

这种设计使得 nano-vLLM 可以灵活支持 1-8 卡张量并行，而无需修改权重加载逻辑。
