# 张量并行与线性层详解

## 一、概述

在大规模语言模型中，模型参数量巨大（如 Qwen3-72B），单个 GPU 无法容纳。**张量并行（Tensor Parallel）** 通过将大矩阵切分到多个 GPU 上并行计算，实现大模型的推理和训练。

### 1.1 nano-vLLM 的并行线性层

| 类型 | 类名 | 切分维度 | 应用场景 |
|------|------|----------|----------|
| 复制式 | `ReplicatedLinear` | 无切分 | 小尺寸层 |
| 列并行 | `ColumnParallelLinear` | 输出维度 | QKV 投影、gate/up 投影 |
| 行并行 | `RowParallelLinear` | 输入维度 | 输出投影（o_proj、down_proj） |
| 合并列并行 | `MergedColumnParallelLinear` | 输出维度 | SwiGLU 门控结构 |
| QKV 并行 | `QKVParallelLinear` | 输出维度 | 多头注意力 QKV 投影 |

---

## 二、张量并行基础

### 2.1 矩阵乘法切分原理

对于线性变换 `Y = X @ W^T`，其中：
- `X`: 输入 `[batch, input_size]`
- `W`: 权重 `[output_size, input_size]`
- `Y`: 输出 `[batch, output_size]`

### 2.2 列并行（Column Parallel）

**切分方式**：按输出维度（行）切分权重矩阵

```
权重矩阵 W 按行切分：
    ┌─────────────────┐
    │     W_0         │  GPU 0
    ├─────────────────┤
    │     W_1         │  GPU 1
    ├─────────────────┤
    │     W_2         │  GPU 2
    └─────────────────┘

计算过程：
    Y_0 = X @ W_0^T   (GPU 0)
    Y_1 = X @ W_1^T   (GPU 1)
    Y_2 = X @ W_2^T   (GPU 2)

输出：每个 GPU 持有完整输出的 1/n
```

**代码实现**：
```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        # 输出维度按 tp_size 切分
        super().__init__(input_size, output_size // tp_size, bias, tp_dim=0)

    def forward(self, x):
        # 所有 GPU 的输入 X 相同
        # 每个 GPU 计算部分输出
        return F.linear(x, self.weight, self.bias)
```

### 2.3 行并行（Row Parallel）

**切分方式**：按输入维度（列）切分权重矩阵

```
权重矩阵 W 按列切分：
    ┌─────┬─────┬─────┐
    │ W_0 │ W_1 │ W_2 │  所有 GPU
    └─────┴─────┴─────┘

输入 X 也按列切分：
    X = [X_0, X_1, X_2]

计算过程：
    Y_0 = X_0 @ W_0^T   (GPU 0)
    Y_1 = X_1 @ W_1^T   (GPU 1)
    Y_2 = X_2 @ W_2^T   (GPU 2)

    Y = Y_0 + Y_1 + Y_2  (AllReduce 求和)

输出：每个 GPU 持有完整的输出 Y
```

**代码实现**：
```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        # 输入维度按 tp_size 切分
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def forward(self, x):
        # x 是部分输入（当前 GPU 对应的分片）
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # AllReduce：将所有 GPU 的部分结果相加
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

---

## 三、特殊并行线性层

### 3.1 MergedColumnParallelLinear

**用途**：合并两个列并行线性层，用于 SwiGLU 门控结构。

**权重布局**：
```
完整权重：W = [W_gate; W_up]  # 按行拼接
分片权重：W_shard = [W_gate_shard; W_up_shard]
```

**使用示例**：
```python
# Qwen3 MLP 的 gate_up_proj
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size=4096,
    output_sizes=[11008, 11008],  # [gate_size, up_size]
    bias=False
)

# 前向传播
gate_up = self.gate_up_proj(x)  # [batch, 22016]
gate, up = gate_up.chunk(2, -1)
output = F.silu(gate) * up
```

### 3.2 QKVParallelLinear

**用途**：合并 Q、K、V 三个投影矩阵，支持 GQA（Grouped Query Attention）。

**权重布局**：
```
完整权重：W_qkv = [W_q; W_k; W_v]

其中：
    W_q: [num_heads * head_dim, hidden_size]
    W_k: [num_kv_heads * head_dim, hidden_size]
    W_v: [num_kv_heads * head_dim, hidden_size]

输出维度 = (num_heads + 2 * num_kv_heads) * head_dim
```

**GQA 配置**：
| 类型 | num_heads | num_kv_heads | 说明 |
|------|-----------|--------------|------|
| MHA | 32 | 32 | 标准多头注意力 |
| GQA | 32 | 8 | 分组查询注意力 |
| MQA | 32 | 1 | 多查询注意力 |

**使用示例**：
```python
self.qkv_proj = QKVParallelLinear(
    hidden_size=4096,
    head_size=128,
    total_num_heads=32,
    total_num_kv_heads=8,  # GQA
    bias=False
)

# 前向传播
qkv = self.qkv_proj(x)  # [batch, (32+8+8)*128]
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
```

---

## 四、并行方式的决定机制

### 4.1 并行方式的确定时机

**答案**：并行方式在模型代码定义时就已经**硬编码确定**，加载权重时根据层的类型自动匹配对应的权重切分策略。

### 4.2 模型定义时硬编码

在 `nanovllm/models/qwen3.py` 中，每一层使用什么并行方式是**写死在代码里**的：

```python
class Qwen3Attention(nn.Module):
    def __init__(self, ...):
        # 硬编码：QKV 投影使用列并行
        self.qkv_proj = QKVParallelLinear(...)
        
        # 硬编码：输出投影使用行并行
        self.o_proj = RowParallelLinear(...)


class Qwen3MLP(nn.Module):
    def __init__(self, ...):
        # 硬编码：gate/up 投影使用列并行（合并）
        self.gate_up_proj = MergedColumnParallelLinear(...)
        
        # 硬编码：down 投影使用行并行
        self.down_proj = RowParallelLinear(...)
```

### 4.3 为什么不能随意选择

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

## 五、权重加载机制

### 5.1 HuggingFace 原始权重

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

### 5.2 权重映射关系

nano-vLLM 通过 `packed_modules_mapping` 知道哪些权重需要合并加载：

```python
# nanovllm/models/qwen3.py
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),      # q_proj → qkv_proj 的 q 部分
    "k_proj": ("qkv_proj", "k"),      # k_proj → qkv_proj 的 k 部分
    "v_proj": ("qkv_proj", "v"),      # v_proj → qkv_proj 的 v 部分
    "gate_proj": ("gate_up_proj", 0), # gate_proj → gate_up_proj 的第 0 部分
    "up_proj": ("gate_up_proj", 1),   # up_proj → gate_up_proj 的第 1 部分
}
```

### 5.3 weight_loader 自动切分

每个并行线性层类都有自己的 `weight_loader` 方法：

| 类名 | weight_loader 行为 | 切分维度 |
|------|-------------------|----------|
| `ColumnParallelLinear` | 按输出维度切分（行） | dim=0 |
| `RowParallelLinear` | 按输入维度切分（列） | dim=1 |
| `MergedColumnParallelLinear` | 按输出维度切分 + 偏移 | dim=0 |
| `QKVParallelLinear` | 按输出维度切分 + QKV 偏移 | dim=0 |
| `ReplicatedLinear` | 不切分，直接复制 | - |

---

## 六、多 GPU 加载示例

### 6.1 2-GPU 加载 Qwen3

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

### 6.2 权重切分示意

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
```

---

## 七、Qwen3 中的并行模式

### 7.1 模型结构

```
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding      # 词汇表并行嵌入
│   ├── Qwen3DecoderLayer × N
│   │   ├── Qwen3Attention
│   │   │   ├── QKVParallelLinear   # qkv_proj (列并行)
│   │   │   └── RowParallelLinear   # o_proj (行并行)
│   │   └── Qwen3MLP
│   │       ├── MergedColumnParallelLinear  # gate_up_proj (列并行)
│   │       └── RowParallelLinear           # down_proj (行并行)
│   └── RMSNorm
└── ParallelLMHead                  # 词汇表并行 LM 头 (Gather)
```

### 7.2 并行方式配对原则

| 层类型 | 输入状态 | 输出状态 | 并行方式 | 通信 |
|--------|----------|----------|----------|------|
| QKV 投影 | 完整 hidden | 部分 QKV | 列并行 | 无 |
| Attention | 部分 QKV | 部分 O | 分片计算 | 无 |
| O 投影 | 部分 O | 完整 hidden | 行并行 | AllReduce |
| GateUp | 完整 hidden | 部分 gate_up | 列并行 | 无 |
| SwiGLU | 部分 gate_up | 部分 output | 分片计算 | 无 |
| Down | 部分 output | 完整 hidden | 行并行 | AllReduce |

**关键观察**：
1. 列并行输出 → 分片计算 → 行并行输入（完美匹配）
2. 行并行输出（完整） → 下一层列并行输入（完整）（完美匹配）
3. 残差连接需要完整 hidden_states，所以输出层必须是行并行

---

## 八、多 GPU 通信模式

### 8.1 通信原语

| 原语 | 说明 | 示意图 |
|------|------|--------|
| AllReduce | 所有 GPU 求和并广播 | `[a], [b], [c]` → `[a+b+c], [a+b+c], [a+b+c]` |
| AllGather | 收集所有 GPU 的数据 | `[a], [b], [c]` → `[a,b,c], [a,b,c], [a,b,c]` |
| ReduceScatter | 求和后切分 | `[a1,a2], [b1,b2], [c1,c2]` → `[a1+b1+c1], [a2+b2+c2]` |
| Broadcast | 从根节点广播 | `[data], [], []` → `[data], [data], [data]` |
| Gather | 收集到根节点 | `[a], [b], [c]` → `[a,b,c], [], []` |

### 8.2 nano-vLLM 中的通信模式

#### 模式 1：AllReduce（行并行输出）

```python
# RowParallelLinear 使用 AllReduce
dist.all_reduce(y)  # 所有 GPU 的 y 相加
```

**应用场景**：
- 注意力输出投影（o_proj）
- MLP 输出投影（down_proj）

#### 模式 2：Gather（LM 头输出）

```python
# ParallelLMHead 使用 Gather
all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
dist.gather(logits, all_logits, 0)  # 收集到 rank=0
logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
```

**应用场景**：
- 语言模型输出头（ParallelLMHead）

#### 模式 3：无通信（列并行输出）

```python
# ColumnParallelLinear 无需通信
return F.linear(x, self.weight, self.bias)
```

**原因**：输出已经在各 GPU 上切分，后续操作可以直接使用部分输出。

---

## 九、总结

### 并行方式选择原则

| 决策点 | 选择 |
|--------|------|
| 输出是否需要完整数据？ | 是 → 行并行，否 → 列并行 |
| 计算是否可分片？ | 是 → 列并行，否 → 特殊设计 |
| 是否是输出层？ | 是 → 词汇并行 + Gather |

### 核心机制

| 机制 | 说明 |
|------|------|
| **模型代码硬编码** | 使用什么并行类在模型定义时确定 |
| **weight_loader 自动切分** | 每个类有自己的 weight_loader |
| **运行时自动获取 TP 信息** | dist.get_rank() 和 dist.get_world_size() |
| **层间匹配** | 列并行→分片计算→行并行，形成完整数据流 |

### 设计思想

1. **模型结构决定并行方式**：不同的层有不同的计算特点
2. **解耦模型和权重加载**：通过 weight_loader 抽象切分逻辑
3. **自动化运行时配置**：通过分布式环境自动获取 TP 信息
4. **最小化通信开销**：能分片计算的操作尽量避免通信
