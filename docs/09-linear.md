# 并行线性层详解

## 1. 概述

在大规模语言模型中，模型参数量巨大（如 Qwen3-72B），单个 GPU 无法容纳。张量并行（Tensor Parallel）通过将大矩阵切分到多个 GPU 上并行计算，实现大模型的推理和训练。

nano-vLLM 实现了四种并行线性层：

| 类型 | 类名 | 切分维度 | 应用场景 |
|------|------|----------|----------|
| 复制式 | `ReplicatedLinear` | 无切分 | 小尺寸层 |
| 列并行 | `ColumnParallelLinear` | 输出维度 | QKV 投影、gate/up 投影 |
| 行并行 | `RowParallelLinear` | 输入维度 | 输出投影（o_proj、down_proj） |
| 合并列并行 | `MergedColumnParallelLinear` | 输出维度 | SwiGLU 门控结构 |
| QKV 并行 | `QKVParallelLinear` | 输出维度 | 多头注意力 QKV 投影 |

---

## 2. 张量并行基础

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

## 3. 多 GPU 通信模式

### 3.1 通信原语

| 原语 | 说明 | 示意图 |
|------|------|--------|
| AllReduce | 所有 GPU 求和并广播 | `[a], [b], [c]` → `[a+b+c], [a+b+c], [a+b+c]` |
| AllGather | 收集所有 GPU 的数据 | `[a], [b], [c]` → `[a,b,c], [a,b,c], [a,b,c]` |
| ReduceScatter | 求和后切分 | `[a1,a2], [b1,b2], [c1,c2]` → `[a1+b1+c1], [a2+b2+c2]` |
| Broadcast | 从根节点广播 | `[data], [], []` → `[data], [data], [data]` |
| Gather | 收集到根节点 | `[a], [b], [c]` → `[a,b,c], [], []` |

### 3.2 nano-vLLM 中的通信模式

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
- 需要将各 GPU 的词汇表 logits 拼接成完整分布

#### 模式 3：无通信（列并行输出）

```python
# ColumnParallelLinear 无需通信
return F.linear(x, self.weight, self.bias)
```

**原因**：输出已经在各 GPU 上切分，后续操作可以直接使用部分输出。

---

## 4. 特殊并行线性层

### 4.1 MergedColumnParallelLinear

**用途**：合并两个列并行线性层，用于 SwiGLU 门控结构。

**权重布局**：
```
完整权重：W = [W_gate; W_up]  # 按行拼接
分片权重：W_shard = [W_gate_shard; W_up_shard]
```

**代码实现**：
```python
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size, output_sizes, bias=False):
        # output_sizes = [intermediate_size, intermediate_size]
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)
    
    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        # loaded_shard_id: 0=gate, 1=up
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        
        # 从合并参数中切分出当前分片对应的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 对输入权重进行张量并行切分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
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

### 4.2 QKVParallelLinear

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

**代码实现**：
```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads=None, bias=False):
        tp_size = dist.get_world_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        
        # 总输出维度 = Q + K + V
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)
    
    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        # loaded_shard_id: "q", "k", "v"
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

**使用示例**：
```python
# Qwen3 Attention 的 QKV 投影
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
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
```

---

## 5. Qwen3 模型中的并行线性层

### 5.1 模型结构

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

### 5.2 数据流示例（2-GPU 张量并行）

```
输入：input_ids [batch, seq_len]

1. 词嵌入 (VocabParallelEmbedding)
   GPU0: 处理词汇 [0, 15999] 的嵌入
   GPU1: 处理词汇 [16000, 31999] 的嵌入
   → AllReduce 后：hidden_states [batch, seq_len, hidden_size] (所有 GPU 相同)

2. Attention 层
   a) QKV 投影 (QKVParallelLinear - 列并行)
      GPU0: qkv_proj_0 → [batch, seq_len, (16+4+4)*head_dim]
      GPU1: qkv_proj_1 → [batch, seq_len, (16+4+4)*head_dim]
      → 无需通信，各 GPU 持有部分 QKV
   
   b) Attention 计算
      GPU0: attn_0 → [batch, seq_len, 16*head_dim]
      GPU1: attn_1 → [batch, seq_len, 16*head_dim]
   
   c) 输出投影 (RowParallelLinear - 行并行)
      GPU0: o_proj_0(attn_0 的部分) → partial_0
      GPU1: o_proj_1(attn_1 的部分) → partial_1
      → AllReduce: output = partial_0 + partial_1

3. MLP 层
   a) gate_up_proj (MergedColumnParallelLinear - 列并行)
      GPU0: gate_up_0 → [batch, seq_len, 2*intermediate/2]
      GPU1: gate_up_1 → [batch, seq_len, 2*intermediate/2]
      → 无需通信
   
   b) SwiGLU 激活
      GPU0: silu(gate_0) * up_0
      GPU1: silu(gate_1) * up_1
   
   c) down_proj (RowParallelLinear - 行并行)
      GPU0: down_0 → partial_0
      GPU1: down_1 → partial_1
      → AllReduce: output = partial_0 + partial_1

4. LM 头 (ParallelLMHead)
   GPU0: logits_0 [batch, vocab/2]
   GPU1: logits_1 [batch, vocab/2]
   → Gather 到 GPU0: logits = [logits_0, logits_1] [batch, vocab]
```

---

## 6. 权重加载

### 6.1 权重分片加载

```python
# ColumnParallelLinear 的权重加载
def weight_loader(self, param, loaded_weight):
    param_data = param.data
    shard_size = param_data.size(self.tp_dim)
    start_idx = self.tp_rank * shard_size
    # 从完整权重中切分出当前 GPU 负责的部分
    loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
    param_data.copy_(loaded_weight)
```

### 6.2 打包权重加载

对于 QKV 和 gate_up 这种合并的权重，需要分别加载：

```python
# QKV 权重加载
qkv_proj.weight_loader(weight_q, "q")   # 加载 Q 权重
qkv_proj.weight_loader(weight_k, "k")   # 加载 K 权重
qkv_proj.weight_loader(weight_v, "v")   # 加载 V 权重

# gate_up 权重加载
gate_up_proj.weight_loader(weight_gate, 0)  # 加载 gate 权重
gate_up_proj.weight_loader(weight_up, 1)    # 加载 up 权重
```

---

## 7. 总结

### 7.1 并行模式选择

| 层类型 | 并行模式 | 通信需求 | 原因 |
|--------|----------|----------|------|
| QKV 投影 | 列并行 | 无 | 输出直接用于多头计算 |
| 注意力输出 | 行并行 | AllReduce | 需要完整输出 |
| gate/up 投影 | 列并行 | 无 | SwiGLU 可分片计算 |
| down 投影 | 行并行 | AllReduce | 需要完整输出 |
| LM 头 | 词汇并行 | Gather | 需要完整词汇分布 |

### 7.2 通信开销优化

1. **减少通信次数**：列并行输出无需通信
2. **融合通信**：AllReduce 可以与计算重叠
3. **选择合适的并行策略**：根据层的特点选择行/列并行

### 7.3 扩展性

- **2-8 GPU**：张量并行效果最佳
- **>8 GPU**：建议结合流水线并行或数据并行
