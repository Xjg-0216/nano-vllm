# 注意力机制 (Attention) 详解

## 一、模块概述

`nanovllm/layers/attention.py` 实现了基于 **Flash Attention** 的 KV Cache 管理注意力机制，是 Nano-vLLM 高效推理的核心组件。

### 1.1 核心功能

| 功能 | 说明 |
|------|------|
| **KV Cache 管理** | 使用 Triton kernel 高效写入 KV Cache |
| **Prefill 模式** | 处理整个 prompt 序列，使用 `flash_attn_varlen_func` |
| **Decode 模式** | 增量解码，使用 `flash_attn_with_kvcache` |
| **前缀缓存优化** | 重用已计算的前缀 KV Cache |

### 1.2 在推理流程中的位置

```
完整推理流程:
─────────────────────────────────────────────────────────
输入 → Qwen3Attention → QKV 投影 → RoPE → Attention → 输出投影
                                      │
                                      ▼
                              Flash Attention
                                      │
                                      ▼
                              KV Cache 管理
```

### 1.3 两个核心模式

```
Prefill 模式 (处理用户输入):
─────────────────────────────────────────────────────────
输入："Hello, how are you"
      └─ 一次性处理所有 token
      └─ 计算并存储 KV Cache
      └─ 使用 flash_attn_varlen_func

Decode 模式 (逐 token 生成):
─────────────────────────────────────────────────────────
输入：[上一个生成的 token]
      └─ 只处理 1 个新 token
      └─ 使用历史 KV Cache
      └─ 使用 flash_attn_with_kvcache
```

---

## 二、核心概念

### 2.1 KV Cache 的作用

```
自回归生成中的重复计算问题:
─────────────────────────────────────────────────────────

Step 1: 输入 "Hello"
        └─ 计算 K₀, V₀

Step 2: 输入 "Hello, how"
        └─ 重新计算 K₀, V₀ (浪费!)
        └─ 计算 K₁, V₁

Step 3: 输入 "Hello, how are"
        └─ 重新计算 K₀, V₀ (浪费!)
        └─ 重新计算 K₁, V₁ (浪费!)
        └─ 计算 K₂, V₂

问题：每一步都重复计算之前的 KV!
```

```
KV Cache 优化:
─────────────────────────────────────────────────────────

Step 1: 输入 "Hello"
        └─ 计算 K₀, V₀ → 存入 KV Cache
        └─ Cache: [K₀, V₀]

Step 2: 输入 "how"
        └─ 从 Cache 读取 K₀, V₀
        └─ 计算 K₁, V₁ → 存入 KV Cache
        └─ Cache: [K₀, V₀, K₁, V₁]

Step 3: 输入 "are"
        └─ 从 Cache 读取 K₀, V₀, K₁, V₁
        └─ 计算 K₂, V₂ → 存入 KV Cache
        └─ Cache: [K₀, V₀, K₁, V₁, K₂, V₂]

优势：避免重复计算，加速生成!
```

### 2.2 Prefill vs Decode

| 特性 | Prefill 模式 | Decode 模式 |
|------|-------------|------------|
| **输入** | 完整 prompt (n 个 token) | 上一个 token (1 个) |
| **KV Cache** | 写入所有 token 的 KV | 只写入新 token 的 KV |
| **注意力计算** | 处理所有 token | 只处理最后一个 token |
| **Flash Attention** | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| **计算量** | O(n²) | O(1) |
| **使用场景** | 处理用户输入 | 逐 token 生成 |

---

## 三、Triton Kernel 详解

### 3.1 为什么使用 Triton？

```
传统 Python 循环写入 KV Cache:
─────────────────────────────────────────────────────────
for i in range(num_tokens):
    k_cache[slot_mapping[i]] = k[i]  # 慢！Python 循环

问题:
- Python 循环开销大
- 无法充分利用 GPU 并行
- 内存访问模式不优化
```

```
Triton Kernel:
─────────────────────────────────────────────────────────
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)  # 每个程序实例处理一个 token
    slot = tl.load(slot_mapping_ptr + idx)
    tl.store(k_cache_ptr + slot * D, key)

优势:
- GPU 并行执行
- 编译优化
- 高效内存访问
```

### 3.2 store_kvcache_kernel 详解

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,           # Key 张量指针
    key_stride,        # Key 张量的 stride（第 0 维）
    value_ptr,         # Value 张量指针
    value_stride,      # Value 张量的 stride（第 0 维）
    k_cache_ptr,       # Key Cache 张量指针
    v_cache_ptr,       # Value Cache 张量指针
    slot_mapping_ptr,  # 槽位映射指针
    D: tl.constexpr,   # 每个 token 的 K/V 总维度
):
    """
    Triton Kernel：将 K/V 张量写入 KV Cache
    
    每个程序实例处理一个 token 的 K/V 向量
    """
```

#### 执行模型

```
GPU 并行执行模型:
─────────────────────────────────────────────────────────

假设有 N 个 token，启动 N 个程序实例:

Program 0: 处理 token 0  ─┐
Program 1: 处理 token 1  ─┤
Program 2: 处理 token 2  ─┼── 并行执行
...                      │
Program N-1: 处理 token N-1 ─┘

每个程序独立执行:
1. 读取自己的 token 索引 (idx)
2. 从 slot_mapping 读取自己的槽位 (slot)
3. 计算 K/V 在输入张量中的位置
4. 加载 K/V 数据
5. 计算在 KV Cache 中的位置
6. 存储到 KV Cache
```

#### 逐步解析

```python
# 步骤 1：获取当前程序实例处理的 token 索引
idx = tl.program_id(0)
# 例如：idx = 5 (当前程序处理第 5 个 token)

# 步骤 2：从 slot_mapping 读取该 token 应存储的 KV Cache 槽位
slot = tl.load(slot_mapping_ptr + idx)
# 例如：slot = 128 (第 5 个 token 应存储到 KV Cache 的第 128 个位置)

# 步骤 3：检查是否需要存储
if slot == -1:
    return
# slot = -1 表示不需要存储（如 warmup 场景）

# 步骤 4：计算当前 token 的 K/V 在输入张量中的偏移
key_offsets = idx * key_stride + tl.arange(0, D)
value_offsets = idx * value_stride + tl.arange(0, D)
# 例如：idx=5, stride=4096, D=4096
# key_offsets = [5*4096+0, 5*4096+1, ..., 5*4096+4095]
#             = [20480, 20481, ..., 24575]

# 步骤 5：加载 K/V 数据
key = tl.load(key_ptr + key_offsets)
value = tl.load(value_ptr + value_offsets)
# 加载第 5 个 token 的完整 K/V 向量（4096 个元素）

# 步骤 6：计算在 KV Cache 中的偏移
cache_offsets = slot * D + tl.arange(0, D)
# 例如：slot=128, D=4096
# cache_offsets = [128*4096+0, 128*4096+1, ..., 128*4096+4095]
#               = [524288, 524289, ..., 528383]

# 步骤 7：将 K/V 存储到 KV Cache
tl.store(k_cache_ptr + cache_offsets, key)
tl.store(v_cache_ptr + cache_offsets, value)
# 将第 5 个 token 的 K/V 写入 KV Cache 的第 128 个位置
```

#### 图示

```
Triton Kernel 执行流程 (N=4 个 token):
─────────────────────────────────────────────────────────

输入:
key:      [k₀, k₁, k₂, k₃]  形状：[4, 4096]
slot_map: [10, 15, -1, 20]  形状：[4]
          └─ token 2 不需要存储

GPU 并行执行:
┌─────────────────────────────────────────────────────────┐
│ Program 0 (idx=0)    Program 1 (idx=1)                  │
│ slot = 10            slot = 15                          │
│ k_cache[10] = k₀     k_cache[15] = k₁                   │
│                                                             │
│ Program 2 (idx=2)    Program 3 (idx=3)                  │
│ slot = -1 (skip)     slot = 20                          │
│                      k_cache[20] = k₃                   │
└─────────────────────────────────────────────────────────┘

输出:
k_cache: [..., k₀@10, ..., k₁@15, ..., k₃@20, ...]
```

### 3.3 store_kvcache 封装函数

```python
def store_kvcache(key: torch.Tensor, value: torch.Tensor, 
                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                  slot_mapping: torch.Tensor):
    """
    将 K/V 张量写入 KV Cache 的封装函数
    
    启动 Triton kernel，每个 token 由一个 GPU 线程块处理。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 展平后的维度
    
    # 验证张量布局连续性
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # 最后一维连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 第二维连续
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # KV Cache 布局匹配
    
    assert slot_mapping.numel() == N  # slot_mapping 长度必须等于 token 数
    
    # 启动 kernel：N 个程序实例，每个处理一个 token
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D
    )
```

#### 参数说明

```
输入张量布局:
─────────────────────────────────────────────────────────

key: (num_tokens, num_heads, head_dim)
     └─ 例如：(1024, 32, 128)

value: (num_tokens, num_kv_heads, head_dim)
       └─ 例如：(1024, 8, 128)  (GQA)

k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
         └─ 例如：(256, 256, 8, 128)

v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
         └─ 例如：(256, 256, 8, 128)

slot_mapping: (num_tokens,)
              └─ 例如：(1024,) 每个元素是 KV Cache 的线性索引
```

#### slot_mapping 详解

```
slot_mapping 的作用:
─────────────────────────────────────────────────────────

将 token 索引映射到 KV Cache 的物理位置

示例:
输入 token:     [0, 1, 2, 3, 4]
                │  │  │  │  │
slot_mapping:   [5, 8, -1, 12, 15]
                │  │  │  │  │
                ▼  ▼  │  ▼  ▼
KV Cache 位置： [5] [8] │ [12] [15]
                    │
                    └─ -1 表示不存储

为什么需要 slot_mapping?
1. KV Cache 是分块管理的
2. token 可能存储在不同的块中
3. 需要灵活的映射关系
```

```
slot_mapping 计算示例:
─────────────────────────────────────────────────────────

假设:
- block_size = 256 (每个块 256 个 token)
- 序列 0: 占用块 [0, 1, 2]
- 序列 1: 占用块 [3, 4]

序列 0 的 token (0-511):
token 0-255:   slot = 0*256 + token_idx = 0-255
token 256-511: slot = 1*256 + (token_idx-256) = 256-511

序列 1 的 token (0-255):
token 0-255:   slot = 3*256 + token_idx = 768-1023

完整 slot_mapping:
[0, 1, 2, ..., 255, 256, 257, ..., 511, 768, 769, ..., 1023]
 └──── 序列 0 块 0 ────┘ └──── 序列 0 块 1 ────┘ └─ 序列 1 ──┘
```

---

## 四、Attention 类详解

### 4.1 类结构

```python
class Attention(nn.Module):
    """
    注意力模块
    
    封装 Flash Attention，支持：
    - Prefill 模式：处理整个序列
    - Decode 模式：每次生成一个 token
    - 前缀缓存：重用已计算的前缀 KV Cache
    """
    
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.num_heads = num_heads        # 注意力头数
        self.head_dim = head_dim          # 每个头的维度
        self.scale = scale                # 缩放因子 (1/sqrt(head_dim))
        self.num_kv_heads = num_kv_heads  # KV 头数 (GQA 支持)
        self.k_cache = self.v_cache = torch.tensor([])  # KV Cache
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 核心注意力计算
        ...
```

### 4.2 forward 方法详解

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    执行注意力计算
    
    根据上下文状态自动选择 Prefill 或 Decode 模式。
    
    Args:
        q: Query 张量
        k: Key 张量
        v: Value 张量
    
    Returns:
        o: 注意力输出张量
    """
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache
    
    # 步骤 1：如果 KV Cache 已分配，将当前 K/V 写入 Cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    
    # 步骤 2：根据模式选择注意力计算方式
    if context.is_prefill:
        # Prefill 模式
        o = flash_attn_varlen_func(...)
    else:
        # Decode 模式
        o = flash_attn_with_kvcache(...)
    
    return o
```

### 4.3 步骤 1：写入 KV Cache

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

**何时执行**：
- KV Cache 已分配（非空）
- 每次有新的 K/V 需要存储

**何时不执行**：
- KV Cache 未分配（空张量）
- 仅使用 KV Cache 推理（如纯 Decode 阶段）

**写入流程**：
```
输入:
k: (num_tokens, num_kv_heads, head_dim)
v: (num_tokens, num_kv_heads, head_dim)
slot_mapping: (num_tokens,)

Triton Kernel 并行执行:
for each token i:
    slot = slot_mapping[i]
    if slot != -1:
        k_cache[slot] = k[i]
        v_cache[slot] = v[i]

输出:
k_cache, v_cache 更新
```

### 4.4 步骤 2：Prefill 模式

```python
if context.is_prefill:
    # Prefill 模式：处理整个序列
    if context.block_tables is not None:
        # 有前缀缓存：需要从 KV Cache 中加载历史 K/V
        k, v = k_cache, v_cache
    
    # 使用 Flash Attention 的变长序列版本
    o = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=context.max_seqlen_q,      # 最大 query 序列长度
        cu_seqlens_q=context.cu_seqlens_q,      # query 累积序列长度
        max_seqlen_k=context.max_seqlen_k,      # 最大 key 序列长度
        cu_seqlens_k=context.cu_seqlens_k,      # key 累积序列长度
        softmax_scale=self.scale,               # 缩放因子
        causal=True,                            # 因果掩码
        block_table=context.block_tables,       # 块表（前缀缓存时使用）
    )
```

#### Prefill 模式参数详解

```
cu_seqlens (累积序列长度):
─────────────────────────────────────────────────────────

作用：标记多个序列的边界

示例：3 个序列
序列 0: 长度 100 (token 0-99)
序列 1: 长度 50  (token 100-149)
序列 2: 长度 80  (token 150-229)

cu_seqlens_q = cu_seqlens_k = [0, 100, 150, 229]
               └─ 起始位置
                  └─ 序列 0 结束 (序列 1 起始)
                     └─ 序列 1 结束 (序列 2 起始)
                        └─ 序列 2 结束

总 token 数：229
```

```
图示:
─────────────────────────────────────────────────────────

cu_seqlens: [0, 100, 150, 229]
            │   │    │    │
            │   │    │    └─ 序列 2 结束
            │   │    └─ 序列 1 结束
            │   └─ 序列 0 结束
            └─ 起始位置

token 索引：
序列 0: [0, 1, 2, ..., 99]
序列 1: [100, 101, ..., 149]
序列 2: [150, 151, ..., 228]
```

#### block_tables (前缀缓存)

```
block_tables 的作用:
─────────────────────────────────────────────────────────

当启用前缀缓存时，KV Cache 不是连续的，而是分块存储的。
block_tables 告诉 Flash Attention 每个序列的 KV Cache 块在哪里。

示例:
序列 0: 占用块 [0, 1, 2]
序列 1: 占用块 [3, 4]
序列 2: 占用块 [0, 1, 5]  (共享序列 0 的前缀)

block_tables = [
    [0, 1, 2],  # 序列 0 的块
    [3, 4],     # 序列 1 的块
    [0, 1, 5],  # 序列 2 的块 (共享前缀)
]
```

```
前缀缓存共享:
─────────────────────────────────────────────────────────

序列 0: "Hello, how are you?"
序列 1: "Hello, how are you doing?"
        └─────┬─────┘
          相同前缀

KV Cache 布局:
块 0: "Hello, "     (序列 0 和 2 共享)
块 1: "how are "    (序列 0 和 2 共享)
块 2: "you?"        (仅序列 0)
块 3: "you"         (仅序列 1)
块 4: " doing?"     (仅序列 1)
块 5: "today?"      (仅序列 2)

block_tables:
序列 0: [0, 1, 2]
序列 1: [0, 1, 3, 4]
序列 2: [0, 1, 5]
```

#### flash_attn_varlen_func

```
函数签名:
─────────────────────────────────────────────────────────

flash_attn_varlen_func(
    q, k, v,                    # Q, K, V 张量
    cu_seqlens_q,               # Query 累积序列长度
    cu_seqlens_k,               # Key 累积序列长度
    max_seqlen_q,               # 最大 Query 序列长度
    max_seqlen_k,               # 最大 Key 序列长度
    softmax_scale,              # 缩放因子
    causal,                     # 因果掩码
    block_table,                # 块表 (可选)
)

返回:
    o: 注意力输出
```

### 4.5 步骤 2：Decode 模式

```python
else:
    # Decode 模式：每次只生成一个 token
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),                         # (batch, 1, num_heads, head_dim)
        k_cache, v_cache,                       # KV Cache
        cache_seqlens=context.context_lens,     # 每个序列的上下文长度
        block_table=context.block_tables,       # 块表
        softmax_scale=self.scale,
        causal=True,
    )
```

#### Decode 模式参数详解

```
q.unsqueeze(1):
─────────────────────────────────────────────────────────

输入 q: (batch_size, num_heads, head_dim)
       └─ 例如：(4, 32, 128)

unsqueeze(1) 后：(batch_size, 1, num_heads, head_dim)
                └─ 例如：(4, 1, 32, 128)
                   └─ 添加序列长度维度 (1 个 token)

为什么需要？
flash_attn_with_kvcache 期望 q 有序列长度维度
```

```
cache_seqlens (上下文长度):
─────────────────────────────────────────────────────────

作用：告诉 Flash Attention 每个序列在 KV Cache 中有多少历史 token

示例：4 个序列
序列 0: 已有 100 个 token
序列 1: 已有 50 个 token
序列 2: 已有 80 个 token
序列 3: 已有 120 个 token

cache_seqlens = [100, 50, 80, 120]

Flash Attention 会使用这些长度的历史 KV 来计算注意力
```

```
block_tables (Decode 模式):
─────────────────────────────────────────────────────────

作用：定位每个序列的 KV Cache 块

示例:
序列 0: 块 [0, 1, 2]  (100 个 token)
序列 1: 块 [3, 4]     (50 个 token)
序列 2: 块 [5, 6]     (80 个 token)
序列 3: 块 [7, 8]     (120 个 token)

block_tables = [
    [0, 1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
]
```

#### flash_attn_with_kvcache

```
函数签名:
─────────────────────────────────────────────────────────

flash_attn_with_kvcache(
    q,                              # Query (batch, 1, heads, head_dim)
    k_cache, v_cache,               # KV Cache
    cache_seqlens,                  # 每个序列的上下文长度
    block_table,                    # 块表
    softmax_scale,                  # 缩放因子
    causal,                         # 因果掩码
)

返回:
    o: 注意力输出 (batch, 1, heads, head_dim)
```

---

## 五、Prefill vs Decode 对比

### 5.1 完整对比表

| 特性 | Prefill 模式 | Decode 模式 |
|------|-------------|------------|
| **触发条件** | `context.is_prefill = True` | `context.is_prefill = False` |
| **输入 q** | `(total_q, heads, dim)` | `(batch, heads, dim)` |
| **输入 k/v** | `(total_k, kv_heads, dim)` | `(batch, kv_heads, dim)` |
| **KV Cache 写入** | 写入所有 token | 只写入新 token |
| **Flash Attention** | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| **序列长度** | 变长 (cu_seqlens) | 固定为 1 |
| **历史 KV** | 从输入 k/v 或 block_table 加载 | 从 k_cache/v_cache 读取 |
| **计算复杂度** | O(n²) | O(1) |
| **使用场景** | 处理用户 prompt | 逐 token 生成 |

### 5.2 数据流对比

```
Prefill 模式数据流:
─────────────────────────────────────────────────────────

输入："Hello, how are you" (5 个 token)

1. QKV 投影:
   q, k, v = qkv_proj(hidden_states)
   形状：(5, num_heads, head_dim)

2. 写入 KV Cache:
   store_kvcache(k, v, k_cache, v_cache, slot_mapping)
   KV Cache 存储 5 个 token 的 KV

3. Flash Attention:
   o = flash_attn_varlen_func(q, k, v, ...)
   计算 5 个 token 的注意力输出

4. 输出:
   每个 token 的注意力结果
```

```
Decode 模式数据流:
─────────────────────────────────────────────────────────

输入："today" (1 个 token，上一个生成的)

1. QKV 投影:
   q, k, v = qkv_proj(hidden_states)
   形状：(1, num_heads, head_dim)

2. 写入 KV Cache:
   store_kvcache(k, v, k_cache, v_cache, slot_mapping)
   KV Cache 只存储 1 个新 token 的 KV

3. Flash Attention:
   o = flash_attn_with_kvcache(q, k_cache, v_cache, ...)
   使用历史 KV Cache + 新 KV 计算注意力

4. 输出:
   只有最后一个 token 的注意力结果
```

### 5.3 模式切换时机

```
完整推理过程的模式切换:
─────────────────────────────────────────────────────────

用户输入："Hello, how are"
期望输出："Hello, how are you today"

Step 1 - Prefill:
─────────────────────────────────────────────────────────
context.is_prefill = True
输入：["Hello", "how", "are"]
处理：计算所有 token 的 KV Cache
输出：生成第一个 token "you"

Step 2 - Decode (第 1 次):
─────────────────────────────────────────────────────────
context.is_prefill = False
输入：["you"]
处理：使用历史 KV Cache + 计算新 KV
输出：生成 " today"

Step 3 - Decode (第 2 次):
─────────────────────────────────────────────────────────
context.is_prefill = False
输入：[" today"]
处理：使用历史 KV Cache + 计算新 KV
输出：生成 "<EOS>"

结束
```

---

## 六、完整执行流程

### 6.1 端到端流程图

```mermaid
flowchart TB
    subgraph 输入层
        Q[Query q]
        K[Key k]
        V[Value v]
    end
    
    subgraph KV Cache 管理
        Check{KV Cache<br/>已分配？}
        Store[store_kvcache<br/>写入 KV Cache]
        KCache[k_cache]
        VCache[v_cache]
    end
    
    subgraph 模式判断
        IsPrefill{is_prefill?}
    end
    
    subgraph Prefill 模式
        CheckBlock{有前缀缓存？}
        LoadCache[从 KV Cache<br/>加载历史 K/V]
        FlashVarlen[flash_attn_varlen_func]
    end
    
    subgraph Decode 模式
        Unsqueeze[q.unsqueeze(1)]
        FlashKVCache[flash_attn_with_kvcache]
    end
    
    subgraph 输出
        O[注意力输出 o]
    end
    
    Q --> Check
    K --> Check
    V --> Check
    
    Check -->|是 | IsPrefill
    Check -->|否 | IsPrefill
    
    IsPrefill -->|True| CheckBlock
    CheckBlock -->|是 | LoadCache
    LoadCache --> FlashVarlen
    CheckBlock -->|否 | FlashVarlen
    
    IsPrefill -->|False| Unsqueeze
    Unsqueeze --> FlashKVCache
    
    FlashVarlen --> O
    FlashKVCache --> O
    
    K -.-> Store
    V -.-> Store
    Store --> KCache
    Store --> VCache
    KCache -.-> FlashKVCache
    VCache -.-> FlashKVCache
```

### 6.2 数值示例

```
完整示例 (简化版):
─────────────────────────────────────────────────────────

配置:
- num_heads = 2
- num_kv_heads = 1 (GQA)
- head_dim = 4
- batch_size = 2

Prefill 阶段:
─────────────────────────────────────────────────────────

输入序列:
序列 0: "Hello" (1 个 token)
序列 1: "How are you" (3 个 token)

输入张量:
q: (4, 2, 4)  # 总共 4 个 token
k: (4, 1, 4)
v: (4, 1, 4)

cu_seqlens_q = cu_seqlens_k = [0, 1, 4]
              └─ 序列 0: token 0
                 └─ 序列 1: token 1-3

写入 KV Cache:
slot_mapping = [0, 1, 2, 3]
k_cache[0:4] = k
v_cache[0:4] = v

Flash Attention:
o = flash_attn_varlen_func(q, k, v, cu_seqlens=[0,1,4], ...)
输出：o (4, 2, 4)

Decode 阶段 (第 1 次):
─────────────────────────────────────────────────────────

输入:
序列 0 生成: " there"
序列 1 生成: "?"

q: (2, 2, 4)  # 2 个序列，每个 1 个 token
k: (2, 1, 4)
v: (2, 1, 4)

写入 KV Cache:
slot_mapping = [4, 5]
k_cache[4:6] = k  # 追加到 KV Cache
v_cache[4:6] = v

context_lens = [1, 3]  # 序列 0 有 1 个历史 token，序列 1 有 3 个

Flash Attention:
o = flash_attn_with_kvcache(
    q.unsqueeze(1),  # (2, 1, 2, 4)
    k_cache, v_cache,
    cache_seqlens=[1, 3],
    block_table=[[0], [1, 2, 3]],
)
输出：o (2, 1, 2, 4)
```

---

## 七、关键设计决策

### 7.1 为什么使用 Flash Attention？

```
标准 Attention vs Flash Attention:
─────────────────────────────────────────────────────────

标准 Attention:
1. 计算 QK^T (O(n²) 内存)
2. Softmax
3. 乘以 V

问题:
- 需要 O(n²) 内存存储 attention matrix
- 长序列时内存爆炸
- 多次 GPU 内存访问

Flash Attention:
1. 分块计算
2. 在线 softmax
3. 单次 GPU 内存访问

优势:
- O(n) 内存复杂度
- 支持长序列
- 2-3 倍速度提升
```

### 7.2 为什么区分 Prefill 和 Decode？

```
Prefill 和 Decode 的不同需求:
─────────────────────────────────────────────────────────

Prefill:
- 处理整个 prompt
- 所有 token 都需要计算
- 需要变长序列支持
- 使用 flash_attn_varlen_func

Decode:
- 只处理 1 个新 token
- 需要访问历史 KV Cache
- 需要增量更新
- 使用 flash_attn_with_kvcache

分开处理的原因:
1. 性能优化（不同的 kernel）
2. 内存管理（KV Cache 使用方式不同）
3. API 差异（Flash Attention 提供不同接口）
```

### 7.3 为什么使用 Triton 写入 KV Cache？

```
Triton vs Python 循环:
─────────────────────────────────────────────────────────

Python 循环:
for i in range(num_tokens):
    k_cache[slot_mapping[i]] = k[i]

问题:
- Python 解释器开销
- 无法 GPU 并行
- 内存访问模式不优化

Triton Kernel:
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)
    tl.store(k_cache_ptr + slot * D, key)

优势:
- GPU 并行执行
- 编译优化
- 高效内存访问
- 10-100 倍速度提升
```

---

## 八、性能优化

### 8.1 内存布局优化

```python
# 验证张量布局连续性
assert key.stride(-1) == 1 and value.stride(-1) == 1  # 最后一维连续
assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 第二维连续
assert k_cache.stride(1) == D and v_cache.stride(1) == D  # KV Cache 布局匹配
```

**为什么重要**：
```
连续内存布局:
- GPU 内存访问更高效
- 更好的缓存利用率
- 避免不必要的内存复制

不连续布局会导致:
- 额外的内存复制
- 降低 kernel 性能
- 可能触发断言失败
```

### 8.2 Torch Compile

```python
@torch.compile
def forward(self, q: torch.Tensor, temperatures: torch.Tensor):
    ...
```

**优势**：
- JIT 编译优化
- 融合多个操作
- 减少 Python 开销

---

## 九、总结

### 9.1 核心要点

| 概念 | 说明 |
|------|------|
| **KV Cache** | 避免重复计算，加速自回归生成 |
| **Prefill 模式** | 处理整个 prompt，使用 `flash_attn_varlen_func` |
| **Decode 模式** | 增量生成，使用 `flash_attn_with_kvcache` |
| **Triton Kernel** | 高效并行写入 KV Cache |
| **slot_mapping** | token 到 KV Cache 位置的映射 |
| **block_tables** | 前缀缓存时的块定位表 |

### 9.2 关键公式

```
注意力计算:
─────────────────────────────────────────────────────────

Attention(Q, K, V) = softmax(QK^T / sqrt(d)) × V

Prefill:
o = flash_attn_varlen_func(q, k, v, cu_seqlens, ...)

Decode:
o = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, ...)
```

### 9.3 与其他模块的关系

```
完整注意力层:
─────────────────────────────────────────────────────────

Qwen3Attention (qwen3.py)
│
├── QKVParallelLinear (qkv 投影)
├── RotaryEmbedding (RoPE 位置编码)
└── Attention (本模块，flash attention)
    │
    ├── store_kvcache (triton kernel)
    ├── flash_attn_varlen_func (prefill)
    └── flash_attn_with_kvcache (decode)
```

</content>