# Attention 机制详解

## 一、模块概述

`nanovllm/layers/attention.py` 实现了基于 **Flash Attention** 的 KV Cache 管理注意力机制，是 Nano-vLLM 高效推理的核心组件。

### 1.1 核心功能

| 功能 | 说明 |
|------|------|
| **KV Cache 管理** | 使用 Triton kernel 高效写入 KV Cache |
| **Prefill 模式** | 处理整个 prompt 序列，使用 `flash_attn_varlen_func` |
| **Decode 模式** | 增量解码，使用 `flash_attn_with_kvcache` |
| **前缀缓存优化** | 重用已计算的前缀 KV Cache |

### 1.2 两个核心模式

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

## 二、KV Cache 基础

### 2.1 为什么需要 KV Cache

**自回归生成中的重复计算问题：**

```
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

**KV Cache 优化：**

```
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

### 2.2 Prefill vs Decode 对比

| 特性 | Prefill 模式 | Decode 模式 |
|------|-------------|------------|
| **输入** | 完整 prompt (n 个 token) | 上一个 token (1 个) |
| **KV Cache** | 写入所有 token 的 KV | 只写入新 token 的 KV |
| **注意力计算** | 处理所有 token | 只处理最后一个 token |
| **Flash Attention API** | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |

---

## 三、Triton 与自定义 Kernel

### 3.1 什么是 Triton

Triton 是由 OpenAI 开发的 GPU 编程框架，允许使用 Python 编写高性能的 GPU kernel。相比 CUDA，Triton 具有以下优势：

| 优势 | 说明 |
|------|------|
| **Python 语法** | 使用 Python 语法编写 GPU kernel，学习曲线平缓 |
| **自动优化** | 编译器自动处理内存合并、共享内存、寄存器分配等优化 |
| **块级编程模型** | 程序员只需关注块内逻辑，编译器自动处理网格划分 |

### 3.2 @triton.jit 装饰器

`@triton.jit` 是 Triton 的核心装饰器，用于将 Python 函数编译为 GPU kernel。

```python
@triton.jit
def my_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # GPU kernel 代码
    pass
```

#### 关键特性

**tl.constexpr**：编译时常量
```python
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE 在编译时确定，可用于循环展开、数组大小等
    offsets = tl.arange(0, BLOCK_SIZE)
```

**程序实例（Program Instance）**：
```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    pid = tl.program_id(0)  # 获取当前程序 ID
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)
```

**启动 Kernel**：
```python
# 启动 N 个程序实例
kernel[(N,)](a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE=1024)
```

---

## 四、KV Cache 写入 Kernel

### 4.1 store_kvcache_kernel

在 `nanovllm/layers/attention.py` 中，使用 Triton 实现高效的 KV Cache 写入：

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    每个程序实例处理一个 token 的 K/V 向量存储。
    """
    idx = tl.program_id(0)  # 当前处理的 token 索引
    slot = tl.load(slot_mapping_ptr + idx)  # 读取 KV Cache 槽位
    if slot == -1:
        return
    
    # 计算 K/V 在输入张量中的偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # 加载 K/V 数据
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # 计算在 KV Cache 中的偏移并存储
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

### 4.2 Python 封装

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """
    Python 封装函数，启动 Triton kernel。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 展平后的维度
    
    # 启动 N 个程序实例，每个处理一个 token
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D
    )
```

### 4.3 为什么使用 Triton

| 对比项 | CUDA | Triton |
|--------|------|--------|
| 编程语言 | C++ | Python |
| 内存管理 | 手动 | 自动 |
| 编译优化 | 手动调优 | 自动优化 |
| 开发效率 | 低 | 高 |
| 运行效率 | 极高 | 接近 CUDA |

---

## 五、Attention 类实现

### 5.1 类结构

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 如果 KV Cache 已分配，将当前 K/V 写入 Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # Prefill 模式
            if context.block_tables is not None:  # 有前缀缓存
                k, v = k_cache, v_cache
            
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            # Decode 模式
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        
        return o
```

### 5.2 Prefill 模式详解

```python
if context.is_prefill:
    if context.block_tables is not None:  # 有前缀缓存
        k, v = k_cache, v_cache
    
    o = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=context.max_seqlen_q,      # 最大 query 序列长度
        cu_seqlens_q=context.cu_seqlens_q,      # query 累积序列长度
        max_seqlen_k=context.max_seqlen_k,      # 最大 key 序列长度
        cu_seqlens_k=context.cu_seqlens_k,      # key 累积序列长度
        softmax_scale=self.scale,
        causal=True,                            # 因果掩码
        block_table=context.block_tables,       # 块表（前缀缓存时使用）
    )
```

#### 变长序列处理

`flash_attn_varlen_func` 支持批处理多个不同长度的序列：

```
序列 1: [token_0, token_1, token_2]           (长度 3)
序列 2: [token_0, token_1]                    (长度 2)
序列 3: [token_0, token_1, token_2, token_3]  (长度 4)

展平后：[t0, t1, t2, t0, t1, t0, t1, t2, t3]
         |-------| |--| |---------|
           序列 1   序列 2    序列 3

cu_seqlens_q = [0, 3, 5, 9]  # 累积序列长度
```

### 5.3 Decode 模式详解

```python
else:  # decode
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),           # (batch, 1, num_heads, head_dim)
        k_cache, v_cache,         # KV Cache
        cache_seqlens=context.context_lens,  # 每个序列的上下文长度
        block_table=context.block_tables,    # 块表
        softmax_scale=self.scale,
        causal=True,
    )
```

---

## 六、上下文管理

Attention 层通过全局上下文获取运行时信息：

```python
from nanovllm.utils.context import get_context

def forward(self, q, k, v):
    context = get_context()
    
    # Prefill 模式
    if context.is_prefill:
        cu_seqlens_q = context.cu_seqlens_q
        cu_seqlens_k = context.cu_seqlens_k
        ...
    
    # Decode 模式
    else:
        context_lens = context.context_lens
        block_tables = context.block_tables
        ...
```

### Context 数据结构

```python
@dataclass
class Context:
    is_prefill: bool              # 是否为 Prefill 模式
    cu_seqlens_q: torch.Tensor    # Query 累积序列长度
    cu_seqlens_k: torch.Tensor    # Key 累积序列长度
    max_seqlen_q: int             # 最大 Query 序列长度
    max_seqlen_k: int             # 最大 Key 序列长度
    slot_mapping: torch.Tensor    # KV Cache 槽位映射
    context_lens: torch.Tensor    # 上下文长度（Decode 模式）
    block_tables: torch.Tensor    # 块表（Decode 模式/前缀缓存）
```

---

## 七、前缀缓存（Prefix Caching）

### 7.1 原理

前缀缓存通过重用已计算的 KV Cache 来优化重复 prompt 的处理：

```python
if context.block_tables is not None:  # 有前缀缓存
    k, v = k_cache, v_cache
```

### 7.2 工作原理

```
请求 1: "今天天气真好" → "，适合出去玩。"
  - 计算并缓存 KV

请求 2: "今天天气真好" → "，但是我要上班。"
  - 检测前缀匹配
  - 直接使用缓存的 KV
  - 只计算新 token 的 KV
```

### 7.3 应用场景

**场景 1：相同系统提示词的多个请求**

```python
system_prompt = "你是一个有帮助的助手。"

llm.generate([
    system_prompt + "问题 1",
    system_prompt + "问题 2",
    system_prompt + "问题 3",
])
```

**场景 2：多轮对话**

```
第 1 轮：[对话历史 (空)][第 1 轮 Q][第 1 轮 A]
第 2 轮：[对话历史 (第 1 轮)][第 2 轮 Q][第 2 轮 A]
              ↑ 缓存命中！复用第 1 轮
第 3 轮：[对话历史 (第 1+2 轮)][第 3 轮 Q][第 3 轮 A]
              ↑ 缓存命中！复用前两轮
```

---

## 八、性能优化技巧

### 8.1 Kernel 融合

将多个操作融合到一个 kernel 中，减少 GPU kernel 启动开销和内存访问。

### 8.2 内存布局优化

```python
# 确保张量布局连续，提高内存访问效率
assert key.stride(-1) == 1  # 最后一维连续
assert key.stride(1) == head_dim  # 第二维连续
```

### 8.3 CUDA Graph

在 Decode 阶段使用 CUDA Graph 捕获计算图，减少 CPU 调度开销：

```python
# 捕获 CUDA Graph
with torch.cuda.graph(graph):
    outputs = model(input_ids, positions)

# 重放 Graph
graph.replay()
```

---

## 九、总结

### Attention 核心组件

| 组件 | 功能 |
|------|------|
| **store_kvcache** | Triton kernel 写入 KV Cache |
| **flash_attn_varlen_func** | Prefill 模式，变长序列注意力 |
| **flash_attn_with_kvcache** | Decode 模式，使用 KV Cache |
| **Context** | 全局上下文管理 |

### 性能优化

| 技术 | 收益 |
|------|------|
| **KV Cache** | 避免重复计算，加速生成 |
| **Triton Kernel** | 高效的 GPU 实现 |
| **前缀缓存** | 复用相同前缀的 KV Cache |
| **CUDA Graph** | 减少 CPU 调度开销 |
