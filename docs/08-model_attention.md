# Triton 与自定义 Attention Kernel

## 1. @triton.jit 装饰器

### 1.1 什么是 Triton

Triton 是由 OpenAI 开发的 GPU 编程框架，允许使用 Python 编写高性能的 GPU kernel。相比 CUDA，Triton 具有以下优势：

- **Python 语法**：使用 Python 语法编写 GPU kernel，学习曲线平缓
- **自动优化**：编译器自动处理内存合并、共享内存、寄存器分配等优化
- **块级编程模型**：程序员只需关注块内逻辑，编译器自动处理网格划分

### 1.2 @triton.jit 装饰器

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

### 1.3 关键特性

#### tl.constexpr

`tl.constexpr` 表示编译时常量，编译器会为该值生成专用代码：

```python
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE 在编译时确定，可用于循环展开、数组大小等
    offsets = tl.arange(0, BLOCK_SIZE)
```

#### 程序实例（Program Instance）

每个 kernel 实例由多个程序（program）并行执行，通过 `tl.program_id()` 获取当前程序 ID：

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

#### 启动 Kernel

```python
# 启动 N 个程序实例
kernel[(N,)](a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE=1024)
```

### 1.4 nano-vLLM 中的 Triton Kernel

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

### 1.5 为什么使用 Triton

| 对比项 | CUDA | Triton |
|--------|------|--------|
| 编程语言 | C++ | Python |
| 内存管理 | 手动 | 自动 |
| 编译优化 | 手动调优 | 自动优化 |
| 开发效率 | 低 | 高 |
| 运行效率 | 极高 | 接近 CUDA |

在 nano-vLLM 中，使用 Triton 编写 KV Cache 写入 kernel 的原因：
1. **简单高效**：只需几十行 Python 代码即可实现高效的 GPU kernel
2. **避免 Python 开销**：直接在 GPU 上操作，避免 CPU-GPU 数据传输
3. **灵活定制**：可以根据具体需求定制 kernel 逻辑

---

## 2. Attention 机制详解

### 2.1 Attention 类型

nano-vLLM 支持两种 Attention 计算模式：

| 模式 | 使用场景 | Flash Attention API |
|------|----------|---------------------|
| Prefill | 处理 prompt | `flash_attn_varlen_func` |
| Decode | 生成 token | `flash_attn_with_kvcache` |

### 2.2 Prefill 模式

Prefill 阶段处理整个 prompt 序列，使用变长序列注意力：

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

### 2.3 Decode 模式

Decode 阶段每次只生成一个 token，使用 KV Cache 避免重复计算：

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

#### KV Cache 优化

```
传统 Attention:          使用 KV Cache:
-----------------       -----------------
Prompt: [t0, t1, t2]    Prompt: [t0, t1, t2]
  - 计算 Q0, K0, V0       - 计算 Q0, K0, V0 → 存入 Cache
  - 计算 Q1, K1, V1       - 计算 Q1, K1, V1 → 存入 Cache
  - 计算 Q2, K2, V2       - 计算 Q2, K2, V2 → 存入 Cache
                          - 生成 t3: 使用 Cache 中的 K0-K2

Decode: 生成 t3          - 只需计算 Q3, K3, V3
  - 重新计算所有 QKV      - 注意力计算使用 Cache
  - 计算注意力
```

### 2.4 前缀缓存（Prefix Caching）

前缀缓存通过重用已计算的 KV Cache 来优化重复 prompt 的处理：

```python
if context.block_tables is not None:  # 有前缀缓存
    k, v = k_cache, v_cache
```

#### 工作原理

```
请求 1: "今天天气真好" → "，适合出去玩。"
  - 计算并缓存 KV

请求 2: "今天天气真好" → "，但是我要上班。"
  - 检测前缀匹配
  - 直接使用缓存的 KV
  - 只计算新 token 的 KV
```

---

## 3. 上下文管理

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

### 上下文数据结构

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

## 4. 性能优化技巧

### 4.1 Kernel 融合

将多个操作融合到一个 kernel 中，减少 GPU kernel 启动开销和内存访问：

```python
# 融合前：多个 kernel
x = linear(x)        # kernel 1
x = layernorm(x)     # kernel 2
x = activation(x)    # kernel 3

# 融合后：一个 kernel
x = fused_linear_layernorm_activation(x)
```

### 4.2 内存布局优化

```python
# 确保张量布局连续，提高内存访问效率
assert key.stride(-1) == 1  # 最后一维连续
assert key.stride(1) == head_dim  # 第二维连续
```

### 4.3 CUDA Graph

在 Decode 阶段使用 CUDA Graph 捕获计算图，减少 CPU 调度开销：

```python
# 捕获 CUDA Graph
with torch.cuda.graph(graph):
    outputs = model(input_ids, positions)

# 重放 Graph
graph.replay()
```

---

## 5. 总结

1. **@triton.jit** 是 Triton 的核心装饰器，用于编写高性能 GPU kernel
2. **KV Cache 写入** 使用 Triton 实现，每个 token 由一个程序实例处理
3. **Prefill/Decode** 使用不同的 Flash Attention API 优化性能
4. **前缀缓存** 通过重用 KV Cache 优化重复 prompt
5. **上下文管理** 提供统一的运行时信息访问接口
