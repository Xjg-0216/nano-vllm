# 旋转位置编码 (RoPE) 与 LRU Cache

## 1. 旋转位置编码 (RoPE) 核心思想

### 1.1 为什么需要位置编码

Transformer 模型本身是**位置无关**的：
- Self-Attention 计算中，token 之间没有顺序概念
- 打乱输入顺序，Attention 输出不变

但语言是有顺序的：
- "我吃饭" ≠ "饭吃我"
- 需要让模型知道每个 token 的位置

### 1.2 位置编码的演进

| 方法 | 代表模型 | 原理 | 缺点 |
|------|----------|------|------|
| 绝对位置编码 | Transformer, BERT | 学习位置向量并相加 | 无法外推到更长序列 |
| 相对位置编码 | T5, Transformer-XL | 编码 token 间的相对距离 | 实现复杂，计算开销大 |
| **旋转位置编码** | RoFormer, LLaMA, Qwen | 通过旋转操作编码位置 | 支持外推，实现简单 |

### 1.3 RoPE 的核心思想

RoPE（Rotary Positional Embedding）的核心思想是：
> **将位置信息编码为复数空间的旋转角度，通过旋转操作实现相对位置编码**

#### 数学原理

对于位置 $m$ 的 token，其 Query 向量 $Q_m$ 和位置 $n$ 的 Key 向量 $K_n$：

**复数形式**：
$$Q_m = Q \cdot e^{i \cdot m \cdot \theta}$$
$$K_n = K \cdot e^{i \cdot n \cdot \theta}$$

**注意力分数**：
$$Q_m \cdot K_n = Q \cdot K \cdot e^{i \cdot (m-n) \cdot \theta}$$

**关键观察**：注意力分数只依赖于**相对位置** $(m-n)$！

#### 实数域等价形式

将向量分成两半 $[x_1, x_2]$，旋转操作等价于：

$$\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

展开得：
$$y_1 = x_1 \cdot \cos\theta - x_2 \cdot \sin\theta$$
$$y_2 = x_1 \cdot \sin\theta + x_2 \cdot \cos\theta$$

---

### 1.3.1 为什么要在最后一维分成两半？

**核心原因：复数旋转是 2D 操作，需要两个实数来表示一个复数（实部 + 虚部）。**

#### 复数旋转的数学推导

复数旋转公式：
$$z' = z \cdot e^{i\theta} = z \cdot (\cos\theta + i\sin\theta)$$

其中复数 $z = a + bi$ 可以表示为二维向量 $[a, b]$。

展开复数乘法：
$$\begin{align}
z' &= (a + bi) \cdot (\cos\theta + i\sin\theta) \\
&= a\cos\theta + ai\sin\theta + bi\cos\theta + bi^2\sin\theta \\
&= a\cos\theta + ai\sin\theta + bi\cos\theta - b\sin\theta \\
&= (a\cos\theta - b\sin\theta) + i(a\sin\theta + b\cos\theta)
\end{align}$$

所以旋转后的实部和虚部分别是：
$$\begin{bmatrix} a' \\ b' \end{bmatrix} = \begin{bmatrix} a\cos\theta - b\sin\theta \\ a\sin\theta + b\cos\theta \end{bmatrix}$$

#### 为什么是 2D 操作？

```
一个复数 = 实部 + 虚部 = 二维向量 [a, b]

旋转操作需要同时改变实部和虚部：
- 新实部 = 原实部×cos - 原虚部×sin
- 新虚部 = 原实部×sin + 原虚部×cos

因此，每个复数需要 2 个维度来进行旋转。
head_dim 必须是偶数，分成两半：[实部，虚部]
```

#### 可视化示例

```
假设 head_dim = 4，位置θ的旋转：

输入向量：[1.0, 2.0, 3.0, 4.0]
           └─复数 1─┘ └─复数 2─┘

分成两半：
x1 = [1.0, 2.0]  ← 两个复数的实部
x2 = [3.0, 4.0]  ← 两个复数的虚部

假设 cosθ=0.5, sinθ=0.866：

y1 = x1*cos - x2*sin
   = [1.0*0.5 - 3.0*0.866, 2.0*0.5 - 4.0*0.866]
   = [-2.098, -2.464]  ← 新的实部

y2 = x1*sin + x2*cos
   = [1.0*0.866 + 3.0*0.5, 2.0*0.866 + 4.0*0.5]
   = [2.366, 3.732]  ← 新的虚部

输出：[-2.098, -2.464, 2.366, 3.732]
       └────复数 1────┘ └────复数 2────┘
```

#### 成对处理的好处

| 设计 | 说明 |
|------|------|
| **保持向量长度不变** | 旋转是正交变换，不改变向量模长 |
| **多尺度编码** | 不同复数对使用不同频率，捕获多尺度位置信息 |
| **数值稳定性** | 正交变换保持数值稳定 |

```python
# 频率分配：不同复数对使用不同频率
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

# 假设 rotary_dim=4, base=10000
# inv_freq = [1.0, 0.01]

# 第 0 对维度（索引 0,1）：频率 1.0（高频）→ 捕获短距离位置
# 第 1 对维度（索引 2,3）：频率 0.01（低频）→ 捕获长距离位置
```

---

### 1.4 nano-vLLM 中的 RoPE 实现

```python
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码
    
    将输入张量 x 按 RoPE 公式进行旋转编码。
    """
    # 将最后一维分成两半
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    
    # 应用旋转公式
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    # 拼接回完整维度并恢复原始精度
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

#### 频率计算

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        # 计算逆频率：1 / (base^(2i/d))
        # 这是一个几何序列，频率从 1 到 1/base
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 计算所有位置的角度
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 外积
        
        # 计算 cos 和 sin
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 拼接并缓存：[max_pos, 1, rotary_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
```

#### 频率可视化

```
假设 head_dim=4, base=10000

inv_freq = [1/10000^(0/4), 1/10000^(2/4)]
         = [1, 1/100]
         = [1.0, 0.01]

位置 0 的角度：
freqs[0] = [0*1.0, 0*0.01] = [0, 0]
cos[0] = [1, 1]
sin[0] = [0, 0]

位置 100 的角度：
freqs[100] = [100*1.0, 100*0.01] = [100, 1]
cos[100] = [cos(100), cos(1)]
sin[100] = [sin(100), sin(1)]
```

#### 前向传播

```python
@torch.compile
def forward(self, positions, query, key):
    # 根据位置索引查找 cos/sin：[num_tokens, 1, 2*rotary_dim]
    cos_sin = self.cos_sin_cache[positions]
    
    # 分离 cos 和 sin：各为 [num_tokens, 1, rotary_dim]
    cos, sin = cos_sin.chunk(2, dim=-1)
    
    # 应用旋转编码（通过广播自动应用到所有头）
    query = apply_rotary_emb(query, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    
    return query, key
```

---

### 1.5 RoPE 的优势

| 优势 | 说明 |
|------|------|
| **相对位置编码** | 注意力分数只依赖相对位置 $(m-n)$ |
| **支持外推** | 推理时可处理比训练更长的序列 |
| **无需参数** | 不需要学习位置向量，节省参数 |
| **计算高效** | 只需简单的乘加运算 |
| **归纳偏置** | 具有良好的长度泛化能力 |

#### 外推性示例

```
训练时：最大长度 4096
推理时：可以处理 8192 甚至更长

原因：
- RoPE 的频率是预先计算的几何序列
- 位置 m 的角度 = m * inv_freq
- 即使 m 超出训练范围，旋转操作仍然有效

而绝对位置编码：
- 位置向量是学习的，只覆盖训练范围
- 超出范围的位置没有对应的向量
- 无法外推
```

---

## 2. LRU Cache 详解

### 2.1 什么是 LRU Cache

**LRU** = **Least Recently Used**（最近最少使用）

LRU Cache 是一种缓存淘汰策略：
> 当缓存满时，淘汰最长时间未被访问的数据

### 2.2 Python 的 `lru_cache` 装饰器

`functools.lru_cache` 是 Python 标准库提供的函数结果缓存装饰器。

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 第一次调用：计算并缓存
fibonacci(10)  # 计算

# 第二次调用：直接从缓存返回
fibonacci(10)  # 缓存命中
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `maxsize` | 最大缓存条目数（None 表示无限制） |
| `typed` | 是否区分参数类型（默认 False） |

#### 常用方法

```python
# 查看缓存统计
fibonacci.cache_info()
# CacheInfo(hits=2, misses=11, maxsize=128, currsize=11)

# 清除缓存
fibonacci.cache_clear()
```

---

### 2.3 nano-vLLM 中的 LRU Cache 应用

```python
from functools import lru_cache

@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取 RoPE 实例（带缓存）
    
    使用 lru_cache 避免重复创建相同配置的 RoPE 实例。
    """
    assert rope_scaling is None
    
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

#### 为什么使用 `@lru_cache(1)`

**问题场景**：
```python
# 在 Qwen3Attention 初始化时调用
self.rotary_emb = get_rope(
    head_dim,
    rotary_dim=head_dim,
    max_position=max_position,
    base=rope_theta,
)

# 如果有多个 Decoder 层，每层都有一个 Qwen3Attention
# 没有缓存：每层都创建新的 RoPE 实例（浪费内存）
# 有缓存：所有层共享同一个 RoPE 实例（节省内存）
```

#### 内存节省示例

```python
# 假设 32 层 Decoder，每层都有 RoPE
# RoPE 缓存大小：8192 * 256 * 2 * 4 bytes ≈ 16 MB

# 没有缓存：32 * 16 MB = 512 MB
# 使用 lru_cache(1)：16 MB（所有层共享）

# 节省：496 MB
```

---

### 2.4 LRU Cache 工作原理

#### 内部数据结构

```
LRU Cache 内部维护：
1. 字典：key → result 映射
2. 双向链表：记录访问顺序

访问顺序：
[最近访问] ←→ ... ←→ [最久未访问]
     ↑                    ↑
   MRU                  LRU
```

#### 工作流程

```python
@lru_cache(maxsize=3)
def func(x):
    return x * 2

# 调用过程：
func(1)  # 缓存：{1: 2}          链表：[1]
func(2)  # 缓存：{1: 2, 2: 4}    链表：[2, 1]
func(3)  # 缓存：{1: 2, 2: 4, 3: 6}  链表：[3, 2, 1]

# 缓存已满，新调用触发淘汰
func(4)  # 淘汰最久未使用的 1
         # 缓存：{2: 4, 3: 6, 4: 8}  链表：[4, 3, 2]

# 访问已缓存的 key，更新顺序
func(2)  # 缓存命中，更新顺序
         # 链表：[2, 4, 3]
```

---

### 2.5 在 RoPE 中使用 LRU Cache 的优势

| 优势 | 说明 |
|------|------|
| **内存节省** | 所有层共享同一个 RoPE 实例 |
| **初始化加速** | 第二次调用直接返回缓存实例 |
| **自动管理** | 无需手动管理单例模式 |
| **线程安全** | Python 3.2+ 的 lru_cache 是线程安全的 |

#### 为什么是 `maxsize=1`

```python
@lru_cache(1)  # 只缓存 1 个实例
def get_rope(head_size, rotary_dim, max_position, base, rope_scaling):
    ...
```

**原因**：
1. 在单个模型中，RoPE 配置通常是**唯一**的
2. 所有 Decoder 层共享相同的 RoPE 参数
3. 缓存 1 个就足够，避免不必要的内存占用

#### 多模型场景

```python
# 如果加载多个不同配置的模型
llm1 = LLM("model_a")  # head_dim=128, max_position=8192
llm2 = LLM("model_b")  # head_dim=256, max_position=16384

# lru_cache(1) 会保留最后一个配置的 RoPE
# 第一个模型的 RoPE 会被淘汰
# 这是可接受的，因为通常只运行一个模型
```

---

## 3. 完整示例

### 3.1 RoPE 计算示例

```python
import torch
from nanovllm.layers.rotary_embedding import RotaryEmbedding

# 初始化 RoPE
rope = RotaryEmbedding(
    head_size=128,
    rotary_dim=128,
    max_position_embeddings=8192,
    base=1000000
)

# 准备输入
batch_size = 2
seq_len = 10
num_heads = 32
head_dim = 128

positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).flatten()
query = torch.randn(batch_size * seq_len, num_heads, head_dim)
key = torch.randn(batch_size * seq_len, num_heads, head_dim)

# 应用 RoPE
query_rotated, key_rotated = rope(positions, query, key)

# 验证旋转操作
# 位置 0 和位置 1 的 query 应该有不同的旋转角度
assert not torch.allclose(query_rotated[0], query_rotated[1])
```

### 3.2 LRU Cache 验证

```python
from nanovllm.layers.rotary_embedding import get_rope

# 第一次调用：创建新实例
rope1 = get_rope(128, 128, 8192, 1000000)
print(get_rope.cache_info())
# CacheInfo(hits=0, misses=1, maxsize=1, currsize=1)

# 第二次调用：缓存命中
rope2 = get_rope(128, 128, 8192, 1000000)
print(get_rope.cache_info())
# CacheInfo(hits=1, misses=1, maxsize=1, currsize=1)

# 验证是同一个实例
assert rope1 is rope2  # True
```

---

## 4. 总结

### 4.1 RoPE 核心思想

| 要点 | 说明 |
|------|------|
| **相对位置编码** | 注意力分数只依赖相对位置 |
| **旋转操作** | 通过复数旋转编码位置信息 |
| **无需参数** | 预计算频率，无需学习 |
| **支持外推** | 可处理更长序列 |

### 4.2 LRU Cache 核心思想

| 要点 | 说明 |
|------|------|
| **缓存淘汰** | 淘汰最久未使用的数据 |
| **内存优化** | 避免重复创建相同对象 |
| **自动管理** | 无需手动管理单例 |
| **性能提升** | 减少重复计算和初始化 |

### 4.3 在 nano-vLLM 中的应用

```python
# RoPE + LRU Cache 的完美结合
@lru_cache(1)
def get_rope(head_size, rotary_dim, max_position, base, rope_scaling):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

# 所有 Decoder 层共享同一个 RoPE 实例
# 节省内存，加速初始化
```
