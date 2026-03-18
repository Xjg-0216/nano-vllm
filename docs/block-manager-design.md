# Block Manager 设计说明

## 概述

BlockManager 是 nano-vLLM 中负责管理 KV Cache 物理内存块的核心组件。它通过**前缀缓存（Prefix Caching）**和**引用计数**技术，实现高效的 GPU 内存复用。

## 核心概念

### 1. 物理块 vs 逻辑块

```
物理块（Physical Block）
├── 固定大小的 GPU 内存区域
├── 容量：block_size tokens（默认 256）
└── 由 BlockManager 统一管理

逻辑块（Logical Block）
├── 序列视角的块划分
├── 第 i 个逻辑块 → 映射到某个物理块
└── 通过 block_table 建立映射关系
```

### 2. Block Table（块表）

每个 Sequence 维护一个 `block_table`，记录逻辑块到物理块的映射：

```python
# 序列的 block_table
seq.block_table = [5, 3, 8]

# 含义：
# 逻辑块 0 → 物理块 5
# 逻辑块 1 → 物理块 3
# 逻辑块 2 → 物理块 8
```

## 前缀缓存（Prefix Caching）

### 问题背景

在 LLM 推理中，多个请求可能共享相同的前缀：

```
请求 1: "请介绍一下 Python" → "Python 是一种编程语言..."
请求 2: "请介绍一下 Python 的历史" → "Python 的历史始于 1989 年..."
请求 3: "请介绍一下 Python 的应用场景" → "Python 可用于 Web 开发..."
         ↑ 相同前缀
```

如果每个请求都独立计算 KV Cache，会造成大量重复计算。前缀缓存通过**哈希去重**复用相同前缀的 KV Cache。

### 链式哈希计算

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
    """
    计算 token 序列的哈希值，支持链式计算。
    
    链式哈希确保：
    - 相同前缀的序列产生相同哈希
    - 哈希可以增量计算，无需重新处理完整数据
    """
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 前缀哈希
    h.update(np.array(token_ids).tobytes())     # 当前块 token
    return h.intdigest()
```

### `h.update()` 详解

`update()` 是哈希算法的**增量更新**操作，用于分块计算哈希。

#### 两种哈希计算方式

**方式 1：一次性计算**
```python
data = b"hello world"
h = xxhash.xxh64(data)  # 直接计算完整数据
```

**方式 2：增量计算（`update`）**
```python
h = xxhash.xxh64()      # 创建空哈希对象
h.update(b"hello ")     # 更新第一部分
h.update(b"world")      # 更新第二部分
result = h.intdigest()  # 得到最终哈希
```

**关键：两种方式结果相同！**

#### 链式哈希原理

```
块 1: [1, 2, 3, ..., 256]
    → hash1 = xxh64([1,2,3,...,256])

块 2: [257, 258, ..., 512]
    → hash2 = xxh64(hash1 + [257,258,...,512])
              ↑
         前缀哈希（8 字节）

块 3: [513, 514, ..., 768]
    → hash3 = xxh64(hash2 + [513,514,...,768])
```

#### 图示

```
update 操作流程：

h = xxh64()
       ↓
h.update(prefix_hash)  ← 8 字节（前一块的哈希，小端序）
       ↓
h.update(token_ids)    ← N*4 字节（当前块 token）
       ↓
h.intdigest()          ← 最终 64 位哈希值
```

#### 为什么要用 `update`？

| 原因 | 说明 |
|------|------|
| **内存效率** | 不需要拼接大缓冲区，避免内存拷贝 |
| **流式计算** | 数据可以分片到达，边接收边计算 |
| **链式结构** | 自然支持前缀哈希的嵌套计算 |

#### 对比：不用 `update` 的写法

```python
# 不推荐：需要拼接缓冲区，额外内存拷贝
prefix_bytes = prefix.to_bytes(8, "little")
token_bytes = np.array(token_ids).tobytes()
data = prefix_bytes + token_bytes  # 内存拷贝
h = xxhash.xxh64(data)

# 推荐：增量更新，无额外拷贝
h = xxhash.xxh64()
h.update(prefix.to_bytes(8, "little"))
h.update(np.array(token_ids).tobytes())
```

### 缓存命中流程

```
1. 计算当前块的哈希值
       ↓
2. 查找 hash_to_block_id 字典
       ↓
3. 如果命中且 token_ids 匹配 → 复用物理块（Cache Hit）
   否则 → 分配新物理块（Cache Miss）
```

## 引用计数管理

### 问题背景

多个序列可能共享同一个物理块（前缀缓存命中时），需要追踪每个块被多少序列使用。

### 引用计数规则

```python
# 分配新块时
block.ref_count = 1  # 初始为 1（分配者持有）

# 缓存命中共享时
block.ref_count += 1  # 增加引用

# 回收时
block.ref_count -= 1  # 减少引用
if block.ref_count == 0:
    # 引用归零，真正回收到空闲池
    deallocate_block(block_id)
```

### 状态转换图

```
空闲块 (ref_count=0)
    ↓ allocate()
已用块 (ref_count=1)
    ↓ 其他序列缓存命中
共享块 (ref_count=2, 3, ...)
    ↓ deallocate() × N
空闲块 (ref_count=0) → 回收到空闲池
```

## 块分配策略

### 分配条件

```python
def can_allocate(self, seq: Sequence) -> bool:
    """检查是否有足够空闲块"""
    return len(self.free_block_ids) >= seq.num_blocks
```

### 分配流程

```python
def allocate(self, seq: Sequence):
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = compute_hash(token_ids, prefix)
        
        # 查找缓存
        block_id = hash_to_block_id.get(h, -1)
        
        if cache_miss:
            # 从空闲池分配新块
            block_id = free_block_ids.popleft()
            _allocate_block(block_id)
        else:
            # 复用已有块
            seq.num_cached_tokens += block_size
            block.ref_count += 1
        
        # 注册到哈希表
        hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

## 块追加（动态扩展）

当序列在 decode 阶段增长时，可能需要追加新块。

### 三种情况

```python
def may_append(self, seq: Sequence):
    if len(seq) % block_size == 1:
        # 情况 1：跨块边界，需要分配新块
        # 例如：256 → 257 tokens
        block_id = free_block_ids.popleft()
        _allocate_block(block_id)
        block_table.append(block_id)
        
    elif len(seq) % block_size == 0:
        # 情况 2：最后一个块刚好填满，计算哈希并注册
        # 例如：len = 256, 512, ...
        token_ids = seq.block(seq.num_blocks - 1)
        h = compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        hash_to_block_id[h] = last_block.block_id
        
    else:
        # 情况 3：块内追加，无需操作
        # 例如：257 → 300 tokens，仍在第 2 个块内
        pass
```

### 图示

```
block_size = 256

len=255 → len=256:  情况 2（填满，计算哈希）
len=256 → len=257:  情况 1（跨边界，分配新块）
len=257 → len=300:  情况 3（块内追加，无操作）
len=511 → len=512:  情况 2（填满，计算哈希）
len=512 → len=513:  情况 1（跨边界，分配新块）
```

## 数据结构

| 数据结构 | 类型 | 用途 |
|---------|------|------|
| `blocks` | `list[Block]` | 所有物理块数组 |
| `hash_to_block_id` | `dict[int, int]` | 哈希 → 物理块 ID 映射 |
| `free_block_ids` | `deque[int]` | 空闲块 ID 队列（FIFO） |
| `used_block_ids` | `set[int]` | 已用块 ID 集合（快速查询） |
| `block_table` | `list[int]` | 序列的逻辑→物理块映射 |

## 实际应用场景

### 场景 1：相同系统提示词的多个请求

当批量处理带有相同系统提示词的请求时，前缀缓存自动复用：

```python
# 系统提示词相同
system_prompt = "你是一个有帮助的助手。请回答用户的问题。"

# 多个用户请求共享相同前缀
llm.generate([
    system_prompt + "问题 1",
    system_prompt + "问题 2",
    system_prompt + "问题 3",
])
```

**缓存复用过程：**

```
请求 1: [系统提示词][问题 1][回答 1]
              ↓ 计算并缓存
请求 2: [系统提示词][问题 2][回答 2]
              ↑ 缓存命中！复用
请求 3: [系统提示词][问题 3][回答 3]
              ↑ 缓存命中！复用
```

### 场景 2：多轮对话（应用层拼接历史）

nano-vLLM 当前是"无状态"的，应用层需要手动拼接对话历史：

```python
# 应用层维护对话历史
history = []

# 第 1 轮
history.append(("介绍一下 Python", "Python 是一种编程语言..."))

# 第 2 轮：手动拼接历史
prompt = "对话历史：\n" + "\n".join(history) + "\n它有什么特点？"

# 第 3 轮
history.append(("它有什么特点？", "它的特点是简洁易学..."))
prompt = "对话历史：\n" + "\n".join(history) + "\n适合做什么？"
```

**前缀缓存优化：**

```
第 1 轮：[对话历史 (空)][第 1 轮 Q][第 1 轮 A]
第 2 轮：[对话历史 (第 1 轮)][第 2 轮 Q][第 2 轮 A]
              ↑ 缓存命中！复用第 1 轮
第 3 轮：[对话历史 (第 1+2 轮)][第 3 轮 Q][第 3 轮 A]
              ↑ 缓存命中！复用前两轮
```

### 场景 3：批量翻译/问答

```python
# 批量翻译场景
prompts = [
    "请翻译以下英文：Hello",
    "请翻译以下英文：World",  
    "请翻译以下英文：Python",
    # ↑ 相同前缀："请翻译以下英文："
]

# BlockManager 会：
# 1. 第 1 个请求：计算"请翻译以下英文："的 KV Cache
# 2. 第 2 个请求：复用相同前缀的 KV Cache
# 3. 第 3 个请求：复用相同前缀的 KV Cache
```

### 场景 4：vLLM 原生连续批处理（Continuous Batching）

完整的 vLLM 支持**连续批处理**，在多轮对话中自动管理上下文：

```
第 1 轮：
  Request 1: [Prompt 1] → 生成回答
           ↓
第 2 轮：
  Request 1: [Prompt 1 + Answer 1 + Prompt 2] → 生成回答
           ↓
第 3 轮：
  Request 1: [Prompt 1 + Answer 1 + Prompt 2 + Answer 2 + Prompt 3] → 生成回答
```

**每轮只需计算新增 prompt 的 KV Cache，历史部分通过前缀缓存复用。**

---

## 总结

### 技术收益

| 技术 | 收益 |
|------|------|
| **分块管理** | 灵活的内存分配，避免碎片化 |
| **前缀缓存** | 复用相同前缀的 KV Cache，减少重复计算 |
| **链式哈希** | 增量计算，内存高效 |
| **引用计数** | 安全的多序列共享机制 |

### 应用收益

| 场景 | 优化效果 |
|------|---------|
| 相同系统提示词 | 复用系统提示词 KV Cache |
| 多轮对话 | 复用历史对话 KV Cache |
| 批量翻译/问答 | 复用指令前缀 KV Cache |
| 连续批处理 | 每轮仅计算新增部分 |

> **注意**：nano-vLLM 当前是无状态的，多轮对话需要应用层手动拼接历史。完整的 vLLM 提供原生的连续批处理支持，自动管理对话上下文。
