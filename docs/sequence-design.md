# Sequence 设计说明

## 问题：为什么叫 Sequence 而不是 Request？

在 nano-vLLM（以及原始 vLLM）中，使用 `Sequence` 类来表示推理的基本单位，而不是 `Request`。这背后的设计考量是什么？

## 当前实现：1:1 关系

在当前的 nano-vLLM 实现中，**一个请求（Request）对应一个序列（Sequence）**：

```python
# 用户侧
llm.generate(["Hello"], sampling_params)  # 提交一个请求

# 引擎侧（llm_engine.py 中）
for prompt in prompts:
    seq = Sequence(prompt_token_ids, sampling_params)  # 1 个 prompt → 1 个 Sequence
    scheduler.add_sequence(seq)
```

| 概念 | 说明 |
|------|------|
| **Request** | 用户调用 `llm.generate()` 时提交的一个请求 |
| **Sequence** | 当前实现中，每个 request 被包装成一个 Sequence 对象 |

## 设计扩展：1:N 关系

使用 `Sequence` 而非 `Request` 的术语，是为了**支持未来高级采样功能**，在这些场景下，**一个 prompt 可以对应多个输出序列**。

### 场景 1：Beam Search（束搜索）

```
beam_size = 3 时，同一个 prompt 同时维护 3 个候选序列：

prompt: "Hello"
    → Sequence 1: "Hello, how are you?"
    → Sequence 2: "Hello, what can I do?"
    → Sequence 3: "Hello, nice to meet you!"

最终选择得分最高的序列作为输出
```

每一步解码时，3 个序列共享相同的 KV Cache（prompt 部分），但 completion 部分各自独立。

### 场景 2：Parallel Sampling（并行采样）

```python
# n=4 时，同一个 prompt 生成 4 个不同的输出（用于多样性）
llm.generate(["Hello"], sampling_params, n=4)

# 内部创建 4 个 Sequence，共享同一个 prompt
prompt_token_ids: [1, 2, 3]  # 共享
    → Sequence 1: [1, 2, 3, 10, 20, 30]  # 不同采样结果
    → Sequence 2: [1, 2, 3, 15, 25, 35]
    → Sequence 3: [1, 2, 3, 12, 22, 32]
    → Sequence 4: [1, 2, 3, 18, 28, 38]
```

### 场景 3：Best-of-N 采样

```
生成 5 个候选序列，选择逻辑得分最高的：

prompt: "Once upon a time"
    → Sequence 1: temperature=0.5 → "there was a king..."
    → Sequence 2: temperature=0.8 → "there lived a dragon..."
    → Sequence 3: temperature=1.0 → "in a faraway land..."
    → ...
```

## vLLM 的完整设计：SequenceGroup

在完整的 vLLM 实现中，有 `SequenceGroup` 概念来管理 1:N 的关系：

```
SequenceGroup (一个请求)
├── Sequence 1 (beam/sampling 候选 1)
├── Sequence 2 (beam/sampling 候选 2)
└── Sequence 3 (beam/sampling 候选 3)
```

- **SequenceGroup**：代表用户的一个请求，管理多个 Sequence
- **Sequence**：推理的基本单位，每个有独立的 KV Cache 和状态

## nano-vLLM 的简化

nano-vLLM 为了代码清晰和简洁，**省略了 `SequenceGroup` 层**，直接采用 1:1 映射：

```
Request → Sequence (直接对应)
```

## 设计收益

使用 `Sequence` 术语的设计带来以下好处：

| 好处 | 说明 |
|------|------|
| **术语准确** | Sequence 准确描述了"一个 token 序列"的本质 |
| **扩展性** | 未来添加 beam search、parallel sampling 时，只需增加 SequenceGroup 层 |
| **与 vLLM 对齐** | 保持与原始 vLLM 一致的术语体系，便于理解和迁移 |
| **KV Cache 管理** | 每个 Sequence 有独立的 block_table，便于细粒度管理 |

## 核心字段：`num_tokens` vs `num_prompt_tokens`

在 `Sequence` 类中，有两个关键字段用于追踪 token 数量，它们的区别如下：

### 定义

```python
# 当前总 token 数（prompt + completion）
self.num_tokens = len(self.token_ids)

# prompt token 数量，用于区分 prompt 和生成部分
self.num_prompt_tokens = len(token_ids)
```

### 区别对比

| 字段 | 含义 | 是否变化 | 用途 |
|------|------|----------|------|
| `num_prompt_tokens` | **初始 prompt 的长度** | ❌ 固定不变 | 区分 prompt 和 completion 的边界 |
| `num_tokens` | **当前序列的总长度** | ✅ 随生成递增 | 判断是否达到 `max_tokens`、分配 KV Cache |

### 状态演变

**初始状态（刚创建时）：**
```python
seq = Sequence([1, 2, 3, 4, 5])  # prompt 有 5 个 tokens

# 初始化时：
self.num_tokens      = 5  # 当前总 token 数
self.num_prompt_tokens = 5  # prompt token 数（固定不变）
```
此时两者**相等**，因为还没有生成任何 completion tokens。

**推理过程中（decode 阶段）：**
```python
# 生成了 3 个新 tokens 后
seq.append_token(10)  # 第 6 个 token
seq.append_token(20)  # 第 7 个 token
seq.append_token(30)  # 第 8 个 token

# 此时：
self.num_tokens      = 8  # 当前总 token 数（5 + 3）
self.num_prompt_tokens = 5  # prompt token 数（始终不变）
```

### 相关属性的使用

```python
@property
def num_completion_tokens(self):
    """生成的 token 数 = 总数 - prompt 数"""
    return self.num_tokens - self.num_prompt_tokens  # 8 - 5 = 3

@property
def prompt_token_ids(self):
    """prompt 部分：[1, 2, 3, 4, 5]"""
    return self.token_ids[:self.num_prompt_tokens]

@property
def completion_token_ids(self):
    """生成部分：[10, 20, 30]"""
    return self.token_ids[self.num_prompt_tokens:]
```

### 图示

```
token_ids: [1, 2, 3, 4, 5, 10, 20, 30]
            ├─────────┬─────────────┤
            │         │             │
            ↓         ↓             ↓
        prompt    completion    total
        (5 个)     (3 个)        (8 个)
          ↑           ↑           ↑
          │           │           │
    num_prompt   num_tokens -  num_tokens
     _tokens     num_prompt_tokens
```

> **总结**：`num_prompt_tokens` 是**锚点**，用于划分 prompt 和 completion；`num_tokens` 是**计数器**，追踪序列的实时长度。

## 总结

> **当前 nano-vLLM 是简化的 1:1 实现，但使用 `Sequence` 这个术语为未来扩展（beam search、parallel sampling）留下了设计空间。**

这种设计体现了"**简单但不简陋**"的原则：
- 代码保持简洁（约 1200 行）
- 架构设计保留了扩展能力
- 术语与工业级 vLLM 保持一致
