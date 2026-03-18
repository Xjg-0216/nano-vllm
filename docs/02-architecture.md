# Nano-vLLM 核心架构详解

## 1. LLMEngine - 推理引擎总控

**文件**: `nanovllm/engine/llm_engine.py`

**职责**: 用户接口层，协调所有组件完成推理任务

**关键流程**:
```python
# 初始化时启动多进程 (张量并行)
for i in range(1, tensor_parallel_size):
    process = Process(target=ModelRunner, args=(config, i, event))
    process.start()

# generate() 主循环
while not is_finished():
    step()  # 调度 → 执行 → 后处理
```

**核心方法**:

| 方法 | 作用 |
|------|------|
| `add_request()` | 将 prompt 转为 Sequence 加入调度队列 |
| `step()` | 执行一步推理 (prefill 或 decode) |
| `generate()` | 完整生成循环，带进度条和吞吐量统计 |

---

## 2. ModelRunner - 模型执行器

**文件**: `nanovllm/engine/model_runner.py`

**职责**: 每个 GPU 一个实例，负责实际模型计算

**多进程通信机制**:
```
┌─────────────┐     共享内存 (SharedMemory)     ┌─────────────┐
│ ModelRunner │ ←─── pickle([method, args]) ───→│ ModelRunner │
│   rank=0    │ ←─────── Event 同步 ────────────│   rank=1    │
└─────────────┘ ←─────── NCCL 分布式 ────────────└─────────────┘
     (主进程)                                       (工作进程)
```

**关键功能**:

### 2.1 KV Cache 分配 (`allocate_kv_cache`)
```python
# 根据 GPU 内存动态计算可用块数
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
num_kvcache_blocks = (可用内存) // block_bytes
```

### 2.2 Prefill vs Decode 准备

**Prefill** (处理整个 prompt，使用 flash attention 格式):
```python
prepare_prefill(seqs):
    input_ids = [所有 cached_tokens 之后的 token]
    cu_seqlens_q/k = [累积序列长度]
    slot_mapping = [block_table 映射到 KV cache 的位置]
```

**Decode** (单 token 生成):
```python
prepare_decode(seqs):
    input_ids = [每个 seq 的最后一个 token]
    context_lens = [每个 seq 的总长度]
    block_tables = [每个 seq 的块表]
```

### 2.3 CUDA 图捕获 (`capture_cudagraph`)
```python
# 为不同 batch size 预捕获计算图 (1,2,4,8,16,32...)
for bs in [1,2,4,8,16...]:
    graph = CUDAGraph()
    with cuda_graph(graph):
        outputs[:bs] = model(input_ids[:bs], ...)
    self.graphs[bs] = graph
```

---

## 3. Scheduler - 序列调度器

**文件**: `nanovllm/engine/scheduler.py`

**职责**: 决定哪些序列参与当前 step，管理 prefill/decode 阶段切换

**调度策略**:
```python
schedule():
    # 阶段 1: 优先 prefill (新请求)
    while waiting 队列:
        if 未超过 max_num_seqs 且 未超过 max_num_batched_tokens:
            分配 KV cache 块
            移到 running 队列
        else:
            break
    
    # 阶段 2: decode (进行中的请求)
    while running 队列:
        if KV cache 不足:
            preempt(抢占)  # 释放块，移回 waiting
        else:
            扩展块表，加入 decode 批次
```

**抢占机制** (`preempt`):
```python
def preempt(seq):
    seq.status = WAITING
    block_manager.deallocate(seq)  # 释放块 (引用计数减 1)
    waiting.appendleft(seq)        # 插队到队首
```

---

## 4. BlockManager - KV Cache 块管理器

**文件**: `nanovllm/engine/block_manager.py`

**职责**: 管理 GPU 显存中的 KV Cache 块，支持前缀缓存优化

**核心数据结构**:
```
Block:
  - block_id: 块 ID
  - ref_count: 引用计数 (支持多序列共享)
  - hash: token_ids 的哈希值
  - token_ids: 实际 token 列表

free_block_ids: [0, 1, 2, ...]  # 空闲块队列
used_block_ids: {5, 7, 9}       # 已用块集合
hash_to_block_id: {hash1: 3, hash2: 8}  # 哈希 → 块映射
```

### 4.1 前缀缓存原理 (`allocate`)
```python
# 为每个完整块计算哈希 (链式哈希，包含前缀)
h = compute_hash(token_ids, prefix_hash)

# 检查是否已缓存
if h in hash_to_block_id:
    # Cache Hit: 共享已有块
    block.ref_count += 1
    seq.num_cached_tokens += block_size
else:
    # Cache Miss: 分配新块
    block = allocate_new_block()
    block.update(h, token_ids)
    hash_to_block_id[h] = block_id
```

### 4.2 引用计数管理
```python
deallocate(seq):
    for block_id in seq.block_table:
        block.ref_count -= 1
        if ref_count == 0:
            free_block_ids.append(block_id)  # 真正释放
```

---

## 5. Sequence - 序列表示

**文件**: `nanovllm/engine/sequence.py`

**职责**: 封装单个请求的完整状态

**关键属性**:
```python
Sequence:
  seq_id              # 唯一标识
  status              # WAITING/RUNNING/FINISHED
  token_ids           # 完整 token 列表 (prompt + completion)
  num_prompt_tokens   # prompt 长度
  num_cached_tokens   # 前缀缓存命中的 token 数
  block_table         # [block_id1, block_id2, ...] KV 块索引表
  temperature         # 采样温度
  max_tokens          # 最大生成长度
```

### 5.1 块表映射示例
```
假设 block_size = 256, seq 有 600 个 token:
  num_blocks = ceil(600 / 256) = 3
  block_table = [5, 8, 12]
  
  block(0) → token_ids[0:256]    → 映射到 KV cache 块 5
  block(1) → token_ids[256:512]  → 映射到 KV cache 块 8
  block(2) → token_ids[512:600]  → 映射到 KV cache 块 12 (不满)
```

### 5.2 序列化优化 (`__getstate__`/`__setstate__`)
```python
# 跨进程传递时只传必要数据
__getstate__():
    if 只有 prompt (无生成):
        传完整 token_ids
    else:
        只传 last_token (节省内存)
```

---

## 组件协作流程

```
用户调用 generate()
       │
       ▼
┌──────────────────┐
│   LLMEngine      │ add_request() → Sequence
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Scheduler      │ schedule() → 选择 seqs, 分配块
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  BlockManager    │ allocate()/may_append()
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ModelRunner     │ run() → 模型前向 → 采样
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Scheduler      │ postprocess() → 检查结束条件
└──────────────────┘
```

---

## 关键设计决策总结

| 组件 | 设计决策 | 优势 |
|------|----------|------|
| **张量并行** | 多进程 + NCCL + 共享内存 | 支持多 GPU 扩展，通信高效 |
| **KV Cache** | 分块管理 + 前缀缓存 | 内存利用率高，支持 cache 复用 |
| **调度策略** | Prefill 优先 + 抢占式 | 保证新请求响应速度 |
| **CUDA 图** | 多 batch size 预捕获 | 减少 launch 开销，提升吞吐 |
