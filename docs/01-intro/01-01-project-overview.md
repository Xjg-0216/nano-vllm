# Nano-vLLM 项目上下文

## 项目概述

Nano-vLLM 是一个轻量级的 vLLM 实现，从头开始构建，用于离线大语言模型推理。

**核心特性：**
- 🚀 **快速离线推理** - 与 vLLM 相当的推理速度
- 📖 **可读代码** - 约 1200 行 Python 代码的清晰实现
- ⚡ **优化套件** - 支持前缀缓存、张量并行、Torch 编译、CUDA 图等优化

**主要技术栈：**
- Python 3.10-3.12
- PyTorch + Triton
- Transformers (HuggingFace)
- Flash Attention
- NCCL (多 GPU 通信)

## 项目结构

```
nano-vllm/
├── nanovllm/
│   ├── __init__.py          # 导出 LLM, SamplingParams
│   ├── llm.py               # LLM 类入口 (继承自 LLMEngine)
│   ├── config.py            # Config 数据类
│   ├── sampling_params.py   # SamplingParams 数据类
│   ├── engine/
│   │   ├── llm_engine.py    # LLMEngine 主引擎类
│   │   ├── model_runner.py  # 模型执行器 (支持多进程张量并行)
│   │   ├── scheduler.py     # 序列调度器 (prefill/decode)
│   │   ├── block_manager.py # KV Cache 块管理器 (支持前缀缓存)
│   │   ├── sequence.py      # Sequence 序列类
│   │   └── ...
│   ├── models/
│   │   └── qwen3.py         # Qwen3 模型实现
│   ├── layers/
│   │   ├── attention.py     # Attention 层
│   │   ├── linear.py        # 并行线性层
│   │   ├── rotary_embedding.py  # RoPE 旋转位置编码
│   │   ├── sampler.py       # 采样器
│   │   └── ...
│   └── utils/
│       ├── context.py       # 上下文管理
│       └── loader.py        # 模型加载工具
├── example.py               # 使用示例
├── bench.py                 # 性能基准测试
└── pyproject.toml           # 项目配置
```

## 安装与运行

### 安装
```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

### 依赖
- torch>=2.4.0
- triton>=3.0.0
- transformers>=4.51.0
- flash-attn
- xxhash

### 运行示例
```bash
python example.py
python bench.py
```

## 核心 API

```python
from nanovllm import LLM, SamplingParams

# 初始化模型
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)

# 配置采样参数
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 生成文本
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 配置参数 (Config)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | - | 模型路径 |
| `max_num_batched_tokens` | 16384 | 最大批处理 token 数 |
| `max_num_seqs` | 512 | 最大序列数 |
| `max_model_len` | 4096 | 最大模型长度 |
| `gpu_memory_utilization` | 0.9 | GPU 内存利用率 |
| `tensor_parallel_size` | 1 | 张量并行大小 (1-8) |
| `enforce_eager` | False | 强制 eager 模式 |
| `kvcache_block_size` | 256 | KV Cache 块大小 |
| `num_kvcache_blocks` | -1 | KV Cache 块数量 |

## 核心架构

### Engine 模块依赖关系

```mermaid
graph TD
    subgraph 第一层：基础数据结构
        Sequence["sequence.py<br/>238 行<br/>序列数据类"]
    end
    
    subgraph 第二层：资源管理
        BlockManager["block_manager.py<br/>347 行<br/>KV Cache 内存管理"]
    end
    
    subgraph 第三层：调度策略
        Scheduler["scheduler.py<br/>71 行<br/>序列调度器"]
    end
    
    subgraph 第四层：模型执行
        ModelRunner["model_runner.py<br/>251 行<br/>模型加载与执行"]
    end
    
    subgraph 第五层：引擎总控
        LLMEngine["llm_engine.py<br/>93 行<br/>统一引擎接口"]
    end
    
    Sequence --> BlockManager
    BlockManager --> Scheduler
    Sequence --> Scheduler
    Scheduler --> LLMEngine
    ModelRunner --> LLMEngine
```

### 模块说明

| 顺序 | 模块 | 文件 | 行数 | 职责 |
|------|------|------|------|------|
| 1 | **Sequence** | `sequence.py` | 238 | 推理的基本单位，管理 token IDs、状态、KV Cache 块表 |
| 2 | **BlockManager** | `block_manager.py` | 347 | KV Cache 内存管理，支持前缀缓存和引用计数 |
| 3 | **Scheduler** | `scheduler.py` | 71 | 序列调度器，管理 prefill/decode 阶段，处理抢占 |
| 4 | **ModelRunner** | `model_runner.py` | 251 | 模型加载与执行，支持多进程张量并行 |
| 5 | **LLMEngine** | `llm_engine.py` | 93 | 统一引擎接口，整合调度和执行 |

### 各模块详解

#### 1. Sequence (`engine/sequence.py`)
- **职责**: 推理的基本单位，表示一个序列
- **核心属性**:
  - `block_table`: KV Cache 物理块 ID 列表
  - `status`: 调度状态 (WAITING → RUNNING → FINISHED)
  - `token_ids`: 序列的所有 token
  - `num_cached_tokens`: 已缓存的 token 数 (前缀缓存优化)

#### 2. BlockManager (`engine/block_manager.py`)
- **职责**: 管理 GPU 上的 KV Cache 内存块
- **核心功能**:
  - 块的分配与回收 (基于引用计数)
  - 前缀缓存 (Prefix Caching): 通过哈希去重复用相同前缀
  - 动态扩展：序列增长时追加新块
- **关键数据结构**:
  - `hash_to_block_id`: 哈希表，用于前缀缓存查找
  - `free_block_ids`: 空闲块队列 (FIFO)
  - `used_block_ids`: 已使用块集合

#### 3. Scheduler (`engine/scheduler.py`)
- **职责**: 决定哪个序列何时执行
- **调度流程**:
  1. **Prefill 阶段**: 从 `waiting` 队列选择序列，分配资源并执行
  2. **Decode 阶段**: 从 `running` 队列继续生成 token
  3. **抢占处理**: 资源不足时将序列从 `running` 移回 `waiting`
- **协作关系**: 依赖 `BlockManager` 进行内存分配

#### 4. ModelRunner (`engine/model_runner.py`)
- **职责**: 模型加载与执行
- **核心功能**:
  - 加载 HuggingFace 模型权重
  - 初始化 KV Cache 张量
  - 执行模型前向传播
  - 支持多进程张量并行 (NCCL 通信)

#### 5. LLMEngine (`engine/llm_engine.py`)
- **职责**: 统一引擎接口，推理主循环
- **工作流程**:
  ```
  1. 添加用户请求到 Scheduler.waiting
  2. 循环执行直到所有序列完成:
     a. Scheduler.schedule() 获取待执行序列
     b. ModelRunner.execute_model() 执行前向传播
     c. 更新序列状态和 token
  3. 返回生成结果
  ```

## 开发约定

### 代码风格
- 使用 dataclass 进行配置管理
- 类型注解 (Type Hints)
- 模块化设计，每层职责单一

### 测试实践
- 通过 `example.py` 验证基本功能
- 通过 `bench.py` 进行性能基准测试

### 关键设计决策
1. **张量并行**: 使用 PyTorch 分布式 (NCCL) 实现
2. **KV Cache**: 分块管理，支持前缀缓存优化
3. **调度策略**: 先 prefill 后 decode，支持抢占
4. **模型支持**: 当前仅支持 Qwen3 模型

## 相关文件

- `README.md` - 项目说明和快速开始
- `pyproject.toml` - 项目配置和依赖
- `example.py` - 使用示例
- `bench.py` - 性能基准测试脚本
