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

### 1. LLMEngine (`engine/llm_engine.py`)
- 主引擎类，管理模型执行和序列调度
- 支持多进程张量并行 (通过 `ModelRunner`)
- 提供 `generate()` 方法用于批量推理

### 2. ModelRunner (`engine/model_runner.py`)
- 模型执行器，每个 GPU 一个实例
- 负责模型加载、KV Cache 分配、CUDA 图捕获
- 通过共享内存进行进程间通信

### 3. Scheduler (`engine/scheduler.py`)
- 序列调度器，管理 prefill 和 decode 阶段
- 实现抢占式调度 (preemption)
- 与 BlockManager 协作管理 KV Cache

### 4. BlockManager (`engine/block_manager.py`)
- KV Cache 块管理器
- 支持前缀缓存 (prefix caching) 通过哈希去重
- 块引用计数管理内存

### 5. Sequence (`engine/sequence.py`)
- 序列表示类
- 管理 token IDs、状态、块表

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
