# Nano-vLLM 文档

欢迎来到 **Nano-vLLM** 项目文档站点！

## 📖 项目简介

Nano-vLLM 是一个轻量级的 vLLM 实现，从头开始构建，用于离线大语言模型推理。

### 核心特性

- 🚀 **快速离线推理** - 与 vLLM 相当的推理速度
- 📖 **可读代码** - 约 1200 行 Python 代码的清晰实现
- ⚡ **优化套件** - 支持前缀缓存、张量并行、Torch 编译、CUDA 图等优化

### 技术栈

- Python 3.10-3.12
- PyTorch + Triton
- Transformers (HuggingFace)
- Flash Attention
- NCCL (多 GPU 通信)

## 📚 文档导航

### 入门指南

- [项目介绍](01-intro.md) - 了解 Nano-vLLM 的项目结构和核心概念
- [架构概览](02-architecture.md) - 整体架构设计和技术选型
- [学习顺序](03-engine-learning-order.md) - 推荐的学习路径

### Engine 模块详解

- [Sequence 设计](04-sequence-design.md) - 序列类的设计和实现
- [BlockManager 设计](05-block-manager-design.md) - KV Cache 块管理器设计
- [BlockManager 流程](06-block-manager-flow.md) - 块管理详细流程
- [Scheduler 流程](07-scheduler-flow.md) - 序列调度器工作原理

### 模型架构

- [Qwen3 模型](08-model-qwen3.md) - Qwen3 模型完整架构解析
- [模型注意力](08-model_attention.md) - 注意力机制实现

### 核心组件

- [张量并行基础](09-tensor-parallel.md) - tp_rank 和 tp_size 详解
- [嵌入层与输出头](10-embed-head.md) - 词汇表并行实现
- [采样器](11-sampler.md) - Gumbel-Max 采样原理
- [注意力机制](12-attention.md) - Flash Attention 与 KV Cache 管理
- [线性层](12-linear.md) - 并行线性层实现
- [并行决策](13-parallel-decision.md) - 张量并行决策过程
- [RoPE 与 LRU 缓存](14-rope-and-lru-cache.md) - 旋转位置编码

## 🚀 快速开始

### 安装

```bash
pip install git+https://github.com/Xjg-0216/nano-vllm.git
```

### 使用示例

```python
from nanovllm import LLM, SamplingParams

# 初始化模型
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)

# 配置采样参数
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 生成文本
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

## 📦 项目结构

```
nano-vllm/
├── nanovllm/              # 核心代码
│   ├── engine/           # 推理引擎
│   ├── models/           # 模型实现
│   ├── layers/           # 网络层
│   └── utils/            # 工具函数
├── docs/                 # 文档目录
├── example.py            # 使用示例
└── bench.py              # 性能测试
```

## 🔗 相关链接

- [GitHub 仓库](https://github.com/Xjg-0216/nano-vllm)
- [vLLM 原项目](https://github.com/vllm-project/vllm)

## 📝 关于文档

本文档使用 [MkDocs](https://www.mkdocs.org/) 和 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 主题构建，支持：

- 📱 响应式设计，支持移动设备
- 🌙 深色/浅色模式切换
- 🔍 全文搜索
- 📊 Mermaid 图表支持
- 📐 MathJax 数学公式支持

---

**最后更新**: 本文档会随项目代码持续更新
