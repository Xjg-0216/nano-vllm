"""
配置管理模块

本模块定义了 Config 数据类，用于管理 Nano-vLLM 的所有配置参数。
配置参数涵盖：
- 模型路径和 HF 配置
- 调度器参数（批处理、序列数限制）
- KV Cache 内存管理
- 张量并行设置
- 其他推理相关配置

使用方式：
    config = Config(
        model="/path/to/model",
        max_num_seqs=256,
        tensor_parallel_size=2
    )
"""

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    Nano-vLLM 配置类

    使用 Python dataclass 装饰器，提供简洁的配置管理。
    所有配置参数在 __post_init__ 中进行验证和初始化。

    配置类别：
    1. 模型配置：model, hf_config
    2. 调度配置：max_num_batched_tokens, max_num_seqs, max_model_len
    3. 内存配置：gpu_memory_utilization, kvcache_block_size, num_kvcache_blocks
    4. 并行配置：tensor_parallel_size
    5. 执行配置：enforce_eager
    6. 其他配置：eos

    属性：
        model (str): 模型路径（必需）
        max_num_batched_tokens (int): 最大批处理 token 数
        max_num_seqs (int): 最大并发序列数
        max_model_len (int): 最大模型长度
        gpu_memory_utilization (float): GPU 内存利用率
        tensor_parallel_size (int): 张量并行大小
        enforce_eager (bool): 是否强制 eager 模式
        hf_config (AutoConfig): HuggingFace 模型配置（自动加载）
        eos (int): 结束 token ID（自动从 HF 配置获取）
        kvcache_block_size (int): KV Cache 块大小
        num_kvcache_blocks (int): KV Cache 块数量
    """

    # ==================== 必需参数 ====================

    model: str
    """
    模型路径（必需）

    本地路径或 HuggingFace 模型名称。
    模型目录应包含：
    - config.json: 模型配置
    - model.safetensors 或 pytorch_model.bin: 模型权重
    - tokenizer.json 或 tokenizer.model: 分词器

    示例：
        "/path/to/Qwen3-8B"
        "Qwen/Qwen3-8B"
    """

    # ==================== 调度器配置 ====================

    max_num_batched_tokens: int = 16384
    """
    最大批处理 token 数

    限制单次迭代中处理的 token 总量（包括 prefill 和 decode 阶段）。
    影响：
    - 值越大：GPU 利用率越高，但内存占用越大
    - 值越小：并发能力越低，但单序列延迟可能更低

    默认值：16384
    推荐值：根据 GPU 显存调整，通常为 max_model_len 的 2-4 倍
    """

    max_num_seqs: int = 512
    """
    最大并发序列数

    限制同时处理的序列（请求）数量。
    影响：
    - 值越大：并发能力越强，但调度开销越大
    - 值越小：调度开销越小，但可能拒绝部分请求

    默认值：512
    推荐值：根据实际并发需求调整，通常为 128-1024
    """

    max_model_len: int = 4096
    """
    最大模型长度

    限制单个序列的最大 token 数（输入 + 输出）。
    影响：
    - 值越大：支持更长的上下文，但 KV Cache 占用越大
    - 值越小：KV Cache 占用越小，但可能截断长文本

    默认值：4096
    注意：__post_init__ 中会与 HF 配置的 max_position_embeddings 取较小值
    """

    # ==================== 内存配置 ====================

    gpu_memory_utilization: float = 0.9
    """
    GPU 内存利用率

    控制 KV Cache 占用的 GPU 显存比例。
    影响：
    - 值越大：KV Cache 容量越大，但可能影响其他操作
    - 值越小：KV Cache 容量越小，但更安全

    默认值：0.9 (90%)
    推荐值：0.8-0.95，根据实际 GPU 显存和模型大小调整
    """

    kvcache_block_size: int = 256
    """
    KV Cache 块大小

    每个 KV Cache 块可容纳的 token 数量。
    影响：
    - 值越大：块管理开销越小，但内存碎片可能越多
    - 值越小：内存利用率越高，但管理开销越大

    默认值：256
    注意：必须是 256 的倍数（__post_init__ 中验证）
    推荐值：256 或 512
    """

    num_kvcache_blocks: int = -1
    """
    KV Cache 块数量

    -1 表示自动计算（根据 gpu_memory_utilization）。
    影响：
    - 值越大：可容纳的序列越多，但可能 OOM
    - 值越小：可容纳的序列越少，但更安全

    默认值：-1 (自动计算)
    推荐值：除非有特殊需求，否则使用默认值
    """

    # ==================== 并行配置 ====================

    tensor_parallel_size: int = 1
    """
    张量并行大小

    使用多少张 GPU 进行张量并行推理。
    影响：
    - 值越大：单请求速度越快，但需要更多 GPU
    - 值越小：GPU 需求越少，但单请求速度越慢

    默认值：1 (单 GPU)
    范围：1-8
    注意：需要 GPU 支持 NCCL 通信
    """

    # ==================== 执行配置 ====================

    enforce_eager: bool = False
    """
    是否强制使用 eager 模式

    False 表示使用 CUDA Graph 等优化技术。
    影响：
    - True: 更易于调试，但性能较低
    - False: 性能更高，但可能增加内存占用

    默认值：False (使用优化)
    推荐值：调试时设为 True，生产环境设为 False
    """

    # ==================== 自动配置（只读） ====================

    hf_config: AutoConfig | None = None
    """
    HuggingFace 模型配置（自动加载）

    在 __post_init__ 中从模型路径加载，包含：
    - 模型架构信息
    - 词表大小
    - 隐藏层维度
    - 注意力头数
    - 最大位置编码数
    等模型相关配置

    注意：此字段为只读，不应手动设置
    """

    eos: int = -1
    """
    结束 token ID (End of Sequence)

    用于判断序列是否应该结束。
    在 __post_init__ 中从 HF 配置自动获取。

    默认值：-1 (未设置)
    注意：如果模型没有 eos_token_id，需要手动设置
    """

    def __post_init__(self):
        """
        初始化后处理：验证配置并加载自动配置

        执行顺序：
        1. 验证模型路径存在
        2. 验证 kvcache_block_size 是 256 的倍数
        3. 验证 tensor_parallel_size 在有效范围内
        4. 加载 HuggingFace 模型配置
        5. 调整 max_model_len 为配置值和 HF 值的较小者
        6. 验证 max_num_batched_tokens 不小于 max_model_len
        7. 加载 eos token ID

        Raises:
            AssertionError: 如果任何验证失败
        """
        # 1. 验证模型路径存在
        # 确保模型目录存在，避免后续加载失败
        assert os.path.isdir(self.model), f"模型路径不存在：{self.model}"

        # 2. 验证 kvcache_block_size 是 256 的倍数
        # 确保块大小与内存对齐要求匹配
        assert self.kvcache_block_size % 256 == 0, \
            f"kvcache_block_size 必须是 256 的倍数：{self.kvcache_block_size}"

        # 3. 验证 tensor_parallel_size 在有效范围内
        # 支持 1-8 张 GPU 并行
        assert 1 <= self.tensor_parallel_size <= 8, \
            f"tensor_parallel_size 必须在 1-8 之间：{self.tensor_parallel_size}"

        # 4. 加载 HuggingFace 模型配置
        # 从模型目录读取 config.json，解析模型架构信息
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # 5. 调整 max_model_len
        # 取配置值和 HF 最大位置编码的较小者，确保不超过模型支持的范围
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings
        )

        # 6. 验证 max_num_batched_tokens 不小于 max_model_len
        # 确保至少能处理一个完整的序列
        assert self.max_num_batched_tokens >= self.max_model_len, \
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) 必须不小于 max_model_len ({self.max_model_len})"

        # 7. 加载 eos token ID
        # 从 HF 配置获取结束 token，用于判断序列是否完成
        if hasattr(self.hf_config, "eos_token_id"):
            self.eos = self.hf_config.eos_token_id
