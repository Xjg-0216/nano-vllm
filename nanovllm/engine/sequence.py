"""
序列管理模块

本模块定义了 Sequence 类，用于表示和管理 LLM 推理过程中的单个序列。
序列是推理的基本单位，包含 token IDs、状态信息、KV Cache 块表等。
一个序列对应一次请求
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举类
    
    定义序列在推理生命周期中的三种状态：
    - WAITING: 等待调度，序列已创建但尚未开始推理
    - RUNNING: 正在执行，序列正在进行 prefill 或 decode 阶段
    - FINISHED: 已完成，序列已生成所有 token 或遇到 EOS
    """
    WAITING = auto()  # 等待状态：序列在调度队列中等待资源
    RUNNING = auto()  # 运行状态：序列正在 GPU 上进行推理
    FINISHED = auto()  # 完成状态：序列推理结束，结果已输出


class Sequence:
    """
    序列类
    
    表示一个大语言模型推理的序列单元，管理：
    - Token IDs：序列中的所有 token
    - 状态机：WAITING -> RUNNING -> FINISHED
    - KV Cache 块表：用于定位 GPU 内存中的 KV Cache 块
    - 采样参数：温度、最大 token 数等
    
    属性：
        block_size (int): 类变量，KV Cache 块大小（默认 256 tokens）
        counter (iterator): 类变量，序列 ID 计数器，确保每个序列有唯一 ID
    """
    block_size = 256  # KV Cache 块大小，每个块可容纳的 token 数
    counter = count()  # 全局序列计数器，从 0 开始递增

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        """
        初始化序列对象
        
        Args:
            token_ids: 输入 prompt 的 token IDs 列表
            sampling_params: 采样参数配置，包括温度、最大生成 token 数等
        """
        # 唯一序列 ID，从全局计数器获取
        self.seq_id = next(Sequence.counter)
        
        # 初始状态为等待调度
        self.status = SequenceStatus.WAITING
        
        # 复制 token IDs，避免外部修改影响内部状态
        self.token_ids = copy(token_ids)
        
        # 最后一个 token，用于快速访问（decode 阶段常用）
        self.last_token = token_ids[-1]
        
        # 当前总 token 数（prompt + completion）
        self.num_tokens = len(self.token_ids)
        
        # prompt token 数量，用于区分 prompt 和生成部分，在初始化方法内是相等的，因为completion=0
        self.num_prompt_tokens = len(token_ids)
        
        # 已缓存的 token 数（前缀缓存优化时使用）
        self.num_cached_tokens = 0
        
        # KV Cache 块表，记录每个逻辑块在物理内存中的索引
        # 例如：block_table = [5, 3, 8] 表示逻辑块 0->物理块 5, 逻辑块 1->物理块 3, ...
        self.block_table = []
        
        # 从采样参数中提取常用配置，便于快速访问
        self.temperature = sampling_params.temperature  # 采样温度
        self.max_tokens = sampling_params.max_tokens    # 最大生成 token 数
        self.ignore_eos = sampling_params.ignore_eos    # 是否忽略 EOS token

    def __len__(self):
        """返回序列的总 token 数"""
        return self.num_tokens

    def __getitem__(self, key):
        """
        支持索引访问 token IDs
        
        Args:
            key: 索引或切片
            
        Returns:
            对应的 token ID 或 token ID 列表
        """
        return self.token_ids[key]

    @property # Python 装饰器，用于将方法转换为只读属性
    def is_finished(self):
        """检查序列是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """
        返回生成的 completion token 数量
        
        Returns:
            总 token 数减去 prompt token 数
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """
        返回 prompt 部分的 token IDs
        
        Returns:
            前 num_prompt_tokens 个 token
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """
        返回生成部分的 token IDs
        
        Returns:
            从 num_prompt_tokens 开始到结尾的所有 token
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """
        返回已缓存的 KV Cache 块数量
        
        用于前缀缓存优化，计算有多少个完整块已被缓存
        
        Returns:
            已缓存的完整块数
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        返回当前序列需要的 KV Cache 块总数
        
        向上取整计算，例如 block_size=256，num_tokens=300 时需要 2 个块
        
        Returns:
            需要的总块数
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        返回最后一个块中的 token 数量
        
        用于分配 KV Cache 时确定最后一个块需要存储多少 token
        
        Returns:
            最后一个块中的 token 数（1 到 block_size 之间）
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个逻辑块对应的 token IDs
        
        Args:
            i: 逻辑块索引（0-based）
            
        Returns:
            第 i 个块包含的 token ID 列表
            
        Raises:
            AssertionError: 如果索引越界
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """
        在序列末尾追加一个新的 token
        
        在 decode 阶段，每生成一个 token 就调用此方法
        
        Args:
            token_id: 新生成的 token ID
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化支持，用于多进程通信
        
        优化序列化数据量：
        - 当 completion 为空时（prefill 阶段），序列化完整 token_ids
        - 当有 completion 时（decode 阶段），只序列化 last_token，减少数据传输
        
        Returns:
            可序列化的状态元组
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            # 优化：prefill 阶段传完整 token_ids，decode 阶段只传 last_token
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state):
        """
        反序列化支持，从状态元组恢复对象
        
        Args:
            state: 由 __getstate__ 返回的状态元组
        """
        # 恢复基本属性
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        
        # 根据 completion token 数量决定如何恢复 token_ids
        if self.num_completion_tokens == 0:
            # Prefill 阶段：state[-1] 是完整的 token_ids 列表
            self.token_ids = state[-1]
        else:
            # Decode 阶段：state[-1] 只是 last_token
            # 注意：此处 token_ids 未完全恢复，依赖后续逻辑补充
            self.last_token = state[-1]
