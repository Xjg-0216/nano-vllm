"""
序列调度器模块

本模块定义了 Scheduler 类，负责管理 LLM 推理过程中序列的调度。
核心功能包括：
- 序列的等待队列和运行队列管理
- Prefill 阶段调度：新序列首次执行，分配 KV Cache 块
- Decode 阶段调度：已运行序列继续生成 token
- 抢占式调度：资源不足时将序列从 running 移回 waiting
- 序列完成处理：释放资源并移除已完成的序列

调度策略：
1. 优先执行 prefill 阶段（新序列首次执行）
2. 然后执行 decode 阶段（已运行序列继续生成）
3. 资源不足时抢占优先级低的序列
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    序列调度器

    管理序列的生命周期和调度决策，是 LLM 推理的核心组件。

    调度队列：
    - waiting: 等待调度的序列队列（新加入的序列或被打断的序列）
    - running: 正在执行的序列队列（已分配 KV Cache 块）

    调度阶段：
    - Prefill: 新序列首次执行，分配 KV Cache 块，处理所有输入 token
    - Decode: 已运行序列每次生成一个 token

    抢占策略：
    - 当资源不足时，将 running 队列中的序列移回 waiting 队列
    - 被抢占的序列保留 KV Cache，下次调度时可继续执行
    """

    def __init__(self, config: Config):
        """
        初始化调度器

        Args:
            config: 配置对象，包含调度相关的参数
        """
        # 最大并发序列数，限制同时处理的序列数量
        self.max_num_seqs = config.max_num_seqs

        # 最大批处理 token 数，限制单次迭代的 token 总量
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # 结束 token ID，用于判断序列是否结束
        self.eos = config.eos

        # KV Cache 块管理器，负责内存分配和回收
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # 等待队列：FIFO 队列，存储等待调度的序列
        # 新序列或被抢占的序列加入此队列
        self.waiting: deque[Sequence] = deque()

        # 运行队列：双端队列，存储正在执行的序列
        # 已分配 KV Cache 块并正在生成的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """
        检查调度器是否已完成所有推理任务

        Returns:
            True 如果 waiting 和 running 队列都为空，表示所有序列已完成
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新序列到等待队列

        用户请求生成时，创建 Sequence 对象并调用此方法加入调度

        Args:
            seq: 要添加的序列对象
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        执行调度决策，决定哪些序列在本轮执行

        调度优先级：
        1. 优先处理 waiting 队列中的序列（prefill 阶段）
        2. 如果 waiting 队列为空或无法分配资源，处理 running 队列（decode 阶段）

        Returns:
            tuple[list[Sequence], bool]:
                - list[Sequence]: 本轮调度的序列列表
                - bool: True 表示执行 prefill，False 表示执行 decode
        """
        # 本轮调度的序列列表
        scheduled_seqs = []

        # 本轮调度的序列数量
        num_seqs = 0

        # 本轮已分配的 token 数量（用于批处理限制）
        num_batched_tokens = 0

        # ==================== Prefill 阶段 ====================
        # 从 waiting 队列中选择序列进行预填充
        # 新序列首次执行，需要分配 KV Cache 块并处理所有输入 token
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 查看队首序列（不弹出）

            # 检查是否可以调度此序列：
            # 1. token 数量不超过批处理限制
            # 2. BlockManager 有足够的空闲块分配
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or
                    not self.block_manager.can_allocate(seq)):
                break  # 资源不足，停止调度

            # 调度此序列
            num_seqs += 1

            # 为序列分配 KV Cache 块（支持前缀缓存优化）
            self.block_manager.allocate(seq)

            # 累加实际需要的 token 数
            # len(seq): 序列总长度（输入 token）
            # seq.num_cached_tokens: 已缓存的 token 数（前缀缓存命中部分）
            # 差值：实际需要计算 KV Cache 的 token 数
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            # 更新序列状态为运行中
            seq.status = SequenceStatus.RUNNING

            # 从 waiting 队列移除，加入 running 队列
            self.waiting.popleft()
            self.running.append(seq)

            # 加入本轮调度列表
            scheduled_seqs.append(seq)

        # 如果有序列被调度（prefill），直接返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # ==================== Decode 阶段 ====================
        # waiting 队列为空或无法分配新序列时，处理 running 队列
        # 已运行的序列每次生成一个 token
        while self.running and num_seqs < self.max_num_seqs:
            # 从 running 队列左侧弹出序列（FIFO 顺序）
            seq = self.running.popleft()

            # 检查是否可以为序列追加新块（序列增长时可能需要新块）
            # 如果无法追加，需要抢占其他序列释放资源
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 还有其他运行序列，抢占队尾的序列（优先级最低）
                    self.preempt(self.running.pop())
                else:
                    # 没有其他序列可抢占，只能抢占当前序列
                    self.preempt(seq)
                    break
            else:
                # 资源充足，可以继续执行
                num_seqs += 1

                # 尝试为序列追加新块（如果需要）
                # 当序列长度跨块边界时分配新块，或计算完整块的哈希
                self.block_manager.may_append(seq)

                # 加入本轮调度列表
                scheduled_seqs.append(seq)

        # 确保至少调度了一个序列（decode 阶段不应全部被抢占）
        assert scheduled_seqs

        # 将调度的序列放回 running 队列左侧（保持执行顺序）
        # 使用 extendleft + reversed 保持原顺序
        self.running.extendleft(reversed(scheduled_seqs))

        # 返回调度结果：False 表示 decode 阶段
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占序列，将其从 running 队列移回 waiting 队列

        当资源不足时调用此方法，释放序列占用的 KV Cache 块，
        但保留序列状态，以便后续恢复执行。

        Args:
            seq: 要被抢占的序列
        """
        # 更新序列状态为等待
        seq.status = SequenceStatus.WAITING

        # 回收序列占用的 KV Cache 块
        # 注意：只是减少引用计数，不一定会真正释放（如果块被共享）
        self.block_manager.deallocate(seq)

        # 将序列加入 waiting 队列头部（高优先级，优先恢复）
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理：处理模型生成的 token，更新序列状态

        在 ModelRunner 执行完前向传播后调用，将生成的 token 追加到序列，
        并检查是否需要结束序列。

        Args:
            seqs: 本轮执行的序列列表
            token_ids: 模型生成的 token IDs（与 seqs 一一对应）

        Returns:
            list[bool]: 每个序列是否完成（True 表示完成）
        """
        is_completed = []

        # 遍历每个序列和对应的生成 token
        for seq, token_id in zip(seqs, token_ids):
            # 将生成的 token 追加到序列
            seq.append_token(token_id)

            # 检查序列是否应该结束：
            # 1. 生成 EOS token 且未忽略 EOS
            # 2. 达到最大生成 token 数
            if ((not seq.ignore_eos and token_id == self.eos) or
                    seq.num_completion_tokens == seq.max_tokens):
                # 更新序列状态为完成
                seq.status = SequenceStatus.FINISHED

                # 回收序列占用的所有 KV Cache 块
                self.block_manager.deallocate(seq)

                # 从 running 队列移除
                self.running.remove(seq)

                is_completed.append(True)
            else:
                is_completed.append(False)

        return is_completed
