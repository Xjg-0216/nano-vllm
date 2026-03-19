"""
ModelRunner - 模型执行器模块

负责模型的加载、初始化、KV Cache 分配以及 CUDA 图捕获。
支持多进程张量并行执行，通过共享内存进行进程间通信。

核心功能：
1. 模型加载与初始化
2. KV Cache 内存分配
3. Prefill/Decode 阶段的数据准备
4. CUDA 图捕获（用于加速 decode 阶段）
5. 多进程同步与通信
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型执行器类
    
    每个 GPU 对应一个 ModelRunner 实例，负责：
    - 模型加载和初始化
    - KV Cache 分配和管理
    - 前向推理执行
    - CUDA 图捕获（可选优化）
    - 多进程张量并行通信
    
    Attributes:
        config: 配置对象，包含模型和运行时配置
        rank: 当前进程的 rank（GPU ID）
        world_size: 张量并行的进程数
        model: Qwen3 因果语言模型
        sampler: 采样器，用于从 logits 生成 token
        kv_cache: KV Cache 张量，存储在所有 GPU 上
        shm: 共享内存，用于多进程通信（rank>0 时）
        event: 同步事件，用于多进程同步
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化 ModelRunner
        
        Args:
            config: 配置对象
            rank: 当前进程 rank（0 到 world_size-1）
            event: 同步事件对象，rank=0 时为事件列表，rank>0 时为单个事件
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size  # KV Cache 块大小（token 数）
        self.enforce_eager = config.enforce_eager  # 是否强制使用 eager 模式（不使用 CUDA 图）
        self.world_size = config.tensor_parallel_size  # 张量并行大小
        self.rank = rank
        self.event = event

        # 初始化分布式进程组（NCCL 后端）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # 设置默认数据类型和设备为 CUDA
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 加载模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # 模型预热
        self.warmup_model()
        
        # 分配 KV Cache
        self.allocate_kv_cache()
        
        # 捕获 CUDA 图（如果不禁用 eager 模式）
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # 恢复默认数据类型和设备
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多进程设置：rank=0 创建共享内存，rank>0 加入并进入循环等待
        if self.world_size > 1:
            if rank == 0:
                # rank=0 创建共享内存用于进程间通信
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # rank>0 等待 barrier 后加入共享内存并进入循环
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """
        清理资源并退出
        
        清理共享内存、CUDA 图、进程组等资源
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        主循环（仅 rank>0 的进程运行）
        
        不断从共享内存读取方法名和参数，执行对应方法。
        当收到 "exit" 方法时退出循环。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取方法调用信息（仅 rank>0 使用）
        
        Returns:
            tuple: (方法名，参数列表)
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待事件触发
        # 读取数据长度（前 4 字节）
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 读取并反序列化数据
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # 清除事件状态
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        向共享内存写入方法调用信息（仅 rank=0 使用）
        
        Args:
            method_name: 要调用的方法名
            *args: 方法参数
        """
        assert self.world_size > 1 and self.rank == 0
        # 序列化数据
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 写入长度和数据
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 触发所有 rank>0 的事件
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        调用指定方法
        
        Args:
            method_name: 方法名
            *args: 方法参数
            
        Returns:
            方法返回值
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        模型预热
        
        创建虚拟序列并执行一次前向传播，以初始化 CUDA 上下文和分配必要的内存。
        这有助于避免首次推理时的额外开销。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 计算预热序列数：受限于最大批处理 token 数和最大序列数
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # 创建满长度的虚拟序列
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)  # 执行 prefill 模式
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配 KV Cache 内存
        
        根据 GPU 可用内存和配置计算可分配的 KV Cache 块数量，
        并将 KV Cache 绑定到模型的注意力层。
        
        KV Cache 形状：(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        - 第 0 维：0 表示 key cache，1 表示 value cache
        """
        config = self.config
        hf_config = config.hf_config
        
        # 获取 GPU 内存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free  # 当前已用内存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 峰值分配
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前分配
        
        # 计算每个 KV Cache 块的大小（字节）
        # 2 表示 key 和 value 各一份
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 张量并行下的 KV 头数
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # 计算可分配的 KV Cache 块数量
        # 公式：(目标内存 - 已用内存 - 峰值 + 当前) / 块大小
        # 这样可以在保留模型权重的同时最大化 KV Cache
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        # 分配 KV Cache 张量
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        # 将 KV Cache 绑定到模型的注意力层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备块表（block tables）
        
        将序列的块表转换为张量格式，用于注意力计算。
        对较短的序列进行 padding 以对齐最大长度。
        
        Args:
            seqs: 序列列表
            
        Returns:
            block_tables: 形状为 (batch_size, max_num_blocks) 的张量
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        # 对每个序列的块表进行 padding，-1 表示无效块
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 prefill 阶段的输入数据
        
        Prefill 阶段处理整个 prompt，需要：
        - 输入 token IDs（未缓存的部分）
        - 位置信息
        - 累积序列长度（用于 Flash Attention）
        - slot_mapping（KV Cache 槽位映射）
        
        Args:
            seqs: 序列列表
            
        Returns:
            input_ids: 输入 token IDs
            positions: 位置信息
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # query 的累积序列长度
        cu_seqlens_k = [0]  # key 的累积序列长度
        max_seqlen_q = 0    # 最大 query 序列长度
        max_seqlen_k = 0    # 最大 key 序列长度
        slot_mapping = []   # KV Cache 槽位映射
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            # 只添加未缓存的 token（跳过 prefix cache 已缓存的部分）
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens  # 本次需要处理的 token 数
            seqlen_k = seqlen  # 完整的序列长度（用于 KV Cache 查找）
            
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:    # warmup 阶段没有块表
                continue
            
            # 构建 slot_mapping：将 token 位置映射到 KV Cache 的物理槽位
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        
        # 如果有前缀缓存（key 序列长度 > query 序列长度），需要准备块表
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # 转换为 CUDA 张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置上下文（供注意力层使用）
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 decode 阶段的输入数据
        
        Decode 阶段每次只生成一个 token，需要：
        - 最后一个 token 的 ID
        - 位置信息（序列长度 - 1）
        - slot_mapping（最后一个 KV Cache 槽位）
        - context_lens（上下文长度，用于注意力计算）
        - block_tables（块表，用于查找历史 KV）
        
        Args:
            seqs: 序列列表
            
        Returns:
            input_ids: 最后一个 token 的 ID
            positions: 位置信息
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)  # 最后一个 token
            positions.append(len(seq) - 1)    # 当前位置
            context_lens.append(len(seq))     # 上下文长度
            # 计算最后一个 token 的 KV Cache 槽位
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        # 转换为 CUDA 张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置上下文（decode 模式）
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数
        
        Args:
            seqs: 序列列表
            
        Returns:
            temperatures: 温度参数张量
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向传播
        
        Prefill 阶段或 eager 模式下直接执行模型。
        Decode 阶段且启用 CUDA 图时使用图执行以加速。
        
        Args:
            input_ids: 输入 token IDs
            positions: 位置信息
            is_prefill: 是否为 prefill 阶段
            
        Returns:
            logits: 模型输出的 logits
        """
        # Prefill、eager 模式或 batch size 过大时直接执行
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 使用 CUDA 图加速 decode 阶段
            bs = input_ids.size(0)
            context = get_context()
            # 选择能容纳当前 batch size 的最小图
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 更新图变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放 CUDA 图
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行一次推理步骤
        
        Args:
            seqs: 序列列表
            is_prefill: 是否为 prefill 阶段
            
        Returns:
            token_ids: 生成的 token IDs（仅 rank=0 返回有效值）
        """
        # 准备输入数据
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 准备采样参数（仅 rank=0 需要）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 执行模型
        logits = self.run_model(input_ids, positions, is_prefill)
        # 采样生成 token
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 重置上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获 CUDA 图
        
        CUDA 图可以显著降低 decode 阶段的 CPU 开销。
        为不同的 batch size 捕获多个图，运行时选择最合适的图。
        
        捕获的图包括：
        - 模型前向传播
        - 注意力计算
        - KV Cache 查找
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 最大 batch size
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size  # 最大块数
        
        # 预分配最大尺寸的张量
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 定义要捕获的 batch size 序列
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小捕获图（有利于内存优化）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 设置上下文
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 预热（避免首次执行的额外开销）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 捕获 CUDA 图
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 保存图池（第一个图创建池，后续图复用）
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存图变量，用于运行时更新
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
