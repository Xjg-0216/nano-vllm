"""
嵌入层与输出头模块

实现词汇表并行的嵌入层和语言模型输出头，支持张量并行分布式训练/推理。

核心组件：
1. VocabParallelEmbedding：词汇表并行的词嵌入层
2. ParallelLMHead：词汇表并行的语言模型输出头

张量并行策略：
- 词汇表按维度切分到不同 GPU 上
- 每个 GPU 只存储部分词汇的嵌入向量
- 通过 gather/all_reduce 操作合并结果
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词汇表并行的词嵌入层
    
    将词汇表均匀切分到多个 GPU 上，每个 GPU 只存储部分词汇的嵌入向量。
    这种并行方式可以显著减少单个 GPU 的内存占用，适用于大词汇表场景。
    
    并行策略：
        - 词汇表维度切分：vocab_size // tensor_parallel_size
        - 每个 GPU 存储连续的词汇块
        - 前向传播时通过掩码处理词汇索引
    
    Attributes:
        tp_rank: 当前进程的张量并行 rank
        tp_size: 张量并行大小（GPU 数量）
        num_embeddings: 完整词汇表大小
        num_embeddings_per_partition: 当前 GPU 存储的词汇数量
        vocab_start_idx: 当前 GPU 负责的词汇起始索引
        vocab_end_idx: 当前 GPU 负责的词汇结束索引
        weight: 嵌入权重矩阵，形状为 (num_embeddings_per_partition, embedding_dim)
    
    Example:
        >>> # 假设 vocab_size=32000, tensor_parallel_size=4
        >>> # 每个 GPU 存储 8000 个词的嵌入
        >>> emb = VocabParallelEmbedding(32000, 4096)
        >>> input_ids = torch.tensor([0, 1000, 16000, 31999])  # 跨多个 partition
        >>> output = emb(input_ids)  # 自动处理跨 GPU 的词汇查找
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """
        初始化词汇表并行嵌入层
        
        Args:
            num_embeddings: 词汇表大小
            embedding_dim: 嵌入向量维度
        """
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0  # 确保可整除
        
        self.num_embeddings = num_embeddings
        # 每个 GPU 存储的词汇数量
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前 GPU 负责的词汇范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 嵌入权重矩阵
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器：从完整权重中加载当前 partition 的部分
        
        Args:
            param: 当前参数（当前 partition 的权重）
            loaded_weight: 完整的预训练权重
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        # 从完整权重中切分出当前 GPU 负责的部分
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        词嵌入前向传播
        
        处理词汇索引并查找嵌入向量。对于不在当前 GPU 词汇范围内的 token，
        通过掩码机制将结果置零，最后通过 all_reduce 合并所有 GPU 的结果。
        
        Args:
            x: 输入 token IDs，形状为 [batch_size, seq_len]
            
        Returns:
            嵌入向量，形状为 [batch_size, seq_len, embedding_dim]
        """
        if self.tp_size > 1:
            # 创建掩码：标记哪些 token 在当前 GPU 的词汇范围内
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将全局词汇索引转换为当前 partition 的局部索引
            # 不在范围内的 token 会被映射到 0（后续会被掩码置零）
            x = mask * (x - self.vocab_start_idx)
        
        # 查找嵌入向量
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 应用掩码：将不在当前 GPU 范围内的 token 嵌入置零
            y = mask.unsqueeze(1) * y
            # 所有 GPU 的结果相加，得到完整的嵌入表示
            dist.all_reduce(y)
        
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    词汇表并行的语言模型输出头
    
    将隐藏状态映射到词汇表上的 logits 分布。
    继承自 VocabParallelEmbedding，共享词汇表并行策略。
    
    特殊处理：
        1. Prefill 模式：只提取每个序列的最后一个 token 的 logits
        2. Decode 模式：直接计算所有 token 的 logits
        3. 张量并行：通过 gather 操作合并各 GPU 的 logits
    
    Attributes:
        继承自 VocabParallelEmbedding
    
    Example:
        >>> lm_head = ParallelLMHead(vocab_size=32000, hidden_size=4096)
        >>> hidden_states = torch.randn(2, 4096)  # [batch, hidden_size]
        >>> logits = lm_head(hidden_states)  # [batch, vocab_size] (rank=0)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        """
        初始化并行 LM 头
        
        Args:
            num_embeddings: 词汇表大小
            embedding_dim: 隐藏层维度
            bias: 是否使用偏置（默认 False）
        """
        assert not bias  # 当前不支持偏置
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        LM 头前向传播：计算词汇表 logits
        
        Args:
            x: 隐藏状态，形状为 [total_tokens, hidden_size]
                - total_tokens 是所有序列的 token 总数（变长）
            
        Returns:
            logits: 词汇表 logits
                - rank=0: [num_seqs, vocab_size]
                - rank>0: None
        """
        context = get_context()
        
        if context.is_prefill:
            # Prefill 模式：只提取每个序列最后一个 token 的 logits
            # cu_seqlens_q[1:] 是每个序列的结束位置
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # 计算 logits：只计算当前 GPU 负责的词汇部分
        logits = F.linear(x, self.weight)
        
        if self.tp_size > 1:
            # 张量并行：收集所有 GPU 的 logits
            if self.tp_rank == 0:
                # rank=0 准备接收所有 GPU 的结果
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            else:
                all_logits = None
            
            # 将所有 GPU 的 logits gather 到 rank=0
            dist.gather(logits, all_logits, 0)
            
            # rank=0 拼接所有 logits，得到完整的词汇表分布
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        
        return logits
