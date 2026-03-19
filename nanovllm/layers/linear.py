"""
并行线性层模块

实现张量并行（Tensor Parallel）的线性层，支持多种并行策略：
1. Column Parallel：按输出维度切分（用于 QKV 投影、gate/up 投影）
2. Row Parallel：按输入维度切分（用于输出投影）
3. Merged Column Parallel：合并多个 Column Parallel（用于 SwiGLU）
4. QKV Parallel：合并 Q/K/V 投影（用于多头注意力）

张量并行原理：
- 将大矩阵乘法切分到多个 GPU 上并行执行
- 通过 AllReduce 操作同步结果
- 保持数值等价于单卡训练/推理
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """
    整除断言函数
    
    确保分子能被分母整除，用于张量并行维度切分验证。
    
    Args:
        numerator: 分子
        denominator: 分母
        
    Returns:
        整除结果
        
    Raises:
        AssertionError: 如果不能整除
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    线性层基类
    
    提供通用的线性层初始化和参数管理功能。
    
    Attributes:
        tp_dim: 张量并行切分维度（0 表示输出维度，1 表示输入维度）
        tp_rank: 当前进程的张量并行 rank
        tp_size: 张量并行大小
        weight: 权重矩阵
        bias: 偏置向量（可选）
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """
        初始化线性层基类
        
        Args:
            input_size: 输入特征维度
            output_size: 输出特征维度
            bias: 是否使用偏置
            tp_dim: 张量并行切分维度（None 表示不切分）
        """
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 权重矩阵：[output_size, input_size]
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（由子类实现）"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制式线性层（无张量并行）
    
    所有 GPU 存储完整的权重矩阵，适用于小尺寸线性层或不需并行的场景。
    
    Example:
        >>> layer = ReplicatedLinear(4096, 4096)
        >>> output = layer(x)  # [batch, 4096]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器：直接复制完整权重
        
        Args:
            param: 当前参数
            loaded_weight: 预训练权重
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准线性变换：y = xW^T + b"""
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层（输出维度切分）
    
    将权重矩阵按输出维度（行）切分到多个 GPU 上：
    - 每个 GPU 存储 [output_size/tp_size, input_size] 的权重
    - 输入 x 在所有 GPU 上相同
    - 输出 y 在各 GPU 上为完整输出的 1/tp_size
    
    并行策略示意：
        W = [W_0; W_1; ...; W_{n-1}]  # 按行切分
        y_i = x @ W_i^T               # 每个 GPU 计算部分输出
    
    适用于：
        - QKV 投影层
        - MLP 的 gate/up 投影
    
    Attributes:
        tp_dim: 切分维度为 0（输出维度）
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输出维度按 tp_size 切分
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器：加载当前 GPU 对应的权重分片
        
        Args:
            param: 当前参数（分片权重）
            loaded_weight: 完整预训练权重
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        # 计算当前 rank 对应的起始索引
        start_idx = self.tp_rank * shard_size
        # 从完整权重中切分出当前 GPU 负责的部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        列并行前向传播
        
        Args:
            x: 输入张量，所有 GPU 相同
            
        Returns:
            部分输出张量（需要后续 AllReduce 或拼接）
        """
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层（用于 SwiGLU 等门控结构）
    
    将两个线性层的权重合并存储，用于 SwiGLU 激活函数：
        gate_up = [gate_proj; up_proj]
        output = SiLU(gate) * up
    
    合并优势：
        - 减少 kernel 启动开销
        - 提高内存访问效率
    
    权重布局：
        - 完整权重：[gate_size + up_size, input_size]
        - 分片权重：[(gate_size + up_size)/tp_size, input_size]
    
    Example:
        >>> # Qwen3/LLaMA 的 MLP 中间层
        >>> gate_up_proj = MergedColumnParallelLinear(4096, [11008, 11008])
        >>> gate_up = gate_up_proj(x)  # [batch, 22016]
        >>> output = silu(gate) * up
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """
        初始化合并列并行线性层
        
        Args:
            input_size: 输入特征维度
            output_sizes: 两个输出维度列表（如 [intermediate_size, intermediate_size]）
            bias: 是否使用偏置
        """
        self.output_sizes = output_sizes  # 保存两个输出维度的大小
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        权重加载器：加载指定分片的权重
        
        Args:
            param: 当前参数（合并后的分片权重）
            loaded_weight: 完整的单个分片权重（gate 或 up）
            loaded_shard_id: 分片标识（0 表示 gate，1 表示 up）
        """
        param_data = param.data
        
        # 计算当前分片在合并权重中的偏移（考虑张量并行切分）
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 计算当前分片的大小（考虑张量并行切分）
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        
        # 从合并参数中切分出当前分片对应的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        
        # 对输入权重进行张量并行切分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 并行投影层（用于多头注意力）
    
    将 Q、K、V 三个投影矩阵合并存储，支持 GQA/MQA 架构：
        - Q 投影：[num_heads * head_dim, hidden_size]
        - K 投影：[num_kv_heads * head_dim, hidden_size]
        - V 投影：[num_kv_heads * head_dim, hidden_size]
    
    合并权重布局：
        W_qkv = [W_q; W_k; W_v]
        输出维度 = (num_heads + 2 * num_kv_heads) * head_dim
    
    GQA 支持：
        - num_kv_heads < num_heads 时为 GQA
        - num_kv_heads = num_heads 时为 MHA
        - num_kv_heads = 1 时为 MQA
    
    Example:
        >>> qkv_proj = QKVParallelLinear(
        ...     hidden_size=4096,
        ...     head_dim=128,
        ...     total_num_heads=32,
        ...     total_num_kv_heads=8,  # GQA
        ... )
        >>> qkv = qkv_proj(x)  # [batch, (32+8+8)*128]
        >>> q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """
        初始化 QKV 并行投影层
        
        Args:
            hidden_size: 隐藏层维度
            head_size: 每个注意力头的维度
            total_num_heads: 总注意力头数
            total_num_kv_heads: KV 头数（None 时等于 num_heads）
            bias: 是否使用偏置
        """
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        self.head_size = head_size
        # 当前 GPU 的头数（张量并行切分）
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        
        # 总输出维度 = Q + K + V
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        权重加载器：加载 Q/K/V 指定分片的权重
        
        Args:
            param: 当前参数（合并的 QKV 分片权重）
            loaded_weight: 完整的单个投影权重（q/k/v）
            loaded_shard_id: 分片标识（"q"、"k" 或 "v"）
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        # 根据分片标识计算偏移和大小
        if loaded_shard_id == "q":
            # Q 投影：位于合并权重的起始位置
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            # K 投影：位于 Q 之后
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # loaded_shard_id == "v"
            # V 投影：位于 K 之后
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # 从合并参数中切分出当前分片对应的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        
        # 对输入权重进行张量并行切分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层（输入维度切分）
    
    将权重矩阵按输入维度（列）切分到多个 GPU 上：
    - 每个 GPU 存储 [output_size, input_size/tp_size] 的权重
    - 输入 x 在各 GPU 上为完整输入的 1/tp_size
    - 输出 y 需要 AllReduce 求和得到完整结果
    
    并行策略示意：
        W = [W_0, W_1, ..., W_{n-1}]  # 按列切分
        x = [x_0, x_1, ..., x_{n-1}]  # 输入也切分
        y = sum(x_i @ W_i^T)          # AllReduce 求和
    
    适用于：
        - 注意力输出投影（o_proj）
        - MLP 输出投影（down_proj）
    
    Attributes:
        tp_dim: 切分维度为 1（输入维度）
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维度按 tp_size 切分
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器：加载当前 GPU 对应的权重分片
        
        Args:
            param: 当前参数（分片权重）
            loaded_weight: 完整预训练权重
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        # 计算当前 rank 对应的起始索引
        start_idx = self.tp_rank * shard_size
        # 从完整权重中切分出当前 GPU 负责的部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行并行前向传播
        
        Args:
            x: 部分输入张量（当前 GPU 对应的分片）
            
        Returns:
            完整输出张量（经过 AllReduce 求和）
        """
        # 只有 rank=0 使用偏置（避免重复加偏置）
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        
        if self.tp_size > 1:
            # AllReduce：将所有 GPU 的部分结果相加
            dist.all_reduce(y)
        
        return y
