"""
层归一化模块

实现 RMSNorm（Root Mean Square Layer Normalization），一种轻量级的归一化方法。
相比标准 LayerNorm，RMSNorm 去除了均值中心化和偏置项，仅使用均方根进行归一化。

RMSNorm 公式：
    RMS = sqrt(mean(x^2) + eps)
    output = (x / RMS) * weight

优势：
    - 计算更高效（少一个均值计算）
    - 内存占用更少（无偏置项）
    - 在大模型训练中表现与 LayerNorm 相当

支持两种模式：
    1. 标准归一化：仅输入 hidden_states
    2. 残差融合归一化：同时处理 hidden_states + residual
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    RMS 归一化层
    
    对输入张量进行 RMSNorm 归一化，支持可选的残差连接融合。
    
    Attributes:
        eps: 数值稳定性常数，防止除零（默认 1e-6）
        weight: 可学习的缩放参数，形状为 (hidden_size,)
    
    Example:
        >>> norm = RMSNorm(hidden_size=4096)
        >>> x = torch.randn(2, 4096)
        >>> output = norm(x)  # 标准归一化
        >>> output, residual = norm(x, residual=torch.randn(2, 4096))  # 残差融合
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """
        初始化 RMSNorm
        
        Args:
            hidden_size: 输入特征的维度
            eps: 数值稳定性常数
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为 1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        标准 RMS 归一化（无残差）
        
        Args:
            x: 输入张量，形状为 [..., hidden_size]
            
        Returns:
            归一化后的张量
        """
        orig_dtype = x.dtype
        # 转换为 float32 进行计算，提高数值稳定性
        x = x.float()
        # 计算均方根：sqrt(mean(x^2) + eps)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 归一化并缩放
        x.mul_(torch.rsqrt(var + self.eps))
        # 恢复原始精度并应用权重
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        残差融合 RMS 归一化
        
        将输入与残差相加后进行归一化，并返回归一化结果和新的残差。
        这种融合操作可以减少 GPU 内存访问，提高效率。
        
        Args:
            x: 输入张量，形状为 [..., hidden_size]
            residual: 残差张量，形状与 x 相同
            
        Returns:
            tuple: (归一化后的张量，新的残差)
        """
        orig_dtype = x.dtype
        # 将 x 和 residual 相加（转换为 float32）
        x = x.float().add_(residual.float())
        # 保存残差结果（用于下一层）
        residual = x.to(orig_dtype)
        # 计算均方根
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 归一化并缩放
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        RMSNorm 前向传播
        
        根据是否提供 residual 参数，自动选择标准模式或残差融合模式。
        
        Args:
            x: 输入张量，形状为 [..., hidden_size]
            residual: 可选的残差张量
            
        Returns:
            - 无 residual 时：返回归一化后的张量
            - 有 residual 时：返回 (归一化张量，新残差) 元组
        """
        if residual is None:
            # 标准模式：仅归一化
            return self.rms_forward(x)
        else:
            # 残差融合模式
            return self.add_rms_forward(x, residual)
