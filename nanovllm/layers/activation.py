"""
激活函数模块

实现 SwiGLU 激活函数（SiLU + Mul），用于 Qwen3 等模型的 MLP 层。
SwiGLU 是一种门控线性单元，公式为：SwiGLU(x) = SiLU(x_gate) * x_up
"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SiLU 激活与乘法融合模块（SwiGLU）
    
    将输入张量沿最后一维分成两半，对前半部分应用 SiLU 激活，
    然后与后半部分相乘。这是 SwiGLU 门控机制的标准实现。
    
    数学公式：
        SwiGLU(x) = SiLU(x[:d]) * x[d:]
        其中 SiLU(x) = x * sigmoid(x)
    
    应用场景：
        - Qwen3、Llama 等模型的 MLP 中间层
        - 与 MergedColumnParallelLinear 配合使用
    
    Attributes:
        无
    
    Example:
        >>> act = SiluAndMul()
        >>> x = torch.randn(2, 1024)  # [batch, 2*intermediate_size]
        >>> output = act(x)           # [batch, intermediate_size]
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 SwiGLU 激活
        
        Args:
            x: 输入张量，形状为 [..., 2*intermediate_size]
            
        Returns:
            输出张量，形状为 [..., intermediate_size]
        """
        # 沿最后一维分成两半：gate 和 up
        x, y = x.chunk(2, -1)
        # SwiGLU(x, y) = SiLU(x) * y
        return F.silu(x) * y
