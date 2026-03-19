"""
旋转位置编码模块（RoPE - Rotary Positional Embedding）

实现 RoPE 旋转位置编码，将位置信息通过旋转矩阵编码到 Q/K 向量中。

RoPE 核心思想：
- 将 token 的位置信息编码为复数空间的旋转角度
- 通过旋转操作实现相对位置编码
- 具有良好的外推性和长度泛化能力

RoPE 公式：
    对于位置 m 的 token，其 Q/K 向量旋转角度 θ：
    Q_m = Q * e^(i*m*θ)
    K_n = K * e^(i*n*θ)
    
    注意力分数：Q_m · K_n = |Q||K| * e^(i*(m-n)*θ)
    只依赖于相对位置 (m-n)

优势：
- 支持长度外推（推理时可处理比训练更长的序列）
- 相对位置编码，具有良好的归纳偏置
- 计算高效，无需额外参数
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码
    
    将输入张量 x 按 RoPE 公式进行旋转编码。
    
    RoPE 旋转公式（实数域等价形式）：
        将向量分成两半 [x1, x2]，旋转后：
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
    
    这相当于复数乘法：(x1 + i*x2) * (cos + i*sin)
    
    Args:
        x: 输入张量，形状为 [..., head_dim]
        cos: 余弦值，形状为 [..., head_dim/2]
        sin: 正弦值，形状为 [..., head_dim/2]
        
    Returns:
        旋转编码后的张量，形状与输入相同
    """
    # 将最后一维分成两半
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    
    # 应用旋转公式
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    # 拼接回完整维度并恢复原始精度
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    RoPE 旋转位置编码层
    
    预先计算所有位置的 sin/cos 值，运行时根据位置索引查找。
    
    频率计算：
        inv_freq[i] = 1 / (base^(2i/d))
        其中 d 是 rotary_dim，base 是基数（通常 10000 或 1000000）
        
        位置 m 的角度：
        freqs[m, i] = m * inv_freq[i]
        cos[m, i] = cos(freqs[m, i])
        sin[m, i] = sin(freqs[m, i])
    
    Attributes:
        head_size: 注意力头维度
        cos_sin_cache: 预计算的 cos/sin 缓存，形状为 [max_position, 1, 2*rotary_dim]
    
    Example:
        >>> rope = RotaryEmbedding(head_size=128, rotary_dim=128, 
        ...                        max_position=8192, base=1000000)
        >>> positions = torch.arange(10)  # 前 10 个位置
        >>> q, k = rope(positions, query, key)  # 应用 RoPE
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """
        初始化 RoPE 层
        
        Args:
            head_size: 注意力头维度
            rotary_dim: 旋转维度（通常等于 head_size）
            max_position_embeddings: 最大位置数
            base: RoPE 基数，控制频率范围
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size  # 当前实现要求 rotary_dim 等于 head_size
        
        # 计算逆频率：1 / (base^(2i/d))
        # 这是一个几何序列，频率从 1 到 1/base
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 计算所有位置的角度
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 外积：[max_pos, rotary_dim/2]
        
        # 计算 cos 和 sin
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 拼接并缓存：[max_pos, 1, rotary_dim]
        # 第 1 维为 1 是为了广播到 [batch, seq_len, head_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        
        # 注册为 buffer（不参与梯度计算）
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        应用 RoPE 到 Q/K 向量
        
        Args:
            positions: 位置索引，形状为 [num_tokens]
            query: Query 向量，形状为 [num_tokens, num_heads, head_dim]
            key: Key 向量，形状为 [num_tokens, num_kv_heads, head_dim]
            
        Returns:
            tuple: (旋转后的 query, 旋转后的 key)
        """
        # 根据位置索引查找 cos/sin：[num_tokens, 1, 2*rotary_dim]
        cos_sin = self.cos_sin_cache[positions]
        
        # 分离 cos 和 sin：各为 [num_tokens, 1, rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        # 应用旋转编码（通过广播自动应用到所有头）
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取 RoPE 实例（带缓存）
    
    使用 lru_cache 避免重复创建相同配置的 RoPE 实例。
    
    Args:
        head_size: 注意力头维度
        rotary_dim: 旋转维度
        max_position: 最大位置数
        base: RoPE 基数
        rope_scaling: RoPE 缩放配置（当前不支持）
        
    Returns:
        RotaryEmbedding 实例
        
    Raises:
        AssertionError: 当 rope_scaling 不为 None 时
    """
    # 当前不支持 RoPE 缩放（如线性缩放、动态缩放等）
    assert rope_scaling is None
    
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
