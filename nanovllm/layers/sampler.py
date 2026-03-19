"""
采样器模块

实现从 logits 概率分布中采样生成 token 的功能。
使用 Gumbel-Max 技巧的高效实现进行温度采样。

采样策略：
- 温度采样（Temperature Sampling）：通过温度参数控制分布的平滑程度
- 多项式采样（Multinomial Sampling）：从概率分布中随机采样

Gumbel-Max 技巧：
    标准的多项式采样需要计算累积分布并搜索，复杂度 O(vocab_size)
    Gumbel-Max 技巧通过以下方式简化采样：
    sample = argmax(logits / temperature - log(-log(epsilon)))
    其中 epsilon ~ Uniform(0, 1)
    
    等价形式（代码中使用）：
    sample = argmax(logits / temperature + Gumbel(0, 1))
    其中 Gumbel(0, 1) = -log(-log(epsilon)) = -log(exponential(1))
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """
    采样器模块
    
    从模型输出的 logits 中采样生成下一个 token。
    使用 Gumbel-Max 技巧实现高效的多项式采样。
    
    Attributes:
        无
    
    Example:
        >>> sampler = Sampler()
        >>> logits = torch.randn(4, 32000)  # [batch_size, vocab_size]
        >>> temperatures = torch.ones(4) * 0.8  # 温度参数
        >>> tokens = sampler(logits, temperatures)  # [batch_size]
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        从 logits 中采样 token
        
        使用 Gumbel-Max 技巧实现温度控制的多项式采样：
            1. 应用温度缩放：logits /= temperature
            2. 计算概率：probs = softmax(logits)
            3. Gumbel 噪声：g = -log(exponential(1))
            4. 采样：argmax(probs / g)
        
        步骤 3-4 是 Gumbel-Max 技巧的简化形式：
            argmax(softmax(logits) / g) = argmax(logits + gumbel_noise)
        
        Args:
            logits: 模型输出的 logits，形状为 [batch_size, vocab_size]
            temperatures: 温度参数，形状为 [batch_size]
                - temperature > 1：分布更平滑，增加随机性
                - temperature < 1：分布更尖锐，减少随机性
                - temperature = 1：原始分布
                
        Returns:
            sample_tokens: 采样的 token IDs，形状为 [batch_size]
        """
        # 步骤 1：应用温度缩放
        # 除以温度值，温度越大分布越平滑
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # 步骤 2：计算 softmax 概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 步骤 3-4：Gumbel-Max 技巧采样
        # 生成指数分布噪声：exponential(1)
        # Gumbel 噪声 = -log(exponential(1))，但这里直接用除法形式
        # probs / gumbel_noise 等价于 log(probs) + gumbel_noise
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        
        return sample_tokens
