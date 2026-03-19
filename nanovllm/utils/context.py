"""
上下文管理模块

提供全局上下文管理，用于在模型前向传播过程中传递注意力计算所需的运行时信息。

设计目的：
1. 避免在多层嵌套调用中传递大量参数
2. 统一管理和访问注意力计算的上下文信息
3. 支持 Prefill 和 Decode 两种模式的上下文切换

使用场景：
- ModelRunner.prepare_prefill/prepare_decode 设置上下文
- Attention.forward 从上下文获取 KV Cache 和块表信息
- 每层 Decoder 共享同一上下文
"""

from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    注意力计算的上下文数据类
    
    存储 Prefill 和 Decode 模式下的运行时信息，供 Attention 层使用。
    
    Attributes:
        is_prefill: 是否为 Prefill 模式
            - True: Prefill 阶段，处理整个 prompt 序列
            - False: Decode 阶段，每次生成一个 token
        
        cu_seqlens_q: Query 累积序列长度（Prefill 模式）
            - 形状：[batch_size + 1]
            - 用途：flash_attn_varlen_func 需要此参数处理变长序列
            - 示例：[0, 3, 5, 9] 表示 3 个序列，长度分别为 3, 2, 4
        
        cu_seqlens_k: Key 累积序列长度（Prefill 模式）
            - 形状：[batch_size + 1]
            - 用途：当有前缀缓存时，key 序列长度可能大于 query
        
        max_seqlen_q: 最大 Query 序列长度（Prefill 模式）
            - 用途：flash_attn_varlen_func 需要此参数分配共享内存
        
        max_seqlen_k: 最大 Key 序列长度（Prefill 模式）
            - 用途：当有前缀缓存时，可能需要更大的共享内存
        
        slot_mapping: KV Cache 槽位映射
            - 形状：[num_tokens]
            - 用途：将每个 token 映射到 KV Cache 的物理槽位
            - 示例：slot_mapping[i] = 128 表示第 i 个 token 存储在 KV Cache 的第 128 个位置
        
        context_lens: 上下文长度（Decode 模式）
            - 形状：[batch_size]
            - 用途：flash_attn_with_kvcache 需要此参数知道每个序列的历史长度
            - 示例：[10, 15, 8] 表示 3 个序列的当前长度
        
        block_tables: 块表（Decode 模式/前缀缓存）
            - 形状：[batch_size, max_num_blocks]
            - 用途：将逻辑块索引映射到物理块索引
            - 示例：block_tables[i, j] = 5 表示第 i 个序列的第 j 个逻辑块存储在物理块 5
            - -1 表示无效块（padding）
    
    Example:
        # Prefill 模式上下文
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 3, 5]),
            cu_seqlens_k=torch.tensor([0, 3, 5]),
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.tensor([0, 1, 2, 256, 257])
        )
        
        # Decode 模式上下文
        set_context(
            is_prefill=False,
            slot_mapping=torch.tensor([10, 20, 30]),
            context_lens=torch.tensor([10, 15, 8]),
            block_tables=torch.tensor([[0, 1, -1], [2, 3, 4], [5, -1, -1]])
        )
    """
    
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


# 全局上下文实例
# 使用模块级全局变量，避免在函数调用链中传递
_CONTEXT = Context()


def get_context():
    """
    获取全局上下文实例
    
    Returns:
        Context: 当前上下文对象
    
    Example:
        >>> context = get_context()
        >>> if context.is_prefill:
        ...     # 执行 Prefill 逻辑
        >>> else:
        ...     # 执行 Decode 逻辑
    """
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None
):
    """
    设置全局上下文
    
    在 ModelRunner.prepare_prefill 或 ModelRunner.prepare_decode 中调用，
    为后续的模型前向传播设置正确的上下文信息。
    
    Args:
        is_prefill: 是否为 Prefill 模式
        cu_seqlens_q: Query 累积序列长度（Prefill 模式）
        cu_seqlens_k: Key 累积序列长度（Prefill 模式）
        max_seqlen_q: 最大 Query 序列长度（Prefill 模式）
        max_seqlen_k: 最大 Key 序列长度（Prefill 模式）
        slot_mapping: KV Cache 槽位映射
        context_lens: 上下文长度（Decode 模式）
        block_tables: 块表（Decode 模式/前缀缓存）
    
    Example:
        # Prefill 模式
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping
        )
        
        # Decode 模式
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables
        )
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables
    )


def reset_context():
    """
    重置全局上下文
    
    在每次推理步骤完成后调用，将上下文恢复为初始状态。
    这可以避免上下文信息泄露到下一次推理。
    
    Example:
        >>> # 执行推理
        >>> logits = model(input_ids, positions)
        >>> # 重置上下文
        >>> reset_context()
    """
    global _CONTEXT
    _CONTEXT = Context()
