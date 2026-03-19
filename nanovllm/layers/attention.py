"""
注意力机制模块

实现基于 Flash Attention 的 KV Cache 管理注意力机制。
支持 Prefill 和 Decode 两种模式，以及前缀缓存优化。

核心功能：
1. 使用 Triton kernel 高效写入 KV Cache
2. Prefill 模式：使用 flash_attn_varlen_func 处理变长序列
3. Decode 模式：使用 flash_attn_with_kvcache 进行增量解码
4. 支持前缀缓存（prefix caching）优化
"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton Kernel：将 K/V 张量写入 KV Cache
    
    每个程序实例处理一个 token 的 K/V 向量，将其存储到 KV Cache 的指定位置。
    
    Args:
        key_ptr: Key 张量指针
        key_stride: Key 张量的 stride（第 0 维）
        value_ptr: Value 张量指针
        value_stride: Value 张量的 stride（第 0 维）
        k_cache_ptr: Key Cache 张量指针
        v_cache_ptr: Value Cache 张量指针
        slot_mapping_ptr: 槽位映射指针，将 token 索引映射到 KV Cache 位置
        D: 每个 token 的 K/V 总维度（num_heads * head_dim）
    """
    # 获取当前程序实例处理的 token 索引
    idx = tl.program_id(0)
    # 从 slot_mapping 中读取该 token 应存储的 KV Cache 槽位
    slot = tl.load(slot_mapping_ptr + idx)
    # 如果 slot 为 -1，表示该 token 不需要存储（如 warmup 场景）
    if slot == -1:
        return
    
    # 计算当前 token 的 K/V 在输入张量中的偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # 加载 K/V 数据
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # 计算在 KV Cache 中的偏移
    cache_offsets = slot * D + tl.arange(0, D)
    
    # 将 K/V 存储到 KV Cache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    将 K/V 张量写入 KV Cache 的封装函数
    
    启动 Triton kernel，每个 token 由一个 GPU 线程块处理。
    
    Args:
        key: Key 张量，形状为 (num_tokens, num_heads, head_dim)
        value: Value 张量，形状为 (num_tokens, num_kv_heads, head_dim)
        k_cache: Key Cache 张量，形状为 (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: Value Cache 张量，形状为 (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: 槽位映射，形状为 (num_tokens,)，每个元素表示 token 在 KV Cache 中的线性索引
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 展平后的维度
    
    # 验证张量布局连续性，确保 Triton kernel 能正确访问
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # 最后一维连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 第二维连续
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # KV Cache 布局匹配
    
    assert slot_mapping.numel() == N  # slot_mapping 长度必须等于 token 数
    
    # 启动 kernel：N 个程序实例，每个处理一个 token
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D
    )


class Attention(nn.Module):
    """
    注意力模块
    
    封装 Flash Attention，支持：
    - Prefill 模式：处理整个 prompt 序列，使用变长序列注意力
    - Decode 模式：每次生成一个 token，使用 KV Cache 加速
    - 前缀缓存：重用已计算的前缀 KV Cache
    
    Attributes:
        num_heads: 注意力头数
        head_dim: 每个头的维度
        scale: 缩放因子（通常为 1/sqrt(head_dim)）
        num_kv_heads: KV 头数（用于 GQA/MQA）
        k_cache: Key Cache 张量
        v_cache: Value Cache 张量
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 初始化空的 KV Cache，实际使用时由外部绑定
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        执行注意力计算
        
        根据上下文状态自动选择 Prefill 或 Decode 模式。
        
        Args:
            q: Query 张量
                - Prefill: (total_q, num_heads, head_dim)
                - Decode: (batch_size, num_heads, head_dim)
            k: Key 张量
                - Prefill: (total_k, num_kv_heads, head_dim)
                - Decode: (batch_size, num_kv_heads, head_dim)
            v: Value 张量
                - Prefill: (total_k, num_kv_heads, head_dim)
                - Decode: (batch_size, num_kv_heads, head_dim)
                
        Returns:
            o: 注意力输出张量
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 如果 KV Cache 已分配，将当前 K/V 写入 Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # Prefill 模式：处理整个序列
            if context.block_tables is not None:
                # 有前缀缓存：需要从 KV Cache 中加载历史 K/V
                k, v = k_cache, v_cache
            
            # 使用 Flash Attention 的变长序列版本
            # 支持多个序列的批处理，每个序列长度可以不同
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,      # 最大 query 序列长度
                cu_seqlens_q=context.cu_seqlens_q,      # query 累积序列长度
                max_seqlen_k=context.max_seqlen_k,      # 最大 key 序列长度
                cu_seqlens_k=context.cu_seqlens_k,      # key 累积序列长度
                softmax_scale=self.scale,               # 缩放因子
                causal=True,                            # 因果掩码（只能看前面的 token）
                block_table=context.block_tables,       # 块表（前缀缓存时使用）
            )
        else:
            # Decode 模式：每次只生成一个 token
            # 使用 KV Cache 存储的历史 K/V，避免重复计算
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),                         # (batch, 1, num_heads, head_dim)
                k_cache, v_cache,                       # KV Cache
                cache_seqlens=context.context_lens,     # 每个序列的上下文长度
                block_table=context.block_tables,       # 块表
                softmax_scale=self.scale,
                causal=True,
            )
        
        return o
