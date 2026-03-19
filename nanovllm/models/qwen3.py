"""
Qwen3 模型实现

实现 Qwen3 大语言模型的完整架构，基于 Transformer Decoder-only 结构。

Qwen3 架构特点：
- Decoder-only Transformer 架构
- RoPE 旋转位置编码
- SwiGLU 激活函数（MLP 层）
- RMSNorm 归一化
- GQA（Grouped Query Attention）支持
- 张量并行支持

模型层级结构：
    Qwen3ForCausalLM
    ├── Qwen3Model
    │   ├── VocabParallelEmbedding (词嵌入)
    │   ├── Qwen3DecoderLayer × N
    │   │   ├── Qwen3Attention
    │   │   │   ├── QKVParallelLinear (QKV 投影)
    │   │   │   ├── RowParallelLinear (输出投影)
    │   │   │   ├── RotaryEmbedding (RoPE)
    │   │   │   └── Attention (Flash Attention)
    │   │   ├── Qwen3MLP
    │   │   │   ├── MergedColumnParallelLinear (gate_up)
    │   │   │   ├── RowParallelLinear (down)
    │   │   │   └── SiluAndMul (SwiGLU 激活)
    │   │   ├── RMSNorm (input_layernorm)
    │   │   └── RMSNorm (post_attention_layernorm)
    │   └── RMSNorm (final_norm)
    └── ParallelLMHead (语言模型输出头)
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 注意力层
    
    实现多头注意力机制，支持 GQA（Grouped Query Attention）和 RoPE 位置编码。
    
    GQA 配置：
        - num_heads: 查询头数
        - num_kv_heads: KV 头数
        - 当 num_kv_heads < num_heads 时为 GQA
        - 当 num_kv_heads = num_heads 时为 MHA（标准多头注意力）
        - 当 num_kv_heads = 1 时为 MQA（多查询注意力）
    
    结构：
        1. QKV 投影：将 hidden_states 投影到 Q、K、V
        2. Q/K 归一化：使用 RMSNorm（可选）
        3. RoPE：应用旋转位置编码
        4. Attention：Flash Attention 计算
        5. 输出投影：将结果投影回 hidden_size
    
    Attributes:
        total_num_heads: 总注意力头数
        num_heads: 当前 GPU 的注意力头数（张量并行切分后）
        total_num_kv_heads: 总 KV 头数
        num_kv_heads: 当前 GPU 的 KV 头数
        head_dim: 每个头的维度
        q_size: Q 投影的输出维度
        kv_size: K/V 投影的输出维度
        scaling: 注意力缩放因子（1/sqrt(head_dim)）
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        """
        初始化 Qwen3 注意力层
        
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            num_kv_heads: KV 头数（GQA 支持）
            max_position: 最大位置数（用于 RoPE）
            head_dim: 每个头的维度（None 时自动计算为 hidden_size/num_heads）
            rms_norm_eps: RMSNorm 的 eps
            qkv_bias: QKV 投影是否使用偏置
            rope_theta: RoPE 基数
            rope_scaling: RoPE 缩放配置
        """
        super().__init__()
        tp_size = dist.get_world_size()
        
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0  # 确保可整除
        self.num_heads = self.total_num_heads // tp_size  # 当前 GPU 的头数
        
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0  # 确保可整除
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 当前 GPU 的 KV 头数
        
        # 计算头维度
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim  # Q 的输出维度
        self.kv_size = self.num_kv_heads * self.head_dim  # K/V 的输出维度
        self.scaling = self.head_dim ** -0.5  # 缩放因子
        self.qkv_bias = qkv_bias

        # QKV 联合投影（合并为一个矩阵乘法）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        
        # 输出投影
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        
        # RoPE 旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # Flash Attention 封装
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        
        # 如果 QKV 投影没有偏置，则使用 Q/K 归一化
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        注意力层前向传播
        
        Args:
            positions: 位置索引，形状为 [num_tokens]
            hidden_states: 隐藏状态，形状为 [num_tokens, hidden_size]
            
        Returns:
            输出张量，形状为 [num_tokens, hidden_size]
        """
        # QKV 联合投影：[num_tokens, (num_heads + 2*num_kv_heads) * head_dim]
        qkv = self.qkv_proj(hidden_states)
        
        # 分离 Q、K、V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 重塑为多头格式：[num_tokens, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # Q/K 归一化（如果 QKV 投影没有偏置）
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # 应用 RoPE 位置编码
        q, k = self.rotary_emb(positions, q, k)
        
        # Flash Attention 计算
        o = self.attn(q, k, v)
        
        # 输出投影（多头展平后投影回 hidden_size）
        output = self.o_proj(o.flatten(1, -1))
        
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 MLP 层（前馈神经网络）
    
    使用 SwiGLU 激活函数的两阶段 MLP：
        1. gate_up_proj: hidden_states -> [gate, up]
        2. SwiGLU: SiLU(gate) * up
        3. down_proj: intermediate -> hidden_size
    
    结构：
        MLP(x) = down_proj(SwiGLU(gate_up_proj(x)))
        其中 SwiGLU(gate, up) = SiLU(gate) * up
    
    Attributes:
        gate_up_proj: 合并的门控和上投影层
        down_proj: 下投影层
        act_fn: SwiGLU 激活函数
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """
        初始化 Qwen3 MLP 层
        
        Args:
            hidden_size: 隐藏层维度
            intermediate_size: 中间层维度（扩展维度）
            hidden_act: 激活函数名称（当前仅支持 "silu"）
        """
        super().__init__()
        
        # 合并的 gate 和 up 投影（输出维度为 2*intermediate_size）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        
        # 下投影（从 intermediate_size 回到 hidden_size）
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        
        # Qwen3 使用 SiLU 激活
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """
        MLP 前向传播
        
        Args:
            x: 输入张量，形状为 [num_tokens, hidden_size]
            
        Returns:
            输出张量，形状为 [num_tokens, hidden_size]
        """
        # 门控和上投影：[num_tokens, 2 * intermediate_size]
        gate_up = self.gate_up_proj(x)
        
        # SwiGLU 激活：SiLU(gate) * up
        x = self.act_fn(gate_up)
        
        # 下投影
        x = self.down_proj(x)
        
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 Decoder 层
    
    Transformer Decoder 的基本单元，包含：
    1. 自注意力层（带残差和归一化）
    2. MLP 层（带残差和归一化）
    
    结构（Pre-Norm 架构）：
        hidden_states, residual = input_layernorm(hidden_states), hidden_states
        hidden_states = self_attn(positions, hidden_states)
        hidden_states, residual = post_attention_layernorm(hidden_states, residual)
        hidden_states = mlp(hidden_states)
        return hidden_states, residual
    
    Attributes:
        self_attn: Qwen3 自注意力层
        mlp: Qwen3 MLP 层
        input_layernorm: 输入层归一化
        post_attention_layernorm: 注意力后归一化
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        初始化 Qwen3 Decoder 层
        
        Args:
            config: Qwen3 配置对象
        """
        super().__init__()
        
        # 自注意力层
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        
        # MLP 层
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        # 归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decoder 层前向传播
        
        Args:
            positions: 位置索引
            hidden_states: 隐藏状态
            residual: 残差（第一层为 None）
            
        Returns:
            tuple: (新的隐藏状态，新的残差)
        """
        # 自注意力块
        if residual is None:
            # 第一层：初始化残差
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：融合残差连接
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # 自注意力计算
        hidden_states = self.self_attn(positions, hidden_states)
        
        # MLP 块
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 基础模型
    
    Transformer Decoder 堆栈，包含：
    - 词嵌入层
    - N 个 Decoder 层
    - 最终归一化层
    
    Attributes:
        embed_tokens: 词嵌入层（词汇表并行）
        layers: Decoder 层列表
        norm: 最终归一化层
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        初始化 Qwen3 基础模型
        
        Args:
            config: Qwen3 配置对象
        """
        super().__init__()
        
        # 词嵌入层（词汇表并行）
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        
        # Decoder 层堆栈
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Qwen3 模型前向传播
        
        Args:
            input_ids: 输入 token IDs，形状为 [batch_size, seq_len]
            positions: 位置索引，形状为 [total_tokens]
            
        Returns:
            隐藏状态，形状为 [total_tokens, hidden_size]
        """
        # 词嵌入查找
        hidden_states = self.embed_tokens(input_ids)
        
        # 初始化残差
        residual = None
        
        # 逐层处理
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 最终归一化（融合残差）
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型
    
    用于自回归语言模型的完整 Qwen3 实现，包含基础模型和语言模型输出头。
    
    Attributes:
        model: Qwen3 基础模型
        lm_head: 语言模型输出头（词汇表并行）
    
    Class Attributes:
        packed_modules_mapping: 权重打包映射，用于从 HuggingFace 格式加载权重
            - q_proj, k_proj, v_proj 打包到 qkv_proj
            - gate_proj, up_proj 打包到 gate_up_proj
    """
    
    # 权重打包映射：用于从 HuggingFace 格式加载权重
    # 格式：{子模块名：(父模块名，分片标识)}
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        """
        初始化 Qwen3 因果语言模型
        
        Args:
            config: Qwen3 配置对象
        """
        super().__init__()
        
        # 基础模型
        self.model = Qwen3Model(config)
        
        # 语言模型输出头
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 权重绑定（如果配置启用）
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        模型前向传播
        
        Args:
            input_ids: 输入 token IDs
            positions: 位置索引
            
        Returns:
            隐藏状态（用于 compute_logits 计算 logits）
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算词汇表 logits
        
        Args:
            hidden_states: 模型输出的隐藏状态
            
        Returns:
            logits: 词汇表上的 logits 分布
        """
        return self.lm_head(hidden_states)
