import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import ModelConfig
from model.rope import RoPE


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Q Projection
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_bias,
        )

        # K Projection
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.use_bias,
        )

        # V Projection
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.use_bias,
        )

        # Output Projection
        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.use_bias,
        )

        self.rope = RoPE(config)

    def repeat_kv(self, x):
        """
        x:
            [B, T, num_kv_heads, head_dim]

        ->
            [B, T, num_heads, head_dim]
        """

        B, T, H, D = x.shape

        x = x.unsqueeze(3)

        x = x.expand(
            B,
            T,
            H,
            self.num_kv_groups,
            D,
        )

        return x.reshape(
            B,
            T,
            self.num_heads,
            D,
        )

    def forward(
        self,
        hidden_states,
        position_ids,
    ):

        B, T, _ = hidden_states.shape

        # ----------------------------
        # QKV Projection
        # ----------------------------

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # ----------------------------
        # Reshape
        # ----------------------------

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )

        k = k.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        )

        v = v.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        )

        # ----------------------------
        # RoPE
        # ----------------------------

        q, k = self.rope(
            q,
            k,
            position_ids,
        )

        # ----------------------------
        # GQA
        # ----------------------------

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # ----------------------------
        # FlashAttention expects
        # [B, H, T, D]
        # ----------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----------------------------
        # FlashAttention
        # ----------------------------

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        # ----------------------------
        # Merge Heads
        # ----------------------------

        output = output.transpose(1, 2)

        output = output.reshape(
            B,
            T,
            self.hidden_size,
        )

        # ----------------------------
        # Output Projection
        # ----------------------------

        output = self.o_proj(output)

        return output