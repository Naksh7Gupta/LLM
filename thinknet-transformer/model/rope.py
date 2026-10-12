"""
Rotary Positional Embeddings (RoPE)

Applies rotary positional embeddings to Query and Key tensors.

Input:
    q : [B, T, H, D]
    k : [B, T, H, D]

Output:
    q_rot : [B, T, H, D]
    k_rot : [B, T, H, D]
"""

import torch
import torch.nn as nn

from configs.model_config import ModelConfig


class RoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.theta = config.rope_theta

        # Inverse frequencies
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.head_dim, 2).float()
                / self.head_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        positions = torch.arange(self.max_seq_len).float()

        freqs = torch.outer(positions, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(
            "cos_cached",
            emb.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            emb.sin(),
            persistent=False,
        )

    def rotate_half(self, x: torch.Tensor):

        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]

        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ):

        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)

        return q, k