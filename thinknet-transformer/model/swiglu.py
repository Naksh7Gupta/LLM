"""
SwiGLU Feed Forward Network

Input:
    [B, T, hidden_size]

Output:
    [B, T, hidden_size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, hidden_size]
        """

        gate = F.silu(self.gate_proj(x))

        up = self.up_proj(x)

        hidden = gate * up

        output = self.down_proj(hidden)

        return output