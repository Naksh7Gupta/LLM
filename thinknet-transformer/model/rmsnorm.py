"""
Root Mean Square Layer Normalization (RMSNorm)

Paper:
https://arxiv.org/abs/1910.07467

Input Shape:
    [batch_size, seq_len, hidden_size]

Output Shape:
    [batch_size, seq_len, hidden_size]
"""

import torch
import torch.nn as nn

from configs.model_config import ModelConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not subtract the mean.
    It only normalizes using the root mean square (RMS).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.eps = config.rms_norm_eps

        # Learnable scaling parameter
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states:
                Shape -> [B, T, H]

        Returns:
            Shape -> [B, T, H]
        """

        # Compute RMS along the hidden dimension
        rms = torch.rsqrt(
            hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        # Normalize and apply learnable scale
        return hidden_states * rms * self.weight