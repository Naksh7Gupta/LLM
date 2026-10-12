"""
Token Embedding Layer

Converts token IDs into dense learnable vectors.

Input:
    [batch_size, seq_len]

Output:
    [batch_size, seq_len, hidden_size]
"""

import torch
import torch.nn as nn

from configs.model_config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Maps token IDs to dense embedding vectors.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:
                Shape -> [B, T]

        Returns:
            embeddings:
                Shape -> [B, T, H]
        """

        return self.embedding(input_ids)