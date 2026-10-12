import torch
import torch.nn as nn

from configs.model_config import ModelConfig
from model.embedding import TokenEmbedding
from model.block import TransformerBlock
from model.rmsnorm import RMSNorm


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Token Embedding
        self.embed_tokens = TokenEmbedding(config)

        # Transformer Blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        # Final RMSNorm
        self.norm = RMSNorm(config)

        # Language Modeling Head
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Weight Tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
    ):

        B, T = input_ids.shape

        position_ids = torch.arange(
            T,
            device=input_ids.device,
        ).unsqueeze(0).expand(B, T)

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids,
            )

        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits