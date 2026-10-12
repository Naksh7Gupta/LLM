"""
ThinkNet Transformer Configuration

This file contains all hyperparameters required to build
our decoder-only transformer architecture.

Changing values here should automatically scale the model
without modifying any model code.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:

    # -----------------------------
    # Vocabulary
    # -----------------------------
    vocab_size: int = 20000

    # -----------------------------
    # Context Length
    # -----------------------------
    max_seq_len: int = 1024

    # -----------------------------
    # Transformer Dimensions
    # -----------------------------
    hidden_size: int = 768

    num_layers: int = 12

    num_heads: int = 12

    # dimension per head (computed if not provided)
    head_dim: int | None = None

    # Grouped Query Attention
    num_kv_heads: int = 4

    # Feed Forward Expansion
    intermediate_size: int = 2304

    # -----------------------------
    # Normalization
    # -----------------------------
    rms_norm_eps: float = 1e-6

    # -----------------------------
    # RoPE
    # -----------------------------
    rope_theta: float = 10000.0

    # -----------------------------
    # Dropout
    # -----------------------------
    dropout: float = 0.0

    # -----------------------------
    # Bias
    # -----------------------------
    use_bias: bool = False

    # -----------------------------
    # Weight Tying
    # -----------------------------
    tie_word_embeddings: bool = True

    def __post_init__(self):
        # compute head_dim from hidden_size and num_heads if not set
        if self.head_dim is None:
            self.head_dim = max(1, self.hidden_size // self.num_heads)