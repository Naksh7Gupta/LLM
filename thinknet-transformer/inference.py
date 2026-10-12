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
    vocab_size = 25000

    max_seq_len = 1024

    hidden_size = 512

    num_layers = 12

    num_heads = 8

    head_dim = 64

    num_kv_heads = 2      # GQA

    intermediate_size = 1536   # Llama-style ~3× hidden

    use_bias = False

    tie_word_embeddings = True

    rms_norm_eps = 1e-6

    rope_theta = 10000.0