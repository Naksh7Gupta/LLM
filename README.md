# ThinkNet Transformer: Decoder-Only Large Language Model from Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture: Decoder--Only](https://img.shields.io/badge/Architecture-Decoder--Only%20LLM-success.svg)](#architecture-overview)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-HQ--Chat--1K-FFD21E.svg)](https://huggingface.co/datasets/ThinkNet/HQ-Chat-1K)

A modern, high-performance **decoder-only Large Language Model (LLM)** built from scratch in PyTorch. **ThinkNet Transformer** incorporates state-of-the-art architectures inspired by LLaMA and Mistral—including **Rotary Position Embeddings (RoPE)**, **Grouped Query Attention (GQA)**, **SwiGLU activations**, and **RMSNorm**—alongside a scalable mixed-precision pretraining and evaluation pipeline.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [Architecture Overview](#architecture-overview)
  - [Model Specifications](#model-specifications)
  - [Core Components](#core-components)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Pipeline](#data-pipeline)
  - [1. Pretraining Corpus (FineWeb-Edu)](#1-pretraining-corpus-fineweb-edu)
  - [2. Instruction Tuning Dataset](#2-instruction-tuning-dataset)
  - [3. Tokenizer](#3-tokenizer)
- [Training](#training)
  - [Features](#features)
  - [Running Training](#running-training)
  - [Configuration Overrides](#configuration-overrides)
- [Inference & Generation](#inference--generation)
- [Testing & Validation](#testing--validation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Key Highlights

- **Modern Architecture**: Implements modern design patterns from contemporary open-source foundational models (LLaMA 2/3, Mistral):
  - **Grouped Query Attention (GQA)**: Reduces KV-cache memory footprint and accelerates inference speed while preserving attention capacity.
  - **Rotary Position Embeddings (RoPE)**: Dynamic, relative position encoding via complex rotation on key/query projections.
  - **SwiGLU Feed-Forward Networks**: Swish-Gated Linear Units for improved non-linear representation over standard GeLU/ReLU MLPs.
  - **RMSNorm (Pre-LayerNorm)**: Root Mean Square Layer Normalization for stable gradient flow and computational efficiency.
  - **FlashAttention Support**: Leverages PyTorch's `F.scaled_dot_product_attention` for memory-efficient causal attention calculation.
  - **Tied Word Embeddings**: Shares weights between token embeddings and the language modeling classification head to minimize parameter overhead.
- **Production-Ready Training Loop**:
  - Automatic Mixed Precision (AMP FP16) with gradient scaling (`GradScaler`).
  - Gradient accumulation for emulating large effective batch sizes on consumer hardware.
  - Cosine learning rate scheduling with warm-up.
  - Gradient norm clipping to prevent exploding gradients.
  - Mid-training generation hooks to monitor qualitative language modeling progress in real time.
- **End-to-End Pipeline**: Includes corpus harvesting, custom Byte-Pair Encoding (BPE) tokenization, dataset serialization, training, checkpointing, and autoregressive generation.

---

## Architecture Overview

### Model Specifications

The architecture is fully parametrized via [`ModelConfig`](thinknet-transformer/configs/model_config.py):

| Hyperparameter | Default Value | Description |
| :--- | :--- | :--- |
| **Vocabulary Size** | `20,000` | Total tokens in BPE vocabulary |
| **Max Sequence Length** (`context`) | `1,024` tokens | Maximum context window |
| **Hidden Dimension** ($d_{\text{model}}$) | `768` | Embedding and hidden representation size |
| **Layers / Transformer Blocks** | `12` | Depth of stacked decoder layers |
| **Attention Query Heads** ($H_Q$) | `12` | Number of parallel query attention heads |
| **Key/Value Heads** ($H_{KV}$) | `4` | Number of key/value heads for GQA (group ratio = 3) |
| **Head Dimension** ($d_{\text{head}}$) | `64` | Dimension per attention head ($768 / 12$) |
| **Intermediate Size** ($d_{\text{ffn}}$) | `2,304` | Hidden dimension of the SwiGLU MLP (~$3 \times d_{\text{model}}$) |
| **RoPE Base Frequency** ($\theta$) | `10,000.0` | Base period for rotary embeddings |
| **Normalization** | RMSNorm ($\epsilon = 10^{-6}$) | Pre-normalization on attention & FFN inputs |
| **Bias** | `False` | Zero bias in all linear projections for efficiency |
| **Weight Tying** | `True` | Shared embedding and LM projection weights |

### Core Components

```
                       Input Token IDs [B, T]
                                 │
                         Token Embedding
                                 │
                       ┌───────────────────┐
                       │   Pre-RMSNorm     │
                       │         │         │
                       │   GQA + RoPE      │ ◄── Causally Masked FlashAttention
                       │         │         │
                       │    Residual (+)   │
                       │         │         │
                       │   Pre-RMSNorm     │
                       │         │         │
                       │   SwiGLU FFN      │ ◄── Gate / Up / Down Projections
                       │         │         │
                       │    Residual (+)   │
                       └─────────┬─────────┘  (Repeated × N Layers)
                                 │
                           Final RMSNorm
                                 │
                      LM Head (Tied Weights)
                                 │
                        Logits [B, T, Vocab]
```

---

## Repository Structure

```text
LLM/
├── checkpoints/                    # Saved model and optimizer state checkpoints
├── data/                           # Data preparation scripts & corpus artifacts
│   ├── corpus_part2.txt            # Raw text corpus for pretraining
│   ├── shuffle_json.py             # Hugging Face dataset upload & preparation script
│   ├── tokens.pt                   # Pretokenized PyTorch tensor dataset
│   ├── tokens_small.pt             # Mini tokenized subset for rapid testing & debugging
│   └── json/
│       └── data.json               # Instruction chat dataset (ThinkNet/HQ-Chat-1K)
├── thinknet-transformer/           # Core model implementation
│   ├── configs/
│   │   └── model_config.py         # Model dataclass configuration & hyperparameters
│   ├── data/
│   │   └── dataset.py              # Streaming data downloader (FineWeb-Edu)
│   ├── model/
│   │   ├── attention.py            # Grouped Query Attention with RoPE & SDPA
│   │   ├── block.py                # Pre-LN Transformer decoder layer
│   │   ├── embedding.py            # Token embedding layer
│   │   ├── rmsnorm.py              # Root Mean Square Layer Normalization
│   │   ├── rope.py                 # Rotary Positional Embedding (RoPE) implementation
│   │   ├── swiglu.py               # Swish-Gated Linear Unit feedforward layer
│   │   └── transformer.py          # Autoregressive Decoder-Only Transformer
│   ├── scripts/
│   │   └── sync_files_safe.py      # Automated file synchronization & syntax check utility
│   ├── training/
│   │   └── trainer.py              # Mixed-precision training loop with scheduling & checkpointing
│   ├── data.py                     # TextDataset PyTorch sequence pair loader
│   ├── inference.py                # Inference generation script & config
│   ├── test.py                     # Component sanity check script
│   └── train.py                    # Training entry point
├── tokenizer/                      # Trained BPE Tokenizer files
│   ├── merges.txt                  # Byte-pair encoding merge rules
│   ├── tokenizer.json              # FastTokenizer configuration
│   └── vocab.json                  # Vocabulary token-to-index mapping
└── README.md                       # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+ or 12.x recommended (CPU execution supported for testing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Naksh7Gupta/LLM.git
   cd LLM
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install datasets tokenizers tqdm huggingface_hub
   ```

---

## Data Pipeline

### 1. Pretraining Corpus (FineWeb-Edu)

Pretraining data is streamed and gathered using the [`thinknet-transformer/data/dataset.py`](thinknet-transformer/data/dataset.py) pipeline from `HuggingFaceFW/fineweb-edu`:

```bash
python thinknet-transformer/data/dataset.py
```
This filters and extracts educational, high-quality content into `data/corpus_part2.txt`.

### 2. Instruction Tuning Dataset

For conversational and alignment fine-tuning, the repository includes a curated instruction dataset (`data/json/data.json`) mirrored on Hugging Face at [`ThinkNet/HQ-Chat-1K`](https://huggingface.co/datasets/ThinkNet/HQ-Chat-1K).

### 3. Tokenizer

A custom BPE tokenizer trained for this vocabulary is located in the [`tokenizer/`](tokenizer/) directory:
- Vocab Size: 20,000 - 25,000 tokens
- Special Tokens: `<pad>` (0), `<bos>` (1), `<eos>` (2)
- Fast execution via the Hugging Face `tokenizers` library.

---

## Training

The model features an optimized pretraining loop in [`thinknet-transformer/training/trainer.py`](thinknet-transformer/training/trainer.py).

### Features

- **Mixed-Precision (AMP FP16)**: Drastically reduces memory usage and leverages Tensor Cores.
- **Gradient Accumulation**: Aggregates backward passes across mini-batches before optimizer steps.
- **Cosine Learning Rate Annealing**: Smooth decay from $3 \times 10^{-4}$ down to near zero with a 500-step linear warm-up.
- **Periodic Checkpoints**: Automatically saves model weights, optimizer states, and step counts to `checkpoints/`.
- **Qualitative Generation Checks**: Evaluates sample completions periodically during training to inspect convergence.

### Running Training

To launch training using the default configurations:

```bash
python thinknet-transformer/train.py
```

### Configuration Overrides

You can easily override training hyperparameters via environment variables without modifying the code:

```bash
# On Linux / macOS:
BATCH_SIZE=8 GRAD_ACCUM_STEPS=4 EPOCHS=5 SAVE_STEPS=1000 python thinknet-transformer/train.py

# On Windows (PowerShell):
$env:BATCH_SIZE="8"; $env:GRAD_ACCUM_STEPS="4"; python thinknet-transformer/train.py
```

---

## Inference & Generation

Generate text autoregressively using a trained checkpoint:

```python
import torch
from configs.model_config import ModelConfig
from model.transformer import Transformer

# 1. Initialize configuration and model
config = ModelConfig()
model = Transformer(config)

# 2. Load trained checkpoint
checkpoint = torch.load("checkpoints/model_500.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

# 3. Autoregressive generation
input_tokens = torch.tensor([[1, 25, 142]], dtype=torch.long) # Replace with prompt token IDs
with torch.no_grad():
    for _ in range(50):
        logits = model(input_tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_tokens = torch.cat([input_tokens, next_token], dim=-1)

print("Generated token sequence:", input_tokens.squeeze().tolist())
```

---

## Testing & Validation

Quickly verify that model components and tensor dimensions behave as expected:

```bash
python thinknet-transformer/test.py
```

To verify code changes and synchronize trees safely with automatic Python syntax verification, use the sync utility:

```bash
python thinknet-transformer/scripts/sync_files_safe.py --src-dir . --dst-dir /path/to/destination
```

---

## Roadmap

- [x] Decoder-only architecture implementation (RMSNorm, RoPE, SwiGLU, GQA)
- [x] PyTorch FlashAttention / SDPA integration
- [x] FP16 mixed-precision training pipeline with gradient accumulation
- [x] Tokenizer pipeline & corpus dataset collectors
- [x] Mid-training qualitative generation monitoring
- [ ] KV-Cache implementation for accelerated $O(1)$ autoregressive step generation
- [ ] Distributed Training (DDP / FSDP / DeepSpeed) support
- [ ] Supervised Fine-Tuning (SFT) script for chat datasets (`HQ-Chat-1K`)
- [ ] Model weights export to Hugging Face `transformers` compatible format

---

## Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
