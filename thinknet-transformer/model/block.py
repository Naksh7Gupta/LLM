import torch
import torch.nn as nn

from configs.model_config import ModelConfig
from model.attention import Attention
from model.swiglu import SwiGLU
from model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
	"""A simple transformer block combining attention and feed-forward."""

	def __init__(self, config: ModelConfig):
		super().__init__()

		self.attn = Attention(config)
		self.ff = SwiGLU(config)

		self.norm1 = RMSNorm(config)
		self.norm2 = RMSNorm(config)

		self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.0)

	def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
		# Attention block
		residual = hidden_states
		x = self.norm1(hidden_states)
		x = self.attn(x, position_ids)
		x = self.dropout(x)
		x = x + residual

		# Feed-forward block
		residual = x
		x = self.norm2(x)
		x = self.ff(x)
		x = self.dropout(x)
		x = x + residual

		return x
