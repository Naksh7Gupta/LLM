import torch

from configs.model_config import ModelConfig
from model.rmsnorm import RMSNorm


config = ModelConfig()

rmsnorm = RMSNorm(config)

x = torch.randn(2, 10, config.hidden_size)

y = rmsnorm(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)