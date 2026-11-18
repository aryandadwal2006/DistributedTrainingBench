from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel

from bench.kernels import activation


class BenchmarkConfig(PretrainedConfig):
    model_type = "distributed-training-bench"

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 256,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.gradient_checkpointing = gradient_checkpointing


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(output)


class FeedForward(nn.Module):
    def __init__(self, config: BenchmarkConfig, kernel: str):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(activation(self.fc1(x), self.kernel))


class TransformerBlock(nn.Module):
    def __init__(self, config: BenchmarkConfig, kernel: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = FeedForward(config, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DistributedTrainingModel(PreTrainedModel):
    config_class = BenchmarkConfig
    base_model_prefix = "bench"
    supports_gradient_checkpointing = False

    def __init__(self, config: BenchmarkConfig, kernel: str = "torch"):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, kernel) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch, length = input_ids.shape
        if length > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.layers:
            if self.training and self.gradient_checkpointing:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        if labels is None:
            return None, logits
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
            labels[:, 1:].contiguous().view(-1),
        )
        return loss, logits
