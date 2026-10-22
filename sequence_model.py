"""Opt-in causal contextual processing built from PyTorch attention layers."""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class SequenceConfig:
    embed_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.1
    max_length: int = 128

    def __post_init__(self) -> None:
        for name in ("embed_dim", "num_layers", "num_heads",
                     "feedforward_dim", "max_length"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be finite and in [0, 1).")


class CausalSequenceProcessor(nn.Module):
    def __init__(self, config: Optional[SequenceConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else SequenceConfig()
        self.position_embeddings = nn.Embedding(
            self.config.max_length, self.config.embed_dim
        )
        self.dropout = nn.Dropout(self.config.dropout)
        # Construct independently initialized layers instead of identical clones.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.config.embed_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.feedforward_dim,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(self.config.num_layers)
        ])
        self.register_buffer(
            "causal_mask",
            torch.ones(self.config.max_length, self.config.max_length,
                       dtype=torch.bool).triu(diagonal=1),
            persistent=False,
        )

    def validate_attention_mask(
        self, features: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if (
            features.ndim != 3 or any(size == 0 for size in features.shape)
            or features.size(-1) != self.config.embed_dim
        ):
            raise ValueError("features must have nonempty shape [batch, sequence, embed_dim].")
        if not features.is_floating_point():
            raise TypeError("features must have floating-point dtype.")
        if features.size(1) > self.config.max_length:
            raise ValueError("Sequence length exceeds configured max_length.")
        if features.device != self.position_embeddings.weight.device:
            raise ValueError("features and sequence processor must be on the same device.")
        if attention_mask is None:
            return torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if attention_mask.shape != features.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, sequence].")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean: True is a valid token.")
        if attention_mask.device != features.device:
            raise ValueError("attention_mask and features must be on the same device.")
        if not attention_mask[:, 0].all():
            raise ValueError("Every sequence must contain a valid first token; all-padding is invalid.")
        if torch.any(~attention_mask[:, :-1] & attention_mask[:, 1:]):
            raise ValueError("attention_mask supports right-padding only.")
        return attention_mask

    def forward(
        self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        valid = self.validate_attention_mask(features, attention_mask)
        length = features.size(1)
        positions = torch.arange(length, device=features.device)
        hidden = self.dropout(features + self.position_embeddings(positions))
        for layer in self.layers:
            hidden = layer(
                hidden,
                src_mask=self.causal_mask[:length, :length],
                src_key_padding_mask=~valid,
            )
        return hidden.masked_fill(~valid.unsqueeze(-1), 0)
