"""Versioned UTF-8 tokenization and deterministic root-pair encoding."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TokenizerConfig:
    format_version: int = 1
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    system_id: int = 3
    user_id: int = 4
    assistant_id: int = 5
    end_turn_id: int = 6
    byte_offset: int = 7
    vocab_size: int = 263
    num_atomic_roots: int = 17

    def __post_init__(self) -> None:
        expected = {
            "format_version": 1, "pad_id": 0, "bos_id": 1, "eos_id": 2,
            "system_id": 3, "user_id": 4, "assistant_id": 5,
            "end_turn_id": 6, "byte_offset": 7, "vocab_size": 263,
        }
        for name, value in asdict(self).items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer.")
            if name in expected and value != expected[name]:
                raise ValueError(f"{name} must be {expected[name]} for format version 1.")
        if self.num_atomic_roots <= 0 or self.num_atomic_roots**2 < self.vocab_size:
            raise ValueError("num_atomic_roots squared must cover the vocabulary.")


class ByteTokenizer:
    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config = config if config is not None else TokenizerConfig()

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a str.")
        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        return (data.astype(np.int64) + self.config.byte_offset).tolist()

    def decode(
        self, token_ids: Sequence[int], *, skip_control_tokens: bool = False
    ) -> str:
        byte_ids: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, (bool, np.bool_)) or not isinstance(
                token_id, (int, np.integer)
            ):
                raise TypeError("Token IDs must be integers, not booleans.")
            if not 0 <= token_id < self.config.vocab_size:
                raise ValueError(f"Token ID out of vocabulary range: {token_id}")
            if token_id < self.config.byte_offset:
                if skip_control_tokens:
                    continue
                raise ValueError("Decoding control tokens requires skip_control_tokens=True.")
            byte_ids.append(int(token_id) - self.config.byte_offset)
        return np.asarray(byte_ids, dtype=np.uint8).tobytes().decode("utf-8")

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(asdict(self.config), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ByteTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {field.name for field in fields(TokenizerConfig)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Tokenizer configuration fields do not match format version 1.")
        return cls(TokenizerConfig(**data))


class TokenToRootEncoder(nn.Module):
    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else TokenizerConfig()
        tokens = torch.arange(self.config.vocab_size, dtype=torch.long)
        self.register_buffer(
            "root_pairs",
            torch.stack(
                (tokens // self.config.num_atomic_roots,
                 tokens % self.config.num_atomic_roots),
                dim=-1,
            ),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or any(size == 0 for size in token_ids.shape):
            raise ValueError("token_ids must have nonempty shape [batch, sequence].")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must have int32 or int64 dtype.")
        if token_ids.device != self.root_pairs.device:
            raise ValueError("token_ids and encoder buffers must be on the same device.")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.config.vocab_size):
            raise ValueError("token_ids contain an index outside the vocabulary.")
        return self.root_pairs[token_ids.long()]
