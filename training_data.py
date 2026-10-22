"""Local licensed corpus preparation and once-shifted causal batches."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from text_encoding import ByteTokenizer, TokenizerConfig


@dataclass(frozen=True)
class DocumentSource:
    path: str
    source: str
    license: str


@dataclass(frozen=True)
class Document:
    text: str
    content_hash: str
    provenance: tuple[DocumentSource, ...]


@dataclass(frozen=True)
class PreparedCorpus:
    train: tuple[Document, ...]
    validation: tuple[Document, ...]
    fingerprint: str

    def metadata(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "splits": {
                name: [
                    {"content_hash": doc.content_hash,
                     "provenance": [asdict(source) for source in doc.provenance]}
                    for doc in documents
                ]
                for name, documents in (("train", self.train), ("validation", self.validation))
            },
        }


class CorpusPreparer:
    def __init__(self, seed: int = 42, validation_fraction: float = 0.1) -> None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if (
            isinstance(validation_fraction, bool)
            or not isinstance(validation_fraction, (int, float))
            or not math.isfinite(validation_fraction)
            or not 0 < validation_fraction < 1
        ):
            raise ValueError("validation_fraction must be finite and between zero and one.")
        self.seed = seed
        self.validation_fraction = validation_fraction

    def prepare(self, manifest: Union[str, Path]) -> PreparedCorpus:
        manifest = Path(manifest)
        grouped: dict[str, tuple[str, list[DocumentSource]]] = {}
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at manifest line {line_number}: {error.msg}") from error
            if (
                not isinstance(record, dict) or set(record) != {"path", "source", "license"}
                or any(not isinstance(value, str) or not value.strip() for value in record.values())
            ):
                raise ValueError(
                    f"Manifest line {line_number} requires nonempty path, source, and license fields."
                )
            raw = (manifest.parent / record["path"]).read_bytes()
            text = raw.decode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            if digest not in grouped:
                grouped[digest] = (text, [])
            grouped[digest][1].append(DocumentSource(**record))
        if len(grouped) < 2:
            raise ValueError("Corpus requires at least two unique documents for a held-out split.")
        documents = [
            Document(text, digest, tuple(sorted(set(sources), key=lambda item: (
                item.path, item.source, item.license
            ))))
            for digest, (text, sources) in sorted(grouped.items())
        ]
        order = np.random.default_rng(self.seed).permutation(len(documents))
        validation_count = min(
            len(documents) - 1, max(1, int(len(documents) * self.validation_fraction))
        )
        validation = tuple(documents[int(index)] for index in order[:validation_count])
        train = tuple(documents[int(index)] for index in order[validation_count:])
        metadata = {
            "format_version": 1, "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "documents": [
                {"content_hash": doc.content_hash,
                 "provenance": [asdict(source) for source in doc.provenance]}
                for doc in documents
            ],
            "train": [doc.content_hash for doc in train],
            "validation": [doc.content_hash for doc in validation],
        }
        fingerprint = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return PreparedCorpus(train, validation, fingerprint)


class TextDataset(Dataset):
    def __init__(
        self, texts: Sequence[str], tokenizer: ByteTokenizer, max_length: int = 128
    ) -> None:
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if not texts:
            raise ValueError("TextDataset requires at least one document.")
        self.tokenizer_config = tokenizer.config
        self.max_length = max_length
        self._streams: list[torch.Tensor] = []
        self._windows: list[tuple[int, int, int]] = []
        for text in texts:
            stream = torch.tensor(
                [tokenizer.config.bos_id, *tokenizer.encode(text), tokenizer.config.eos_id],
                dtype=torch.long,
            )
            document_index = len(self._streams)
            self._streams.append(stream)
            # Adjacent windows share one predecessor token, but no supervised target.
            for start in range(0, stream.numel() - 1, max_length):
                self._windows.append((
                    document_index, start, min(start + max_length + 1, stream.numel())
                ))
        digest = hashlib.sha256()
        digest.update(json.dumps(asdict(tokenizer.config), sort_keys=True).encode())
        digest.update(str(max_length).encode())
        for stream in self._streams:
            digest.update(str(stream.numel()).encode() + b":")
            digest.update(stream.numpy().astype("<i8", copy=False).tobytes())
        self.fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        document, start, end = self._windows[index]
        return self._streams[document][start:end].clone()


class CausalBatchCollator:
    def __init__(self, config: TokenizerConfig = TokenizerConfig()) -> None:
        self.config = config

    def __call__(self, samples: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("Cannot collate an empty batch.")
        for sample in samples:
            if sample.ndim != 1 or sample.numel() < 2:
                raise ValueError("Each sample must be a token stream of at least two IDs.")
            if sample.dtype != torch.long:
                raise TypeError("Samples must contain int64 token IDs.")
            if sample.device.type != "cpu":
                raise ValueError("Dataset samples must be CPU tensors.")
            if torch.any(sample < 0) or torch.any(sample >= self.config.vocab_size):
                raise ValueError("Sample contains an out-of-vocabulary token ID.")
            if torch.any(sample == self.config.pad_id):
                raise ValueError("Unpadded sample streams must not contain PAD.")
        inputs = pad_sequence(
            [sample[:-1] for sample in samples], batch_first=True,
            padding_value=self.config.pad_id,
        )
        targets = pad_sequence(
            [sample[1:] for sample in samples], batch_first=True, padding_value=-100
        )
        return {
            "input_ids": inputs, "target_tokens": targets,
            "attention_mask": inputs != self.config.pad_id,
        }
