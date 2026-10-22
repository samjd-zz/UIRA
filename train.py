"""Reproducible local next-token training and validation."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sequence_model import SequenceConfig
from text_encoding import ByteTokenizer, TokenToRootEncoder
from training_data import CorpusPreparer, TextDataset, CausalBatchCollator
from uira import DeviceSelector, UIRAFullArchitecture


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if name == "weight_decay" else value <= 0)
            ):
                raise ValueError(f"{name} must be finite and in its valid positive range.")


class LanguageModelTrainer:
    def __init__(
        self, model: UIRAFullArchitecture, encoder: TokenToRootEncoder,
        train_dataset: TextDataset, validation_dataset: TextDataset,
        config: TrainingConfig, corpus_fingerprint: str,
        corpus_metadata: Optional[dict] = None,
    ) -> None:
        if model.sequence_processor is None:
            raise ValueError("Language training requires a contextual model.")
        if not isinstance(corpus_fingerprint, str) or not corpus_fingerprint:
            raise ValueError("corpus_fingerprint must be a nonempty string.")
        for dataset in (train_dataset, validation_dataset):
            if dataset.tokenizer_config != encoder.config:
                raise ValueError("Dataset and encoder tokenizer configurations differ.")
            if dataset.max_length > model.sequence_processor.config.max_length:
                raise ValueError("Dataset windows exceed model context length.")
        if (
            model.task_head.out_features != encoder.config.vocab_size
            or model.yetzirah_engine.atomic_roots.size(0) != encoder.config.num_atomic_roots
        ):
            raise ValueError("Model vocabulary/root counts do not match the tokenizer.")
        if model.lambda_iso != 0:
            raise ValueError("Text-only training requires lambda_iso=0 without a reference metric.")
        self.model = model
        self.encoder = encoder
        self.device = next(model.parameters()).device
        if encoder.root_pairs.device != self.device:
            raise ValueError("Encoder and model must be on the same device.")
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config
        self.corpus_fingerprint = corpus_fingerprint
        self.corpus_metadata = corpus_metadata if corpus_metadata is not None else {}
        self.collator = CausalBatchCollator(encoder.config)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.step = 0
        processor = model.sequence_processor
        self.model_config = {
            "num_atomic_roots": encoder.config.num_atomic_roots,
            "embed_dim": model.yetzirah_engine.embed_dim,
            "num_projections": model.babel_partitioner.num_projections,
            "vocab_size": model.task_head.out_features,
            "lambda_iso": model.lambda_iso, "lambda_consensus": model.lambda_consensus,
            "sequence_config": asdict(processor.config),
        }

    @classmethod
    def create(
        cls, train_dataset: TextDataset, validation_dataset: TextDataset,
        corpus_fingerprint: str, sequence_config: Optional[SequenceConfig] = None,
        config: Optional[TrainingConfig] = None, device: str = "cpu",
        num_projections: int = 4, lambda_consensus: float = 0.1,
        corpus_metadata: Optional[dict] = None,
    ) -> "LanguageModelTrainer":
        config = config if config is not None else TrainingConfig()
        sequence_config = sequence_config if sequence_config is not None else SequenceConfig()
        if type(num_projections) is not int or num_projections <= 0:
            raise ValueError("num_projections must be a positive integer.")
        if (
            isinstance(lambda_consensus, bool)
            or not isinstance(lambda_consensus, (int, float))
            or not math.isfinite(lambda_consensus) or lambda_consensus < 0
        ):
            raise ValueError("lambda_consensus must be finite and nonnegative.")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        selected = DeviceSelector().select(device)
        tokenizer_config = train_dataset.tokenizer_config
        model = UIRAFullArchitecture(
            num_atomic_roots=tokenizer_config.num_atomic_roots,
            embed_dim=sequence_config.embed_dim, num_projections=num_projections,
            vocab_size=tokenizer_config.vocab_size, lambda_iso=0,
            lambda_consensus=lambda_consensus, sequence_config=sequence_config,
        ).to(selected)
        encoder = TokenToRootEncoder(tokenizer_config).to(selected)
        return cls(
            model, encoder, train_dataset, validation_dataset, config,
            corpus_fingerprint, corpus_metadata,
        )

    def train_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        batch = {key: value.to(self.device) for key, value in batch.items()}
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(
            self.encoder(batch["input_ids"]), target_tokens=batch["target_tokens"],
            attention_mask=batch["attention_mask"],
        )
        if not torch.isfinite(outputs["total_loss"]):
            raise FloatingPointError("Non-finite training loss; optimizer step refused.")
        outputs["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.step += 1
        return {key: outputs[key].detach().item()
                for key in ("total_loss", "loss_task", "loss_iso", "loss_consensus")}

    def fit(self, steps: int) -> list[dict[str, float]]:
        return list(self.iter_steps(steps))

    def iter_steps(self, steps: int) -> Iterator[dict[str, float]]:
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer.")
        batches_per_epoch = math.ceil(len(self.train_dataset) / self.config.batch_size)
        remaining = steps
        while remaining:
            epoch, offset = divmod(self.step, batches_per_epoch)
            generator = torch.Generator().manual_seed(self.config.seed + epoch)
            order = torch.randperm(len(self.train_dataset), generator=generator).tolist()
            loader = DataLoader(
                self.train_dataset, batch_size=self.config.batch_size,
                sampler=order, collate_fn=self.collator, generator=generator,
            )
            for index, batch in enumerate(loader):
                if index < offset:
                    continue
                metrics = self.train_batch(batch)
                remaining -= 1
                yield metrics
                if remaining == 0:
                    break

    def evaluate(self, dataset: Optional[TextDataset] = None) -> dict[str, float]:
        dataset = self.validation_dataset if dataset is None else dataset
        previous_mode = self.model.training
        self.model.eval()
        task_sum = byte_sum = consensus_sum = 0.0
        token_count = byte_count = valid_count = 0
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, collate_fn=self.collator,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        try:
            with torch.inference_mode():
                for batch in loader:
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    out = self.model(
                        self.encoder(batch["input_ids"]),
                        target_tokens=batch["target_tokens"],
                        attention_mask=batch["attention_mask"],
                    )
                    labels = batch["target_tokens"]
                    losses = F.cross_entropy(
                        out["logits"].reshape(-1, out["logits"].size(-1)),
                        labels.reshape(-1), reduction="none", ignore_index=-100,
                    ).reshape_as(labels)
                    supervised = labels != -100
                    byte_mask = labels >= self.encoder.config.byte_offset
                    task_sum += losses[supervised].sum().item()
                    token_count += supervised.sum().item()
                    byte_sum += losses[byte_mask].sum().item()
                    byte_count += byte_mask.sum().item()
                    valid = batch["attention_mask"].sum().item()
                    consensus_sum += out["loss_consensus"].item() * valid
                    valid_count += valid
        finally:
            self.model.train(previous_mode)
        if token_count == 0 or byte_count == 0:
            raise ValueError("Evaluation requires supervised targets and at least one byte target.")
        task = task_sum / token_count
        consensus = consensus_sum / valid_count
        if not all(math.isfinite(value) for value in (task, consensus, byte_sum)):
            raise FloatingPointError("Non-finite validation metrics.")
        return {
            "loss_task": task, "loss_iso": 0.0, "loss_consensus": consensus,
            "total_loss": task + self.model.lambda_consensus * consensus,
            "perplexity": math.exp(task), "bits_per_byte": byte_sum / byte_count / math.log(2),
            "byte_perplexity": math.exp(byte_sum / byte_count),
            "supervised_tokens": token_count, "byte_tokens": byte_count,
        }


class TrainingApplication:
    def run(self, argv: Optional[Sequence[str]] = None) -> dict[str, float]:
        from checkpoints import CheckpointManager

        parser = argparse.ArgumentParser(description="Train UIRA on licensed local UTF-8 documents.")
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--resume", type=Path)
        parser.add_argument("--steps", type=int, default=100)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--learning-rate", type=float, default=1e-3)
        parser.add_argument("--weight-decay", type=float, default=0.01)
        parser.add_argument("--gradient-clip", type=float, default=1.0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--validation-fraction", type=float, default=0.1)
        parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
        parser.add_argument("--embed-dim", type=int, default=64)
        parser.add_argument("--layers", type=int, default=2)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--feedforward-dim", type=int, default=256)
        parser.add_argument("--context-length", type=int, default=128)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--lambda-consensus", type=float, default=0.1)
        parser.add_argument("--num-projections", type=int, default=4)
        args = parser.parse_args(argv)
        if args.steps < 0:
            parser.error("--steps must be nonnegative.")
        corpus = CorpusPreparer(args.seed, args.validation_fraction).prepare(args.manifest)
        tokenizer = ByteTokenizer()
        train = TextDataset([doc.text for doc in corpus.train], tokenizer, args.context_length)
        validation = TextDataset(
            [doc.text for doc in corpus.validation], tokenizer, args.context_length
        )
        trainer = LanguageModelTrainer.create(
            train, validation, corpus.fingerprint,
            sequence_config=SequenceConfig(
                embed_dim=args.embed_dim, num_layers=args.layers, num_heads=args.heads,
                feedforward_dim=args.feedforward_dim, max_length=args.context_length,
                dropout=args.dropout,
            ),
            config=TrainingConfig(
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                weight_decay=args.weight_decay, gradient_clip=args.gradient_clip,
                seed=args.seed,
            ),
            device=args.device, lambda_consensus=args.lambda_consensus,
            num_projections=args.num_projections,
            corpus_metadata=corpus.metadata(),
        )
        manager = CheckpointManager()
        if args.resume is not None:
            manager.load(args.resume, trainer)
        # Stream step metrics instead of retaining an unbounded application history.
        for metrics in trainer.iter_steps(args.steps):
            print(json.dumps({"step": trainer.step, "train": metrics}, allow_nan=False))
        validation_metrics = trainer.evaluate()
        manager.save(args.output, trainer)
        print(json.dumps(
            {"step": trainer.step, "validation": validation_metrics,
             "checkpoint": str(args.output)}, allow_nan=False
        ))
        return validation_metrics


if __name__ == "__main__":
    TrainingApplication().run()
