import math

import pytest
import torch
import torch.nn.functional as F

from sequence_model import SequenceConfig
from text_encoding import ByteTokenizer
from training_data import TextDataset, CausalBatchCollator
from train import LanguageModelTrainer, TrainingConfig


def make_trainer(dropout=0.0, batch_size=2, device="cpu"):
    tokenizer = ByteTokenizer()
    train = TextDataset(["abab", "baba", "abba"], tokenizer, max_length=8)
    validation = TextDataset(["aba", "b"], tokenizer, max_length=8)
    return LanguageModelTrainer.create(
        train, validation, corpus_fingerprint="test-corpus",
        sequence_config=SequenceConfig(
            embed_dim=16, num_layers=1, num_heads=2,
            feedforward_dim=32, dropout=dropout, max_length=8,
        ),
        config=TrainingConfig(batch_size=batch_size, learning_rate=0.02, seed=12),
        num_projections=2, lambda_consensus=0,
        device=device,
    )


class TestTrainer:
    @pytest.mark.parametrize("change", [
        {"batch_size": 0}, {"learning_rate": 0}, {"gradient_clip": -1},
        {"weight_decay": -1}, {"seed": True}, {"learning_rate": float("nan")},
        {"batch_size": 1.5}, {"gradient_clip": float("inf")},
    ])
    def test_invalid_config(self, change):
        with pytest.raises(ValueError):
            TrainingConfig(**change)

    def test_fixed_tiny_corpus_overfits(self):
        trainer = make_trainer()
        # One deterministic document avoids irreducible ambiguity at BOS.
        trainer.train_dataset = TextDataset(["abab"], ByteTokenizer(), max_length=8)
        initial = trainer.evaluate(trainer.train_dataset)
        history = trainer.fit(200)
        final = trainer.evaluate(trainer.train_dataset)
        assert final["loss_task"] < 0.1
        assert final["loss_task"] < initial["loss_task"]
        assert trainer.step == 200
        assert len(history) == 200
        assert all(math.isfinite(row["loss_task"]) for row in history)
        assert history[-1]["loss_iso"] == 0

    def test_validation_matches_token_weighted_manual_loss(self):
        trainer = make_trainer(batch_size=1)
        trainer.model.train()
        expected_sum = 0.0
        expected_count = 0
        byte_sum = 0.0
        byte_count = 0
        trainer.model.eval()
        for sample in trainer.validation_dataset:
            batch = trainer.collator([sample])
            with torch.inference_mode():
                out = trainer.model(
                    trainer.encoder(batch["input_ids"]),
                    attention_mask=batch["attention_mask"]
                )
                targets = batch["target_tokens"]
                losses = F.cross_entropy(
                    out["logits"].reshape(-1, 263), targets.reshape(-1),
                    reduction="none"
                ).reshape_as(targets)
            expected_sum += losses.sum().item()
            expected_count += targets.numel()
            byte_mask = targets >= 7
            byte_sum += losses[byte_mask].sum().item()
            byte_count += byte_mask.sum().item()
        trainer.model.train()
        metrics = trainer.evaluate()
        assert trainer.model.training
        assert metrics["loss_task"] == pytest.approx(expected_sum / expected_count)
        assert metrics["perplexity"] == pytest.approx(math.exp(metrics["loss_task"]))
        assert metrics["bits_per_byte"] == pytest.approx(byte_sum / byte_count / math.log(2))
        assert metrics["byte_perplexity"] == pytest.approx(math.exp(byte_sum / byte_count))
        assert metrics["supervised_tokens"] == expected_count

    def test_reproducible_training_and_order(self):
        first = make_trainer(dropout=0.1)
        first.fit(5)
        state = {name: value.clone() for name, value in first.model.state_dict().items()}
        second = make_trainer(dropout=0.1)
        second.fit(5)
        for name, value in second.model.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)

    def test_refuses_zero_supervision_and_invalid_step_counts(self):
        trainer = make_trainer()
        batch = CausalBatchCollator()([torch.tensor([1, 10, 2])])
        batch["target_tokens"].fill_(-100)
        with pytest.raises(ValueError, match="supervised"):
            trainer.train_batch(batch)
        for count in (-1, True, 1.5):
            with pytest.raises(ValueError):
                trainer.fit(count)
        assert trainer.fit(0) == []

    def test_empty_byte_evaluation_reports_error_and_restores_mode(self):
        trainer = make_trainer()
        trainer.model.train()
        dataset = TextDataset([""], ByteTokenizer(), max_length=8)
        with pytest.raises(ValueError, match="byte"):
            trainer.evaluate(dataset)
        assert trainer.model.training

    @pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu"])
    def test_device_training_and_checkpoint_resume(self, device_type, tmp_path):
        from checkpoints import CheckpointManager

        backend = getattr(torch, device_type, None)
        if device_type != "cpu" and (backend is None or not backend.is_available()):
            pytest.skip(f"{device_type} hardware unavailable")
        trainer = make_trainer(device=device_type)
        trainer.fit(2)
        path = tmp_path / "model.pt"
        CheckpointManager().save(path, trainer)
        restored = make_trainer(device=device_type)
        CheckpointManager().load(path, restored)
        assert restored.step == 2
        restored.fit(1)
        assert math.isfinite(restored.evaluate()["loss_task"])
        assert next(restored.model.parameters()).device.type == device_type
