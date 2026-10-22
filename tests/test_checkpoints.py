import random

import numpy as np
import pytest
import torch

from checkpoints import CheckpointManager
from test_training import make_trainer


class TestCheckpoints:
    def test_exact_resume_including_optimizer_rng_and_data_position(self, tmp_path):
        original = make_trainer(dropout=0.1)
        original.fit(3)
        path = tmp_path / "model.pt"
        manager = CheckpointManager()
        manager.save(path, original)
        expected_rng = (random.random(), np.random.random(), torch.rand(3))
        original.fit(4)
        expected = {key: value.clone() for key, value in original.model.state_dict().items()}
        restored = make_trainer(dropout=0.1)
        manager.load(path, restored)
        assert restored.step == 3
        assert restored.optimizer.state_dict()["state"]
        assert random.random() == expected_rng[0]
        assert np.random.random() == expected_rng[1]
        torch.testing.assert_close(torch.rand(3), expected_rng[2], atol=0, rtol=0)
        restored.fit(4)
        for key, value in restored.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        assert restored.step == 7

    def test_reload_preserves_eval_logits_and_format(self, tmp_path):
        trainer = make_trainer()
        trainer.fit(2)
        batch = trainer.collator([trainer.validation_dataset[0]])
        trainer.model.eval()
        with torch.inference_mode():
            expected = trainer.model(trainer.encoder(batch["input_ids"]))["logits"]
        path = tmp_path / "model.pt"
        CheckpointManager().save(path, trainer)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["format_version"] == 1
        assert payload["tokenizer_config"]["num_atomic_roots"] == 17
        assert payload["model_config"]["sequence_config"]["max_length"] == 8
        restored = make_trainer()
        CheckpointManager().load(path, restored)
        restored.model.eval()
        with torch.inference_mode():
            actual = restored.model(restored.encoder(batch["input_ids"]))["logits"]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("field,value", [
        ("format_version", 99), ("corpus_fingerprint", "different"),
        ("step", -1), ("tokenizer_config", {}), ("model_config", {}),
        ("training_config", {}), ("model_state", {}),
    ])
    def test_incompatible_artifacts_rejected_before_model_mutation(self, tmp_path, field, value):
        trainer = make_trainer()
        path = tmp_path / "bad.pt"
        manager = CheckpointManager()
        manager.save(path, trainer)
        payload = torch.load(path, weights_only=True)
        payload[field] = value
        torch.save(payload, path)
        before = {key: tensor.clone() for key, tensor in trainer.model.state_dict().items()}
        with pytest.raises(ValueError):
            manager.load(path, trainer)
        for key, tensor in trainer.model.state_dict().items():
            torch.testing.assert_close(tensor, before[key])

    @pytest.mark.parametrize("corruption", ["moments", "optimizer_step", "mapping", "rng"])
    def test_corrupt_state_is_rejected_before_mutation(self, tmp_path, corruption):
        trainer = make_trainer()
        trainer.fit(2)
        path = tmp_path / "bad.pt"
        manager = CheckpointManager()
        manager.save(path, trainer)
        payload = torch.load(path, weights_only=True)
        if corruption == "moments":
            next(iter(payload["optimizer_state"]["state"].values()))["exp_avg"] = torch.zeros(1)
        elif corruption == "optimizer_step":
            next(iter(payload["optimizer_state"]["state"].values()))["step"] = torch.tensor(99.0)
        elif corruption == "mapping":
            payload["encoder_state"]["root_pairs"][0] = torch.tensor([1, 1])
        else:
            payload["rng_state"] = {}
        torch.save(payload, path)
        before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
        with pytest.raises(ValueError):
            manager.load(path, trainer)
        assert trainer.step == 2
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_changed_dataset_order_rejects_resume(self, tmp_path):
        from text_encoding import ByteTokenizer
        from training_data import TextDataset

        trainer = make_trainer()
        path = tmp_path / "model.pt"
        manager = CheckpointManager()
        manager.save(path, trainer)
        trainer.train_dataset = TextDataset(
            ["abba", "baba", "abab"], ByteTokenizer(), max_length=8
        )
        with pytest.raises(ValueError, match="dataset_fingerprints"):
            manager.load(path, trainer)

    def test_failed_write_preserves_existing_checkpoint_and_cleans_temporary(self, tmp_path, monkeypatch):
        trainer = make_trainer()
        path = tmp_path / "model.pt"
        manager = CheckpointManager()
        manager.save(path, trainer)
        original = path.read_bytes()

        def fail_save(*args, **kwargs):
            raise OSError("simulated storage failure")

        monkeypatch.setattr(torch, "save", fail_save)
        with pytest.raises(OSError, match="storage"):
            manager.save(path, trainer)
        assert path.read_bytes() == original
        assert list(tmp_path.iterdir()) == [path]
