"""Versioned checkpoints for trusted local training artifacts."""

from dataclasses import asdict
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from train import LanguageModelTrainer


class CheckpointManager:
    format_version = 1

    def save(self, path: Union[str, Path], trainer: "LanguageModelTrainer") -> None:
        payload = {
            "format_version": self.format_version,
            "model_config": trainer.model_config,
            "tokenizer_config": asdict(trainer.encoder.config),
            "training_config": asdict(trainer.config),
            "corpus_fingerprint": trainer.corpus_fingerprint,
            "corpus_metadata": trainer.corpus_metadata,
            "dataset_fingerprints": {
                "train": trainer.train_dataset.fingerprint,
                "validation": trainer.validation_dataset.fingerprint,
            },
            "step": trainer.step,
            "device_type": trainer.device.type,
            "runtime": self._runtime(),
            "model_state": trainer.model.state_dict(),
            "encoder_state": trainer.encoder.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "rng_state": self._capture_rng(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                torch.save(payload, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load(self, path: Union[str, Path], trainer: "LanguageModelTrainer") -> None:
        # Even restricted loading is not a reason to accept untrusted artifacts.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self._validate(payload, trainer)
        trainer.model.load_state_dict(payload["model_state"], strict=True)
        trainer.encoder.load_state_dict(payload["encoder_state"], strict=True)
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        trainer.step = payload["step"]
        self._restore_rng(payload["rng_state"])

    def _validate(self, payload: dict, trainer: "LanguageModelTrainer") -> None:
        expected = {
            "format_version": self.format_version,
            "model_config": trainer.model_config,
            "tokenizer_config": asdict(trainer.encoder.config),
            "training_config": asdict(trainer.config),
            "corpus_fingerprint": trainer.corpus_fingerprint,
            "corpus_metadata": trainer.corpus_metadata,
            "dataset_fingerprints": {
                "train": trainer.train_dataset.fingerprint,
                "validation": trainer.validation_dataset.fingerprint,
            },
            "device_type": trainer.device.type,
            "runtime": self._runtime(),
        }
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must contain a versioned dictionary.")
        required = set(expected) | {
            "step", "model_state", "encoder_state", "optimizer_state", "rng_state"
        }
        if set(payload) != required:
            raise ValueError("Checkpoint fields do not match the supported format.")
        for key, value in expected.items():
            if payload[key] != value:
                raise ValueError(f"Incompatible checkpoint {key}.")
        if type(payload["step"]) is not int or payload["step"] < 0:
            raise ValueError("Checkpoint step must be a nonnegative integer.")
        for name, current in (
            ("model_state", trainer.model.state_dict()),
            ("encoder_state", trainer.encoder.state_dict()),
        ):
            saved = payload[name]
            if not isinstance(saved, dict) or set(saved) != set(current):
                raise ValueError(f"Incompatible checkpoint {name} keys.")
            for key, tensor in current.items():
                candidate = saved[key]
                if (
                    not isinstance(candidate, torch.Tensor)
                    or candidate.shape != tensor.shape or candidate.dtype != tensor.dtype
                    or not torch.isfinite(candidate).all()
                ):
                    raise ValueError(f"Incompatible checkpoint tensor {name}.{key}.")
        if not torch.equal(payload["encoder_state"]["root_pairs"],
                           trainer.encoder.root_pairs.cpu()):
            raise ValueError("Checkpoint root mapping does not match tokenizer configuration.")
        optimizer = payload["optimizer_state"]
        current_optimizer = trainer.optimizer.state_dict()
        if (
            not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}
            or not isinstance(optimizer["state"], dict)
            or not isinstance(optimizer["param_groups"], list)
            or len(optimizer["param_groups"]) != len(current_optimizer["param_groups"])
        ):
            raise ValueError("Invalid checkpoint optimizer state.")
        for saved_group, current_group in zip(
            optimizer["param_groups"], current_optimizer["param_groups"]
        ):
            if saved_group != current_group:
                raise ValueError("Incompatible checkpoint optimizer parameter groups.")
        parameter_ids = [
            index for group in optimizer["param_groups"] for index in group["params"]
        ]
        parameters = [parameter for group in trainer.optimizer.param_groups
                      for parameter in group["params"]]
        if set(optimizer["state"]) - set(parameter_ids):
            raise ValueError("Unknown checkpoint optimizer parameter IDs.")
        for index, parameter in zip(parameter_ids, parameters):
            state = optimizer["state"].get(index)
            if state is None:
                if payload["step"] != 0:
                    raise ValueError("Missing checkpoint optimizer state for trained parameters.")
                continue
            if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Invalid checkpoint AdamW state fields.")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if (
                    not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not torch.isfinite(value).all()
                ):
                    raise ValueError("Invalid checkpoint AdamW moment tensor.")
            step = state["step"]
            if (
                not isinstance(step, torch.Tensor) or step.numel() != 1
                or not torch.isfinite(step).all() or step.item() != payload["step"]
            ):
                raise ValueError("Invalid checkpoint AdamW step.")
        self._validate_rng(payload["rng_state"])

    def _runtime(self) -> dict[str, str]:
        return {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        }

    def _capture_rng(self) -> dict:
        numpy_state = np.random.get_state()
        accelerators = {}
        for name in ("cuda", "xpu"):
            backend = getattr(torch, name, None)
            if backend is not None and backend.is_available():
                accelerators[name] = backend.get_rng_state_all()
        return {
            "python": random.getstate(),
            "numpy": {
                "algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                "position": numpy_state[2], "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "torch": torch.get_rng_state(), "accelerators": accelerators,
        }

    def _validate_rng(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {
            "python", "numpy", "torch", "accelerators"
        }:
            raise ValueError("Invalid checkpoint RNG state.")
        python_rng = random.Random()
        python_rng.setstate(state["python"])
        numpy_rng = np.random.RandomState()
        numpy_rng.set_state(self._numpy_state(state["numpy"]))
        torch.Generator().set_state(state["torch"])
        if not isinstance(state["accelerators"], dict):
            raise ValueError("Invalid checkpoint accelerator RNG state.")
        for name, tensors in state["accelerators"].items():
            if name not in ("cuda", "xpu"):
                raise ValueError("Unknown checkpoint accelerator RNG backend.")
            backend = getattr(torch, name, None)
            if (
                backend is None or not backend.is_available()
                or len(tensors) != backend.device_count()
            ):
                raise ValueError(f"Checkpoint requires matching {name} hardware for RNG resume.")

    def _numpy_state(self, state: dict) -> tuple:
        if not isinstance(state, dict) or set(state) != {
            "algorithm", "keys", "position", "has_gauss", "cached_gaussian"
        }:
            raise ValueError("Invalid checkpoint NumPy RNG state.")
        return (
            state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
            state["position"], state["has_gauss"], state["cached_gaussian"],
        )

    def _restore_rng(self, state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(self._numpy_state(state["numpy"]))
        torch.set_rng_state(state["torch"])
        for name, tensors in state["accelerators"].items():
            getattr(torch, name).set_rng_state_all(tensors)
