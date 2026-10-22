from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from uira import (
    DeviceSelector,
    SyntheticTrainingExample,
    UIRAFullArchitecture,
    YetzirahPermutationEngine,
    select_device,
)


class TestRootComposition:
    @pytest.mark.parametrize("size", [0, 1, 3, True, 2.0])
    def test_rejects_unsupported_combination_size(self, size):
        with pytest.raises((ValueError, TypeError)):
            YetzirahPermutationEngine(4, 8, size)

    @pytest.mark.parametrize(
        "roots",
        [torch.ones(1, 2, dtype=torch.long),
         torch.ones(1, 2, 3, dtype=torch.long),
         torch.ones(1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
         torch.tensor([[[-1, 0]]]), torch.tensor([[[0, 4]]]),
         torch.empty(1, 0, 2, dtype=torch.long)]
    )
    def test_rejects_invalid_roots(self, roots):
        with pytest.raises((ValueError, TypeError)):
            YetzirahPermutationEngine(4, 8)(roots)

    def test_composition_matches_original_pairwise_operation(self):
        engine = YetzirahPermutationEngine(4, 8)
        roots = torch.randint(0, 4, (2, 3, 2))
        selected = engine.atomic_roots[roots]
        products = torch.bmm(
            selected[:, :, 0].reshape(-1, 8, 1),
            selected[:, :, 1].reshape(-1, 1, 8)
        ).reshape(2, 3, 64)
        torch.testing.assert_close(
            engine(roots), engine.permutation_projection(products)
        )
        torch.testing.assert_close(engine(roots.int()), engine(roots))


class TestArchitectureCompatibility:
    def test_shapes_losses_and_backward(self):
        model = UIRAFullArchitecture(4, 8, 2, 10).double()
        roots = torch.randint(0, 4, (2, 3, 2))
        targets = torch.randint(0, 10, (2, 3))
        distances = torch.cdist(torch.randn(2, 2, dtype=torch.double),
                                torch.randn(2, 2, dtype=torch.double))
        outputs = model(roots, distances, targets)
        assert set(outputs) == {
            "logits", "z_final", "total_loss", "loss_task",
            "loss_iso", "loss_consensus"
        }
        assert outputs["logits"].shape == (2, 3, 10)
        assert outputs["z_final"].shape == (2, 3, 8)
        expected_task = F.cross_entropy(outputs["logits"].reshape(-1, 10),
                                        targets.reshape(-1))
        torch.testing.assert_close(outputs["loss_task"], expected_task)
        for value in outputs.values():
            assert value.dtype == torch.double
            assert torch.isfinite(value).all()
        outputs["total_loss"].backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_refactor_matches_original_objective(self):
        model = UIRAFullArchitecture(4, 8, 2, 10)
        roots = torch.randint(0, 4, (2, 3, 2))
        targets = torch.randint(0, 10, (2, 3))
        distances = torch.rand(2, 2)
        z = model.yetzirah_engine(roots)
        views = model.babel_partitioner(z)
        final = model.consensus_gate(torch.cat(views, dim=-1))
        logits = model.task_head(final)
        iso = F.mse_loss(torch.cdist(z.mean(dim=1), z.mean(dim=1)), distances)
        consensus = F.mse_loss(views[0], views[1])
        task = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        outputs = model(roots, distances, targets)
        torch.testing.assert_close(outputs["logits"], logits)
        torch.testing.assert_close(
            outputs["total_loss"], task + 0.1 * iso + 0.1 * consensus
        )

    def test_optional_losses_follow_model_dtype(self):
        model = UIRAFullArchitecture(4, 8, 1, 10).double()
        outputs = model(torch.zeros(1, 2, 2, dtype=torch.long))
        for name in ("loss_iso", "loss_task", "loss_consensus"):
            assert outputs[name].dtype == torch.double
            assert outputs[name].item() == 0


class TestDeviceSelection:
    def test_cpu_and_compatibility_wrapper(self):
        assert DeviceSelector().select("cpu") == torch.device("cpu")
        assert select_device("cpu") == torch.device("cpu")

    @pytest.mark.parametrize(
        "xpu_available,cuda_available,expected",
        [(True, True, "xpu"), (False, True, "cuda"), (False, False, "cpu")]
    )
    def test_auto_priority(self, monkeypatch, xpu_available, cuda_available, expected):
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(
            is_available=lambda: xpu_available
        ), raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
        assert DeviceSelector().select().type == expected

    @pytest.mark.parametrize("backend", ["cuda", "xpu"])
    def test_unavailable_device_is_explicit(self, monkeypatch, backend):
        monkeypatch.setattr(torch, "xpu", None, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="available"):
            DeviceSelector().select(backend)

    def test_unsupported_backend(self):
        with pytest.raises(ValueError, match="Unsupported"):
            DeviceSelector().select("meta")

    def test_synthetic_application_preserves_cli_output(self, capsys):
        outputs = SyntheticTrainingExample().run(["--device", "cpu"])
        captured = capsys.readouterr().out
        assert "UIRA Pipeline Execution Successful" in captured
        assert "Device:           cpu" in captured
        assert "torch.Size([4, 8, 250])" in captured
        assert torch.isfinite(outputs["total_loss"])
