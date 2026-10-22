import pytest
import torch
import torch.nn.functional as F

from sequence_model import SequenceConfig
from uira import UIRAFullArchitecture


class TestContextualArchitecture:
    def setup_method(self):
        torch.manual_seed(42)
        self.config = SequenceConfig(
            embed_dim=8, num_heads=2, num_layers=2,
            feedforward_dim=16, dropout=0, max_length=8,
        )
        self.model = UIRAFullArchitecture(
            5, 8, 2, 10, sequence_config=self.config
        ).eval()

    def test_opt_in_and_width_validation(self):
        legacy = UIRAFullArchitecture(5, 8, 2, 10)
        assert legacy.sequence_processor is None
        assert not any("sequence_processor" in key for key in legacy.state_dict())
        with pytest.raises(ValueError, match="embed_dim"):
            UIRAFullArchitecture(5, 16, 2, 10, sequence_config=self.config)
        with pytest.raises(ValueError, match="contextual"):
            legacy(torch.zeros(1, 2, 2, dtype=torch.long),
                   attention_mask=torch.ones(1, 2, dtype=torch.bool))

    def test_causality_and_context_dependency(self):
        roots = torch.randint(0, 5, (1, 4, 2))
        future = roots.clone()
        future[:, 2:] = (future[:, 2:] + 1) % 5
        previous = roots.clone()
        previous[:, 0] = (previous[:, 0] + 1) % 5
        with torch.inference_mode():
            original = self.model(roots)["logits"]
            changed = self.model(future)["logits"]
            context_changed = self.model(previous)["logits"]
        torch.testing.assert_close(
            original[:, :2], changed[:, :2], atol=1e-6, rtol=1e-5
        )
        assert not torch.allclose(original[:, -1], context_changed[:, -1])

    def test_right_padding_preserves_all_losses_and_valid_outputs(self):
        roots = torch.randint(0, 5, (2, 3, 2))
        targets = torch.randint(0, 10, (2, 3))
        distances = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        padded = torch.cat([roots, torch.randint(0, 5, (2, 2, 2))], dim=1)
        mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
        padded_targets = torch.cat([targets, torch.full((2, 2), -999)], dim=1)
        with torch.inference_mode():
            expected = self.model(roots, distances, targets)
            actual = self.model(padded, distances, padded_targets,
                                attention_mask=mask)
        assert set(expected) == set(actual)
        for key in ("logits", "z_final"):
            torch.testing.assert_close(
                expected[key], actual[key][:, :3], atol=1e-6, rtol=1e-5
            )
        for key in ("total_loss", "loss_task", "loss_iso", "loss_consensus"):
            torch.testing.assert_close(expected[key], actual[key], atol=1e-6, rtol=1e-5)

    def test_variable_lengths_match_individual_predictions_and_reductions(self):
        roots = torch.randint(0, 5, (2, 4, 2))
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        targets = torch.randint(0, 10, (2, 4))
        outputs = self.model(roots, target_tokens=targets, attention_mask=mask)
        single = self.model(roots[1:2, :2])
        torch.testing.assert_close(outputs["logits"][1, :2], single["logits"][0],
                                   atol=1e-6, rtol=1e-5)
        expected_task = F.cross_entropy(outputs["logits"][mask], targets[mask])
        views = self.model.babel_partitioner(self.model.yetzirah_engine(roots))
        expected_consensus = F.mse_loss(views[0][mask], views[1][mask])
        torch.testing.assert_close(outputs["loss_task"], expected_task)
        torch.testing.assert_close(outputs["loss_consensus"], expected_consensus)

    def test_padding_has_zero_input_gradient(self):
        roots = torch.tensor([[[0, 1], [1, 0], [4, 4]]])
        output = self.model(
            roots, target_tokens=torch.tensor([[1, 2, -100]]),
            attention_mask=torch.tensor([[True, True, False]])
        )
        output["total_loss"].backward()
        assert torch.count_nonzero(self.model.yetzirah_engine.atomic_roots.grad[4]) == 0
        for parameter in self.model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    def test_ignored_labels_and_invalid_supervision(self):
        roots = torch.randint(0, 5, (1, 3, 2))
        targets = torch.tensor([[1, -100, 2]])
        outputs = self.model(roots, target_tokens=targets)
        torch.testing.assert_close(
            outputs["loss_task"],
            F.cross_entropy(outputs["logits"][:, [0, 2]].reshape(-1, 10),
                            torch.tensor([1, 2]))
        )
        for invalid in (torch.full((1, 3), -100), torch.ones(1, 2, dtype=torch.long),
                        torch.ones(1, 3), torch.tensor([[1, -1, 2]])):
            with pytest.raises((ValueError, TypeError)):
                self.model(roots, target_tokens=invalid)

    def test_rejects_metric_broadcasting_and_invalid_masks(self):
        roots = torch.randint(0, 5, (2, 3, 2))
        with pytest.raises(ValueError, match="distance"):
            self.model(roots, env_distance_matrix=torch.ones(2, 1))
        for mask in (torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.bool),
                     torch.tensor([[True, False, True], [True, True, True]])):
            with pytest.raises((ValueError, TypeError)):
                self.model(roots, attention_mask=mask)

    def test_context_stack_state_round_trip(self):
        restored = UIRAFullArchitecture(5, 8, 2, 10, sequence_config=self.config).eval()
        restored.load_state_dict(self.model.state_dict(), strict=True)
        roots = torch.randint(0, 5, (2, 4, 2))
        torch.testing.assert_close(self.model(roots)["logits"], restored(roots)["logits"])

    def test_learns_prefix_dependent_prediction(self):
        torch.manual_seed(7)
        config = SequenceConfig(
            embed_dim=16, num_heads=2, num_layers=1,
            feedforward_dim=32, dropout=0, max_length=2,
        )
        model = UIRAFullArchitecture(
            5, 16, 2, 10, lambda_consensus=0, sequence_config=config
        ).train()
        # The final root pair is identical: only the preceding pair identifies the label.
        roots = torch.tensor([[[0, 1], [2, 2]], [[1, 0], [2, 2]]])
        targets = torch.tensor([[-100, 3], [-100, 4]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(roots, target_tokens=targets)["total_loss"]
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            outputs = model(roots, target_tokens=targets)
        assert outputs["loss_task"].item() < 0.05
        assert outputs["logits"][:, -1].argmax(-1).tolist() == [3, 4]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu"])
    def test_contextual_model_device_dtype_gradients(self, dtype, device_type):
        backend = getattr(torch, device_type, None)
        if device_type != "cpu" and (backend is None or not backend.is_available()):
            pytest.skip(f"{device_type} hardware unavailable")
        model = self.model.to(device=device_type, dtype=dtype).train()
        roots = torch.randint(0, 5, (2, 3, 2), device=device_type)
        labels = torch.randint(0, 10, (2, 3), device=device_type)
        distances = torch.tensor([[0, 1], [1, 0]], device=device_type, dtype=dtype)
        mask = torch.tensor([[True, True, False], [True, True, True]], device=device_type)
        outputs = model(roots, distances, labels, attention_mask=mask)
        assert outputs["logits"].shape == (2, 3, 10)
        for value in outputs.values():
            assert value.dtype == dtype and value.device.type == device_type
            assert torch.isfinite(value).all()
        outputs["total_loss"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())

    def test_masked_metric_pooling_matches_reference(self):
        roots = torch.randint(0, 5, (2, 3, 2))
        mask = torch.tensor([[True, False, False], [True, True, True]])
        distances = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        outputs = self.model(roots, distances, attention_mask=mask)
        composed = self.model.yetzirah_engine(roots)
        pooled = torch.stack([composed[0, :1].mean(0), composed[1].mean(0)])
        torch.testing.assert_close(
            outputs["loss_iso"], F.mse_loss(torch.cdist(pooled, pooled), distances)
        )
