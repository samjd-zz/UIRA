from dataclasses import asdict

import pytest
import torch

from sequence_model import CausalSequenceProcessor, SequenceConfig


class TestSequenceConfig:
    @pytest.mark.parametrize(
        "change",
        [{"embed_dim": 0}, {"num_heads": 3}, {"num_layers": 0},
         {"feedforward_dim": -1}, {"max_length": 0}, {"dropout": -0.1},
         {"dropout": 1.0}, {"dropout": float("nan")}, {"num_heads": True},
         {"num_layers": 1.5}, {"dropout": True}]
    )
    def test_invalid_configuration(self, change):
        with pytest.raises((ValueError, TypeError)):
            SequenceConfig(**change)

    def test_configuration_round_trip(self):
        config = SequenceConfig(embed_dim=8, num_heads=2)
        assert SequenceConfig(**asdict(config)) == config


class TestCausalSequenceProcessor:
    def setup_method(self):
        torch.manual_seed(123)
        self.config = SequenceConfig(
            embed_dim=8, num_heads=2, num_layers=2,
            feedforward_dim=16, dropout=0.0, max_length=8,
        )
        self.processor = CausalSequenceProcessor(self.config).eval()

    def test_future_isolation_in_evaluation_and_inference_modes(self):
        inputs = torch.randn(2, 5, 8)
        changed = inputs.clone()
        changed[:, 3:] += 10
        for inference in (False, True):
            with torch.inference_mode(inference):
                original = self.processor(inputs)
                result = self.processor(changed)
            torch.testing.assert_close(
                original[:, :3], result[:, :3], atol=1e-6, rtol=1e-5
            )

    def test_previous_context_affects_later_output(self):
        inputs = torch.randn(1, 3, 8)
        changed = inputs.clone()
        changed[:, 0, 0] += 5
        assert not torch.allclose(
            self.processor(inputs)[:, -1], self.processor(changed)[:, -1]
        )

    def test_positions_distinguish_identical_inputs(self):
        result = self.processor(torch.zeros(1, 3, 8))
        assert not torch.allclose(result[:, 0], result[:, 1])

    def test_padding_and_state_reload(self):
        inputs = torch.randn(2, 3, 8)
        padded = torch.cat((inputs, torch.randn(2, 2, 8)), dim=1)
        mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
        with torch.inference_mode():
            expected = self.processor(inputs)
            output = self.processor(padded, mask)
        torch.testing.assert_close(expected, output[:, :3], atol=1e-6, rtol=1e-5)
        assert torch.count_nonzero(output[:, 3:]) == 0
        restored = CausalSequenceProcessor(self.config).eval()
        restored.load_state_dict(self.processor.state_dict())
        torch.testing.assert_close(restored(inputs), expected)

    @pytest.mark.parametrize(
        "shape,mask",
        [((0, 3, 8), None), ((1, 0, 8), None), ((1, 9, 8), None),
         ((1, 3, 7), None), ((3, 8), None),
         ((1, 3, 8), torch.ones(1, 2, dtype=torch.bool)),
         ((1, 3, 8), torch.ones(1, 3)),
         ((1, 3, 8), torch.zeros(1, 3, dtype=torch.bool)),
         ((1, 3, 8), torch.tensor([[False, True, True]])),
         ((1, 3, 8), torch.tensor([[True, False, True]]))]
    )
    def test_invalid_inputs(self, shape, mask):
        with pytest.raises((ValueError, TypeError)):
            self.processor(torch.randn(shape), mask)

    def test_integer_features_rejected(self):
        with pytest.raises(TypeError, match="floating"):
            self.processor(torch.ones(1, 3, 8, dtype=torch.long))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu"])
    def test_device_dtype_and_backward(self, dtype, device_type):
        backend = getattr(torch, device_type, None)
        if device_type != "cpu" and (backend is None or not backend.is_available()):
            pytest.skip(f"{device_type} hardware unavailable")
        processor = self.processor.to(device=device_type, dtype=dtype).train()
        inputs = torch.randn(2, 3, 8, device=device_type, dtype=dtype,
                             requires_grad=True)
        output = processor(inputs)
        assert output.dtype == dtype and output.device.type == device_type
        output.square().mean().backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in processor.parameters())
