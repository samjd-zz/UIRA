import json

import numpy as np
import pytest
import torch

from text_encoding import ByteTokenizer, TokenizerConfig, TokenToRootEncoder
from uira import UIRAFullArchitecture


class TestByteTokenizer:
    @pytest.mark.parametrize(
        "text", ["", "hello\nworld", "caf\u00e9 \u4e16\u754c", "\U0001f680",
                 "<|SYSTEM|> [ASSISTANT] END_TURN", "\x00\x7f"]
    )
    def test_round_trip(self, text):
        tokenizer = ByteTokenizer()
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text
        assert all(7 <= token < 263 for token in ids)
        assert ids == (np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(
            np.int64
        ) + 7).tolist()

    def test_control_tokens_require_explicit_handling(self):
        tokenizer = ByteTokenizer()
        config = tokenizer.config
        ids = [config.bos_id, *tokenizer.encode("hi"), config.eos_id]
        with pytest.raises(ValueError, match="control"):
            tokenizer.decode(ids)
        assert tokenizer.decode(ids, skip_control_tokens=True) == "hi"
        assert tokenizer.encode("BOS") != [config.bos_id]

    def test_all_control_ids_are_distinct_and_separate_from_bytes(self):
        config = TokenizerConfig()
        assert [
            config.pad_id, config.bos_id, config.eos_id, config.system_id,
            config.user_id, config.assistant_id, config.end_turn_id
        ] == list(range(7))
        assert config.byte_offset == 7
        assert config.vocab_size == 263

    def test_configuration_round_trip(self, tmp_path):
        tokenizer = ByteTokenizer()
        path = tmp_path / "tokenizer.json"
        tokenizer.save(path)
        saved = json.loads(path.read_text())
        assert saved["format_version"] == 1
        restored = ByteTokenizer.load(path)
        assert restored.config == tokenizer.config
        assert restored.encode("test") == tokenizer.encode("test")

    @pytest.mark.parametrize(
        "change",
        [{"format_version": 99}, {"bos_id": 0}, {"byte_offset": 8},
         {"vocab_size": 262}, {"pad_id": True}, {"num_atomic_roots": 16},
         {"num_atomic_roots": 0}, {"num_atomic_roots": 17.0}]
    )
    def test_rejects_invalid_configuration(self, change):
        with pytest.raises((ValueError, TypeError)):
            TokenizerConfig(**change)

    def test_load_rejects_unknown_or_missing_configuration_fields(self, tmp_path):
        path = tmp_path / "tokenizer.json"
        ByteTokenizer().save(path)
        data = json.loads(path.read_text())
        data["unknown"] = 1
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="fields"):
            ByteTokenizer.load(path)
        data.pop("unknown")
        data.pop("bos_id")
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="fields"):
            ByteTokenizer.load(path)

    @pytest.mark.parametrize("ids", [[-1], [263], [1.5], [True]])
    def test_rejects_invalid_token_ids(self, ids):
        with pytest.raises((ValueError, TypeError)):
            ByteTokenizer().decode(ids)

    def test_invalid_utf8_is_not_silently_replaced(self):
        with pytest.raises(UnicodeDecodeError):
            ByteTokenizer().decode([7 + 0xFF])

    def test_invalid_text_type(self):
        with pytest.raises(TypeError, match="str"):
            ByteTokenizer().encode(b"bytes")


class TestTokenToRootEncoder:
    def test_exhaustive_unique_mapping(self):
        config = TokenizerConfig()
        encoder = TokenToRootEncoder(config)
        tokens = torch.arange(config.vocab_size).unsqueeze(0)
        roots = encoder(tokens)
        assert roots.shape == (1, 263, 2)
        assert roots.dtype == torch.long
        assert roots.min() >= 0 and roots.max() < config.num_atomic_roots
        assert roots.reshape(-1, 2).unique(dim=0).size(0) == config.vocab_size
        torch.testing.assert_close(
            roots[..., 0] * config.num_atomic_roots + roots[..., 1], tokens
        )
        assert "root_pairs" in dict(encoder.named_buffers())
        assert not list(encoder.parameters())

    def test_configuration_and_state_round_trip(self, tmp_path):
        tokenizer = ByteTokenizer(TokenizerConfig(num_atomic_roots=20))
        path = tmp_path / "config.json"
        tokenizer.save(path)
        original = TokenToRootEncoder(tokenizer.config)
        restored = TokenToRootEncoder(ByteTokenizer.load(path).config)
        restored.load_state_dict(original.state_dict())
        ids = torch.tensor([[0, 5, 100, 262]], dtype=torch.int32)
        torch.testing.assert_close(original(ids), restored(ids))

    @pytest.mark.parametrize(
        "tokens",
        [torch.tensor([1, 2]), torch.ones(1, 2, 1, dtype=torch.long),
         torch.tensor([[-1]]), torch.tensor([[263]]),
         torch.tensor([[1.0]]), torch.tensor([[True]]),
         torch.empty(0, 2, dtype=torch.long), torch.empty(1, 0, dtype=torch.long)]
    )
    def test_rejects_invalid_tensor(self, tokens):
        with pytest.raises((ValueError, TypeError)):
            TokenToRootEncoder()(tokens)

    @pytest.mark.parametrize("device_type", ["cpu", "cuda", "xpu"])
    def test_device_and_model_integration(self, device_type):
        backend = getattr(torch, device_type, None)
        if device_type != "cpu" and (backend is None or not backend.is_available()):
            pytest.skip(f"{device_type} hardware unavailable")
        device = torch.device(device_type)
        tokenizer = ByteTokenizer()
        config = tokenizer.config
        encoder = TokenToRootEncoder(config).to(device)
        ids = torch.tensor([tokenizer.encode("hello")], device=device)
        roots = encoder(ids)
        assert roots.device.type == device_type
        model = UIRAFullArchitecture(
            num_atomic_roots=config.num_atomic_roots, embed_dim=8,
            num_projections=2, vocab_size=config.vocab_size
        ).to(device)
        outputs = model(roots, target_tokens=ids)
        assert outputs["logits"].shape == (1, 5, 263)
        assert all(torch.isfinite(value).all() for value in outputs.values())
        outputs["total_loss"].backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
