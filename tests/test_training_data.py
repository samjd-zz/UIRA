import json

import pytest
import torch

from text_encoding import ByteTokenizer
from training_data import CorpusPreparer, TextDataset, CausalBatchCollator


class TestCorpusPreparer:
    def test_deduplicates_before_reproducible_split_and_tracks_provenance(self, tmp_path):
        records = []
        for name, text in [("a", "hello"), ("b", "hello"), ("c", "world"), ("d", "other")]:
            (tmp_path / name).write_text(text)
            records.append({"path": name, "source": f"local:{name}", "license": "CC0-1.0"})
        manifest = tmp_path / "corpus.jsonl"
        manifest.write_text("\n".join(json.dumps(record) for record in records))
        preparer = CorpusPreparer(seed=42, validation_fraction=0.34)
        corpus = preparer.prepare(manifest)
        assert len(corpus.train) == 2 and len(corpus.validation) == 1
        all_docs = corpus.train + corpus.validation
        assert len({doc.content_hash for doc in all_docs}) == 3
        assert sum(len(doc.provenance) for doc in all_docs) == 4
        assert not {doc.content_hash for doc in corpus.train} & {
            doc.content_hash for doc in corpus.validation
        }
        assert corpus.fingerprint == preparer.prepare(manifest).fingerprint
        manifest.write_text("\n".join(json.dumps(record) for record in reversed(records)))
        assert corpus.fingerprint == preparer.prepare(manifest).fingerprint
        (tmp_path / "d").write_text("changed")
        assert corpus.fingerprint != preparer.prepare(manifest).fingerprint

    @pytest.mark.parametrize("record", [
        {"path": "a", "source": "local:a"}, {"path": "a", "source": "", "license": "CC0"},
        {"path": "a", "source": "local:a", "license": ""}, {"unknown": 1},
    ])
    def test_invalid_manifest_records(self, tmp_path, record):
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="line 1"):
            CorpusPreparer().prepare(path)

    def test_invalid_json_and_insufficient_unique_documents(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text("{")
        with pytest.raises(ValueError, match="line 1"):
            CorpusPreparer().prepare(path)
        path.write_text("")
        with pytest.raises(ValueError, match="two"):
            CorpusPreparer().prepare(path)

    @pytest.mark.parametrize("fraction", [0, 1, float("nan"), -0.1, True])
    def test_invalid_split_fraction(self, fraction):
        with pytest.raises(ValueError):
            CorpusPreparer(validation_fraction=fraction)


class TestTextDataset:
    def test_windows_supervise_each_next_token_once(self):
        tokenizer = ByteTokenizer()
        dataset = TextDataset(["abcdefgh", "", "\u4e16\u754c"], tokenizer, max_length=3)
        expected_targets = []
        for text in ["abcdefgh", "", "\u4e16\u754c"]:
            stream = [tokenizer.config.bos_id, *tokenizer.encode(text), tokenizer.config.eos_id]
            expected_targets.extend(stream[1:])
        observed = []
        for sample in dataset:
            assert 2 <= sample.numel() <= 4
            observed.extend(sample[1:].tolist())
        assert observed == expected_targets
        # Only the first window of a document gets BOS.
        assert dataset[1][0].item() != tokenizer.config.bos_id

    def test_dataset_returns_independent_samples(self):
        dataset = TextDataset(["abc"], ByteTokenizer(), max_length=8)
        dataset[0][0] = 100
        assert dataset[0][0] == 1

    @pytest.mark.parametrize("texts,length", [([], 2), (["a"], 0), (["a"], True)])
    def test_invalid_configuration(self, texts, length):
        with pytest.raises(ValueError):
            TextDataset(texts, ByteTokenizer(), max_length=length)

    def test_collation_shifts_once_and_pads(self):
        collator = CausalBatchCollator()
        batch = collator([torch.tensor([1, 10, 11, 2]), torch.tensor([1, 20, 2])])
        assert batch["input_ids"].tolist() == [[1, 10, 11], [1, 20, 0]]
        assert batch["target_tokens"].tolist() == [[10, 11, 2], [20, 2, -100]]
        assert batch["attention_mask"].tolist() == [[True, True, True], [True, True, False]]

    @pytest.mark.parametrize("samples", [
        [], [torch.tensor([1])], [torch.tensor([1.0, 2.0])],
        [torch.tensor([[1, 2]])], [torch.tensor([1, 263])], [torch.tensor([1, 0])],
    ])
    def test_invalid_collation(self, samples):
        with pytest.raises((ValueError, TypeError)):
            CausalBatchCollator()(samples)
