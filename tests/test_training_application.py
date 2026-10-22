import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from train import TrainingApplication


class TestTrainingApplication:
    def test_train_save_and_resume_local_corpus(self, tmp_path, capsys):
        # Original test fixtures; no external dataset or download required.
        records = []
        for index, text in enumerate(("A small original test.", "A second local fixture.")):
            name = f"document-{index}.txt"
            (tmp_path / name).write_text(text)
            records.append({"path": name, "source": f"test:{index}", "license": "CC0-1.0"})
        manifest = tmp_path / "corpus.jsonl"
        manifest.write_text("\n".join(json.dumps(record) for record in records))
        checkpoint = tmp_path / "checkpoint.pt"
        args = [
            "--manifest", str(manifest), "--output", str(checkpoint),
            "--steps", "2", "--device", "cpu", "--embed-dim", "8",
            "--heads", "2", "--layers", "1", "--feedforward-dim", "16",
            "--context-length", "8", "--dropout", "0",
        ]
        metrics = TrainingApplication().run(args)
        assert metrics["supervised_tokens"] > 0
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert [line["step"] for line in lines] == [1, 2, 2]
        payload = torch.load(checkpoint, weights_only=True)
        metadata = payload["corpus_metadata"]
        assert metadata["fingerprint"] == payload["corpus_fingerprint"]
        assert metadata["splits"]["train"][0]["provenance"][0]["license"] == "CC0-1.0"
        assert "text" not in metadata["splits"]["train"][0]
        TrainingApplication().run([*args, "--resume", str(checkpoint)])
        assert torch.load(checkpoint, weights_only=True)["step"] == 4
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert [line["step"] for line in lines] == [3, 4, 4]

    def test_negative_steps_fail_before_io(self, tmp_path):
        with pytest.raises(SystemExit):
            TrainingApplication().run([
                "--manifest", str(tmp_path / "missing.jsonl"),
                "--output", str(tmp_path / "model.pt"), "--steps", "-1",
            ])

    def test_module_cli_help(self):
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        )
        assert "--resume" in result.stdout
        assert "--manifest" in result.stdout
