from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mres_fer.cli import main
from mres_fer.config import Config


def _config_file(config: Config, path: Path) -> Path:
    path.write_text(yaml.safe_dump(config.to_dict()))
    return path


def test_loso_runs_one_fold_per_subject(
    tiny_config: Config, tiny_dataset: Path, tmp_path: Path
) -> None:
    config_path = _config_file(tiny_config, tmp_path / "config.yaml")
    output_dir = Path(tiny_config.run.output_dir)

    exit_code = main(
        [
            "loso",
            "--config",
            str(config_path),
            "--manifest",
            str(tiny_dataset / "train.json"),
            "--override",
            "optim.batch_size=1",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "loso_summary.json").read_text())
    assert sorted(summary["folds"]) == ["s0", "s1"]
    assert "macro_uf1" in summary["mean"]
    assert (output_dir / "fold_s0" / "best.pt").exists()


def test_prepare_mmew_writes_manifest(
    tmp_path: Path, mmew_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "manifests"
    exit_code = main(["prepare-mmew", "--root", str(mmew_root), "--out", str(out)])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["micro_clips"] == 2
    assert report["macro_clips"] == 1
    records = json.loads((out / "manifest.json").read_text())
    assert {r["clip_id"] for r in records} == {
        "micro_S03-01-002",
        "micro_S05-07-001",
        "macro_S03-02-001",
    }
    assert json.loads((out / "labels.json").read_text())["macro"]["anger"] == 0
