from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mres_fer.cli import main
from mres_fer.config import Config, load_config


def _config_file(config: Config, path: Path) -> Path:
    path.write_text(yaml.safe_dump(config.to_dict()))
    return path


@pytest.mark.parametrize(
    "name", ["base", "flow_only", "magnified", "mmew", "mmew_joint", "mmew_macro_only"]
)
def test_shipped_configs_load(name: str) -> None:
    load_config(Path(__file__).parents[1] / "configs" / f"{name}.yaml")


def test_transfer_config_sets_the_two_stage_schedule() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "mmew_transfer.yaml")
    assert config.transfer.freeze == "encoder"
    assert config.data.sampling == "apex_centered"  # inherited from configs/mmew.yaml


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
    (mmew_root / "MMEW_Micro_Exp.csv").write_text(
        "Subject,Filename,OnsetFrame,ApexFrame,OffsetFrame\nS03,S03-01-002,100,103,105\n"
    )
    exit_code = main(["prepare-mmew", "--root", str(mmew_root), "--out", str(out)])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["micro_clips"], report["macro_clips"]) == (3, 2)
    assert (report["model.num_micro_classes"], report["model.num_macro_classes"]) == (3, 2)
    assert report["micro_annotations"].endswith("MMEW_Micro_Exp.csv")

    records = {r["clip_id"]: r for r in json.loads((out / "manifest.json").read_text())}
    assert set(records) == {
        "micro_S03-01-002",
        "micro_S05-07-001",
        "micro_S13-07-001",
        "macro_S03-07-001",
        "macro_S04-05-001",
    }
    assert records["micro_S03-01-002"]["apex"] == 3  # auto-detected annotations applied
    assert json.loads((out / "labels.json").read_text())["macro"]["anger"] == 0
