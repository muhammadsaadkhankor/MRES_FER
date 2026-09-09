from __future__ import annotations

from pathlib import Path

import pytest

from mres_fer.config import load_config


def test_load_config_applies_defaults_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("model:\n  fusion_dim: 128\n")
    config = load_config(path, {"optim.epochs": 3, "run.device": "cpu"})
    assert config.model.fusion_dim == 128
    assert config.optim.epochs == 3
    assert config.run.device == "cpu"
    assert config.data.num_frames == 16


def test_extends_merges_section_keys_over_the_base(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model:\n  fusion_dim: 128\n  dropout: 0.5\n")
    child = tmp_path / "ablation.yaml"
    child.write_text("extends: base.yaml\nmodel:\n  dropout: 0.0\nloss:\n  micro_weight: 0.0\n")
    config = load_config(child)
    assert (config.model.fusion_dim, config.model.dropout) == (128, 0.0)
    assert config.loss.micro_weight == 0.0


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("model:\n  not_a_key: 1\n")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(path)
