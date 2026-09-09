from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mres_fer.config import Config
from mres_fer.engine.transfer import apply_freeze, run_transfer
from mres_fer.models.mres_fer import build_model


def test_freeze_encoder_leaves_only_the_macro_head_trainable(tiny_config: Config) -> None:
    model = build_model(tiny_config.model, tiny_config.data.image_size)
    apply_freeze(model, "encoder")
    trainable = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"macro_head"}


def test_freeze_appearance_keeps_the_motion_path_trainable(tiny_config: Config) -> None:
    model = build_model(tiny_config.model, tiny_config.data.image_size)
    apply_freeze(model, "appearance")
    frozen = {name.split(".")[0] for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"appearance"}


def test_unknown_freeze_policy_is_rejected(tiny_config: Config) -> None:
    model = build_model(tiny_config.model, tiny_config.data.image_size)
    with pytest.raises(ValueError, match="unknown freeze policy"):
        apply_freeze(model, "half")


def test_transfer_runs_both_stages_and_keeps_the_micro_encoder_fixed(tiny_config: Config) -> None:
    tiny_config.transfer.stage1_epochs = 1
    tiny_config.transfer.stage2_epochs = 1
    tiny_config.transfer.freeze = "encoder"

    summary = run_transfer(tiny_config)
    assert set(summary) == {"stage1_micro", "stage2_macro"}
    assert "macro_uf1" in summary["stage2_macro"]

    root = Path(tiny_config.run.output_dir)
    assert (root / "transfer_summary.json").exists()
    pretrained = torch.load(root / "stage1_micro" / "best.pt", map_location="cpu")["model"]
    finetuned = torch.load(root / "stage2_macro" / "last.pt", map_location="cpu")["model"]
    key = "temporal.cls_token"
    assert torch.equal(pretrained[key], finetuned[key])
