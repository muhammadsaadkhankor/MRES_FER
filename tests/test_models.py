from __future__ import annotations

import pytest
import torch

from mres_fer.config import Config
from mres_fer.data.motion_magnification import EulerianMagnification
from mres_fer.models.fusion import GatedFusion
from mres_fer.models.motion import FlowEncoder
from mres_fer.models.mres_fer import build_model
from mres_fer.models.temporal import TemporalTransformer


def test_flow_encoder_matches_the_appearance_grid() -> None:
    encoder = FlowEncoder(grid_size=(7, 7), embed_dim=32, width=8)
    tokens = encoder(torch.randn(2, 3, 2, 56, 56))
    assert tokens.shape == (2, 3, 49, 32)


def test_gate_scales_the_motion_stream() -> None:
    fusion = GatedFusion(appearance_dim=8, motion_dim=8, output_dim=16, use_gate=True)
    fused, gate = fusion(torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 8))
    assert fused.shape == (2, 3, 4, 16)
    assert gate.shape == (2, 3, 4, 8)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_fusion_needs_at_least_one_stream() -> None:
    fusion = GatedFusion(appearance_dim=8, motion_dim=8, output_dim=16)
    with pytest.raises(ValueError):
        fusion(None, None)


def test_temporal_transformer_returns_clip_and_frame_features() -> None:
    temporal = TemporalTransformer(dim=16, depth=1, heads=2)
    clip, frames = temporal(torch.randn(2, 5, 9, 16))
    assert clip.shape == (2, 16)
    assert frames.shape == (2, 5, 16)


def test_magnification_preserves_shape_and_range() -> None:
    magnifier = EulerianMagnification(factor=5.0, pyramid_levels=2)
    clip = torch.rand(1, 8, 3, 32, 32)
    out = magnifier(clip)
    assert out.shape == clip.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_magnification_amplifies_temporal_variation() -> None:
    magnifier = EulerianMagnification(factor=10.0, pyramid_levels=1)
    time = torch.arange(16).float()
    signal = 0.5 + 0.01 * torch.sin(2 * torch.pi * 0.2 * time)
    clip = signal.view(1, 16, 1, 1, 1).expand(1, 16, 3, 16, 16).contiguous()
    out = magnifier(clip)
    assert out.std(dim=1).mean() > clip.std(dim=1).mean()


def test_full_model_forward_and_backward(tiny_config: Config) -> None:
    model = build_model(tiny_config.model, image_size=tiny_config.data.image_size)
    frames = torch.rand(2, tiny_config.data.num_frames, 3, 32, 32)
    flow = torch.randn(2, tiny_config.data.num_frames, 2, 32, 32)
    output = model(frames, flow)
    assert output.micro_logits.shape == (2, tiny_config.model.num_micro_classes)
    assert output.macro_logits.shape == (2, tiny_config.model.num_macro_classes)
    assert output.gate is not None

    output.macro_logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_motion_stream_requires_flow(tiny_config: Config) -> None:
    model = build_model(tiny_config.model, image_size=tiny_config.data.image_size)
    with pytest.raises(ValueError, match="no flow"):
        model(torch.rand(1, tiny_config.data.num_frames, 3, 32, 32), None)


def test_appearance_can_be_disabled(tiny_config: Config) -> None:
    config = tiny_config.model
    config.use_appearance = False
    model = build_model(config, image_size=tiny_config.data.image_size)
    output = model(
        torch.rand(1, tiny_config.data.num_frames, 3, 32, 32),
        torch.randn(1, tiny_config.data.num_frames, 2, 32, 32),
    )
    assert output.macro_logits.shape == (1, config.num_macro_classes)
    assert output.gate is None
