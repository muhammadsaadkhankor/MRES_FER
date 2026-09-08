"""End-to-end model wiring the pipeline blocks together."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from mres_fer.config import MagnificationConfig, ModelConfig
from mres_fer.data.motion_magnification import EulerianMagnification
from mres_fer.models.appearance import AppearanceBranch
from mres_fer.models.fusion import GatedFusion
from mres_fer.models.heads import ClipHead, MicroGuidedMacroHead
from mres_fer.models.motion import FlowEncoder
from mres_fer.models.temporal import TemporalTransformer

# Fallback token grid when the appearance branch is disabled and there is no ViT to
# borrow a patch layout from.
DEFAULT_GRID = (14, 14)


@dataclass
class ModelOutput:
    micro_logits: Tensor
    macro_logits: Tensor
    gate: Tensor | None
    clip_embedding: Tensor


class MresFer(nn.Module):
    """Micro-expression guided macro facial expression recogniser.

    ``frames`` are ``(B, T, 3, H, W)`` in ``[0, 1]`` and ``flow`` is ``(B, T, 2, H, W)``
    in pixels. Either stream may be disabled through :class:`ModelConfig`.
    """

    def __init__(
        self,
        config: ModelConfig,
        image_size: int = 224,
        magnification: MagnificationConfig | None = None,
    ) -> None:
        super().__init__()
        if not (config.use_appearance or config.use_motion):
            raise ValueError("at least one of use_appearance / use_motion must be True")
        self.config = config

        self.magnifier: EulerianMagnification | None = None
        if config.use_motion_magnification:
            mag = magnification or MagnificationConfig()
            self.magnifier = EulerianMagnification(
                factor=mag.factor,
                low_cut=mag.low_cut,
                high_cut=mag.high_cut,
                pyramid_levels=mag.pyramid_levels,
                attenuate_chrominance=mag.attenuate_chrominance,
            )

        self.appearance: AppearanceBranch | None = None
        appearance_dim = 0
        grid = DEFAULT_GRID
        if config.use_appearance:
            self.appearance = AppearanceBranch(
                backbone=config.appearance_backbone,
                pretrained=config.appearance_pretrained,
                image_size=image_size,
                freeze=config.freeze_appearance,
            )
            appearance_dim = self.appearance.embed_dim
            grid = self.appearance.grid_size

        self.motion: FlowEncoder | None = None
        motion_dim = 0
        if config.use_motion:
            motion_dim = config.fusion_dim
            self.motion = FlowEncoder(
                grid_size=grid, embed_dim=motion_dim, width=config.motion_width
            )

        self.fusion = GatedFusion(
            appearance_dim=appearance_dim,
            motion_dim=motion_dim,
            output_dim=config.fusion_dim,
            use_gate=config.use_gate and config.use_motion and config.use_appearance,
            dropout=config.dropout,
        )
        self.temporal = TemporalTransformer(
            dim=config.fusion_dim,
            depth=config.temporal_layers,
            heads=config.temporal_heads,
            dropout=config.dropout,
        )
        self.micro_head = ClipHead(config.fusion_dim, config.num_micro_classes, config.dropout)
        self.macro_head: nn.Module
        if config.macro_from_micro:
            self.macro_head = MicroGuidedMacroHead(
                config.fusion_dim,
                config.num_macro_classes,
                config.num_micro_classes,
                config.dropout,
            )
        else:
            self.macro_head = ClipHead(config.fusion_dim, config.num_macro_classes, config.dropout)

    def forward(self, frames: Tensor, flow: Tensor | None = None) -> ModelOutput:
        if self.magnifier is not None:
            frames = self.magnifier(frames)

        appearance_tokens = self.appearance(frames) if self.appearance is not None else None
        motion_tokens = None
        if self.motion is not None:
            if flow is None:
                raise ValueError("model was configured with use_motion=True but no flow was given")
            motion_tokens = self.motion(flow)

        fused, gate = self.fusion(appearance_tokens, motion_tokens)
        clip_embedding, _ = self.temporal(fused)

        micro_logits = self.micro_head(clip_embedding)
        if isinstance(self.macro_head, MicroGuidedMacroHead):
            macro_logits = self.macro_head(clip_embedding, micro_logits)
        else:
            macro_logits = self.macro_head(clip_embedding)

        return ModelOutput(
            micro_logits=micro_logits,
            macro_logits=macro_logits,
            gate=gate if self.fusion.use_gate else None,
            clip_embedding=clip_embedding,
        )


def build_model(
    config: ModelConfig, image_size: int = 224, magnification: MagnificationConfig | None = None
) -> MresFer:
    return MresFer(config, image_size=image_size, magnification=magnification)
