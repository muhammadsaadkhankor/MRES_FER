"""Token-level fusion of the appearance and motion streams.

The baseline is plain concatenation to ``T x N x (Dv + Dm)`` followed by a projection.
With ``use_gate`` a per-token sigmoid gate is learned from the concatenated features and
decides how much motion evidence to admit, so uninformative flow (blur, head motion) can
be suppressed instead of averaged in.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GatedFusion(nn.Module):
    def __init__(
        self,
        appearance_dim: int,
        motion_dim: int,
        output_dim: int,
        use_gate: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.appearance_dim = appearance_dim
        self.motion_dim = motion_dim
        concat_dim = appearance_dim + motion_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = (
            nn.Sequential(nn.Linear(concat_dim, motion_dim), nn.Sigmoid()) if use_gate else None
        )

    def forward(self, appearance: Tensor | None, motion: Tensor | None) -> tuple[Tensor, Tensor]:
        """Return fused tokens ``(B, T, N, D)`` and the gate values used."""
        if appearance is None and motion is None:
            raise ValueError("at least one of the two streams must be enabled")
        if appearance is None:
            assert motion is not None
            appearance = motion.new_zeros((*motion.shape[:-1], self.appearance_dim))
        if motion is None:
            motion = appearance.new_zeros((*appearance.shape[:-1], self.motion_dim))

        joint = torch.cat([appearance, motion], dim=-1)
        if self.gate is not None:
            gate = self.gate(joint)
            joint = torch.cat([appearance, motion * gate], dim=-1)
        else:
            gate = motion.new_ones(motion.shape)
        return self.proj(joint), gate
