"""Temporal aggregation over per-frame token summaries."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalTransformer(nn.Module):
    """Pool tokens per frame, then attend across time and return a clip embedding.

    ``forward`` also returns the per-frame sequence so downstream heads can inspect
    where in the clip the evidence came from.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        heads: int = 6,
        max_frames: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.dim() != 4:
            raise ValueError(f"expected (B, T, N, D), got {tuple(tokens.shape)}")
        batch, num_frames = tokens.shape[:2]
        if num_frames + 1 > self.pos_embed.shape[1]:
            raise ValueError(f"clip has {num_frames} frames, more than max_frames")

        frame_features = tokens.mean(dim=2)
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, frame_features], dim=1)
        sequence = sequence + self.pos_embed[:, : num_frames + 1]
        encoded = self.norm(self.encoder(sequence))
        return encoded[:, 0], encoded[:, 1:]
