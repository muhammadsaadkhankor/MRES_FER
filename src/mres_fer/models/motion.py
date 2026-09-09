"""Motion branch: a light convolutional encoder over dense optical flow."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


def _block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class FlowEncoder(nn.Module):
    """Encode ``(B, T, 2, H, W)`` flow into tokens ``(B, T, N, D)``.

    The output is resampled onto ``grid_size`` so each motion token corresponds to the
    same image region as the appearance token with the same index.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        embed_dim: int = 256,
        width: int = 64,
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.stem = _block(in_channels, width, stride=2)
        self.stage2 = _block(width, width * 2, stride=2)
        self.stage3 = _block(width * 2, width * 4, stride=2)
        self.proj = nn.Conv2d(width * 4, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, flow: Tensor) -> Tensor:
        batch, num_frames = flow.shape[:2]
        x = flow.flatten(0, 1)
        # Flow magnitudes are unbounded; a soft squash keeps the stem well conditioned.
        x = x.sign() * (x.abs() + 1.0).log()
        x = self.stage3(self.stage2(self.stem(x)))
        x = self.proj(x)
        if x.shape[-2:] != self.grid_size:
            x = F.interpolate(x, size=self.grid_size, mode="bilinear", align_corners=False)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens.reshape(batch, num_frames, tokens.shape[1], tokens.shape[2])
