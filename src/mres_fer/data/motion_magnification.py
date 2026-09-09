"""Differentiable Eulerian motion magnification.

Linear Eulerian magnification (Wu et al., 2012) amplifies small intensity changes over
time inside a spatial band, which makes micro-expression motion visible to the frame
branch. Implemented in torch so it can sit inside the model graph as an optional block.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_BINOMIAL = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])


def _gaussian_kernel(device: torch.device, dtype: torch.dtype) -> Tensor:
    kernel = _BINOMIAL.to(device=device, dtype=dtype)
    kernel2d = torch.outer(kernel, kernel)
    return kernel2d / kernel2d.sum()


def _blur(x: Tensor, kernel: Tensor) -> Tensor:
    channels = x.shape[1]
    weight = kernel.expand(channels, 1, *kernel.shape)
    x = F.pad(x, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(x, weight, groups=channels)


def gaussian_pyramid_level(x: Tensor, level: int) -> Tensor:
    """Blur-and-downsample ``x`` ``level`` times."""
    kernel = _gaussian_kernel(x.device, x.dtype)
    for _ in range(level):
        x = _blur(x, kernel)[:, :, ::2, ::2]
    return x


class EulerianMagnification(nn.Module):
    """Amplify temporal variation of a clip inside ``[low_cut, high_cut]`` cycles/frame.

    Input and output are ``(B, T, C, H, W)`` tensors in ``[0, 1]``.
    """

    def __init__(
        self,
        factor: float = 8.0,
        low_cut: float = 0.05,
        high_cut: float = 0.4,
        pyramid_levels: int = 3,
        attenuate_chrominance: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= low_cut < high_cut <= 0.5:
            raise ValueError("expected 0 <= low_cut < high_cut <= 0.5 (cycles per frame)")
        self.factor = factor
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.pyramid_levels = pyramid_levels
        self.attenuate_chrominance = attenuate_chrominance

    def _bandpass(self, x: Tensor) -> Tensor:
        """Ideal temporal bandpass filter along dim 1 of a ``(B, T, ...)`` tensor."""
        num_frames = x.shape[1]
        spectrum = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(num_frames, d=1.0, device=x.device)
        mask = ((freqs >= self.low_cut) & (freqs <= self.high_cut)).to(spectrum.dtype)
        mask = mask.view(1, -1, *([1] * (x.dim() - 2)))
        return torch.fft.irfft(spectrum * mask, n=num_frames, dim=1)

    def forward(self, clip: Tensor) -> Tensor:
        if clip.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(clip.shape)}")
        batch, num_frames, channels, height, width = clip.shape
        if num_frames < 4:
            return clip

        flat = clip.reshape(batch * num_frames, channels, height, width)
        low = gaussian_pyramid_level(flat, self.pyramid_levels)
        low = low.reshape(batch, num_frames, channels, *low.shape[-2:])

        residual = self._bandpass(low) * self.factor
        if self.attenuate_chrominance and channels == 3:
            weights = torch.tensor([1.0, 0.5, 0.5], device=clip.device, dtype=clip.dtype)
            residual = residual * weights.view(1, 1, -1, 1, 1)

        residual = F.interpolate(
            residual.reshape(batch * num_frames, channels, *residual.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, num_frames, channels, height, width)
        return (clip + residual).clamp(0.0, 1.0)
