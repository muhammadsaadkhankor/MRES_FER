"""Losses of the framework: InfoNCE over window views + temporal consistency."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent between two views of the same items.

    z_a, z_b: [N, D] projections; item i in z_a is positive with item i in z_b,
    everything else in the batch is a negative. No labels are required, so the
    micro-clue space is shaped by *dynamics*, not by micro-expression classes.
    """
    if z_a.numel() == 0:
        return z_a.new_zeros(())
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.t() / temperature
    targets = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def temporal_consistency(clues: torch.Tensor) -> torch.Tensor:
    """Penalises abrupt jumps between latent clues of consecutive windows.

    clues: [B, N, D]. Overlapping windows describe a continuous facial motion,
    so their latents should drift smoothly rather than flip between windows.
    """
    if clues.size(1) < 2:
        return clues.new_zeros(())
    clues = F.normalize(clues, dim=-1)
    return (clues[:, 1:] - clues[:, :-1]).pow(2).sum(-1).mean()
