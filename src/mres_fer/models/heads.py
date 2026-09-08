"""Classification heads.

The hypothesis under test is that fine-grained micro-expression evidence helps infer the
macro expression, so the macro head can be conditioned on the micro head's output
(``macro_from_micro``) rather than reading the clip embedding alone.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ClipHead(nn.Module):
    """Two-layer classifier over a clip embedding."""

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MicroGuidedMacroHead(nn.Module):
    """Macro classifier that consumes the clip embedding plus the micro evidence."""

    def __init__(
        self,
        dim: int,
        num_macro_classes: int,
        num_micro_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.micro_proj = nn.Sequential(
            nn.Linear(num_micro_classes, dim),
            nn.GELU(),
        )
        self.head = ClipHead(dim * 2, num_macro_classes, dropout)

    def forward(self, embedding: Tensor, micro_logits: Tensor) -> Tensor:
        micro_evidence = self.micro_proj(micro_logits.softmax(dim=-1))
        return self.head(torch.cat([embedding, micro_evidence], dim=-1))
