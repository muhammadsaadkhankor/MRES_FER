r"""Model of the framework diagram.

    ViT features ->  short overlapping windows -> MicroEncoder -> latent micro-clues
                 \                                                     |
                  -> full sequence ------------------> MacroEncoder <--/
                                                            |
                                                        Classifier

The ViT backbone itself is frozen and run offline by feature_extractor.py, so
this module only ever sees pre-computed frame features.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positions, added to the projected frame features."""

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        return x + self.pe[:, : x.size(1)]


def _encoder(dim: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=dim * 4,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class MicroEncoder(nn.Module):
    """Motion / temporal encoder over a short overlapping window of frames.

    The window is represented by its frame-to-frame differences (optionally
    magnified), which is what makes the branch sensitive to the brief,
    low-amplitude dynamics that characterise micro-expressions.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        micro_dim: int,
        proj_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        magnification: float,
    ) -> None:
        super().__init__()
        self.magnification = magnification
        self.input_proj = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(hidden_dim)
        self.encoder = _encoder(hidden_dim, heads, layers, dropout)
        self.to_clue = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, micro_dim))
        self.projector = nn.Sequential(
            nn.Linear(micro_dim, micro_dim),
            nn.GELU(),
            nn.Linear(micro_dim, proj_dim),
        )

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """windows: [B, N, W, F] -> (clues [B, N, micro_dim], proj [B, N, proj_dim])."""
        batch, num_windows, window, feat = windows.shape
        x = windows.reshape(batch * num_windows, window, feat)
        motion = torch.zeros_like(x)
        motion[:, 1:] = (x[:, 1:] - x[:, :-1]) * self.magnification
        x = self.input_proj(torch.cat([x, motion], dim=-1))
        x = self.encoder(self.pos(x))
        clue = self.to_clue(x.mean(dim=1))
        proj = self.projector(clue)
        return clue.view(batch, num_windows, -1), proj.view(batch, num_windows, -1)


class MacroEncoder(nn.Module):
    """Transformer over the full clip, conditioned on the latent micro-clues."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        micro_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        use_micro_tokens: bool,
    ) -> None:
        super().__init__()
        self.use_micro_tokens = use_micro_tokens
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim)
        )
        self.clue_proj = nn.Sequential(nn.LayerNorm(micro_dim), nn.Linear(micro_dim, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_embed = nn.Parameter(torch.zeros(2, hidden_dim))  # frame vs. clue token
        self.pos = PositionalEncoding(hidden_dim)
        self.encoder = _encoder(hidden_dim, heads, layers, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(
        self, frames: torch.Tensor, clues: torch.Tensor | None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """frames: [B, T, F], clues: [B, N, micro_dim], mask: [B, T] (True = padding)."""
        tokens = self.pos(self.frame_proj(frames)) + self.type_embed[0]
        pad_mask = mask
        if self.use_micro_tokens and clues is not None:
            clue_tokens = self.clue_proj(clues) + self.type_embed[1]
            tokens = torch.cat([tokens, clue_tokens], dim=1)
            if pad_mask is not None:
                clue_pad = torch.zeros(
                    clue_tokens.shape[:2], dtype=torch.bool, device=tokens.device
                )
                pad_mask = torch.cat([pad_mask, clue_pad], dim=1)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        if pad_mask is not None:
            cls_pad = torch.zeros((tokens.size(0), 1), dtype=torch.bool, device=tokens.device)
            pad_mask = torch.cat([cls_pad, pad_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=pad_mask)
        return self.norm(encoded[:, 0])


class MicroGuidedFER(nn.Module):
    """Full model: micro branch + macro branch + linear classifier."""

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.micro_encoder = MicroEncoder(
            feature_dim=model_cfg["feature_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            micro_dim=model_cfg["micro_dim"],
            proj_dim=model_cfg["proj_dim"],
            layers=model_cfg["micro_layers"],
            heads=model_cfg["micro_heads"],
            dropout=model_cfg["dropout"],
            magnification=model_cfg["motion_magnification"],
        )
        self.macro_encoder = MacroEncoder(
            feature_dim=model_cfg["feature_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            micro_dim=model_cfg["micro_dim"],
            layers=model_cfg["macro_layers"],
            heads=model_cfg["macro_heads"],
            dropout=model_cfg["dropout"],
            use_micro_tokens=model_cfg["use_micro_tokens"],
        )
        self.classifier = nn.Linear(model_cfg["hidden_dim"], num_classes)

    def forward(self, frames: torch.Tensor, windows: torch.Tensor) -> dict[str, torch.Tensor]:
        clues, proj = self.micro_encoder(windows)
        embedding = self.macro_encoder(frames, clues)
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
            "clues": clues,
            "proj": proj,
        }
