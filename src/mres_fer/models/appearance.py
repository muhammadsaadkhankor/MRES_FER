"""Frame-appearance branch: a ViT applied to every frame of the clip."""

from __future__ import annotations

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

from mres_fer.data.transforms import IMAGENET_MEAN, IMAGENET_STD


class AppearanceBranch(nn.Module):
    """Encode ``(B, T, 3, H, W)`` frames into patch tokens ``(B, T, N, D)``.

    The CLS token is dropped so the tokens stay spatially aligned with the motion
    branch, which is what makes token-level gated fusion meaningful.
    """

    def __init__(
        self,
        backbone: str = "vit_small_patch16_224",
        pretrained: bool = True,
        image_size: int = 224,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        created = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, img_size=image_size
        )
        if not isinstance(created, VisionTransformer):
            raise ValueError(f"appearance backbone must be a timm ViT, got '{backbone}'")
        vit = created
        self.backbone: VisionTransformer = vit
        self.num_prefix_tokens = int(vit.num_prefix_tokens)
        self.embed_dim = int(vit.num_features)
        grid = vit.patch_embed.grid_size
        self.grid_size = (int(grid[0]), int(grid[1]))
        if freeze:
            for param in vit.parameters():
                param.requires_grad_(False)
        self.mean: Tensor
        self.std: Tensor
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def num_tokens(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def forward(self, clip: Tensor) -> Tensor:
        batch, num_frames = clip.shape[:2]
        flat = clip.flatten(0, 1)
        flat = (flat - self.mean) / self.std
        tokens = self.backbone.forward_features(flat)
        tokens = tokens[:, self.num_prefix_tokens :]
        return tokens.reshape(batch, num_frames, tokens.shape[1], tokens.shape[2])
