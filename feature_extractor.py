"""Step 1 - frozen ViT frame features for every macro and micro clip.

    python feature_extractor.py --config configs/default.yaml
    python feature_extractor.py --config configs/default.yaml --domain micro

Writes one [T, feature_dim] .npy per clip plus a manifest CSV that the training
scripts consume, so the ViT runs exactly once over the dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mres_fer.config import Workspace, add_config_args, load_config  # noqa: E402
from mres_fer.data_index import Clip, load_annotations, scan_macro, scan_micro  # noqa: E402
from mres_fer.utils import get_logger, resolve_device  # noqa: E402


class FrameDataset(Dataset):
    """Reads and preprocesses the frames of a single clip."""

    def __init__(self, frames: list[Path], image_size: int, crop: bool, crop_ratio: float) -> None:
        from torchvision import transforms

        self.frames = frames
        steps = []
        if crop:
            steps.append(transforms.CenterCrop(int(image_size / max(crop_ratio, 1e-3))))
        steps += [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.frames[index]) as image:
            return self.transform(image.convert("RGB"))


class ViTFeatureExtractor(torch.nn.Module):
    """Frozen ViT returning the CLS embedding of each frame."""

    def __init__(self, backbone: str, weights: str, device: torch.device) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.is_timm = False
        if backbone.startswith("vit_") and hasattr(__import__("torchvision").models, backbone):
            from torchvision import models

            builder = getattr(models, backbone)
            self.model = builder(weights=weights or None)
            self.feature_dim = self.model.hidden_dim
            self.model.heads = torch.nn.Identity()
        else:  # fall back to timm when the name is not a torchvision ViT
            import timm

            self.model = timm.create_model(backbone, pretrained=bool(weights), num_classes=0)
            self.feature_dim = self.model.num_features
            self.is_timm = True
        self.model.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def extract_clip(
    clip: Clip,
    extractor: ViTFeatureExtractor,
    device: torch.device,
    cfg: dict,
) -> np.ndarray:
    dataset = FrameDataset(
        clip.frames,
        cfg["image_size"],
        cfg["face_crop"],
        cfg["face_crop_ratio"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    chunks = [extractor(batch.to(device, non_blocking=True)).cpu() for batch in loader]
    return torch.cat(chunks, dim=0).numpy().astype(cfg["dtype"])


def run_domain(
    clips: list[Clip],
    out_dir: Path,
    index_csv: Path,
    extractor: ViTFeatureExtractor,
    device: torch.device,
    cfg: dict,
    annotations: dict[str, dict[str, int]],
    overwrite: bool,
    logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, clip in enumerate(clips, start=1):
        feature_path = out_dir / f"{clip.clip_id}.npy"
        if feature_path.exists() and not overwrite:
            features = np.load(feature_path, mmap_mode="r")
        else:
            features = extract_clip(clip, extractor, device, cfg)
            np.save(feature_path, features)
        meta = annotations.get(clip.clip_id, {})
        rows.append(
            {
                "clip_id": clip.clip_id,
                "subject": clip.subject,
                "label": clip.label,
                "domain": clip.domain,
                "num_frames": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "apex": int(meta.get("apex", -1)) - 1 if meta.get("apex", -1) > 0 else -1,
                "onset": meta.get("onset", -1),
                "offset": meta.get("offset", -1),
                "feature_path": str(feature_path.resolve()),
            }
        )
        if position % 25 == 0 or position == len(clips):
            logger.info("%s: %d/%d clips", clip.domain, position, len(clips))
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    logger.info("wrote manifest %s (%d clips)", index_csv, len(rows))


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--domain", choices=["macro", "micro", "both"], default="both")
    parser.add_argument("--overwrite", action="store_true", help="re-extract existing .npy files")
    parser.add_argument("--limit", type=int, default=0, help="debug: only process N clips")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    workspace = Workspace(cfg)
    workspace.create()
    logger = get_logger("extract", workspace.logs / "feature_extractor.log")

    device = resolve_device(cfg.extractor["device"])
    extractor = ViTFeatureExtractor(
        cfg.extractor["backbone"], cfg.extractor["weights"], device
    )
    logger.info(
        "backbone=%s feature_dim=%d device=%s",
        cfg.extractor["backbone"],
        extractor.feature_dim,
        device,
    )
    if extractor.feature_dim != cfg.model["feature_dim"]:
        logger.warning(
            "config model.feature_dim=%d but backbone outputs %d - update the config",
            cfg.model["feature_dim"],
            extractor.feature_dim,
        )

    data_root = Path(cfg.paths["data_root"]).expanduser()
    annotations: dict[str, dict[str, int]] = {}
    for key in ("micro_annotations", "macro_annotations"):
        sheet = cfg.paths.get(key, "")
        if sheet:
            annotations.update(load_annotations(sheet))
            logger.info("loaded %s (%d rows)", sheet, len(annotations))

    if args.domain in {"macro", "both"}:
        clips = scan_macro(data_root / cfg.paths["macro_dir"], cfg.labels["macro_classes"])
        logger.info("macro clips found: %d", len(clips))
        run_domain(
            clips[: args.limit or None],
            workspace.macro_features,
            workspace.macro_index,
            extractor,
            device,
            cfg.extractor,
            annotations,
            args.overwrite,
            logger,
        )

    if args.domain in {"micro", "both"}:
        clips = scan_micro(data_root / cfg.paths["micro_dir"], cfg.labels["micro_ignore"])
        logger.info("micro clips found: %d", len(clips))
        run_domain(
            clips[: args.limit or None],
            workspace.micro_features,
            workspace.micro_index,
            extractor,
            device,
            cfg.extractor,
            annotations,
            args.overwrite,
            logger,
        )


if __name__ == "__main__":
    main()
