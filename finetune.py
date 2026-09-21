"""Optional step - end-to-end fine-tuning of the last ViT blocks.

    python finetune.py --config configs/default.yaml
    python finetune.py --config configs/default.yaml --fold 0

Unlike train.py this reads the JPEG frames directly, so the backbone can adapt to
faces instead of being stuck with frozen ImageNet features. Everything above the
backbone is the same MicroGuidedFER model. Checkpoints land in
<work_root>/checkpoints/finetune[_fold<i>].pt.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mres_fer.config import Workspace, add_config_args, load_config  # noqa: E402
from mres_fer.data_index import Clip, scan_macro  # noqa: E402
from mres_fer.datasets import kfold_subject_splits, mixup, read_splits, uniform_indices  # noqa: E402
from mres_fer.losses import temporal_consistency  # noqa: E402
from mres_fer.models import MicroGuidedFER  # noqa: E402
from mres_fer.utils import count_parameters, get_logger, resolve_device, save_json, set_seed  # noqa: E402


class MacroFrameDataset(Dataset):
    """Macro clips as raw frames: returns [T, 3, H, W] plus the label."""

    def __init__(self, clips: list[Clip], class_to_idx: dict[str, int], cfg, train: bool) -> None:
        from torchvision import transforms

        self.clips = [c for c in clips if c.label in class_to_idx]
        self.class_to_idx = class_to_idx
        self.num_frames = cfg.finetune["frames"]
        self.train = train
        size = cfg.extractor["image_size"]
        crop_ratio = cfg.extractor["face_crop_ratio"] if cfg.extractor["face_crop"] else 1.0
        self.crop_ratio = crop_ratio
        normalise = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.ColorJitter(0.2, 0.2, 0.2),
                    transforms.ToTensor(),
                    normalise,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((size, size)), transforms.ToTensor(), normalise]
            )

    def __len__(self) -> int:
        return len(self.clips)

    def _read(self, path: Path) -> torch.Tensor:
        from torchvision.transforms import functional as F

        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.crop_ratio < 1.0:
                image = F.center_crop(image, int(min(image.size) * self.crop_ratio))
            return self.transform(image)

    def __getitem__(self, index: int) -> dict:
        clip = self.clips[index]
        idx = uniform_indices(len(clip.frames), self.num_frames)
        if self.train:
            idx = np.clip(idx + np.random.randint(-1, 2, size=idx.shape), 0, len(clip.frames) - 1)
        frames = torch.stack([self._read(clip.frames[i]) for i in idx])
        if self.train and random.random() < 0.5:  # mirror the whole clip consistently
            frames = torch.flip(frames, dims=[-1])
        return {
            "images": frames,
            "label": torch.tensor(self.class_to_idx[clip.label], dtype=torch.long),
            "clip_id": clip.clip_id,
            "subject": clip.subject,
        }


class FineTunedFER(nn.Module):
    """ViT-B/16 with its last `trainable_blocks` blocks unfrozen + MicroGuidedFER on top."""

    def __init__(self, cfg, num_classes: int) -> None:
        super().__init__()
        from torchvision import models

        builder = getattr(models, cfg.extractor["backbone"])
        self.backbone = builder(weights=cfg.extractor["weights"] or None)
        self.backbone.heads = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        blocks = list(self.backbone.encoder.layers)
        for block in blocks[-cfg.finetune["trainable_blocks"] :]:
            for param in block.parameters():
                param.requires_grad_(True)
        for param in self.backbone.encoder.ln.parameters():
            param.requires_grad_(True)

        self.head = MicroGuidedFER(cfg, num_classes)
        self.window_size = cfg.sampling["window_size"]
        self.window_stride = cfg.sampling["window_stride"]

    def windows(self, features: torch.Tensor) -> torch.Tensor:
        """[B, T, F] -> [B, N, window_size, F] of overlapping windows."""
        size = min(self.window_size, features.size(1))
        stride = max(1, min(self.window_stride, size))
        return features.unfold(1, size, stride).permute(0, 1, 3, 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, frames = images.shape[:2]
        features = self.backbone(images.flatten(0, 1)).view(batch, frames, -1)
        return self.head(features, self.windows(features))


@torch.no_grad()
def test_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    preds, targets = [], []
    for batch in loader:
        preds += model(batch["images"].to(device))["logits"].argmax(1).cpu().tolist()
        targets += batch["label"].tolist()
    return {
        "num_clips": len(targets),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            logits = model(batch["images"].to(device))["logits"]
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--fold", type=int, default=None, help="use fold i of split.num_folds")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    workspace = Workspace(cfg)
    workspace.create()
    logger = get_logger("finetune", workspace.logs / "finetune.log")
    set_seed(cfg.train["seed"])
    device = resolve_device(cfg.train["device"])
    suffix = "" if args.fold is None else f"_fold{args.fold}"

    classes = list(cfg.labels["macro_classes"])
    class_to_idx = {name: i for i, name in enumerate(classes)}
    clips = scan_macro(
        Path(cfg.paths["data_root"]).expanduser() / cfg.paths["macro_dir"], classes
    )
    if args.fold is None:
        splits = read_splits(workspace.splits)
    else:
        subjects = [clip.subject for clip in clips]
        splits = kfold_subject_splits(subjects, cfg.split["num_folds"], cfg.train["seed"])[
            args.fold
        ]
    logger.info(
        "%d macro clips | train %d / val %d subjects",
        len(clips),
        len(splits["train"]),
        len(splits["val"]),
    )

    def loader(name: str, train: bool) -> DataLoader:
        keep = set(splits[name])
        dataset = MacroFrameDataset(
            [c for c in clips if c.subject in keep], class_to_idx, cfg, train
        )
        return DataLoader(
            dataset,
            batch_size=cfg.finetune["batch_size"],
            shuffle=train,
            num_workers=cfg.train["num_workers"],
            pin_memory=device.type == "cuda",
        )

    train_loader, val_loader = loader("train", True), loader("val", False)

    model = FineTunedFER(cfg, len(classes)).to(device)
    logger.info("trainable parameters: %.2fM", count_parameters(model) / 1e6)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.finetune["backbone_lr"]},
            {"params": model.head.parameters(), "lr": cfg.train["lr"]},
        ],
        weight_decay=cfg.train["weight_decay"],
    )
    epochs = cfg.finetune["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=[cfg.finetune["backbone_lr"], cfg.train["lr"]],
        total_steps=max(epochs * max(len(train_loader), 1), 1),
        pct_start=0.2,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.loss["label_smoothing"])
    use_amp = bool(cfg.train["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mixup_alpha = float(cfg.get("augment", {}).get("mixup_alpha", 0.0))
    patience = int(cfg.train.get("early_stopping_patience", 0))

    history, best_acc, best_epoch = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        seen, correct, loss_sum = 0, 0, 0.0
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["label"].to(device)
            images, _, labels_b, lam = mixup(images, images, labels, mixup_alpha)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(images)
                cls_loss = lam * criterion(output["logits"], labels) + (1 - lam) * criterion(
                    output["logits"], labels_b
                )
                loss = cls_loss + cfg.loss["w_temporal"] * temporal_consistency(output["clues"])

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train["grad_clip"])
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            seen += labels.size(0)
            correct += (output["logits"].argmax(1) == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.info(
            "epoch %02d/%d | loss %.4f | train acc %.3f | val loss %.4f acc %.3f | %.1fs",
            epoch,
            epochs,
            loss_sum / max(seen, 1),
            correct / max(seen, 1),
            val_loss,
            val_acc,
            time.time() - started,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(seen, 1),
                "train_acc": correct / max(seen, 1),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": dict(cfg),
                    "classes": classes,
                    "splits": splits,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                workspace.checkpoints / f"finetune{suffix}.pt",
            )
        elif patience and epoch - best_epoch >= patience:
            logger.info("early stop: no validation improvement for %d epochs", patience)
            break

    save_json(workspace.results / f"finetune_history{suffix}.json", history)

    checkpoint = torch.load(
        workspace.checkpoints / f"finetune{suffix}.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    metrics = test_metrics(model, loader("test", False), device)
    metrics.update(best_val_acc=best_acc, best_epoch=best_epoch, subjects=splits["test"])
    save_json(workspace.results / f"finetune_metrics_test{suffix}.json", metrics)
    logger.info(
        "done. best val acc %.3f (epoch %d) | test acc %.3f macro-F1 %.3f",
        best_acc,
        best_epoch,
        metrics["accuracy"],
        metrics["macro_f1"],
    )


if __name__ == "__main__":
    main()
