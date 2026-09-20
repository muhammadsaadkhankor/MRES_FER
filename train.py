"""Step 2 - train the micro-guided macro-expression recogniser.

    python train.py --config configs/default.yaml

Objective per step:
    cross-entropy on the macro label
  + InfoNCE over two views of every short window (macro windows AND micro clips
    share the micro encoder, so real micro dynamics shape the latent space)
  + temporal consistency between the clues of consecutive overlapping windows

Saves checkpoints/last.pt (after the final epoch) and checkpoints/best.pt.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mres_fer.config import Workspace, add_config_args, load_config  # noqa: E402
from mres_fer.datasets import (  # noqa: E402
    MacroClipDataset,
    MicroClipDataset,
    augment_views,
    subject_split,
    write_splits,
)
from mres_fer.losses import info_nce, temporal_consistency  # noqa: E402
from mres_fer.models import MicroGuidedFER  # noqa: E402
from mres_fer.utils import count_parameters, get_logger, resolve_device, save_json, set_seed  # noqa: E402


def build_loaders(cfg, workspace: Workspace, class_to_idx: dict[str, int], logger):
    full = MacroClipDataset(workspace.macro_index, class_to_idx, cfg.sampling, train=False)
    subjects = [r.subject for r in full.records]
    if not subjects:
        raise RuntimeError(f"no macro clips in {workspace.macro_index}; run feature_extractor.py")

    splits = subject_split(
        subjects, cfg.split["val_ratio"], cfg.split["test_ratio"], cfg.train["seed"]
    )
    write_splits(workspace.splits, splits)
    logger.info(
        "subject-independent split -> train %d / val %d / test %d subjects",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    def macro_loader(name: str, train: bool) -> DataLoader:
        dataset = MacroClipDataset(
            workspace.macro_index, class_to_idx, cfg.sampling, splits[name], train=train
        )
        return DataLoader(
            dataset,
            batch_size=cfg.train["batch_size"],
            shuffle=train,
            drop_last=train and len(dataset) > cfg.train["batch_size"],
            num_workers=cfg.train["num_workers"],
        )

    micro_loader = None
    if workspace.micro_index.exists():
        micro_dataset = MicroClipDataset(workspace.micro_index, cfg.sampling)
        if len(micro_dataset) > 0:
            micro_loader = DataLoader(
                micro_dataset,
                batch_size=cfg.train["micro_batch_size"],
                shuffle=True,
                drop_last=False,
                num_workers=cfg.train["num_workers"],
            )
            logger.info("micro corpus: %d clips", len(micro_dataset))
    if micro_loader is None:
        logger.warning("no micro features found - training the macro branch only")

    return macro_loader("train", True), macro_loader("val", False), micro_loader, splits


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            windows = batch["windows"].to(device)
            labels = batch["label"].to(device)
            logits = model(frames, windows)["logits"]
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--resume", default="", help="checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    workspace = Workspace(cfg)
    workspace.create()
    logger = get_logger("train", workspace.logs / "train.log")
    set_seed(cfg.train["seed"])
    device = resolve_device(cfg.train["device"])

    classes = list(cfg.labels["macro_classes"])
    class_to_idx = {name: i for i, name in enumerate(classes)}
    train_loader, val_loader, micro_loader, splits = build_loaders(
        cfg, workspace, class_to_idx, logger
    )

    model = MicroGuidedFER(cfg, num_classes=len(classes)).to(device)
    logger.info("trainable parameters: %.2fM", count_parameters(model) / 1e6)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"])
        logger.info("resumed from %s", args.resume)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train["lr"], weight_decay=cfg.train["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=cfg.train["lr"],
        total_steps=max(cfg.train["epochs"] * max(len(train_loader), 1), 1),
        pct_start=min(0.3, cfg.train["warmup_epochs"] / max(cfg.train["epochs"], 1)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.loss["label_smoothing"])
    use_amp = bool(cfg.train["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    micro_iter = itertools.cycle(micro_loader) if micro_loader is not None else None

    history: list[dict] = []
    best_acc = -1.0
    for epoch in range(1, cfg.train["epochs"] + 1):
        model.train()
        started = time.time()
        sums = {"loss": 0.0, "cls": 0.0, "con": 0.0, "tmp": 0.0}
        seen, correct = 0, 0

        for batch in train_loader:
            frames = batch["frames"].to(device)
            windows = batch["windows"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(frames, windows)
                cls_loss = criterion(output["logits"], labels)

                # --- micro branch: two views of the same windows -> InfoNCE ----
                _, proj_view = model.micro_encoder(augment_views(windows))
                proj = output["proj"].flatten(0, 1)
                proj_view = proj_view.flatten(0, 1)
                if micro_iter is not None:
                    micro_windows = next(micro_iter)["windows"].to(device)
                    _, micro_a = model.micro_encoder(micro_windows)
                    _, micro_b = model.micro_encoder(augment_views(micro_windows))
                    proj = torch.cat([proj, micro_a.flatten(0, 1)], dim=0)
                    proj_view = torch.cat([proj_view, micro_b.flatten(0, 1)], dim=0)
                con_loss = info_nce(proj, proj_view, cfg.loss["temperature"])
                tmp_loss = temporal_consistency(output["clues"])

                loss = (
                    cfg.loss["w_cls"] * cls_loss
                    + cfg.loss["w_contrastive"] * con_loss
                    + cfg.loss["w_temporal"] * tmp_loss
                )

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train["grad_clip"])
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            batch_size = labels.size(0)
            seen += batch_size
            correct += (output["logits"].argmax(1) == labels).sum().item()
            sums["loss"] += loss.item() * batch_size
            sums["cls"] += cls_loss.item() * batch_size
            sums["con"] += con_loss.detach().item() * batch_size
            sums["tmp"] += tmp_loss.detach().item() * batch_size

        train_acc = correct / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.info(
            "epoch %02d/%d | loss %.4f (cls %.4f con %.4f tmp %.4f) | train acc %.3f | "
            "val loss %.4f acc %.3f | %.1fs",
            epoch,
            cfg.train["epochs"],
            sums["loss"] / max(seen, 1),
            sums["cls"] / max(seen, 1),
            sums["con"] / max(seen, 1),
            sums["tmp"] / max(seen, 1),
            train_acc,
            val_loss,
            val_acc,
            time.time() - started,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": sums["loss"] / max(seen, 1),
                "cls_loss": sums["cls"] / max(seen, 1),
                "contrastive_loss": sums["con"] / max(seen, 1),
                "temporal_loss": sums["tmp"] / max(seen, 1),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        checkpoint = {
            "model": model.state_dict(),
            "config": dict(cfg),
            "classes": classes,
            "splits": splits,
            "epoch": epoch,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, workspace.checkpoints / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, workspace.checkpoints / "best.pt")

    save_json(workspace.results / "training_history.json", history)
    logger.info(
        "done. best val acc %.3f | checkpoints: %s",
        best_acc,
        workspace.checkpoints,
    )


if __name__ == "__main__":
    main()
