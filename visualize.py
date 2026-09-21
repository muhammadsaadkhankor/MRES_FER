"""Step 4 - figures for the test split.

    python visualize.py --config configs/default.yaml --checkpoint workdir/checkpoints/last.pt

Produces in <work_root>/figures:
  training_curves.png     losses and accuracy per epoch
  confusion_matrix.png    normalised confusion matrix
  embeddings_tsne.png     t-SNE of the macro clip embeddings
  micro_clue_timeline.png per-window micro-clue activity for a few test clips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evaluation import infer, load_model  # noqa: E402
from mres_fer.config import Workspace, add_config_args, load_config  # noqa: E402
from mres_fer.datasets import MacroClipDataset, read_splits  # noqa: E402
from mres_fer.utils import get_logger, load_json, resolve_device  # noqa: E402


def plot_training_curves(history_path: Path, out_path: Path) -> None:
    if not history_path.exists():
        return
    history = load_json(history_path)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [
        ("cls_loss", "cross entropy"),
        ("contrastive_loss", "InfoNCE"),
        ("temporal_loss", "temporal consistency"),
    ]:
        axes[0].plot(epochs, [h[key] for h in history], label=label)
    axes[0].set(xlabel="epoch", ylabel="loss", title="Training losses")
    axes[0].legend()
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(labels, preds, classes: list[str], out_path: Path) -> None:
    matrix = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xlabel="predicted",
        ylabel="true",
        title="Confusion matrix (row-normalised)",
    )
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{normalised[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, classes: list[str], out_path: Path) -> None:
    if len(embeddings) < 5:
        return
    perplexity = max(2, min(30, len(embeddings) // 3))
    points = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(
        embeddings
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    for index, name in enumerate(classes):
        mask = labels == index
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], label=name, s=28, alpha=0.8)
    ax.set_title("t-SNE of macro clip embeddings (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_micro_clue_timeline(model, dataset, device, classes, out_path: Path, num_clips: int = 4) -> None:
    """Shows how strongly each short window deviates from the clip's mean clue."""
    count = min(num_clips, len(dataset))
    if count == 0:
        return
    fig, axes = plt.subplots(count, 1, figsize=(7, 2.2 * count), squeeze=False)
    for row in range(count):
        sample = dataset[row]
        frames = sample["frames"].unsqueeze(0).to(device)
        windows = sample["windows"].unsqueeze(0).to(device)
        output = model(frames, windows)
        clues = torch.nn.functional.normalize(output["clues"][0], dim=-1)
        activity = (clues - clues.mean(0, keepdim=True)).norm(dim=-1).cpu().numpy()
        predicted = classes[int(output["logits"].argmax(1))]
        truth = classes[int(sample["label"])]
        ax = axes[row][0]
        ax.bar(range(len(activity)), activity, color="steelblue")
        ax.set(
            xlabel="window index (short overlapping windows)",
            ylabel="clue deviation",
            title=f"{sample['clip_id']} | true={truth} pred={predicted}",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    workspace = Workspace(cfg)
    workspace.create()
    logger = get_logger("visualize", workspace.logs / "visualize.log")
    device = resolve_device(cfg.train["device"])

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else workspace.checkpoints / "last.pt"
    model, classes, _ = load_model(checkpoint_path, cfg, device)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    splits = read_splits(workspace.splits)
    dataset = MacroClipDataset(
        workspace.macro_index, class_to_idx, cfg.sampling, splits[args.split], train=False
    )
    loader = DataLoader(dataset, batch_size=cfg.train["batch_size"], num_workers=cfg.train["num_workers"])
    out = infer(model, loader, device)

    plot_training_curves(workspace.results / "training_history.json", workspace.figures / "training_curves.png")
    plot_confusion(out["labels"], out["preds"], classes, workspace.figures / "confusion_matrix.png")
    plot_tsne(out["embeddings"], out["labels"], classes, workspace.figures / "embeddings_tsne.png")
    plot_micro_clue_timeline(model, dataset, device, classes, workspace.figures / "micro_clue_timeline.png")
    logger.info("figures written to %s", workspace.figures)


if __name__ == "__main__":
    main()
