"""Creates a tiny fake MMEW_Final tree so the pipeline can be smoke-tested.

    python scripts/make_dummy_dataset.py --root /tmp/MMEW_Dummy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def write_frames(directory: Path, names: list[str], size: int = 64) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        array = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        Image.fromarray(array).save(directory / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--subjects", type=int, default=6)
    args = parser.parse_args()
    root = Path(args.root)

    for subject in range(1, args.subjects + 1):
        sid = f"S{subject:02d}"
        for label in CLASSES:
            for session in (7, 8):
                names = [f"{sid}-{session:02d}-{i:03d}.jpg" for i in range(1, 13)]
                write_frames(root / "Macro_Expression" / sid / label, names)

    for subject in range(1, args.subjects + 1):
        sid = f"S{subject:02d}"
        for label in CLASSES:
            clip = root / "Micro_Expression" / label / f"{sid}-07-001"
            write_frames(clip, [f"{i}.jpg" for i in range(1, 21)])

    print(f"dummy dataset written to {root}")


if __name__ == "__main__":
    main()
