"""Precompute and cache optical flow for every clip in a manifest.

Flow extraction dominates dataloading cost, so run this once per (manifest, sampling,
algorithm) combination before training::

    python scripts/precompute_flow.py --config configs/base.yaml --split train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mres_fer.config import load_config
from mres_fer.data.dataset import frame_paths, is_still, read_manifest
from mres_fer.data.optical_flow import FlowCache, compute_flow_sequence
from mres_fer.data.sampling import apex_centered_indices, uniform_indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config).data
    if config.flow_cache_dir is None:
        raise SystemExit("data.flow_cache_dir must be set to precompute flow")

    manifest = config.train_manifest if args.split == "train" else config.val_manifest
    records = read_manifest(manifest)
    root = Path(config.root)
    cache = FlowCache(config.flow_cache_dir, config.flow_algorithm)

    for record in tqdm(records, desc=f"flow[{args.split}]"):
        if is_still(root, record):  # a single image has no motion to cache
            continue
        paths = frame_paths(root, record)
        if config.sampling == "apex_centered" and record.apex is not None:
            indices = apex_centered_indices(
                len(paths), config.num_frames, record.apex, record.onset, record.offset
            )
        else:
            indices = uniform_indices(len(paths), config.num_frames)

        frames = []
        for index in indices:
            image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"could not read {paths[index]}")
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        stacked = np.stack(frames)
        size = (stacked.shape[1], stacked.shape[2])

        if not args.overwrite and cache.load(record.clip_id, indices, size) is not None:
            continue
        flow = compute_flow_sequence(list(stacked), config.flow_algorithm)
        cache.save(record.clip_id, indices, size, flow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
