"""Step 0 - create the working directories used by every other script.

    python setup_dirs.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mres_fer.config import Workspace, add_config_args, load_config  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    workspace = Workspace(cfg)
    for directory in workspace.create():
        print(f"ready: {directory}")

    data_root = Path(cfg.paths["data_root"]).expanduser()
    macro = data_root / cfg.paths["macro_dir"]
    micro = data_root / cfg.paths["micro_dir"]
    print(f"\ndataset root : {data_root}  ({'found' if data_root.is_dir() else 'MISSING'})")
    print(f"macro folder : {macro}  ({'found' if macro.is_dir() else 'MISSING'})")
    print(f"micro folder : {micro}  ({'found' if micro.is_dir() else 'MISSING'})")


if __name__ == "__main__":
    main()
