"""Tiny YAML config loader with dotted CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict that also supports attribute and dotted access."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _coerce(text: str) -> Any:
    return yaml.safe_load(text)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    with open(path) as handle:
        cfg = Config(yaml.safe_load(handle))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        cfg.set_path(key.strip(), _coerce(value.strip()))
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/default.yaml", help="path to the YAML config")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. train.epochs=50 run.device=cpu",
    )
    return parser


class Workspace:
    """Resolves and creates the output directory layout declared in the config."""

    def __init__(self, cfg: Config) -> None:
        paths = cfg.paths
        self.root = Path(paths["work_root"]).expanduser().resolve()
        self.features = self.root / paths["features_dir"]
        self.manifests = self.root / paths["manifests_dir"]
        self.checkpoints = self.root / paths["checkpoints_dir"]
        self.results = self.root / paths["results_dir"]
        self.figures = self.root / paths["figures_dir"]
        self.logs = self.root / paths["logs_dir"]

    @property
    def macro_features(self) -> Path:
        return self.features / "macro"

    @property
    def micro_features(self) -> Path:
        return self.features / "micro"

    @property
    def macro_index(self) -> Path:
        return self.manifests / "macro_index.csv"

    @property
    def micro_index(self) -> Path:
        return self.manifests / "micro_index.csv"

    @property
    def splits(self) -> Path:
        return self.manifests / "splits.json"

    def all_dirs(self) -> list[Path]:
        return [
            self.root,
            self.features,
            self.macro_features,
            self.micro_features,
            self.manifests,
            self.checkpoints,
            self.results,
            self.figures,
            self.logs,
        ]

    def create(self) -> list[Path]:
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self.all_dirs()
