"""Console and JSON-lines logging for training runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logging(output_dir: str | Path, name: str = "mres_fer") -> logging.Logger:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(directory / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class JsonlWriter:
    """Append one JSON object per line, for later plotting."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
