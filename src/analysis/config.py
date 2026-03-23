"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """Return an absolute path inside the repository."""

    return ROOT_DIR.joinpath(*parts)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {} if data is None else data

