"""Configuration loading with explicit error messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def require_section(cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = cfg.get(name)
    if not isinstance(section, dict):
        raise KeyError(f"Missing required config section '{name}'")
    return dict(section)


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge CLI overrides into a config dictionary."""
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out
