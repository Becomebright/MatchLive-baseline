"""Load a YAML config and resolve its paths against the repo root.

The config stays a plain nested dict; only the `paths` section is rewritten into
absolute `Path`s so no module ever depends on the current working directory.

A config may start from another one: `base: default.yaml` (relative to the config's own folder)
takes every setting of that file, and the config's own settings replace them, section by section
and key by key (configs/qwen.yaml changes only the model and its endpoint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# matchlive/common/config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """The config at `path` (relative to the repo root unless absolute), with every entry of
    `paths` made absolute (relative ones are taken from the repo root)."""
    config = _read(Path(path) if Path(path).is_absolute() else REPO_ROOT / path)
    config["paths"] = {name: REPO_ROOT / value for name, value in config["paths"].items()}
    return config


def _read(path: Path) -> dict[str, Any]:
    """The YAML at `path`, merged over its `base` config if it names one."""
    with open(path) as f:
        config = yaml.safe_load(f)
    base = config.pop("base", None)
    return _merge(_read(path.parent / base), config) if base else config


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`base` with `over`'s settings in place of its own; nested sections are merged key by key."""
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out
