"""
Public load entrypoints for config_v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .resolve import resolve_config
from .types import ResolvedConfig
from .validate import validate_config


def load_config(raw: Mapping[str, object], mode: str | None = None) -> ResolvedConfig:
    canonical = validate_config(raw, mode=mode)
    return resolve_config(canonical, mode=mode)


def load_config_file(path: str | Path, mode: str | None = None) -> ResolvedConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level JSON object in {p}, got {type(raw).__name__}")
    return load_config(raw, mode=mode)

