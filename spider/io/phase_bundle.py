from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import polars as pl
import torch

_BUNDLE_VERSION = 1


@dataclass
class Phase2Bundle:
    params: Optional[Dict[str, Any]]
    origins0: pl.DataFrame
    dtimes: pl.DataFrame
    dX_src: torch.Tensor


def save_phase2_bundle(
    *,
    path: str,
    params: Optional[Dict[str, Any]],
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    dX_src: torch.Tensor,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    origins_path = f"{path}.origins0.parquet"
    dtimes_path = f"{path}.dtimes.parquet"
    origins0.write_parquet(origins_path, compression="zstd")
    dtimes.write_parquet(dtimes_path, compression="zstd")
    payload: Dict[str, Any] = {
        "bundle_version": int(_BUNDLE_VERSION),
        "created_unix_s": float(time.time()),
        "origins0_parquet": str(origins_path),
        "dtimes_parquet": str(dtimes_path),
        "dX_src": dX_src.detach().to("cpu", dtype=torch.float32).contiguous(),
        "params": params if isinstance(params, dict) else None,
    }
    torch.save(payload, path)
    return path


def load_phase2_bundle(*, path: str) -> Phase2Bundle:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Invalid bundle: expected dict payload")
    ver = int(payload.get("bundle_version", -1))
    if ver != _BUNDLE_VERSION:
        raise ValueError(f"Unsupported bundle_version={payload.get('bundle_version')}; expected {_BUNDLE_VERSION}")
    origins_path = str(payload.get("origins0_parquet", ""))
    dtimes_path = str(payload.get("dtimes_parquet", ""))
    if not origins_path or not os.path.exists(origins_path):
        raise FileNotFoundError(f"Bundle missing origins0 parquet: {origins_path}")
    if not dtimes_path or not os.path.exists(dtimes_path):
        raise FileNotFoundError(f"Bundle missing dtimes parquet: {dtimes_path}")
    origins0 = pl.read_parquet(origins_path)
    dtimes = pl.read_parquet(dtimes_path)
    dX_src = payload["dX_src"]
    if not isinstance(dX_src, torch.Tensor):
        raise ValueError("Invalid bundle: dX_src must be a torch.Tensor")
    params = payload.get("params", None)
    if params is not None and not isinstance(params, dict):
        params = None
    return Phase2Bundle(params=params, origins0=origins0, dtimes=dtimes, dX_src=dX_src.to(torch.float32))
