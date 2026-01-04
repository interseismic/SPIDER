from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import polars as pl
import torch


_BUNDLE_VERSION = 2


@dataclass
class Phase2Bundle:
    """Everything needed to start Phase 2 without re-running Phase 1."""

    params: Dict[str, Any]
    origins0: pl.DataFrame
    dtimes: pl.DataFrame
    dX_src: torch.Tensor                     # (Ne,4) float32 CPU
    noise_log_scale: Optional[torch.Tensor]  # (2,) float32 CPU or None
    phase1_optimizer_state_dict: Dict[str, Any]
    global_step_count: int


def save_phase2_bundle(
    *,
    path: str,
    params: Dict[str, Any],
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    dX_src: torch.Tensor,
    noise_log_scale: Optional[torch.Tensor],
    phase1_optimizer_state_dict: Dict[str, Any],
    global_step_count: int,
) -> str:
    """
    Save a Phase-2 bundle.

    Format:
      - `path` is a small torch-saved dict (pickle) containing metadata + tensor payloads.
      - Large tables are written as sidecar parquet files next to `path`:
          - `<path>.origins0.parquet`
          - `<path>.dtimes.parquet`

    Rationale: some datasets (e.g. ridgecrest) have dtimes > 4GiB, which cannot be
    safely embedded as a single pickled byte-string without huge RAM overhead.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    origins_path = f"{path}.origins0.parquet"
    dtimes_path = f"{path}.dtimes.parquet"
    # Parquet is robust for very large tables and avoids a single gigantic pickle blob.
    # (Compression is beneficial for storage but can be CPU-heavy; zstd is a good default.)
    origins0.write_parquet(origins_path, compression="zstd")
    dtimes.write_parquet(dtimes_path, compression="zstd")

    dX_cpu = dX_src.detach().to("cpu", dtype=torch.float32).contiguous()
    nls_cpu = None
    if noise_log_scale is not None:
        nls_cpu = noise_log_scale.detach().to("cpu", dtype=torch.float32).contiguous()

    payload: Dict[str, Any] = {
        "bundle_version": int(_BUNDLE_VERSION),
        "created_unix_s": float(time.time()),
        "params": params,
        "origins0_parquet": str(origins_path),
        "dtimes_parquet": str(dtimes_path),
        "dX_src": dX_cpu,
        "noise_log_scale": nls_cpu,
        "phase1_optimizer_state_dict": phase1_optimizer_state_dict,
        "global_step_count": int(global_step_count),
    }
    torch.save(payload, path)
    return path


def load_phase2_bundle(*, path: str) -> Phase2Bundle:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Invalid bundle: expected dict payload")
    ver = int(payload.get("bundle_version", -1))
    if ver not in {1, 2}:
        raise ValueError(f"Unsupported bundle_version={payload.get('bundle_version')}; expected 1 or 2")
    params = payload.get("params", None)
    if not isinstance(params, dict):
        raise ValueError("Invalid bundle: missing/invalid params")
    if ver == 1:
        # Legacy v1 (embedded IPC bytes). Kept for backward compatibility for small datasets.
        import io

        bio0 = io.BytesIO(payload["origins0_ipc"])
        origins0 = pl.read_ipc(bio0)
        bio1 = io.BytesIO(payload["dtimes_ipc"])
        dtimes = pl.read_ipc(bio1)
    else:
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
    noise_log_scale = payload.get("noise_log_scale", None)
    if noise_log_scale is not None and not isinstance(noise_log_scale, torch.Tensor):
        raise ValueError("Invalid bundle: noise_log_scale must be a torch.Tensor or None")
    opt_state = payload.get("phase1_optimizer_state_dict", {})
    if not isinstance(opt_state, dict):
        opt_state = {}
    gsc = int(payload.get("global_step_count", 0))
    return Phase2Bundle(
        params=params,
        origins0=origins0,
        dtimes=dtimes,
        dX_src=dX_src.to(torch.float32),
        noise_log_scale=(noise_log_scale.to(torch.float32) if noise_log_scale is not None else None),
        phase1_optimizer_state_dict=opt_state,
        global_step_count=gsc,
    )


