"""Lightweight helpers for W&B diagnostics gating and safe metric emission.

These are intentionally dependency-free so they can be imported from both
`spider.core.locate` and `spider.core.epoch_runner` without creating cycles.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping
import math


def want_wandb_group(params: Mapping[str, Any], group: str) -> bool:
    """
    True iff W&B is active at runtime AND this metric group is enabled.

    Runtime flag:
      - params["_wandb_runtime_enabled"] set in `spider/cli.py` after calling init_wandb_if_enabled().

    Config-controlled categories:
      - params["_wandb_diag_enabled"] materialized from `inference.diagnostics.wandb.enabled` (default true)
      - params["_wandb_diag_groups"] materialized from `inference.diagnostics.wandb.groups`
        Default is {"all"} for backward compatibility.
    """
    try:
        if not bool(params.get("_wandb_runtime_enabled", False)):
            return False
        if not bool(params.get("_wandb_diag_enabled", True)):
            return False
        groups = params.get("_wandb_diag_groups", None)
        if groups is None:
            return True
        gset = {str(x).strip().lower() for x in groups if str(x).strip()}
        if ("all" in gset) or ("*" in gset):
            return True
        return str(group).strip().lower() in gset
    except Exception:
        return False


def wb_add_if_finite(d: MutableMapping[str, float], key: str, val: Any) -> None:
    """Add a scalar to wandb metrics only if it is a finite float (prevents empty NaN-only plots)."""
    try:
        fv = float(val)
        if math.isfinite(fv):
            d[key] = fv
    except Exception:
        return


