from __future__ import annotations

from typing import Any

import torch


def _model_cfg(params: dict[str, Any]) -> dict[str, Any]:
    model = params.get("model", None)
    if not isinstance(model, dict):
        return {}
    eikonet = model.get("eikonet", None)
    if not isinstance(eikonet, dict):
        return {}
    return eikonet


def load_eikonet_model(params: dict[str, Any], device: str | int | torch.device) -> torch.nn.Module:
    """
    Load an EikoNet model strictly through the external `eikonet` package.

    SPIDER no longer supports the legacy in-repo EikoNet implementation.
    """
    try:
        from eikonet.io import load_model
    except Exception as e:
        raise RuntimeError(
            "Missing required dependency `eikonet`. "
            "Install it first (for local development: `pip install -e /path/to/eikonet`)."
        ) from e

    model_file = str(params["model_file"])
    cfg = _model_cfg(params)

    x_max = cfg.get("x_max", None)
    y_max = cfg.get("y_max", None)
    scale = params.get("scale", None)
    if x_max is None and scale is not None:
        x_max = scale
    if y_max is None and scale is not None:
        y_max = scale
    if x_max is None or y_max is None:
        raise RuntimeError(
            "EikoNet state_dict checkpoints require x_max/y_max. "
            "Provide model.eikonet.x_max/y_max (preferred), or model.domain.scale as a fallback."
        )

    model_params: dict[str, Any] = {
        "x_max": float(x_max),
        "y_max": float(y_max),
        "z_min": float(params["z_min"]),
        "z_max": float(params["z_max"]),
        "n_hidden": int(cfg.get("n_hidden", 128)),
        "n_blocks": int(cfg.get("n_blocks", 5)),
        "n_fourier": int(cfg.get("n_fourier", 4)),
        "use_fourier": bool(cfg.get("use_fourier", True)),
        "phase_emb_dim": int(cfg.get("phase_emb_dim", 8)),
        "t2_log_scale": float(cfg.get("t2_log_scale", 0.1)),
        "vp": float(cfg.get("vp", 6.0)),
        "vs": float(cfg.get("vs", 3.2)),
    }

    model_kind_raw = cfg.get("model_kind", "1d")
    model_kind = str(model_kind_raw).strip().lower() if model_kind_raw is not None else None
    if model_kind not in {None, "1d", "3d"}:
        raise ValueError("model.eikonet.model_kind must be one of: '1d', '3d'")

    return load_model(
        model_path=model_file,
        params=model_params,
        device=device,
        model_kind=model_kind,
    )

