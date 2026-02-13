"""
Strict (nested-only) prior configuration schema for SPIDER.

User intent: priors must be fully specified in the JSON config; no legacy flat keys
and no implicit defaults for prior hyperparameters/modes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# Legacy keys that must NOT appear in configs once priors are nested.
# (We only forbid keys that correspond to priors / their hyperparameters / modes.)
_FORBIDDEN_TOPLEVEL_KEYS: Tuple[str, ...] = (
    # Prior enable toggles
    "prior_event_enable",
    "prior_centroid_enable",
    "prior_noise_enable",
    "prior_laplacian_enable",
    "prior_laplacian_phase4_only",
    # Event prior params
    "prior_event_std",
    "prior_centroid_std",
    # Hierarchical event prior
    "hierarchical_event_prior",
    "hierarchical_prior_dof",
    # Noise prior params/mode
    "noise_prior",
    "noise_prior_loc",
    "noise_prior_scale",
    # Laplacian prior was removed; keep legacy flat keys forbidden so old configs fail fast.
    "laplacian_prior_sigma",
    "laplacian_prior_dims",
    "laplacian_graph_mode",
    "laplacian_edge_weight_mode",
    "laplacian_pair_count_cap",
    "laplacian_edge_chunk_size",
    "laplacian_log_edge_rms",
    "laplacian_tau_gibbs_enable",
    "laplacian_hyperprior",
    "laplacian_tau_prior_a",
    "laplacian_tau_prior_b",
    "laplacian_wishart_prior_df",
    "laplacian_wishart_prior_sigma",
)


def _err(path: str, msg: str) -> ValueError:
    return ValueError(f"Invalid priors config at `{path}`: {msg}")


def _require(d: Dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise _err(path, f"missing required key `{key}`")
    return d[key]


def _require_dict(v: Any, path: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise _err(path, f"expected object/dict, got {type(v).__name__}")
    return v


def _require_bool(v: Any, path: str) -> bool:
    if not isinstance(v, bool):
        raise _err(path, f"expected boolean, got {type(v).__name__}")
    return bool(v)


def _require_str(v: Any, path: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise _err(path, "expected non-empty string")
    return str(v).strip()


def _require_pos_float(v: Any, path: str) -> float:
    if not isinstance(v, (int, float)):
        raise _err(path, f"expected number, got {type(v).__name__}")
    x = float(v)
    if not (x > 0.0):
        raise _err(path, "must be > 0")
    return x


def _require_float_list(v: Any, path: str, *, length: int) -> List[float]:
    if not isinstance(v, list):
        raise _err(path, f"expected list, got {type(v).__name__}")
    if len(v) != length:
        raise _err(path, f"expected list of length {length}, got {len(v)}")
    out: List[float] = []
    for i, xi in enumerate(v):
        if not isinstance(xi, (int, float)):
            raise _err(f"{path}[{i}]", f"expected number, got {type(xi).__name__}")
        out.append(float(xi))
    return out


def _require_pos_float_or_pos_float_list(v: Any, path: str, *, length: int) -> float | List[float]:
    """
    Accept either a single positive float or a list of positive floats of fixed length.
    Used to support scalar-vs-vector hyperparameters in a backwards compatible way.
    """
    if isinstance(v, (int, float)):
        return _require_pos_float(v, path)
    if isinstance(v, list):
        out = _require_float_list(v, path, length=length)
        for i, x in enumerate(out):
            if not (x > 0.0):
                raise _err(f"{path}[{i}]", "must be > 0")
        return out
    raise _err(path, f"expected number or list[{length}] of numbers, got {type(v).__name__}")


def _geo_mean_positive(xs: List[float], path: str) -> float:
    import math
    if not xs:
        raise _err(path, "expected non-empty list of positive floats")
    s = 0.0
    for i, x in enumerate(xs):
        if not (x > 0.0) or not math.isfinite(float(x)):
            raise _err(f"{path}[{i}]", "must be finite and > 0")
        s += math.log(float(x))
    return float(math.exp(s / float(len(xs))))


def validate_and_materialize_priors(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate nested `params['model']['priors']` and materialize the internal flat keys used by the codebase.

    This function intentionally forbids legacy top-level prior keys and requires all prior
    hyperparameters/modes to be explicitly specified (no implicit defaults).

    Returns:
        The same dict (mutated) with materialized internal keys for downstream code.
    """
    # Forbid legacy prior keys
    forbidden_present = [k for k in _FORBIDDEN_TOPLEVEL_KEYS if k in params]
    if forbidden_present:
        lap_keys = [k for k in forbidden_present if "laplacian" in str(k)]
        if lap_keys:
            raise ValueError(
                "Laplacian prior support has been removed; delete these legacy keys from your config:\n"
                + "\n".join(f"- {k}" for k in sorted(lap_keys))
            )
        raise ValueError(
            "Legacy prior keys are not allowed. Move these settings under `priors`:\n"
            + "\n".join(f"- {k}" for k in sorted(forbidden_present))
        )

    # Hard break: `priors` moved under `model.priors`.
    if "priors" in params:
        raise _err("priors", "moved; put this under `model.priors` (top-level `priors` is no longer supported)")

    model = _require_dict(_require(params, "model", "model"), "model")
    priors = _require_dict(_require(model, "priors", "model"), "model.priors")
    if "laplacian" in priors:
        raise _err("priors.laplacian", "Laplacian prior support has been removed; delete this block from your config.")

    # ---- Event prior ----
    ev = _require_dict(_require(priors, "event", "priors"), "priors.event")
    ev_enabled = _require_bool(_require(ev, "enabled", "priors.event"), "priors.event.enabled")
    ev_type = _require_str(_require(ev, "type", "priors.event"), "priors.event.type").lower()
    if ev_type not in {"gaussian"}:
        raise _err("priors.event.type", "supported types: 'gaussian'")
    ev_params = _require_dict(_require(ev, "params", "priors.event"), "priors.event.params")
    if ev_enabled:
        ev_std = _require_float_list(_require(ev_params, "std", "priors.event.params"), "priors.event.params.std", length=4)
    else:
        ev_std = None

    # Hard break: per-phase prior scheduling was removed (it was too easy to misconfigure).
    # Priors are active in all phases when enabled.
    if "schedule" in ev:
        raise _err("priors.event.schedule", "removed; priors are active in all phases when enabled (delete this block)")

    ev_hyper = _require_dict(_require(ev, "hyper", "priors.event"), "priors.event.hyper")
    ev_hyper_enabled = _require_bool(_require(ev_hyper, "enabled", "priors.event.hyper"), "priors.event.hyper.enabled")
    if ev_hyper_enabled:
        ev_hyper_type = _require_str(_require(ev_hyper, "type", "priors.event.hyper"), "priors.event.hyper.type").lower()
        if ev_hyper_type not in {"wishart_precision"}:
            raise _err("priors.event.hyper.type", "supported types: 'wishart_precision'")
        ev_hyper_params = _require_dict(_require(ev_hyper, "params", "priors.event.hyper"), "priors.event.hyper.params")
        ev_hyper_df = _require_pos_float(_require(ev_hyper_params, "df", "priors.event.hyper.params"), "priors.event.hyper.params.df")
        # Wishart(df, ·) is only well-defined for df > p-1, where p is the dimension (here p=4 for [dx,dy,dz,dt]).
        # Using a smaller df can make sampling/updates unstable or undefined.
        if not (float(ev_hyper_df) > 3.0):
            raise _err("priors.event.hyper.params.df", "must be > 3 (Wishart dof constraint for 4D event prior)")
        # Explicit scale for hyperprior (no implicit coupling to base std)
        ev_hyper_scale_std = _require_float_list(
            _require(ev_hyper_params, "scale_std", "priors.event.hyper.params"),
            "priors.event.hyper.params.scale_std",
            length=4,
        )
        # Update cadence (explicit; no default)
        ev_hyper_update = _require_dict(_require(ev_hyper, "update", "priors.event.hyper"), "priors.event.hyper.update")
        ev_hyper_every = int(_require_pos_float(_require(ev_hyper_update, "every_epochs", "priors.event.hyper.update"), "priors.event.hyper.update.every_epochs"))
        if ev_hyper_every < 1:
            raise _err("priors.event.hyper.update.every_epochs", "must be >= 1")
        # Hard break: per-phase hyper-update scheduling removed; updates run in all phases when enabled.
        if "active_phases" in ev_hyper_update:
            raise _err(
                "priors.event.hyper.update.active_phases",
                "removed; hyper-updates run in all phases when enabled (delete this key)",
            )
    else:
        ev_hyper_df = None
        ev_hyper_scale_std = None
        ev_hyper_every = None

    # ---- Centroid prior (optional) ----
    centroid = priors.get("centroid", None)
    if centroid is None:
        centroid_enabled = False
        centroid_std = None
    else:
        centroid = _require_dict(centroid, "priors.centroid")
        centroid_enabled = _require_bool(_require(centroid, "enabled", "priors.centroid"), "priors.centroid.enabled")
        centroid_type = _require_str(_require(centroid, "type", "priors.centroid"), "priors.centroid.type").lower()
        if centroid_type not in {"gaussian"}:
            raise _err("priors.centroid.type", "supported types: 'gaussian'")
        centroid_params = _require_dict(_require(centroid, "params", "priors.centroid"), "priors.centroid.params")
        if centroid_enabled:
            centroid_std = _require_float_list(
                _require(centroid_params, "std", "priors.centroid.params"),
                "priors.centroid.params.std",
                length=4,
            )
        else:
            centroid_std = None

    # Noise prior removed (start fresh): fixed phase_unc only, no σ learning.
    if "noise" in priors:
        raise _err("priors.noise", "removed; delete this block from your config")

    # ---- Materialize internal flat keys (used elsewhere in the codebase) ----
    # Enables
    params["prior_event_enable"] = ev_enabled

    # Event prior
    if ev_enabled:
        params["prior_event_std"] = ev_std
    # Hierarchical event prior (explicit)
    params["hierarchical_event_prior"] = bool(ev_hyper_enabled)
    if ev_hyper_enabled:
        params["hierarchical_prior_dof"] = float(ev_hyper_df)
        params["_hierarchical_scale_std"] = ev_hyper_scale_std
        params["_hierarchical_update_every_epochs"] = int(ev_hyper_every)

    # Centroid prior (optional)
    params["prior_centroid_enable"] = bool(centroid_enabled)
    if centroid_std is not None:
        params["prior_centroid_std"] = centroid_std

    return params


