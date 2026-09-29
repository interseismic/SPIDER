"""
Validation for the canonical config_v2 schema.

This module intentionally does not implement compatibility shims.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

from .errors import ConfigIssue, single_issue_error
from .types import (
    BatchingConfig,
    CanonicalConfig,
    ComputeConfig,
    DeviceSpecifier,
    DomainConfig,
    InferenceConfig,
    IOConfig,
    ModelConfig,
    ObservabilityConfig,
    RuntimeConfig,
    SafetyConfig,
    SamplerConfig,
    WandbConfig,
)


_TOPLEVEL_KEYS = {"io", "model", "inference", "observability", "synth"}
_SAMPLER_BACKENDS = {"psgld", "sghmc"}
# Only these solver names enter the correlated (whitened) likelihood path in
# spider/core/modeling.py compute_likelihood_loss; anything else would silently
# degrade to the independent Gaussian, so validation rejects other values.
_SHARED_EVENT_SOLVER_KINDS = ("pcg", "pcg_sparse", "pcg-sparse")


def _as_dict(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise single_issue_error(path=path, code="type_error", message=f"Expected object/dict, got {type(value).__name__}")
    return value


def _reject_unknown_keys(path: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(k for k in data.keys() if k not in allowed)
    if unknown:
        raise single_issue_error(
            path=path,
            code="unknown_key",
            message=f"Unknown key(s): {', '.join(unknown)}",
            suggestion="Remove unknown keys or add them to config_v2 schema.",
        )


def _require_str(data: Mapping[str, Any], key: str, path: str) -> str:
    if key not in data:
        raise single_issue_error(path=path, code="missing_key", message=f"Missing required key `{key}`")
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise single_issue_error(path=f"{path}.{key}", code="type_error", message="Expected non-empty string")
    return value.strip()


def _require_bool(data: Mapping[str, Any], key: str, path: str) -> bool:
    if key not in data:
        raise single_issue_error(path=path, code="missing_key", message=f"Missing required key `{key}`")
    value = data[key]
    if not isinstance(value, bool):
        raise single_issue_error(path=f"{path}.{key}", code="type_error", message=f"Expected boolean, got {type(value).__name__}")
    return bool(value)


def _require_num(data: Mapping[str, Any], key: str, path: str) -> float:
    if key not in data:
        raise single_issue_error(path=path, code="missing_key", message=f"Missing required key `{key}`")
    value = data[key]
    if not isinstance(value, (int, float)):
        raise single_issue_error(path=f"{path}.{key}", code="type_error", message=f"Expected number, got {type(value).__name__}")
    num = float(value)
    if not math.isfinite(num):
        raise single_issue_error(path=f"{path}.{key}", code="value_error", message="Expected finite number")
    return num


def _optional_num(data: Mapping[str, Any], key: str, path: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, (int, float)):
        raise single_issue_error(path=f"{path}.{key}", code="type_error", message=f"Expected number, got {type(value).__name__}")
    num = float(value)
    if not math.isfinite(num):
        raise single_issue_error(path=f"{path}.{key}", code="value_error", message="Expected finite number")
    return num


def _require_int_list_len(data: Mapping[str, Any], key: str, path: str, length: int) -> list[int]:
    if key not in data:
        raise single_issue_error(path=path, code="missing_key", message=f"Missing required key `{key}`")
    value = data[key]
    if not isinstance(value, list):
        raise single_issue_error(path=f"{path}.{key}", code="type_error", message=f"Expected list, got {type(value).__name__}")
    if len(value) != length:
        raise single_issue_error(path=f"{path}.{key}", code="value_error", message=f"Expected list length {length}, got {len(value)}")
    out: list[int] = []
    for i, item in enumerate(value):
        if not isinstance(item, int):
            raise single_issue_error(path=f"{path}.{key}[{i}]", code="type_error", message=f"Expected integer, got {type(item).__name__}")
        out.append(item)
    return out


def _require_positive_num_list_len(data: Mapping[str, Any], key: str, path: str, length: int) -> list[float]:
    if key not in data:
        raise single_issue_error(path=path, code="missing_key", message=f"Missing required key `{key}`")
    value = data[key]
    if not isinstance(value, list):
        raise single_issue_error(path=f"{path}.{key}", code="type_error", message=f"Expected list, got {type(value).__name__}")
    if len(value) != length:
        raise single_issue_error(path=f"{path}.{key}", code="value_error", message=f"Expected list length {length}, got {len(value)}")
    out: list[float] = []
    for i, item in enumerate(value):
        if not isinstance(item, (int, float)):
            raise single_issue_error(path=f"{path}.{key}[{i}]", code="type_error", message=f"Expected number, got {type(item).__name__}")
        fv = float(item)
        if not math.isfinite(fv) or fv <= 0.0:
            raise single_issue_error(path=f"{path}.{key}[{i}]", code="value_error", message="Expected finite value > 0")
        out.append(fv)
    return out


def _parse_devices(data: Mapping[str, Any], path: str) -> list[DeviceSpecifier]:
    if "devices" not in data:
        return []
    value = data["devices"]
    if not isinstance(value, list):
        raise single_issue_error(path=f"{path}.devices", code="type_error", message=f"Expected list, got {type(value).__name__}")
    parsed: list[DeviceSpecifier] = []
    for i, dev in enumerate(value):
        if isinstance(dev, int):
            parsed.append(dev)
            continue
        if isinstance(dev, str) and dev.strip():
            parsed.append(dev.strip())
            continue
        raise single_issue_error(path=f"{path}.devices[{i}]", code="type_error", message="Expected int or non-empty string")
    return parsed


def _parse_io(raw: Mapping[str, Any]) -> IOConfig:
    io_path = "io"
    _reject_unknown_keys(
        io_path,
        raw,
        {
            "dtime_file",
            "station_file",
            "catalog_infile",
            "catalog_outfile",
            "samples_outfile",
            "checkpoint_dir",
            "checkpoint_interval",
            "sample_write_interval",
            "save_every_n",
            "write_samples",
        },
    )
    checkpoint_interval = int(_require_num(raw, "checkpoint_interval", io_path))
    if checkpoint_interval < 1:
        raise single_issue_error(path="io.checkpoint_interval", code="value_error", message="Must be >= 1")
    save_every_n = int(_require_num(raw, "save_every_n", io_path))
    if save_every_n < 1:
        raise single_issue_error(path="io.save_every_n", code="value_error", message="Must be >= 1")
    sample_write_interval_num = _optional_num(raw, "sample_write_interval", io_path)
    sample_write_interval = None if sample_write_interval_num is None else int(sample_write_interval_num)
    if sample_write_interval is not None and sample_write_interval < 0:
        raise single_issue_error(path="io.sample_write_interval", code="value_error", message="Must be >= 0")
    return IOConfig(
        dtime_file=_require_str(raw, "dtime_file", io_path),
        station_file=_require_str(raw, "station_file", io_path),
        catalog_infile=_require_str(raw, "catalog_infile", io_path),
        catalog_outfile=_require_str(raw, "catalog_outfile", io_path),
        samples_outfile=_require_str(raw, "samples_outfile", io_path),
        checkpoint_dir=_require_str(raw, "checkpoint_dir", io_path),
        checkpoint_interval=checkpoint_interval,
        save_every_n=save_every_n,
        write_samples=_require_bool(raw, "write_samples", io_path),
        sample_write_interval=sample_write_interval,
    )


def _parse_domain(raw: Mapping[str, Any]) -> DomainConfig:
    path = "model.domain"
    _reject_unknown_keys(path, raw, {"lon_min", "lat_min", "z_min", "z_max", "scale"})
    z_min = _require_num(raw, "z_min", path)
    z_max = _require_num(raw, "z_max", path)
    if z_max <= z_min:
        raise single_issue_error(path=path, code="value_error", message="Expected z_max > z_min")
    scale = _require_num(raw, "scale", path)
    if scale <= 0:
        raise single_issue_error(path="model.domain.scale", code="value_error", message="Must be > 0")
    return DomainConfig(
        lon_min=_require_num(raw, "lon_min", path),
        lat_min=_require_num(raw, "lat_min", path),
        z_min=z_min,
        z_max=z_max,
        scale=scale,
    )


def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
    path = "model"
    _reject_unknown_keys(path, raw, {"model_file", "domain", "priors", "likelihoods", "filters", "eikonet"})
    domain_raw = _as_dict(raw.get("domain"), "model.domain")
    priors = raw.get("priors", {})
    likelihoods = raw.get("likelihoods", {})
    filters = raw.get("filters", {})
    eikonet = raw.get("eikonet", {})
    for name, block in (
        ("model.priors", priors),
        ("model.likelihoods", likelihoods),
        ("model.filters", filters),
        ("model.eikonet", eikonet),
    ):
        if not isinstance(block, dict):
            raise single_issue_error(path=name, code="type_error", message=f"Expected object/dict, got {type(block).__name__}")
    _validate_shared_event_re_shape(likelihoods)
    return ModelConfig(
        model_file=_require_str(raw, "model_file", path),
        domain=_parse_domain(domain_raw),
        priors=dict(priors),
        likelihoods=dict(likelihoods),
        filters=dict(filters),
        eikonet=dict(eikonet),
    )


def _validate_shared_event_re_shape(likelihoods: Mapping[str, Any]) -> None:
    sample = likelihoods.get("sample", None)
    if not isinstance(sample, dict):
        return
    se = sample.get("shared_event_re", None)
    if not isinstance(se, dict):
        return

    legacy_top = {
        "grouping",
        "cluster_mode",
        "cluster_k",
        "tau_s",
        "max_nodes_per_group",
        "max_rows_per_group",
        "fallback_to_diag",
        "abort_on_pcg_fallback",
        "jitter0",
        "jitter_max",
        "station_phase_re",
    }
    found_top = sorted([k for k in legacy_top if k in se])
    if found_top:
        raise single_issue_error(
            path="model.likelihoods.sample.shared_event_re",
            code="legacy_key",
            message=(
                "Legacy keys not allowed: "
                + ", ".join(found_top)
                + ". Use sub-blocks: model, limits, fallback, numerics, station_phase_term."
            ),
        )

    allowed_top = {
        "enabled",
        "model",
        "limits",
        "fallback",
        "numerics",
        "station_phase_term",
        "solver",
        "edge_weights",
        "autotune",
        "logging",
    }
    _reject_unknown_keys("model.likelihoods.sample.shared_event_re", se, allowed_top)

    solver = se.get("solver", {})
    if isinstance(solver, dict):
        if "kind" in solver:
            kind = solver.get("kind")
            kind_norm = str(kind).strip().lower() if isinstance(kind, str) else None
            if kind_norm not in _SHARED_EVENT_SOLVER_KINDS:
                raise single_issue_error(
                    path="model.likelihoods.sample.shared_event_re.solver.kind",
                    code="enum_error",
                    message=(
                        f"Unsupported solver kind {kind!r}. Accepted values: "
                        + ", ".join(_SHARED_EVENT_SOLVER_KINDS)
                        + ". Any other name would silently disable the correlated likelihood "
                        "(the runtime falls back to the independent Gaussian without a warning)."
                    ),
                    suggestion="Use solver.kind = \"pcg\" (pcg_sparse / pcg-sparse are aliases).",
                )
        legacy_solver = {"bucket_nodes", "merge_sparse_edge_bins", "min_groups_per_edge_bin", "max_edge_bins_per_node"}
        found_solver = sorted([k for k in legacy_solver if k in solver])
        if found_solver:
            raise single_issue_error(
                path="model.likelihoods.sample.shared_event_re.solver",
                code="legacy_key",
                message=(
                    "Legacy keys not allowed: "
                    + ", ".join(found_solver)
                    + ". Use node_bin_edges, merge_sparse_node_bins, min_groups_per_node_bin, max_node_bins_per_node."
                ),
            )

    autotune = se.get("autotune", {})
    if isinstance(autotune, dict):
        legacy_at = {"max_bins", "min_bin_groups", "min_bucket_node"}
        found_at = sorted([k for k in legacy_at if k in autotune])
        if found_at:
            raise single_issue_error(
                path="model.likelihoods.sample.shared_event_re.autotune",
                code="legacy_key",
                message=(
                    "Legacy keys not allowed: "
                    + ", ".join(found_at)
                    + ". Use max_node_bins, min_groups_per_node_bin, min_node_bin."
                ),
            )


def _parse_sampler(raw: Mapping[str, Any]) -> SamplerConfig:
    path = "inference.sampler"
    _reject_unknown_keys(
        path,
        raw,
        {
            "backend",
            "epochs_per_phase",
            "lr",
            "temperature",
            "beta",
            "eps",
            "freeze_preconditioner_sampling",
            "sghmc_alpha",
            "noise_scale_mult",
            "grad_clip_norm",
            "preconditioning",
            "reparameterization",
            "overrides",
            "lr_schedule",
            "phase1_two_pass",
            "weak_event_mh",
        },
    )
    wm = raw.get("weak_event_mh", {})
    if not isinstance(wm, Mapping):
        raise single_issue_error(path=f"{path}.weak_event_mh", code="type_error", message="Expected object/dict")
    _reject_unknown_keys(
        f"{path}.weak_event_mh", wm,
        {"enabled", "every_epochs", "start_epoch", "max_dtimes", "proposal", "gn_iters", "t_dof", "scale_inflation"},
    )
    if int(wm.get("max_dtimes", 50)) < 1:
        raise single_issue_error(path=f"{path}.weak_event_mh.max_dtimes", code="value_error",
                                 message="Must be >= 1")
    _wm_prop = str(wm.get("proposal", "conditional")).strip().lower()
    if _wm_prop not in {"conditional", "prior"}:
        raise single_issue_error(path=f"{path}.weak_event_mh.proposal", code="enum_error",
                                 message="Supported values are: conditional (default), prior")
    _wm_gn = wm.get("gn_iters", 4)
    if isinstance(_wm_gn, bool) or not isinstance(_wm_gn, int) or int(_wm_gn) < 1:
        raise single_issue_error(path=f"{path}.weak_event_mh.gn_iters", code="value_error", message="Must be an integer >= 1")
    _wm_dof = wm.get("t_dof", 5.0)
    if _wm_dof is not None and (isinstance(_wm_dof, bool) or not isinstance(_wm_dof, (int, float)) or float(_wm_dof) <= 2.0):
        raise single_issue_error(path=f"{path}.weak_event_mh.t_dof", code="value_error",
                                 message="Must be a number > 2 (Student-t degrees of freedom) or null for a Gaussian proposal")
    _wm_sc = wm.get("scale_inflation", 1.2)
    if isinstance(_wm_sc, bool) or not isinstance(_wm_sc, (int, float)) or float(_wm_sc) <= 0.0:
        raise single_issue_error(path=f"{path}.weak_event_mh.scale_inflation", code="value_error", message="Must be > 0")
    backend = _require_str(raw, "backend", path).lower()
    if backend not in _SAMPLER_BACKENDS:
        raise single_issue_error(
            path="inference.sampler.backend",
            code="enum_error",
            message="Supported values are: psgld, sghmc",
        )
    epochs = _require_int_list_len(raw, "epochs_per_phase", path, length=4)
    for i, v in enumerate(epochs):
        if v < 0:
            raise single_issue_error(path=f"inference.sampler.epochs_per_phase[{i}]", code="value_error", message="Must be >= 0")
    lr = _require_positive_num_list_len(raw, "lr", path, length=4)
    temperature = _require_num(raw, "temperature", path)
    if temperature < 0:
        raise single_issue_error(path="inference.sampler.temperature", code="value_error", message="Must be >= 0")
    beta = _require_num(raw, "beta", path)
    if not (0.0 <= beta < 1.0):
        raise single_issue_error(path="inference.sampler.beta", code="value_error", message="Must satisfy 0 <= beta < 1")
    eps = _require_num(raw, "eps", path)
    if eps <= 0.0:
        raise single_issue_error(path="inference.sampler.eps", code="value_error", message="Must be > 0")
    freeze_preconditioner_sampling = _require_bool(raw, "freeze_preconditioner_sampling", path)

    if "sghmc_alpha" not in raw:
        raise single_issue_error(path=path, code="missing_key", message="Missing required key `sghmc_alpha`")
    sghmc_alpha_raw = raw.get("sghmc_alpha", None)
    if backend == "sghmc":
        if not isinstance(sghmc_alpha_raw, (int, float)):
            raise single_issue_error(path="inference.sampler.sghmc_alpha", code="type_error", message="Expected number for SGHMC backend")
        sghmc_alpha = float(sghmc_alpha_raw)
        if (not math.isfinite(sghmc_alpha)) or (sghmc_alpha <= 0.0):
            raise single_issue_error(path="inference.sampler.sghmc_alpha", code="value_error", message="Must be finite and > 0")
    else:
        if sghmc_alpha_raw is None:
            sghmc_alpha = None
        elif isinstance(sghmc_alpha_raw, (int, float)):
            sghmc_alpha = float(sghmc_alpha_raw)
        else:
            raise single_issue_error(path="inference.sampler.sghmc_alpha", code="type_error", message="Expected number or null")

    noise_scale_mult = _optional_num(raw, "noise_scale_mult", path)
    if noise_scale_mult is not None and noise_scale_mult <= 0.0:
        raise single_issue_error(path="inference.sampler.noise_scale_mult", code="value_error", message="Must be > 0")
    grad_clip_norm = _optional_num(raw, "grad_clip_norm", path)
    if grad_clip_norm is not None and grad_clip_norm < 0.0:
        raise single_issue_error(path="inference.sampler.grad_clip_norm", code="value_error", message="Must be >= 0")

    preconditioning = raw.get("preconditioning", {})
    reparameterization = raw.get("reparameterization", {})
    overrides = raw.get("overrides", {})
    if not isinstance(preconditioning, dict):
        raise single_issue_error(path="inference.sampler.preconditioning", code="type_error", message="Expected object/dict")
    _reject_unknown_keys(
        "inference.sampler.preconditioning",
        preconditioning,
        {"enabled", "type", "include_gamma"},
    )
    precond_enabled = bool(preconditioning.get("enabled", False))
    precond_type = str(preconditioning.get("type", "none")).strip().lower()
    if precond_enabled and precond_type != "rmsprop":
        raise single_issue_error(
            path="inference.sampler.preconditioning.type",
            code="enum_error",
            message="Supported value is: rmsprop (or disable preconditioning). The lrd, component_lrd and block_gn "
                    "preconditioners were removed.",
        )
    if not isinstance(overrides, dict):
        raise single_issue_error(path="inference.sampler.overrides", code="type_error", message="Expected object/dict")
    if not isinstance(reparameterization, dict):
        raise single_issue_error(path="inference.sampler.reparameterization", code="type_error", message="Expected object/dict")
    _reject_unknown_keys(
        "inference.sampler.reparameterization",
        reparameterization,
        {"enabled", "spatial_scale", "dt_scale"},
    )
    reparam_enabled = bool(reparameterization.get("enabled", False))
    if "spatial_scale" in reparameterization:
        spatial_scale = _optional_num(reparameterization, "spatial_scale", "inference.sampler.reparameterization")
        if spatial_scale is not None and spatial_scale <= 0.0:
            raise single_issue_error(
                path="inference.sampler.reparameterization.spatial_scale",
                code="value_error",
                message="Must be > 0",
            )
    if "dt_scale" in reparameterization:
        dt_scale = _optional_num(reparameterization, "dt_scale", "inference.sampler.reparameterization")
        if dt_scale is not None and dt_scale <= 0.0:
            raise single_issue_error(
                path="inference.sampler.reparameterization.dt_scale",
                code="value_error",
                message="Must be > 0",
            )
    # Optional Phase-1 (MAP) learning-rate schedule.
    lr_schedule = raw.get("lr_schedule", {})
    if lr_schedule is None:
        lr_schedule = {}
    if not isinstance(lr_schedule, dict):
        raise single_issue_error(path="inference.sampler.lr_schedule", code="type_error", message="Expected object/dict")
    _reject_unknown_keys("inference.sampler.lr_schedule", lr_schedule, {"phase1"})
    p1_sched = lr_schedule.get("phase1", {})
    if p1_sched is None:
        p1_sched = {}
    if not isinstance(p1_sched, dict):
        raise single_issue_error(path="inference.sampler.lr_schedule.phase1", code="type_error", message="Expected object/dict")
    _reject_unknown_keys("inference.sampler.lr_schedule.phase1", p1_sched, {"type", "eta_min", "T_max"})
    p1_sched_type = str(p1_sched.get("type", "none")).strip().lower()
    if p1_sched_type not in {"none", "cosine"}:
        raise single_issue_error(
            path="inference.sampler.lr_schedule.phase1.type",
            code="enum_error",
            message="Supported values are: none, cosine",
        )
    if "eta_min" in p1_sched:
        eta_min = _optional_num(p1_sched, "eta_min", "inference.sampler.lr_schedule.phase1")
        if eta_min is not None and eta_min < 0.0:
            raise single_issue_error(
                path="inference.sampler.lr_schedule.phase1.eta_min", code="value_error", message="Must be >= 0"
            )
    t_max = p1_sched.get("T_max", None)
    if t_max is not None:
        if isinstance(t_max, bool) or not isinstance(t_max, int) or t_max < 1:
            raise single_issue_error(
                path="inference.sampler.lr_schedule.phase1.T_max",
                code="value_error",
                message="Must be null or an integer >= 1",
            )

    # Optional Phase-1 two-pass MAP: MAP -> post-MAP filters at relocated positions -> MAP again.
    two_pass = raw.get("phase1_two_pass", {})
    if two_pass is None:
        two_pass = {}
    if not isinstance(two_pass, dict):
        raise single_issue_error(path="inference.sampler.phase1_two_pass", code="type_error", message="Expected object/dict")
    _reject_unknown_keys("inference.sampler.phase1_two_pass", two_pass, {"enabled", "epochs", "warm_start"})
    if "enabled" in two_pass and not isinstance(two_pass.get("enabled"), bool):
        raise single_issue_error(path="inference.sampler.phase1_two_pass.enabled", code="type_error", message="Expected bool")
    if "warm_start" in two_pass and not isinstance(two_pass.get("warm_start"), bool):
        raise single_issue_error(path="inference.sampler.phase1_two_pass.warm_start", code="type_error", message="Expected bool")
    tp_epochs = two_pass.get("epochs", None)
    if tp_epochs is not None:
        if isinstance(tp_epochs, bool) or not isinstance(tp_epochs, int) or tp_epochs < 1:
            raise single_issue_error(
                path="inference.sampler.phase1_two_pass.epochs",
                code="value_error",
                message="Must be null (defaults to epochs_per_phase[0]) or an integer >= 1",
            )

    return SamplerConfig(
        backend=backend,
        epochs_per_phase=epochs,
        lr=lr,
        temperature=temperature,
        beta=beta,
        eps=eps,
        freeze_preconditioner_sampling=freeze_preconditioner_sampling,
        sghmc_alpha=sghmc_alpha,
        noise_scale_mult=noise_scale_mult,
        grad_clip_norm=grad_clip_norm,
        preconditioning=dict(preconditioning),
        overrides=dict(overrides),
        extras={
            "reparameterization": dict(reparameterization),
            "reparameterization_enabled": bool(reparam_enabled),
            "lr_schedule": {"phase1": dict(p1_sched)},
            "phase1_two_pass": dict(two_pass),
            "weak_event_mh": dict(wm),
        },
    )


def _parse_batching(raw: Mapping[str, Any]) -> BatchingConfig:
    path = "inference.batching"
    _reject_unknown_keys(path, raw, {"standard", "event_batches"})
    standard = raw.get("standard", {})
    event_batches = raw.get("event_batches", {})
    if not isinstance(standard, dict):
        raise single_issue_error(path="inference.batching.standard", code="type_error", message="Expected object/dict")
    if not isinstance(event_batches, dict):
        raise single_issue_error(path="inference.batching.event_batches", code="type_error", message="Expected object/dict")
    return BatchingConfig(standard=dict(standard), event_batches=dict(event_batches))


def _parse_runtime(raw: Mapping[str, Any]) -> RuntimeConfig:
    path = "inference.runtime"
    _reject_unknown_keys(
        path,
        raw,
        {
            "cuda_empty_cache_every",
            "reset_batch_numbers",
            "clear_samples_on_reset",
            "min_samples_to_save",
            "verbose",
            "cluster_events",
            "seed",
            "gauge_projection",
            "torch",
        },
    )
    cuda_empty_cache_every = int(_require_num(raw, "cuda_empty_cache_every", path))
    if cuda_empty_cache_every < 0:
        raise single_issue_error(path="inference.runtime.cuda_empty_cache_every", code="value_error", message="Must be >= 0")
    reset_batch_numbers = _require_bool(raw, "reset_batch_numbers", path)
    clear_samples_on_reset = _require_bool(raw, "clear_samples_on_reset", path)
    # `min_samples_to_save` and `cluster_events` are accepted for backward compatibility only:
    # nothing in the runtime consumes them. They are optional and type-checked when present.
    min_samples_to_save_num = _optional_num(raw, "min_samples_to_save", path)
    min_samples_to_save = 0 if min_samples_to_save_num is None else int(min_samples_to_save_num)
    if min_samples_to_save < 0:
        raise single_issue_error(path="inference.runtime.min_samples_to_save", code="value_error", message="Must be >= 0")
    verbose = _require_bool(raw, "verbose", path)
    cluster_events = False
    if raw.get("cluster_events", None) is not None:
        cluster_events = _require_bool(raw, "cluster_events", path)
    seed_v = _optional_num(raw, "seed", path)
    seed = None if seed_v is None else int(seed_v)
    gauge_projection = raw.get("gauge_projection", {})
    if not isinstance(gauge_projection, dict):
        raise single_issue_error(path="inference.runtime.gauge_projection", code="type_error", message="Expected object/dict")
    torch_cfg = raw.get("torch", {})
    if not isinstance(torch_cfg, dict):
        raise single_issue_error(path="inference.runtime.torch", code="type_error", message="Expected object/dict")
    return RuntimeConfig(
        cuda_empty_cache_every=cuda_empty_cache_every,
        reset_batch_numbers=reset_batch_numbers,
        clear_samples_on_reset=clear_samples_on_reset,
        min_samples_to_save=min_samples_to_save,
        verbose=verbose,
        cluster_events=cluster_events,
        seed=seed,
        gauge_projection=dict(gauge_projection),
        torch=dict(torch_cfg),
        extras={},
    )


def _parse_shift_guard(raw: Any) -> Dict[str, Any]:
    """
    `inference.safety.shift_guard`: {enabled: bool (default false), factor: number > 0 (default 5.0)}.

    Materialized by the legacy bridge to `shift_guard_enable` / `shift_guard_factor`, which
    spider/core/locate.py `_shift_guard_check` reads. The CLI flags `--shift-guard` /
    `--shift-guard-factor` override these after materialization.
    """
    path = "inference.safety.shift_guard"
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise single_issue_error(path=path, code="type_error", message=f"Expected object/dict, got {type(raw).__name__}")
    _reject_unknown_keys(path, raw, {"enabled", "factor"})
    enabled = False
    if raw.get("enabled", None) is not None:
        enabled = _require_bool(raw, "enabled", path)
    factor_num = _optional_num(raw, "factor", path)
    factor = 5.0 if factor_num is None else float(factor_num)
    if factor <= 0.0:
        raise single_issue_error(path=f"{path}.factor", code="value_error", message="Must be > 0")
    return {"enabled": bool(enabled), "factor": float(factor)}


def _parse_safety(raw: Mapping[str, Any]) -> SafetyConfig:
    path = "inference.safety"
    _reject_unknown_keys(path, raw, {"max_abs_dX", "shift_guard"})
    extras: Dict[str, Any] = {"shift_guard": _parse_shift_guard(raw.get("shift_guard", None))}
    max_abs = raw.get("max_abs_dX", None)
    if max_abs is None:
        return SafetyConfig(max_abs_dX=None, extras=extras)
    if not isinstance(max_abs, list) or len(max_abs) != 4:
        raise single_issue_error(path="inference.safety.max_abs_dX", code="type_error", message="Expected list[4] of numbers or null")
    out: list[float] = []
    for i, item in enumerate(max_abs):
        if not isinstance(item, (int, float)):
            raise single_issue_error(path=f"inference.safety.max_abs_dX[{i}]", code="type_error", message="Expected number")
        out.append(float(item))
    return SafetyConfig(max_abs_dX=out, extras=extras)


def _parse_inference(raw: Mapping[str, Any]) -> InferenceConfig:
    path = "inference"
    _reject_unknown_keys(path, raw, {"sampler", "batching", "runtime", "safety", "compute"})
    sampler = _parse_sampler(_as_dict(raw.get("sampler"), "inference.sampler"))
    batching = _parse_batching(_as_dict(raw.get("batching"), "inference.batching"))
    runtime = _parse_runtime(_as_dict(raw.get("runtime"), "inference.runtime"))
    safety = _parse_safety(_as_dict(raw.get("safety"), "inference.safety"))
    compute_raw = raw.get("compute", {})
    if not isinstance(compute_raw, dict):
        raise single_issue_error(path="inference.compute", code="type_error", message="Expected object/dict")
    compute = ComputeConfig(devices=_parse_devices(compute_raw, "inference.compute"))
    return InferenceConfig(
        sampler=sampler,
        batching=batching,
        runtime=runtime,
        safety=safety,
        compute=compute,
    )


def _parse_observability(raw: Mapping[str, Any]) -> ObservabilityConfig:
    path = "observability"
    _reject_unknown_keys(path, raw, {"wandb", "diagnostics"})
    wb_raw = _as_dict(raw.get("wandb"), "observability.wandb")
    _reject_unknown_keys("observability.wandb", wb_raw, {"enabled", "project_name", "run_name"})
    enabled = _require_bool(wb_raw, "enabled", "observability.wandb")
    if "project_name" not in wb_raw:
        raise single_issue_error(path="observability.wandb", code="missing_key", message="Missing required key `project_name`")
    if "run_name" not in wb_raw:
        raise single_issue_error(path="observability.wandb", code="missing_key", message="Missing required key `run_name`")
    project_name = wb_raw["project_name"]
    run_name = wb_raw["run_name"]
    if project_name is not None and (not isinstance(project_name, str) or not project_name.strip()):
        raise single_issue_error(path="observability.wandb.project_name", code="type_error", message="Expected non-empty string or null")
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
        raise single_issue_error(path="observability.wandb.run_name", code="type_error", message="Expected non-empty string or null")
    diagnostics = raw.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise single_issue_error(path="observability.diagnostics", code="type_error", message="Expected object/dict")
    wandb = WandbConfig(
        enabled=enabled,
        project_name=None if project_name is None else project_name.strip(),
        run_name=None if run_name is None else run_name.strip(),
    )
    return ObservabilityConfig(wandb=wandb, diagnostics=dict(diagnostics))


def parse_canonical_config(raw: Mapping[str, Any]) -> CanonicalConfig:
    if not isinstance(raw, Mapping):
        raise single_issue_error(path="<root>", code="type_error", message=f"Expected object/dict, got {type(raw).__name__}")
    _reject_unknown_keys("<root>", raw, _TOPLEVEL_KEYS)
    io = _parse_io(_as_dict(raw.get("io"), "io"))
    model = _parse_model(_as_dict(raw.get("model"), "model"))
    inference = _parse_inference(_as_dict(raw.get("inference"), "inference"))
    observability = _parse_observability(_as_dict(raw.get("observability"), "observability"))
    # Optional `synth` block (consumed only by `spider synth`); pass-through dict, keys validated by the CLI.
    synth = raw.get("synth", None)
    if synth is not None and not isinstance(synth, Mapping):
        raise single_issue_error(path="synth", code="type_error", message="Expected object/dict or omit the block")
    return CanonicalConfig(
        io=io,
        model=model,
        inference=inference,
        observability=observability,
        synth=(dict(synth) if synth is not None else None),
    )


def validate_cross_field_config(cfg: CanonicalConfig, mode: str | None = None) -> None:
    # Explicit cross-field checks should live here.
    # Keep this minimal initially and extend as config_v2 adoption grows.
    if cfg.observability.wandb.enabled and cfg.observability.wandb.project_name is None:
        raise single_issue_error(
            path="observability.wandb.project_name",
            code="required_when_enabled",
            message="Must be non-null when observability.wandb.enabled=true",
        )
    if cfg.model.domain.z_max <= cfg.model.domain.z_min:
        raise single_issue_error(
            path="model.domain",
            code="value_error",
            message="Expected z_max > z_min",
        )
    if mode is not None and not isinstance(mode, str):
        raise single_issue_error(path="<mode>", code="type_error", message="Mode must be string or null")


def validate_config(raw: Mapping[str, Any], mode: str | None = None) -> CanonicalConfig:
    cfg = parse_canonical_config(raw)
    validate_cross_field_config(cfg, mode=mode)
    return cfg


# ---------------------------------------------------------------------------
# Non-fatal unknown-key warnings for pass-through blocks
# ---------------------------------------------------------------------------
#
# The blocks below are not strictly key-checked (the runtime reads them with
# defaults), so a misspelled key is silently ignored. The known-key sets are
# derived from what spider/core/config_v2/legacy_bridge.py and the runtime
# consumers (eikonet_loader.py, cli.py torch settings, epoch_runner.py,
# analyze_resid.py, locate.py) actually read. A `None` value means the key is
# an opaque sub-block whose contents are not inspected here.

def _leaves(*names: str) -> Dict[str, Any]:
    """Spec helper: plain keys whose values are not inspected further."""
    return {n: True for n in names}


_SHARED_EVENT_RE_SPEC: Dict[str, Any] = {
    **_leaves("enabled"),
    "model": {**_leaves("group_by", "tau_s"), "cluster": _leaves("mode", "k")},
    "limits": _leaves("max_nodes", "max_rows"),
    "fallback": _leaves("to_diag", "abort_on_pcg_fallback"),
    "numerics": _leaves("jitter0", "jitter_max"),
    "station_phase_term": _leaves("enabled", "tau_s"),
    "solver": {
        **_leaves(
            "kind", "max_iters", "min_iters", "tol", "batched", "node_bin_edges", "warm_start",
            "cache_max_entries", "prefetch_grouping", "profile_micro_steps",
            "merge_sparse_node_bins", "min_groups_per_node_bin", "max_node_bins_per_node",
        ),
        "precompute": _leaves("enabled", "device"),
    },
    "edge_weights": _leaves("mode", "ell_km", "eps_km", "power", "scale_km", "global_scale", "normalize"),
    "autotune": _leaves(
        "enabled", "observe_epochs", "latest_epoch", "min_groups", "max_node_bins",
        "min_groups_per_node_bin", "min_node_bin", "min_gain", "raise_nodes_cap", "nodes_cap_max",
    ),
    "logging": _leaves("quiet", "stats_log_every_epochs"),
}

# Spec grammar: {key: True} = known leaf; {key: {...}} = sub-block inspected recursively;
# {key: None} = known opaque sub-block (contents not inspected).
_PASSTHROUGH_KNOWN_KEYS: Dict[str, Any] = {
    "model.eikonet": _leaves(
        "model_kind", "x_max", "y_max", "n_hidden", "n_blocks", "n_fourier", "use_fourier",
        "phase_emb_dim", "t2_log_scale", "vp", "vs",
    ),
    "model.filters": {
        "dtimes": _leaves("remove_duplicates", "max_abs_input_dt", "dtime_thin_frac", "flip_dt_sign", "cc_min"),
        "events": {
            **_leaves(
                "min_dtimes", "min_unique_phase_per_event", "min_dtimes_per_pair", "min_event_degree",
                "min_events_per_cluster", "max_pair_station_ratio", "ratio_filter_phase", "lat_bounds",
                "lon_bounds",
            ),
            "linearization_error": _leaves(
                "enabled", "phase", "batch_size", "sample_size", "log_every_batches", "max_ratio"
            ),
        },
        "residual": _leaves("enabled", "method", "mad_sigma", "abs_max", "phase"),
    },
    "model.likelihoods": {
        "locate_map": {**_leaves("type", "phase_unc", "huber_delta"), "student_t": _leaves("nu")},
        "sample": {
            **_leaves("type", "phase_unc", "huber_delta"),
            "student_t": _leaves("nu"),
            "shared_event_re": _SHARED_EVENT_RE_SPEC,
        },
    },
    "inference.batching": {
        "standard": _leaves("warmup", "sgld", "shuffle"),
        "event_batches": _leaves(
            "enabled", "events_per_batch", "max_edges_per_batch", "bucket_reorder_all", "bucket_reuse_epochs"
        ),
    },
    "inference.runtime.gauge_projection": _leaves("enabled", "mode", "dims", "apply_noise", "apply_momentum"),
    "inference.runtime.torch": _leaves(
        "allow_tf32", "matmul_precision", "compile_eikonet", "compile_mode", "compile_backend",
        "compile_dynamic", "compile_fullgraph",
    ),
    "inference.sampler.overrides": {
        "core": _leaves("lr_mult", "temperature_mult", "eps", "freeze_preconditioner_sampling"),
    },
    "observability.diagnostics": {
        **_leaves(
            "pair_count_stats_enable", "display_precond_every", "profile_shared_event_re",
            # accepted for backward compatibility, no effect
            "sgld_log_gnoise", "sgld_log_temperature", "hierarchical_log_every_epochs",
        ),
        "wandb": _leaves("enabled", "groups"),
        "ess_online": _leaves("enabled", "every_n_samples", "n_events", "seed", "window", "max_lag", "dims"),
        "spatial": _leaves("enabled", "min_obs", "permutations", "seed"),
        "pathcorr": _leaves(
            "enabled", "min_obs_per_event", "seed", "max_pairs", "bin_width", "top_k_print", "max_pi", "outfile"
        ),
        # analyze-resid sub-diagnostics and removed blocks: opaque, contents not inspected
        "truth_catalog": None,
        "truth_locations": None,
        "shared_event_re_tau": None,
        "event_pair_station_corr": None,
        "eikonet_v1d": None,
        "resid_distribution": None,
        "resid_scalar_metrics": None,
        "shared_event_legcorr2d": None,
    },
}


def _walk_passthrough(path: str, data: Any, spec: Any, out: List[ConfigIssue]) -> None:
    if not isinstance(spec, dict) or not isinstance(data, Mapping):
        return
    unknown = sorted(str(k) for k in data.keys() if k not in spec)
    if unknown:
        out.append(
            ConfigIssue(
                path=path,
                code="unknown_key",
                message=f"Unknown key(s) ignored by the runtime: {', '.join(unknown)}",
                severity="warning",
                suggestion=f"Known keys: {', '.join(sorted(spec.keys()))}",
            )
        )
    for name, sub_spec in spec.items():
        if isinstance(sub_spec, dict) and name in data:
            _walk_passthrough(f"{path}.{name}", data[name], sub_spec, out)


def _get_path(raw: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = raw
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def collect_unknown_key_issues(raw: Mapping[str, Any]) -> List[ConfigIssue]:
    """
    Return non-fatal `ConfigIssue(severity="warning")` entries for keys inside the
    pass-through blocks that no code reads. Never raises; safe to call on any dict.
    """
    out: List[ConfigIssue] = []
    if not isinstance(raw, Mapping):
        return out
    for block_path, spec in _PASSTHROUGH_KNOWN_KEYS.items():
        _walk_passthrough(block_path, _get_path(raw, block_path), spec, out)
    return out


def collect_unknown_key_warnings(raw: Mapping[str, Any]) -> List[str]:
    """
    Human-readable unknown-key warnings for the pass-through config blocks, one string per
    block with unknown keys, e.g.
    ``[warning] model.filters.dtimes (unknown_key): Unknown key(s) ignored by the runtime: cc_mn | suggestion: Known keys: ...``.

    Intended for `spider validate-config` to print after a successful validation. Returns an
    empty list when every pass-through key is recognized.
    """
    return [issue.render() for issue in collect_unknown_key_issues(raw)]
