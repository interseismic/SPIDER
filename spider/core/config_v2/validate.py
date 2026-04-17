"""
Validation for the canonical config_v2 schema.

This module intentionally does not implement compatibility shims.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from .errors import single_issue_error
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


_TOPLEVEL_KEYS = {"io", "model", "inference", "observability"}
_SAMPLER_BACKENDS = {"psgld", "sghmc"}


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
            "dt_lr_mult",
            "grad_clip_norm",
            "preconditioning",
            "overrides",
        },
    )
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
    dt_lr_mult = _optional_num(raw, "dt_lr_mult", path)
    if dt_lr_mult is not None and dt_lr_mult <= 0.0:
        raise single_issue_error(path="inference.sampler.dt_lr_mult", code="value_error", message="Must be > 0")
    grad_clip_norm = _optional_num(raw, "grad_clip_norm", path)
    if grad_clip_norm is not None and grad_clip_norm < 0.0:
        raise single_issue_error(path="inference.sampler.grad_clip_norm", code="value_error", message="Must be >= 0")

    preconditioning = raw.get("preconditioning", {})
    overrides = raw.get("overrides", {})
    if not isinstance(preconditioning, dict):
        raise single_issue_error(path="inference.sampler.preconditioning", code="type_error", message="Expected object/dict")
    _reject_unknown_keys(
        "inference.sampler.preconditioning",
        preconditioning,
        {"enabled", "type", "include_gamma", "lrd"},
    )
    precond_enabled = bool(preconditioning.get("enabled", False))
    precond_type = str(preconditioning.get("type", "none")).strip().lower()
    if precond_enabled and precond_type not in {"rmsprop", "lrd"}:
        raise single_issue_error(
            path="inference.sampler.preconditioning.type",
            code="enum_error",
            message="Supported values are: rmsprop, lrd (or disable preconditioning).",
        )
    lrd_cfg = preconditioning.get("lrd", {})
    if lrd_cfg is not None and not isinstance(lrd_cfg, dict):
        raise single_issue_error(
            path="inference.sampler.preconditioning.lrd",
            code="type_error",
            message="Expected object/dict",
        )
    if not isinstance(overrides, dict):
        raise single_issue_error(path="inference.sampler.overrides", code="type_error", message="Expected object/dict")
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
        dt_lr_mult=dt_lr_mult,
        grad_clip_norm=grad_clip_norm,
        preconditioning=dict(preconditioning),
        overrides=dict(overrides),
        extras={},
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
    min_samples_to_save = int(_require_num(raw, "min_samples_to_save", path))
    if min_samples_to_save < 0:
        raise single_issue_error(path="inference.runtime.min_samples_to_save", code="value_error", message="Must be >= 0")
    verbose = _require_bool(raw, "verbose", path)
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


def _parse_safety(raw: Mapping[str, Any]) -> SafetyConfig:
    path = "inference.safety"
    _reject_unknown_keys(path, raw, {"max_abs_dX"})
    max_abs = raw.get("max_abs_dX", None)
    if max_abs is None:
        return SafetyConfig(max_abs_dX=None, extras={})
    if not isinstance(max_abs, list) or len(max_abs) != 4:
        raise single_issue_error(path="inference.safety.max_abs_dX", code="type_error", message="Expected list[4] of numbers or null")
    out: list[float] = []
    for i, item in enumerate(max_abs):
        if not isinstance(item, (int, float)):
            raise single_issue_error(path=f"inference.safety.max_abs_dX[{i}]", code="type_error", message="Expected number")
        out.append(float(item))
    return SafetyConfig(max_abs_dX=out, extras={})


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
    return CanonicalConfig(
        io=io,
        model=model,
        inference=inference,
        observability=observability,
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

