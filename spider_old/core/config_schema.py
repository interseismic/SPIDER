"""
Strict nested configuration schema for SPIDER (incremental migration).

Block 1: `wandb`, `io`, `model`, `domain`
 - Nested-only for these blocks (legacy flat keys forbidden in user JSON)
 - No implicit defaults: required keys must be present
 - For now, we materialize the legacy flat keys AFTER validation so existing
   implementation can keep reading params[...] without refactors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


_FORBIDDEN_BLOCK1_TOPLEVEL_KEYS: Tuple[str, ...] = (
    # wandb (legacy)
    "use_wandb",
    "wandb_project_name",
    "wandb_run_name",
    # io (legacy)
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
    # model (legacy)
    "model_file",
    # domain (legacy)
    "lon_min",
    "lat_min",
    "z_min",
    "z_max",
    "scale",
)

_FORBIDDEN_BLOCK2_TOPLEVEL_KEYS: Tuple[str, ...] = (
    # phases (legacy)
    "phase1_epochs",
    "phase2_epochs",
    "phase3_epochs",
    "phase4_epochs",
    "lr_warmup",
    # sampler (legacy)
    "lr_sampler",
    "sampler_backend",
    "sampler_temperature",
    "sampler_preconditioning",
    "sampler_preconditioner",
    "sampler_beta",
    "sampler_eps",
    "freeze_preconditioner_sampling",
    "sghmc_alpha",
    # SGNHT (removed)
    "sgnht_diffusion",
    "sgnht_thermostat_mass",
)

_FORBIDDEN_BLOCK3_TOPLEVEL_KEYS: Tuple[str, ...] = (
    # likelihood (legacy flat keys; `likelihood` itself is now a nested object)
    "phase_unc",
    "learn_noise_scale",
    "learn_phase_unc",
    # batching (legacy)
    "batch_size_warmup",
    "batch_size_sgld",
    "event_batch_enable",
    "event_batch_size",
    "event_batch_max_edges",
    "event_bucket_reorder_all",
    "event_bucket_reuse_epochs",
    # filters (legacy; input filters)
    "remove_duplicates",
    "max_abs_input_dt",
    "dtime_thin_frac",
    "flip_dt_sign",
    "cc_min",
    "min_dtimes",
    "min_unique_phase_per_event",
    "min_dtimes_per_pair",
    "min_event_degree",
    "min_node_degree",
    "min_events_per_cluster",
    "max_pair_station_ratio",
    "ratio_filter_phase",
    # filters (legacy; residual)
    "residual_filter_enable",
    "residual_filter_method",
    "residual_filter_mad_sigma",
    "residual_filter_abs_max",
)

_FORBIDDEN_BLOCK4_TOPLEVEL_KEYS: Tuple[str, ...] = (
    # diagnostics/runtime (legacy)
    "pair_count_stats_enable",
    "svrg_enable",
    "fim_enable_phase2",
    "fim_filter_threshold",
    "phase2_mads_interval",
    "phase3_mads_interval",
    "phase4_mads_interval",
    "cuda_empty_cache_every",
    "display_precond_every",
    "sgld_log_gnoise",
    "max_abs_dX",
    "min_samples_to_save",
    "reset_batch_numbers",
    "clear_samples_on_reset",
    "verbose",
    "cluster_events",
)

_FORBIDDEN_BLOCK5_TOPLEVEL_KEYS: Tuple[str, ...] = (
    "devices",
    "dd_prec_enable",
)


def _err(path: str, msg: str) -> ValueError:
    return ValueError(f"Invalid config at `{path}`: {msg}")


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


def _require_num(v: Any, path: str) -> float:
    if not isinstance(v, (int, float)):
        raise _err(path, f"expected number, got {type(v).__name__}")
    return float(v)


def validate_and_materialize_block1(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Block 1 (hard-break schema): strict nested `io`, `wandb`, and `model`.

    Required nested blocks:
      - params['io']
      - params['wandb']
      - params['model'].model_file
      - params['model'].domain
    """
    forbidden_present = [k for k in _FORBIDDEN_BLOCK1_TOPLEVEL_KEYS if k in params]
    if forbidden_present:
        raise ValueError(
            "Legacy Block-1 keys are not allowed. Move these under nested blocks:\n"
            + "\n".join(f"- {k}" for k in sorted(forbidden_present))
        )

    # Hard break: `domain` moved under `model.domain`.
    if "domain" in params:
        raise _err("domain", "moved; put this under `model.domain` (top-level `domain` is no longer supported)")

    # ---- io ----
    io = _require_dict(_require(params, "io", "io"), "io")
    dtime_file = _require_str(_require(io, "dtime_file", "io"), "io.dtime_file")
    station_file = _require_str(_require(io, "station_file", "io"), "io.station_file")
    catalog_infile = _require_str(_require(io, "catalog_infile", "io"), "io.catalog_infile")
    catalog_outfile = _require_str(_require(io, "catalog_outfile", "io"), "io.catalog_outfile")
    samples_outfile = _require_str(_require(io, "samples_outfile", "io"), "io.samples_outfile")
    checkpoint_dir = _require_str(_require(io, "checkpoint_dir", "io"), "io.checkpoint_dir")
    checkpoint_interval = int(_require_num(_require(io, "checkpoint_interval", "io"), "io.checkpoint_interval"))
    # Optional: allow sample flush cadence to differ from checkpoint cadence.
    # If not provided, preserve historical behavior: flush samples when checkpointing.
    sample_write_interval_v = io.get("sample_write_interval", None)
    if sample_write_interval_v is None:
        sample_write_interval = int(checkpoint_interval)
    else:
        sample_write_interval = int(_require_num(sample_write_interval_v, "io.sample_write_interval"))
    if sample_write_interval < 0:
        raise _err("io.sample_write_interval", "must be >= 0 (0 means 'only flush at final write')")
    save_every_n = int(_require_num(_require(io, "save_every_n", "io"), "io.save_every_n"))
    write_samples = _require_bool(_require(io, "write_samples", "io"), "io.write_samples")

    # ---- model ----
    model = _require_dict(_require(params, "model", "model"), "model")
    model_file = _require_str(_require(model, "model_file", "model"), "model.model_file")

    # ---- domain ----
    dom = _require_dict(_require(model, "domain", "model"), "model.domain")
    lon_min = _require_num(_require(dom, "lon_min", "model.domain"), "model.domain.lon_min")
    lat_min = _require_num(_require(dom, "lat_min", "model.domain"), "model.domain.lat_min")
    z_min = _require_num(_require(dom, "z_min", "model.domain"), "model.domain.z_min")
    z_max = _require_num(_require(dom, "z_max", "model.domain"), "model.domain.z_max")
    scale = _require_num(_require(dom, "scale", "model.domain"), "model.domain.scale")

    # ---- wandb ----
    wb = _require_dict(_require(params, "wandb", "wandb"), "wandb")
    wb_enabled = _require_bool(_require(wb, "enabled", "wandb"), "wandb.enabled")
    # No defaults: keys must exist even if disabled; values can be null when disabled.
    project_name_v = _require(wb, "project_name", "wandb")
    run_name_v = _require(wb, "run_name", "wandb")
    if wb_enabled:
        wb_project = _require_str(project_name_v, "wandb.project_name")
        # allow empty? no; if enabled require non-empty string
        wb_run = None if run_name_v is None else _require_str(run_name_v, "wandb.run_name")
    else:
        wb_project = None if project_name_v is None else str(project_name_v)
        wb_run = None if run_name_v is None else str(run_name_v)

    # ---- materialize legacy flat keys (implementation detail) ----
    params["dtime_file"] = dtime_file
    params["station_file"] = station_file
    params["catalog_infile"] = catalog_infile
    params["catalog_outfile"] = catalog_outfile
    params["samples_outfile"] = samples_outfile
    params["checkpoint_dir"] = checkpoint_dir
    params["checkpoint_interval"] = checkpoint_interval
    params["sample_write_interval"] = sample_write_interval
    params["save_every_n"] = save_every_n
    params["write_samples"] = write_samples

    params["model_file"] = model_file

    params["lon_min"] = lon_min
    params["lat_min"] = lat_min
    params["z_min"] = z_min
    params["z_max"] = z_max
    params["scale"] = scale

    params["use_wandb"] = bool(wb_enabled)
    params["wandb_project_name"] = wb_project
    params["wandb_run_name"] = wb_run

    return params


def validate_and_materialize_block2(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Block 2 (hard-break schema): strict nested `inference.sampler` only.

    Required:
      sampler.epochs_per_phase (list[int], length 4, each >=0)
      sampler.lr (list[float], length 4, each >0)
        - lr[0] is Phase 1 (MAP), lr[1] Phase 2, lr[2] Phase 3, lr[3] Phase 4

      sampler.backend in {"psgld","sghmc","adaptive_sghmc","sgnht","adsgld_adam"}
      sampler.temperature (float>=0)
      sampler.lr_mode (optional str): "absolute" (default) or "per_obs" (psgld/sghmc/adaptive_sghmc/adsgld_adam helper; see below)
      sampler.preconditioning.enabled (bool)
      sampler.preconditioning.type (string) if enabled
        - supported (psgld): "rmsprop", "blockdiag_fisher" (alias: "matrix_ema"), "monge", "shampoo"
        - supported (sghmc): "rmsprop"
      sampler.preconditioning.include_gamma (optional bool, default True):
        If true, include the diagonal Γ(θ) correction term in pSGLD when using diagonal
        preconditioners (rmsprop). This is a low-cost correction from the pSGLD paper.
      sampler.preconditioning.include_gamma_proxy (optional bool, default False):
        If true and preconditioning.type == "blockdiag_fisher", add a cheap diagonal Γ(θ) proxy
        drift correction even though the preconditioner is matrix-valued. This is NOT the exact
        matrix divergence term from the pSGLD paper, but can reduce spurious drift when the
        preconditioner is adapting (beta<1, not frozen).
      sampler.preconditioning.blockdiag_fisher.partitioning.max_cluster_size (optional int>=1):
        Only used when sampler.preconditioning.type == "blockdiag_fisher".
        This controls how we *split each connected component* (based on shared-observation connectivity)
        into disjoint blocks for the block-diagonal Fisher preconditioner. It does NOT affect component
        detection itself.
      sampler.beta (0<=beta<1), sampler.eps (>0)
      sampler.freeze_preconditioner_sampling (bool)
      sampler.sghmc_alpha (>0) iff backend=="sghmc" (no default)
      sampler.dt_lr_mult (optional float>0, default 1.0): multiplier applied to the ΔT gradient (dimension 3)
        to effectively use a different learning rate for the origin-time correction component.
      sampler.adaptive_drift (required iff backend=="adsgld_adam"):
        {beta1 (0<=b1<1), beta2 (0<=b2<1), eps (>0), scale (>0)}
    """
    forbidden_present = [k for k in _FORBIDDEN_BLOCK2_TOPLEVEL_KEYS if k in params]
    if forbidden_present:
        raise ValueError(
            "Legacy Block-2 keys are not allowed. Move these under nested blocks:\n"
            + "\n".join(f"- {k}" for k in sorted(forbidden_present))
        )

    # Hard break: these blocks moved under `inference.*`.
    if "phases" in params:
        raise _err(
            "phases",
            "removed; use `inference.sampler.epochs_per_phase` and `inference.sampler.lr` instead",
        )
    if "sampler" in params:
        raise _err("sampler", "moved; put this under `inference.sampler` (top-level `sampler` is no longer supported)")

    inf = _require_dict(_require(params, "inference", "inference"), "inference")
    if "phases" in inf:
        raise _err(
            "inference.phases",
            "removed; use `inference.sampler.epochs_per_phase` and `inference.sampler.lr` instead",
        )

    def _epochs(v: Any, path: str) -> int:
        if not isinstance(v, int):
            raise _err(path, f"expected integer, got {type(v).__name__}")
        if v < 0:
            raise _err(path, "must be >= 0")
        return int(v)

    sampler = _require_dict(_require(inf, "sampler", "inference"), "inference.sampler")
    epochs_v = _require(sampler, "epochs_per_phase", "inference.sampler")
    if not isinstance(epochs_v, list):
        raise _err("inference.sampler.epochs_per_phase", f"expected list, got {type(epochs_v).__name__}")
    if len(epochs_v) != 4:
        raise _err("inference.sampler.epochs_per_phase", f"expected list of length 4, got {len(epochs_v)}")
    epochs_per_phase: List[int] = []
    for i, xi in enumerate(epochs_v):
        if not isinstance(xi, int):
            raise _err(f"inference.sampler.epochs_per_phase[{i}]", f"expected integer, got {type(xi).__name__}")
        if xi < 0:
            raise _err(f"inference.sampler.epochs_per_phase[{i}]", "must be >= 0")
        epochs_per_phase.append(int(xi))

    lr_per_phase = _require_float_list(
        _require(sampler, "lr", "inference.sampler"),
        "inference.sampler.lr",
        length=4,
    )
    for i, lr_i in enumerate(lr_per_phase):
        if not (math.isfinite(lr_i) and lr_i > 0.0):
            raise _err(f"inference.sampler.lr[{i}]", "must be finite and > 0")

    phase1_epochs = epochs_per_phase[0]
    phase2_epochs = epochs_per_phase[1]
    phase3_epochs = epochs_per_phase[2]
    phase4_epochs = epochs_per_phase[3]

    lr_warmup = float(lr_per_phase[0])
    lr_phase2 = float(lr_per_phase[1])
    lr_phase3 = float(lr_per_phase[2])
    lr_phase4 = float(lr_per_phase[3])

    if not (lr_warmup > 0.0):
        raise _err("inference.sampler.lr[0]", "must be > 0")
    backend = _require_str(_require(sampler, "backend", "inference.sampler"), "inference.sampler.backend").lower()
    if backend not in {"psgld", "sghmc", "adaptive_sghmc", "sgnht", "adsgld_adam"}:
        raise _err("sampler.backend", "supported: 'psgld', 'sghmc', 'adaptive_sghmc', 'sgnht', 'adsgld_adam'")
    lr_mode = str(sampler.get("lr_mode", "absolute")).strip().lower()
    if lr_mode not in {"absolute", "per_obs"}:
        raise _err("sampler.lr_mode", "supported: 'absolute', 'per_obs'")
    temperature = _require_num(_require(sampler, "temperature", "sampler"), "sampler.temperature")
    if temperature < 0.0:
        raise _err("sampler.temperature", "must be >= 0")

    # Optional: extra multiplier for injected Langevin noise (Phase 3/4).
    # This scales the sampler's internal `noise_scale` (not the model noise σ_p/σ_s).
    # It is useful when you want to keep temperature=1.0 but increase injected noise
    # to improve mixing / effective temperature for large problems.
    noise_scale_mult = 1.0
    if "noise_scale_mult" in sampler and sampler.get("noise_scale_mult", None) is not None:
        noise_scale_mult = float(_require_num(sampler.get("noise_scale_mult"), "sampler.noise_scale_mult"))
        if not (noise_scale_mult > 0.0) or (not math.isfinite(noise_scale_mult)):
            raise _err("sampler.noise_scale_mult", "must be finite and > 0")

    # Optional: per-dimension LR multiplier for ΔT (origin time correction).
    # Implemented as a gradient scaler in the epoch runner so it works across Adam/PSGLD/SGHMC.
    dt_lr_mult = 1.0
    if "dt_lr_mult" in sampler and sampler.get("dt_lr_mult", None) is not None:
        dt_lr_mult = float(_require_num(sampler.get("dt_lr_mult"), "sampler.dt_lr_mult"))
        if not (dt_lr_mult > 0.0) or not math.isfinite(dt_lr_mult):
            raise _err("sampler.dt_lr_mult", "must be finite and > 0")

    # Optional: gradient clipping for sampler phases (2/3/4).
    sampler_grad_clip_norm = 0.0
    if "grad_clip_norm" in sampler and sampler.get("grad_clip_norm", None) is not None:
        sampler_grad_clip_norm = float(_require_num(sampler.get("grad_clip_norm"), "sampler.grad_clip_norm"))
        if not (sampler_grad_clip_norm >= 0.0):
            raise _err("sampler.grad_clip_norm", "must be >= 0")

    # Adaptive drift (Adam variant) parameters.
    adaptive_drift_cfg = sampler.get("adaptive_drift", None)
    if backend == "adsgld_adam":
        if not isinstance(adaptive_drift_cfg, dict):
            raise _err("inference.sampler.adaptive_drift", "required when sampler.backend='adsgld_adam'")
        ad_beta1 = _require_num(_require(adaptive_drift_cfg, "beta1", "inference.sampler.adaptive_drift"), "inference.sampler.adaptive_drift.beta1")
        ad_beta2 = _require_num(_require(adaptive_drift_cfg, "beta2", "inference.sampler.adaptive_drift"), "inference.sampler.adaptive_drift.beta2")
        ad_eps = _require_num(_require(adaptive_drift_cfg, "eps", "inference.sampler.adaptive_drift"), "inference.sampler.adaptive_drift.eps")
        ad_scale = _require_num(_require(adaptive_drift_cfg, "scale", "inference.sampler.adaptive_drift"), "inference.sampler.adaptive_drift.scale")
    else:
        if isinstance(adaptive_drift_cfg, dict):
            ad_beta1 = _require_num(adaptive_drift_cfg.get("beta1", 0.9), "inference.sampler.adaptive_drift.beta1")
            ad_beta2 = _require_num(adaptive_drift_cfg.get("beta2", 0.999), "inference.sampler.adaptive_drift.beta2")
            ad_eps = _require_num(adaptive_drift_cfg.get("eps", 1e-8), "inference.sampler.adaptive_drift.eps")
            ad_scale = _require_num(adaptive_drift_cfg.get("scale", 1.0), "inference.sampler.adaptive_drift.scale")
        else:
            ad_beta1 = 0.9
            ad_beta2 = 0.999
            ad_eps = 1e-8
            ad_scale = 1.0
    if not (0.0 <= ad_beta1 < 1.0):
        raise _err("inference.sampler.adaptive_drift.beta1", "must satisfy 0 <= beta1 < 1")
    if not (0.0 <= ad_beta2 < 1.0):
        raise _err("inference.sampler.adaptive_drift.beta2", "must satisfy 0 <= beta2 < 1")
    if not (ad_eps > 0.0) or not math.isfinite(float(ad_eps)):
        raise _err("inference.sampler.adaptive_drift.eps", "must be finite and > 0")
    if not (ad_scale > 0.0) or not math.isfinite(float(ad_scale)):
        raise _err("inference.sampler.adaptive_drift.scale", "must be finite and > 0")

    precond = _require_dict(_require(sampler, "preconditioning", "sampler"), "sampler.preconditioning")
    precond_enabled = _require_bool(_require(precond, "enabled", "sampler.preconditioning"), "sampler.preconditioning.enabled")
    # Optional gamma correction toggles (pSGLD; mostly relevant when backend == 'psgld')
    precond_include_gamma = True
    if "include_gamma" in precond:
        precond_include_gamma = _require_bool(precond.get("include_gamma"), "sampler.preconditioning.include_gamma")
    precond_include_gamma_proxy = False
    if "include_gamma_proxy" in precond:
        precond_include_gamma_proxy = _require_bool(precond.get("include_gamma_proxy"), "sampler.preconditioning.include_gamma_proxy")
    if precond_enabled:
        precond_type = _require_str(_require(precond, "type", "sampler.preconditioning"), "sampler.preconditioning.type").strip().lower()
        if precond_type == "matrix":
            raise _err(
                "sampler.preconditioning.type",
                "preconditioner type 'matrix' has been removed (it triggered the full-dataset FIM workflow). "
                "Use 'blockdiag_fisher' (alias: 'matrix_ema') for the online 4x4 block preconditioner, or 'rmsprop'.",
            )
        # Backwards-compatible alias
        if precond_type == "matrix_ema":
            precond_type = "blockdiag_fisher"
        if precond_type not in {"rmsprop", "blockdiag_fisher", "monge", "shampoo"}:
            raise _err("sampler.preconditioning.type", "supported: 'rmsprop','blockdiag_fisher' (alias: 'matrix_ema'),'monge','shampoo'")

        # Backend-specific support
        if backend in {"sghmc", "adaptive_sghmc", "sgnht"} and precond_type == "blockdiag_fisher":
            raise _err(
                "sampler.preconditioning.type",
                "'blockdiag_fisher' is currently supported for backend='psgld' only. "
                "For SGHMC/AdaptiveSGHMC/SGNHT use 'rmsprop' preconditioning (AdaptiveSGHMC has its own diagonal "
                "preconditioner) unless/until a true preconditioned-SGHMC block-metric implementation is added.",
            )
        if backend in {"sghmc", "adaptive_sghmc", "sgnht"} and precond_type in {"monge", "shampoo"}:
            raise _err(
                "sampler.preconditioning.type",
                "'monge' and 'shampoo' preconditioners are currently supported for backend='psgld' only.",
            )
    else:
        # still require key presence, but value can be null/empty
        _require(precond, "type", "sampler.preconditioning")
        precond_type = "none"

    # Optional: monge/shampoo preconditioner knobs (psgld only).
    monge_alpha = 1.0
    shampoo_beta = 0.99
    shampoo_eps = 1e-6
    shampoo_update_every = 10
    shampoo_max_dim = 512
    if isinstance(precond, dict):
        if "monge_alpha" in precond and precond.get("monge_alpha", None) is not None:
            monge_alpha = float(_require_num(precond.get("monge_alpha"), "sampler.preconditioning.monge_alpha"))
        if "shampoo_beta" in precond and precond.get("shampoo_beta", None) is not None:
            shampoo_beta = float(_require_num(precond.get("shampoo_beta"), "sampler.preconditioning.shampoo_beta"))
        if "shampoo_eps" in precond and precond.get("shampoo_eps", None) is not None:
            shampoo_eps = float(_require_num(precond.get("shampoo_eps"), "sampler.preconditioning.shampoo_eps"))
        if "shampoo_update_every" in precond and precond.get("shampoo_update_every", None) is not None:
            shampoo_update_every = int(_require_num(precond.get("shampoo_update_every"), "sampler.preconditioning.shampoo_update_every"))
        if "shampoo_max_dim" in precond and precond.get("shampoo_max_dim", None) is not None:
            shampoo_max_dim = int(_require_num(precond.get("shampoo_max_dim"), "sampler.preconditioning.shampoo_max_dim"))
    if not (monge_alpha > 0.0) or not math.isfinite(monge_alpha):
        raise _err("sampler.preconditioning.monge_alpha", "must be finite and > 0")
    if not (0.0 <= shampoo_beta < 1.0):
        raise _err("sampler.preconditioning.shampoo_beta", "must satisfy 0 <= beta < 1")
    if not (shampoo_eps > 0.0) or not math.isfinite(shampoo_eps):
        raise _err("sampler.preconditioning.shampoo_eps", "must be finite and > 0")
    if shampoo_update_every < 1:
        raise _err("sampler.preconditioning.shampoo_update_every", "must be >= 1")
    if shampoo_max_dim < 1:
        raise _err("sampler.preconditioning.shampoo_max_dim", "must be >= 1")

    beta = _require_num(_require(sampler, "beta", "sampler"), "sampler.beta")
    if not (0.0 <= beta < 1.0):
        raise _err("sampler.beta", "must satisfy 0 <= beta < 1")
    eps = _require_num(_require(sampler, "eps", "sampler"), "sampler.eps")
    if not (eps > 0.0):
        raise _err("sampler.eps", "must be > 0")

    freeze_preconditioner_sampling = _require_bool(
        _require(sampler, "freeze_preconditioner_sampling", "sampler"),
        "sampler.freeze_preconditioner_sampling",
    )

    # ---- blockdiag_fisher partitioning knobs ----
    blockdiag_fisher_max_cluster_size = 1
    blockdiag_fisher_partition_method = "auto"
    if precond_enabled and precond_type == "blockdiag_fisher":
        try:
            bdf = sampler.get("preconditioning", {}).get("blockdiag_fisher", None)
            if isinstance(bdf, dict):
                # We intentionally do NOT support the legacy alias
                # `sampler.preconditioning.blockdiag_fisher.include_gamma_proxy`.
                # Use `sampler.preconditioning.include_gamma_proxy` instead.
                if "include_gamma_proxy" in bdf:
                    raise _err(
                        "sampler.preconditioning.blockdiag_fisher.include_gamma_proxy",
                        "this key has been removed; use `sampler.preconditioning.include_gamma_proxy` instead",
                    )
                part = bdf.get("partitioning", None)
                if isinstance(part, dict):
                    if "max_cluster_size" in part:
                        mcs = part.get("max_cluster_size")
                        if not isinstance(mcs, int):
                            raise _err("sampler.preconditioning.blockdiag_fisher.partitioning.max_cluster_size", f"expected integer, got {type(mcs).__name__}")
                        if mcs < 1:
                            raise _err("sampler.preconditioning.blockdiag_fisher.partitioning.max_cluster_size", "must be >= 1")
                        blockdiag_fisher_max_cluster_size = int(mcs)
                    if "method" in part:
                        meth = str(part.get("method", "auto")).strip().lower()
                        if meth not in {"auto", "greedy", "spectral"}:
                            raise _err("sampler.preconditioning.blockdiag_fisher.partitioning.method", "supported: 'auto','greedy','spectral'")
                        blockdiag_fisher_partition_method = meth
        except ValueError:
            raise
        except Exception as e:
            raise _err("sampler.preconditioning.blockdiag_fisher", f"invalid: {e}")

    # sghmc_alpha is required for SGHMC, and we also reuse it as the friction/mdecay knob
    # for adaptive_sghmc (BOHAMIANN-style) to avoid introducing a separate required field.
    if backend in {"sghmc", "adaptive_sghmc"}:
        alpha = _require_num(_require(sampler, "sghmc_alpha", "sampler"), "sampler.sghmc_alpha")
        if not (alpha > 0.0):
            raise _err("sampler.sghmc_alpha", "must be > 0")
    else:
        # still require key presence; can be null if not used
        _require(sampler, "sghmc_alpha", "sampler")
        alpha = 0.0

    # Optional SGNHT parameters (used only when backend == 'sgnht').
    sgnht_diffusion = 0.01
    if "sgnht_diffusion" in sampler and sampler.get("sgnht_diffusion", None) is not None:
        sgnht_diffusion = float(_require_num(sampler.get("sgnht_diffusion"), "sampler.sgnht_diffusion"))
        if not (sgnht_diffusion > 0.0) or (not math.isfinite(sgnht_diffusion)):
            raise _err("sampler.sgnht_diffusion", "must be finite and > 0")
    sgnht_thermostat_mass = 1.0
    if "sgnht_thermostat_mass" in sampler and sampler.get("sgnht_thermostat_mass", None) is not None:
        sgnht_thermostat_mass = float(_require_num(sampler.get("sgnht_thermostat_mass"), "sampler.sgnht_thermostat_mass"))
        if not (sgnht_thermostat_mass > 0.0) or (not math.isfinite(sgnht_thermostat_mass)):
            raise _err("sampler.sgnht_thermostat_mass", "must be finite and > 0")

    # ---- materialize legacy flat keys (implementation detail) ----
    params["phase1_epochs"] = phase1_epochs
    params["phase2_epochs"] = phase2_epochs
    params["phase3_epochs"] = phase3_epochs
    params["phase4_epochs"] = phase4_epochs
    params["lr_warmup"] = lr_warmup

    params["lr_sampler"] = lr_phase2
    # lr_mode='per_obs' is a convenience: when using pSGLD/SGHMC/AdaptiveSGHMC (which use minibatch-mean gradients
    # and internally scale the drift by N via n_obs/scale_grad), we apply lr_eff = lr_sampler / N at runtime.
    # This preserves the true posterior target but makes
    # the *user-provided* lr less sensitive to dataset size.
    params["sampler_lr_mode"] = lr_mode
    params["sampler_backend"] = backend
    params["sampler_temperature"] = temperature
    params["sampler_noise_scale_mult"] = float(noise_scale_mult)
    params["dt_lr_mult"] = float(dt_lr_mult)
    params["sampler_grad_clip_norm"] = float(sampler_grad_clip_norm)
    params["sampler_preconditioning"] = bool(precond_enabled)
    params["sampler_preconditioner"] = precond_type if precond_enabled else "none"
    params["sampler_beta"] = beta
    params["sampler_eps"] = eps
    params["freeze_preconditioner_sampling"] = bool(freeze_preconditioner_sampling)
    params["sghmc_alpha"] = alpha
    params["sgnht_diffusion"] = float(sgnht_diffusion)
    params["sgnht_thermostat_mass"] = float(sgnht_thermostat_mass)
    params["blockdiag_fisher_max_cluster_size"] = int(blockdiag_fisher_max_cluster_size)
    params["blockdiag_fisher_partition_method"] = str(blockdiag_fisher_partition_method)
    params["sampler_preconditioning_include_gamma"] = bool(precond_include_gamma)
    params["sampler_preconditioning_include_gamma_proxy"] = bool(precond_include_gamma_proxy)
    params["_sampler_lr_per_phase"] = list(lr_per_phase)
    params["adaptive_drift_beta1"] = float(ad_beta1)
    params["adaptive_drift_beta2"] = float(ad_beta2)
    params["adaptive_drift_eps"] = float(ad_eps)
    params["adaptive_drift_scale"] = float(ad_scale)
    params["sampler_preconditioning_monge_alpha"] = float(monge_alpha)
    params["sampler_preconditioning_shampoo_beta"] = float(shampoo_beta)
    params["sampler_preconditioning_shampoo_eps"] = float(shampoo_eps)
    params["sampler_preconditioning_shampoo_update_every"] = int(shampoo_update_every)
    params["sampler_preconditioning_shampoo_max_dim"] = int(shampoo_max_dim)

    # ---- sampler parameter-group overrides (hard-break schema) ----
    # These are inference-only controls. They let high-dimensional latent groups use smaller step/noise.
    overrides = sampler.get("overrides", None)
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise _err("inference.sampler.overrides", "expected object/dict or null")
    if "shared_event_latent" in overrides:
        raise _err(
            "inference.sampler.overrides.shared_event_latent",
            "removed; shared_event_latent has been removed from SPIDER (delete this block)",
        )

    # Optional: correlated forward-model error latent (corr_error) group overrides.
    # Defaults chosen to be conservative for a large latent.
    corr_lr_mult = 0.05
    corr_temp_mult = 0.25
    corr_eps = 1e-3
    corr_freeze_precond = False
    corr_overrides_active = False
    corr_ov = overrides.get("corr_error", None)
    if corr_ov is not None:
        if not isinstance(corr_ov, dict):
            raise _err("inference.sampler.overrides.corr_error", "expected object/dict or null")
        for k in ("lr_mult", "temperature_mult", "eps", "freeze_preconditioner_sampling"):
            if k in corr_ov and corr_ov.get(k, None) is not None:
                corr_overrides_active = True
            break
        if "lr_mult" in corr_ov and corr_ov.get("lr_mult", None) is not None:
            corr_lr_mult = float(_require_num(corr_ov.get("lr_mult"), "inference.sampler.overrides.corr_error.lr_mult"))
            if not (corr_lr_mult > 0.0):
                raise _err("inference.sampler.overrides.corr_error.lr_mult", "must be > 0")
        if "temperature_mult" in corr_ov and corr_ov.get("temperature_mult", None) is not None:
            corr_temp_mult = float(
                _require_num(corr_ov.get("temperature_mult"), "inference.sampler.overrides.corr_error.temperature_mult")
        )
            if not (corr_temp_mult > 0.0):
                raise _err("inference.sampler.overrides.corr_error.temperature_mult", "must be > 0")
        if "eps" in corr_ov and corr_ov.get("eps", None) is not None:
            corr_eps = float(_require_num(corr_ov.get("eps"), "inference.sampler.overrides.corr_error.eps"))
            if not (corr_eps >= 0.0):
                raise _err("inference.sampler.overrides.corr_error.eps", "must be >= 0")
        if "freeze_preconditioner_sampling" in corr_ov and corr_ov.get("freeze_preconditioner_sampling", None) is not None:
            corr_freeze_precond = _require_bool(
                corr_ov.get("freeze_preconditioner_sampling"),
                "inference.sampler.overrides.corr_error.freeze_preconditioner_sampling",
        )

    params["_corr_error_lr_mult"] = float(corr_lr_mult)
    params["_corr_error_temperature_mult"] = float(corr_temp_mult)
    params["_corr_error_eps"] = float(corr_eps)
    params["_corr_error_freeze_preconditioner_sampling"] = bool(corr_freeze_precond)
    params["_corr_error_sampler_overrides_active"] = bool(corr_overrides_active)

    # Optional: generic overrides for other parameter groups (core, slowness_re, dd_graph_re).
    # For these groups we only apply keys explicitly provided by the user.
    group_overrides: Dict[str, Dict[str, Any]] = {}
    any_group_override_active = False

    if corr_overrides_active:
        group_overrides["corr_error"] = {
            "lr_mult": float(corr_lr_mult),
            "temperature_mult": float(corr_temp_mult),
            "eps": float(corr_eps),
            "freeze_preconditioner_sampling": bool(corr_freeze_precond),
        }
        any_group_override_active = True

    def _parse_group_override(group_name: str) -> None:
        nonlocal any_group_override_active
        ov = overrides.get(group_name, None)
        if ov is None:
            return
        if not isinstance(ov, dict):
            raise _err(f"inference.sampler.overrides.{group_name}", "expected object/dict or null")
        out: Dict[str, Any] = {}
        if "lr_mult" in ov and ov.get("lr_mult", None) is not None:
            lr_mult = float(_require_num(ov.get("lr_mult"), f"inference.sampler.overrides.{group_name}.lr_mult"))
            if not (lr_mult > 0.0):
                raise _err(f"inference.sampler.overrides.{group_name}.lr_mult", "must be > 0")
            out["lr_mult"] = float(lr_mult)
        if "temperature_mult" in ov and ov.get("temperature_mult", None) is not None:
            temp_mult = float(_require_num(ov.get("temperature_mult"), f"inference.sampler.overrides.{group_name}.temperature_mult"))
            if not (temp_mult > 0.0):
                raise _err(f"inference.sampler.overrides.{group_name}.temperature_mult", "must be > 0")
            out["temperature_mult"] = float(temp_mult)
        if "eps" in ov and ov.get("eps", None) is not None:
            eps_g = float(_require_num(ov.get("eps"), f"inference.sampler.overrides.{group_name}.eps"))
            if not (eps_g >= 0.0):
                raise _err(f"inference.sampler.overrides.{group_name}.eps", "must be >= 0")
            out["eps"] = float(eps_g)
        if "freeze_preconditioner_sampling" in ov and ov.get("freeze_preconditioner_sampling", None) is not None:
            freeze_g = _require_bool(
                ov.get("freeze_preconditioner_sampling"),
                f"inference.sampler.overrides.{group_name}.freeze_preconditioner_sampling",
            )
            out["freeze_preconditioner_sampling"] = bool(freeze_g)
        if out:
            group_overrides[group_name] = out
            any_group_override_active = True

    _parse_group_override("core")
    _parse_group_override("slowness_re")
    _parse_group_override("dd_graph_re")

    params["_sampler_group_overrides"] = dict(group_overrides)
    params["_sampler_group_overrides_active"] = bool(any_group_override_active)

    return params


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


def validate_and_materialize_block3(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Block 3 (hard-break schema): strict nested `model.likelihood` + `model.filters` + `inference.batching`.

    Likelihood:
      model.likelihood.type (str)
      model.likelihood.phase_unc ([float,float])
      model.likelihood.learn_noise_scale (bool)

    Filters:
      model.filters.dtimes.{remove_duplicates,max_abs_input_dt,dtime_thin_frac,flip_dt_sign,cc_min}
      model.filters.events.{min_dtimes,min_unique_phase_per_event,min_dtimes_per_pair,min_event_degree,min_events_per_cluster,max_pair_station_ratio,ratio_filter_phase}
      model.filters.residual.{enabled,method,mad_sigma,abs_max} (other fields may be null if enabled=false)

    Batching:
      inference.batching.standard.{warmup,sgld} (ints)
      inference.batching.event_batches.{enabled,events_per_batch,max_edges_per_batch,bucket_reorder_all,bucket_reuse_epochs}
        - if enabled=false, numeric fields may be null (but must be present)
    """
    forbidden_present = [k for k in _FORBIDDEN_BLOCK3_TOPLEVEL_KEYS if k in params]
    if forbidden_present:
        raise ValueError(
            "Legacy Block-3 keys are not allowed. Move these under nested blocks:\n"
            + "\n".join(f"- {k}" for k in sorted(forbidden_present))
        )

    # Hard break: these blocks moved under `model.*` / `inference.*`.
    if "likelihood" in params:
        raise _err("likelihood", "moved; put this under `model.likelihood` (top-level `likelihood` is no longer supported)")
    if "filters" in params:
        raise _err("filters", "moved; put this under `model.filters` (top-level `filters` is no longer supported)")
    if "batching" in params:
        raise _err("batching", "moved; put this under `inference.batching` (top-level `batching` is no longer supported)")

    model = _require_dict(_require(params, "model", "model"), "model")
    inf = _require_dict(_require(params, "inference", "inference"), "inference")

    # ---- likelihood ----
    lk = _require_dict(_require(model, "likelihood", "model"), "model.likelihood")
    lk_type = _require_str(_require(lk, "type", "model.likelihood"), "model.likelihood.type").lower().strip()
    # Accept common aliases; `compute_likelihood_loss` handles the mapping.
    if lk_type in {"student-t", "studentt"}:
        lk_type = "student_t"
    # Convenience alias for the collapsed correlated likelihood (shared_event_re).
    lk_correlated = False
    if lk_type in {"correlated", "correlated_gaussian"}:
        lk_correlated = True
        lk_type = "gaussian"
    if lk_type not in {"huber", "l2", "gaussian", "laplace", "l1", "mae", "mse", "student_t"}:
        raise _err(
            "model.likelihood.type",
            "supported: 'huber', 'l2'/'gaussian' (aliases: 'mse'), 'laplace' (aliases: 'l1','mae'), 'student_t', "
            "'correlated'/'correlated_gaussian'",
        )
    phase_unc = _require_float_list(_require(lk, "phase_unc", "model.likelihood"), "model.likelihood.phase_unc", length=2)
    # We keep only the fixed scalar phase uncertainty (phase_unc). Noise learning is removed.
    if "learn_noise_scale" in lk:
        raise _err("model.likelihood.learn_noise_scale", "removed; SPIDER now uses fixed `model.likelihood.phase_unc` only")
    # Internal legacy variable used by some older compatibility checks below.
    learn_noise_scale = False

    # Optional: Student-t likelihood parameters.
    #
    # NLL per obs (up to a constant) is:
    #   log(sigma) + (nu+1)/2 * log(1 + (r/sigma)^2 / nu)
    #
    # We keep nu fixed (not learned) for now.
    student_t_cfg = lk.get("student_t", None)
    student_t_nu = 4.0
    if student_t_cfg is not None:
        if not isinstance(student_t_cfg, dict):
            raise _err("model.likelihood.student_t", "expected object/dict or null")
        if "nu" in student_t_cfg and student_t_cfg.get("nu", None) is not None:
            student_t_nu = float(_require_num(student_t_cfg.get("nu"), "model.likelihood.student_t.nu"))
            if (not math.isfinite(student_t_nu)) or (not (student_t_nu > 0.0)):
                raise _err("model.likelihood.student_t.nu", "must be finite and > 0")
    if lk_type == "student_t":
        if not isinstance(student_t_cfg, dict):
            raise _err("model.likelihood.student_t", "required when model.likelihood.type='student_t'")
        if ("nu" not in student_t_cfg) or (student_t_cfg.get("nu", None) is None):
            raise _err("model.likelihood.student_t.nu", "required when model.likelihood.type='student_t'")

    # Optional: Student-t scale-mixture (per-row lambda) for robust likelihoods.
    student_t_scale_cfg = lk.get("student_t_scale", None)
    student_t_scale_enabled = False
    student_t_scale_nu = float(student_t_nu)
    student_t_scale_update_every = 1
    student_t_scale_batch_size = 200_000
    student_t_scale_init = "ones"
    student_t_scale_min_lambda = 1e-6
    student_t_scale_max_lambda = 1e6
    if student_t_scale_cfg is not None:
        if not isinstance(student_t_scale_cfg, dict):
            raise _err("model.likelihood.student_t_scale", "expected object/dict or null")
        if "enabled" in student_t_scale_cfg and student_t_scale_cfg.get("enabled", None) is not None:
            student_t_scale_enabled = _require_bool(student_t_scale_cfg.get("enabled"), "model.likelihood.student_t_scale.enabled")
        if "nu" in student_t_scale_cfg and student_t_scale_cfg.get("nu", None) is not None:
            student_t_scale_nu = float(_require_num(student_t_scale_cfg.get("nu"), "model.likelihood.student_t_scale.nu"))
        if "update_every_epochs" in student_t_scale_cfg and student_t_scale_cfg.get("update_every_epochs", None) is not None:
            student_t_scale_update_every = int(_require_num(student_t_scale_cfg.get("update_every_epochs"), "model.likelihood.student_t_scale.update_every_epochs"))
        if "batch_size" in student_t_scale_cfg and student_t_scale_cfg.get("batch_size", None) is not None:
            student_t_scale_batch_size = int(_require_num(student_t_scale_cfg.get("batch_size"), "model.likelihood.student_t_scale.batch_size"))
        if "init" in student_t_scale_cfg and student_t_scale_cfg.get("init", None) is not None:
            student_t_scale_init = _require_str(student_t_scale_cfg.get("init"), "model.likelihood.student_t_scale.init")
        if "min_lambda" in student_t_scale_cfg and student_t_scale_cfg.get("min_lambda", None) is not None:
            student_t_scale_min_lambda = float(_require_num(student_t_scale_cfg.get("min_lambda"), "model.likelihood.student_t_scale.min_lambda"))
        if "max_lambda" in student_t_scale_cfg and student_t_scale_cfg.get("max_lambda", None) is not None:
            student_t_scale_max_lambda = float(_require_num(student_t_scale_cfg.get("max_lambda"), "model.likelihood.student_t_scale.max_lambda"))
    if student_t_scale_enabled:
        if lk_type not in {"gaussian", "l2", "mse"}:
            raise _err(
                "model.likelihood.type",
                "must be 'gaussian'/'l2'/'mse' when model.likelihood.student_t_scale.enabled=true "
                "(scale-mixture is applied on top of Gaussian noise)",
            )
        if (not math.isfinite(student_t_scale_nu)) or (not (student_t_scale_nu > 0.0)):
            raise _err("model.likelihood.student_t_scale.nu", "must be finite and > 0")
        if student_t_scale_update_every < 0:
            raise _err("model.likelihood.student_t_scale.update_every_epochs", "must be >= 0")
        if student_t_scale_batch_size <= 0:
            raise _err("model.likelihood.student_t_scale.batch_size", "must be > 0")
        if (not math.isfinite(student_t_scale_min_lambda)) or (student_t_scale_min_lambda <= 0.0):
            raise _err("model.likelihood.student_t_scale.min_lambda", "must be finite and > 0")
        if (not math.isfinite(student_t_scale_max_lambda)) or (student_t_scale_max_lambda <= 0.0):
            raise _err("model.likelihood.student_t_scale.max_lambda", "must be finite and > 0")
        if student_t_scale_max_lambda < student_t_scale_min_lambda:
            raise _err("model.likelihood.student_t_scale.max_lambda", "must be >= min_lambda")

    # Tempering removed (start fresh; keep core residual distributions only).
    if "tempering" in lk:
        raise _err("model.likelihood.tempering", "removed; delete this block from your config")

    # Heteroscedastic sigma inflation removed (start fresh).
    if "sigma_inflation" in lk:
        raise _err("model.likelihood.sigma_inflation", "removed; delete this block from your config")

    # Residual-correlation block removed entirely (no backward compatibility).
    if "residual_correlation" in lk:
        raise _err("model.likelihood.residual_correlation", "removed; structured residual correlation models are no longer supported")

    # Structured likelihood components removed (start fresh).
    if "shared_event_latent" in lk:
        raise _err("model.likelihood.shared_event_latent", "removed; delete this block from your config")
    # slowness_re is supported (scalar slowness random effects).
    if "latent_field" in lk:
        raise _err("model.likelihood.latent_field", "removed; delete this block from your config")

    # Optional: correlated forward-model error latent (low-rank station basis × radius-subsampled GMRF).
    #
    # Models residual correlations as:
    #   r_e = (w_s · (b_j - b_i)) + eps
    # where w_s is a fixed station basis vector and b_i is an event latent vector with a GMRF prior.
    corr_cfg = lk.get("corr_error", None)
    corr_enabled = False
    corr_r = 0
    corr_tau_ps = [0.0, 0.0]  # seconds
    corr_rho_ps = 0.0
    # Optional: hierarchical Gibbs update for tau (P/S covariance).
    hier_tau_enabled = False
    hier_tau_dof = 10.0
    hier_tau_scale_ps = [0.01, 0.01]  # prior std in seconds
    hier_tau_update_every = 5
    hier_tau_damping = 0.2
    hier_tau_start_after = 0
    hier_tau_min_ps = None
    hier_tau_max_ps = None
    # Station basis (fixed, from station geometry)
    corr_sta_ell_km = 0.0
    corr_sta_jitter = 1e-6
    corr_sta_method = "eigh_rbf"
    corr_sta_basis_enabled = True
    # Event graph (radius-r with capped-k uniform neighbor sampling; refreshed periodically)
    corr_graph_source = "geometry"  # 'geometry' (default) or 'dtimes'
    corr_graph_enabled = True
    corr_radius_km = 0.0
    corr_k = 16
    corr_refresh_every = 10
    corr_symmetrize = True
    corr_cell_size_km = None  # default: radius_km
    corr_cell_hops = 2
    corr_max_tries_per_neighbor = 64
    corr_q_diag = 1e-3
    # Optional edge-weighting scheme for the Laplacian smoothness term.
    # Default preserves historical behavior: per-node uniform weights (random-walk normalization).
    corr_weighting = "uniform_degree"  # 'uniform_degree' | 'rbf' | 'inv_dist'
    corr_weight_ell_km = None          # required for 'rbf'
    corr_weight_eps_km = 1e-3          # used for 'inv_dist'
    corr_weight_normalize = True       # normalize outgoing weights to sum to 1 per node
    # Optional: enable corr_error during Phase-1 MAP (locate-map).
    # Default False for backward compatibility.
    corr_enable_in_phase1 = False
    # Optional: LR multiplier for corr_error_b when optimized with Adam in Phase 1.
    corr_phase1_lr_mult = 0.1
    # Optional: exact elliptical slice sampling (ESS) updates for corr_error_b (blocked sampler step).
    # This is intended for robust likelihoods (Huber/Student/Laplace) where b|ΔX is not Gaussian,
    # but b has a Gaussian prior.
    corr_ess_enabled = False
    corr_ess_update_every = 1
    corr_ess_start_after = 0
    corr_ess_sweeps = 1
    corr_ess_batch_size = 50_000
    corr_ess_max_bracket_steps = 64
    corr_ess_block_by_station = True
    corr_ess_top_k_stations = 0  # 0 => all
    corr_ess_seed = 0
    corr_ess_freeze_sampler_group = True
    if isinstance(corr_cfg, dict):
        corr_enabled = bool(corr_cfg.get("enabled", False))
        if "r" in corr_cfg and corr_cfg.get("r", None) is not None:
            corr_r = int(_require_num(corr_cfg.get("r"), "model.likelihood.corr_error.r"))
            if corr_r < 1:
                raise _err("model.likelihood.corr_error.r", "must be >= 1")
        if "tau_s" in corr_cfg and corr_cfg.get("tau_s", None) is not None:
            v = corr_cfg.get("tau_s")
            if isinstance(v, (int, float)):
                f = float(v)
                corr_tau_ps = [f, f]
            elif isinstance(v, list):
                corr_tau_ps = _require_float_list(v, "model.likelihood.corr_error.tau_s", length=2)
            else:
                raise _err("model.likelihood.corr_error.tau_s", f"expected number or [P,S] list, got {type(v).__name__}")
        if not (corr_tau_ps[0] >= 0.0 and corr_tau_ps[1] >= 0.0):
            raise _err("model.likelihood.corr_error.tau_s", "must be >= 0")
        if "rho_ps" in corr_cfg and corr_cfg.get("rho_ps", None) is not None:
            corr_rho_ps = float(_require_num(corr_cfg.get("rho_ps"), "model.likelihood.corr_error.rho_ps"))
            if not (-0.999 < corr_rho_ps < 0.999):
                raise _err("model.likelihood.corr_error.rho_ps", "must satisfy -0.999 < rho_ps < 0.999")

        sta = corr_cfg.get("station_basis", None)
        if corr_enabled and (not isinstance(sta, dict)):
            raise _err("model.likelihood.corr_error.station_basis", "required when corr_error.enabled=true (expected object/dict)")
        if isinstance(sta, dict):
            if "enabled" in sta and sta.get("enabled", None) is not None:
                corr_sta_basis_enabled = bool(_require_bool(sta.get("enabled"), "model.likelihood.corr_error.station_basis.enabled"))
            if corr_sta_basis_enabled:
                corr_sta_ell_km = float(_require_num(_require(sta, "ell_km", "model.likelihood.corr_error.station_basis"), "model.likelihood.corr_error.station_basis.ell_km"))
                if not (math.isfinite(corr_sta_ell_km) and corr_sta_ell_km > 0.0):
                    raise _err("model.likelihood.corr_error.station_basis.ell_km", "must be finite and > 0")
                if "jitter" in sta and sta.get("jitter", None) is not None:
                    corr_sta_jitter = float(_require_num(sta.get("jitter"), "model.likelihood.corr_error.station_basis.jitter"))
                    if not (math.isfinite(corr_sta_jitter) and corr_sta_jitter >= 0.0):
                        raise _err("model.likelihood.corr_error.station_basis.jitter", "must be finite and >= 0")
                if "method" in sta and sta.get("method", None) is not None:
                    corr_sta_method = str(sta.get("method", "eigh_rbf")).strip().lower()
                if corr_sta_method not in {"eigh_rbf"}:
                    raise _err("model.likelihood.corr_error.station_basis.method", "supported: 'eigh_rbf'")
            else:
                # Per-station coefficients mode: no geometry basis. The runtime will require corr_error.r == n_stations.
                corr_sta_ell_km = 0.0
                corr_sta_jitter = 0.0
                corr_sta_method = "none"

        eg = corr_cfg.get("event_graph", None)
        if corr_enabled and (not isinstance(eg, dict)):
            raise _err("model.likelihood.corr_error.event_graph", "required when corr_error.enabled=true (expected object/dict)")
        if isinstance(eg, dict):
            # Allow disabling the event graph entirely (IID Gaussian prior on b).
            # When disabled, we skip graph parameter requirements and use Q = I in the corr_error prior,
            # so tau_s is interpretable as the per-(event,basis,phase) std in seconds.
            if "enabled" in eg and eg.get("enabled", None) is not None:
                corr_graph_enabled = bool(_require_bool(eg.get("enabled"), "model.likelihood.corr_error.event_graph.enabled"))

            # Hard break: old kNN-weighted graph keys are removed (no backward compatibility).
            if "knn" in eg:
                raise _err("model.likelihood.corr_error.event_graph.knn", "removed; use radius_km + k")
            if "ell_km" in eg:
                raise _err("model.likelihood.corr_error.event_graph.ell_km", "removed; use radius_km (neighbors are uniform-in-radius with unity weights)")
            if "max_edges_per_step" in eg:
                raise _err("model.likelihood.corr_error.event_graph.max_edges_per_step", "removed; use refresh_every_epochs to control stochastic graph updates")

            if corr_graph_enabled:
                if "source" in eg and eg.get("source", None) is not None:
                    corr_graph_source = str(eg.get("source")).strip().lower()
                    if corr_graph_source not in {"geometry", "dtimes"}:
                        raise _err("model.likelihood.corr_error.event_graph.source", "supported: 'geometry', 'dtimes'")

                # radius_km is required for the geometry graph; for dtimes graphs it is optional
                # (if provided, it is used as an optional distance filter when coordinates are available).
                if corr_graph_source != "dtimes":
                    corr_radius_km = float(_require_num(_require(eg, "radius_km", "model.likelihood.corr_error.event_graph"), "model.likelihood.corr_error.event_graph.radius_km"))
                    if not (math.isfinite(corr_radius_km) and corr_radius_km > 0.0):
                        raise _err("model.likelihood.corr_error.event_graph.radius_km", "must be finite and > 0")
                else:
                    if "radius_km" in eg and eg.get("radius_km", None) is not None:
                        corr_radius_km = float(_require_num(eg.get("radius_km"), "model.likelihood.corr_error.event_graph.radius_km"))
                        if not math.isfinite(corr_radius_km):
                            raise _err("model.likelihood.corr_error.event_graph.radius_km", "must be finite when provided")
                    else:
                        corr_radius_km = 0.0

                corr_k = int(_require_num(_require(eg, "k", "model.likelihood.corr_error.event_graph"), "model.likelihood.corr_error.event_graph.k"))
                if corr_k < 1:
                    raise _err("model.likelihood.corr_error.event_graph.k", "must be >= 1")
                if "refresh_every_epochs" in eg and eg.get("refresh_every_epochs", None) is not None:
                    corr_refresh_every = int(_require_num(eg.get("refresh_every_epochs"), "model.likelihood.corr_error.event_graph.refresh_every_epochs"))
                    if corr_refresh_every < 0:
                        raise _err("model.likelihood.corr_error.event_graph.refresh_every_epochs", "must be >= 0 (0 disables refresh)")
                if "symmetrize" in eg and eg.get("symmetrize", None) is not None:
                    corr_symmetrize = bool(_require_bool(eg.get("symmetrize"), "model.likelihood.corr_error.event_graph.symmetrize"))
                if "cell_size_km" in eg and eg.get("cell_size_km", None) is not None:
                    corr_cell_size_km = float(_require_num(eg.get("cell_size_km"), "model.likelihood.corr_error.event_graph.cell_size_km"))
                    if not (math.isfinite(corr_cell_size_km) and corr_cell_size_km > 0.0):
                        raise _err("model.likelihood.corr_error.event_graph.cell_size_km", "must be finite and > 0")
                if "cell_hops" in eg and eg.get("cell_hops", None) is not None:
                    corr_cell_hops = int(_require_num(eg.get("cell_hops"), "model.likelihood.corr_error.event_graph.cell_hops"))
                    if corr_cell_hops < 1:
                        raise _err("model.likelihood.corr_error.event_graph.cell_hops", "must be >= 1")
                if "max_tries_per_neighbor" in eg and eg.get("max_tries_per_neighbor", None) is not None:
                    corr_max_tries_per_neighbor = int(_require_num(eg.get("max_tries_per_neighbor"), "model.likelihood.corr_error.event_graph.max_tries_per_neighbor"))
                    if corr_max_tries_per_neighbor < 1:
                        raise _err("model.likelihood.corr_error.event_graph.max_tries_per_neighbor", "must be >= 1")
                if "q_diag" in eg and eg.get("q_diag", None) is not None:
                    corr_q_diag = float(_require_num(eg.get("q_diag"), "model.likelihood.corr_error.event_graph.q_diag"))
                    if not (math.isfinite(corr_q_diag) and corr_q_diag >= 0.0):
                        raise _err("model.likelihood.corr_error.event_graph.q_diag", "must be finite and >= 0")
                if "weighting" in eg and eg.get("weighting", None) is not None:
                    corr_weighting = str(eg.get("weighting")).strip().lower()
                    if corr_weighting in {"degree", "deg", "uniform"}:
                        corr_weighting = "uniform_degree"
                    if corr_weighting not in {"uniform_degree", "rbf", "inv_dist"}:
                        raise _err("model.likelihood.corr_error.event_graph.weighting", "supported: 'uniform_degree', 'rbf', 'inv_dist'")
                if "weight_ell_km" in eg and eg.get("weight_ell_km", None) is not None:
                    corr_weight_ell_km = float(_require_num(eg.get("weight_ell_km"), "model.likelihood.corr_error.event_graph.weight_ell_km"))
                    if not (math.isfinite(corr_weight_ell_km) and corr_weight_ell_km > 0.0):
                        raise _err("model.likelihood.corr_error.event_graph.weight_ell_km", "must be finite and > 0")
                if "weight_eps_km" in eg and eg.get("weight_eps_km", None) is not None:
                    corr_weight_eps_km = float(_require_num(eg.get("weight_eps_km"), "model.likelihood.corr_error.event_graph.weight_eps_km"))
                    if not (math.isfinite(corr_weight_eps_km) and corr_weight_eps_km > 0.0):
                        raise _err("model.likelihood.corr_error.event_graph.weight_eps_km", "must be finite and > 0")
                if "weight_normalize" in eg and eg.get("weight_normalize", None) is not None:
                    corr_weight_normalize = bool(_require_bool(eg.get("weight_normalize"), "model.likelihood.corr_error.event_graph.weight_normalize"))
            else:
                # IID mode: set dummy graph params (not used), and use Q = I in the prior.
                corr_graph_source = "none"
                corr_radius_km = 0.0
                corr_k = 0
                corr_refresh_every = 0
                corr_symmetrize = True
                corr_cell_size_km = None
                corr_cell_hops = 1
                corr_max_tries_per_neighbor = 1
                corr_q_diag = 1.0
                corr_weighting = "uniform_degree"
                corr_weight_ell_km = None
                corr_weight_eps_km = 1e-3
                corr_weight_normalize = True

        if corr_enabled:
            if corr_r <= 0:
                raise _err("model.likelihood.corr_error.r", "required and must be >= 1 when corr_error.enabled=true")
            if corr_graph_enabled and (corr_graph_source != "dtimes"):
                if not (corr_radius_km > 0.0):
                    raise _err("model.likelihood.corr_error.event_graph.radius_km", "required and must be > 0 when corr_error.enabled=true")

        if "enable_in_phase1" in corr_cfg and corr_cfg.get("enable_in_phase1", None) is not None:
            corr_enable_in_phase1 = _require_bool(corr_cfg.get("enable_in_phase1"), "model.likelihood.corr_error.enable_in_phase1")
        if "phase1_lr_mult" in corr_cfg and corr_cfg.get("phase1_lr_mult", None) is not None:
            corr_phase1_lr_mult = float(_require_num(corr_cfg.get("phase1_lr_mult"), "model.likelihood.corr_error.phase1_lr_mult"))
            if not (math.isfinite(corr_phase1_lr_mult) and corr_phase1_lr_mult > 0.0):
                raise _err("model.likelihood.corr_error.phase1_lr_mult", "must be finite and > 0")

        # Optional: hierarchical Gibbs update for tau (P/S covariance).
        hier_tau_cfg = corr_cfg.get("hierarchical", None)
        if isinstance(hier_tau_cfg, dict):
            hier_tau_enabled = bool(hier_tau_cfg.get("enabled", False))
            if "tau_prior_dof" in hier_tau_cfg and hier_tau_cfg.get("tau_prior_dof", None) is not None:
                hier_tau_dof = float(_require_num(hier_tau_cfg.get("tau_prior_dof"), "model.likelihood.corr_error.hierarchical.tau_prior_dof"))
                if not (hier_tau_dof > 1.0):
                    raise _err("model.likelihood.corr_error.hierarchical.tau_prior_dof", "must be > 1.0")
            if "tau_prior_scale" in hier_tau_cfg and hier_tau_cfg.get("tau_prior_scale", None) is not None:
                v = hier_tau_cfg.get("tau_prior_scale")
                if isinstance(v, (int, float)):
                    f = float(v)
                    hier_tau_scale_ps = [f, f]
                elif isinstance(v, list):
                    hier_tau_scale_ps = _require_float_list(v, "model.likelihood.corr_error.hierarchical.tau_prior_scale", length=2)
                else:
                    raise _err("model.likelihood.corr_error.hierarchical.tau_prior_scale", f"expected number or [P,S] list, got {type(v).__name__}")
            if "update_every_epochs" in hier_tau_cfg and hier_tau_cfg.get("update_every_epochs", None) is not None:
                hier_tau_update_every = int(_require_num(hier_tau_cfg.get("update_every_epochs"), "model.likelihood.corr_error.hierarchical.update_every_epochs"))
                if hier_tau_update_every < 1:
                    raise _err("model.likelihood.corr_error.hierarchical.update_every_epochs", "must be >= 1")
            if "damping" in hier_tau_cfg and hier_tau_cfg.get("damping", None) is not None:
                hier_tau_damping = float(_require_num(hier_tau_cfg.get("damping"), "model.likelihood.corr_error.hierarchical.damping"))
                if not (math.isfinite(hier_tau_damping) and (0.0 < hier_tau_damping <= 1.0)):
                    raise _err("model.likelihood.corr_error.hierarchical.damping", "must satisfy 0 < damping <= 1")
            if "start_after_epochs" in hier_tau_cfg and hier_tau_cfg.get("start_after_epochs", None) is not None:
                hier_tau_start_after = int(_require_num(hier_tau_cfg.get("start_after_epochs"), "model.likelihood.corr_error.hierarchical.start_after_epochs"))
                if hier_tau_start_after < 0:
                    raise _err("model.likelihood.corr_error.hierarchical.start_after_epochs", "must be >= 0")
            # Optional safety clamps for the learned tau (seconds).
            if "min_tau_s" in hier_tau_cfg and hier_tau_cfg.get("min_tau_s", None) is not None:
                v = hier_tau_cfg.get("min_tau_s")
                if isinstance(v, (int, float)):
                    f = float(v)
                    hier_tau_min_ps = [f, f]
                elif isinstance(v, list):
                    hier_tau_min_ps = _require_float_list(v, "model.likelihood.corr_error.hierarchical.min_tau_s", length=2)
                else:
                    raise _err("model.likelihood.corr_error.hierarchical.min_tau_s", f"expected number or [P,S] list, got {type(v).__name__}")
                if not (hier_tau_min_ps[0] >= 0.0 and hier_tau_min_ps[1] >= 0.0):
                    raise _err("model.likelihood.corr_error.hierarchical.min_tau_s", "must be >= 0")
            if "max_tau_s" in hier_tau_cfg and hier_tau_cfg.get("max_tau_s", None) is not None:
                v = hier_tau_cfg.get("max_tau_s")
                if isinstance(v, (int, float)):
                    f = float(v)
                    hier_tau_max_ps = [f, f]
                elif isinstance(v, list):
                    hier_tau_max_ps = _require_float_list(v, "model.likelihood.corr_error.hierarchical.max_tau_s", length=2)
                else:
                    raise _err("model.likelihood.corr_error.hierarchical.max_tau_s", f"expected number or [P,S] list, got {type(v).__name__}")
                if not (hier_tau_max_ps[0] > 0.0 and hier_tau_max_ps[1] > 0.0):
                    raise _err("model.likelihood.corr_error.hierarchical.max_tau_s", "must be > 0")

        # Optional: ESS updates for corr_error_b (blocked sampler step).
        ess_cfg = corr_cfg.get("ess", None)
        if isinstance(ess_cfg, dict):
            if "enabled" in ess_cfg and ess_cfg.get("enabled", None) is not None:
                corr_ess_enabled = bool(_require_bool(ess_cfg.get("enabled"), "model.likelihood.corr_error.ess.enabled"))
            if "update_every_epochs" in ess_cfg and ess_cfg.get("update_every_epochs", None) is not None:
                corr_ess_update_every = int(_require_num(ess_cfg.get("update_every_epochs"), "model.likelihood.corr_error.ess.update_every_epochs"))
                if corr_ess_update_every < 1:
                    raise _err("model.likelihood.corr_error.ess.update_every_epochs", "must be >= 1")
            if "start_after_epochs" in ess_cfg and ess_cfg.get("start_after_epochs", None) is not None:
                corr_ess_start_after = int(_require_num(ess_cfg.get("start_after_epochs"), "model.likelihood.corr_error.ess.start_after_epochs"))
                if corr_ess_start_after < 0:
                    raise _err("model.likelihood.corr_error.ess.start_after_epochs", "must be >= 0")
            if "sweeps_per_update" in ess_cfg and ess_cfg.get("sweeps_per_update", None) is not None:
                corr_ess_sweeps = int(_require_num(ess_cfg.get("sweeps_per_update"), "model.likelihood.corr_error.ess.sweeps_per_update"))
                if corr_ess_sweeps < 1:
                    raise _err("model.likelihood.corr_error.ess.sweeps_per_update", "must be >= 1")
            if "batch_size" in ess_cfg and ess_cfg.get("batch_size", None) is not None:
                corr_ess_batch_size = int(_require_num(ess_cfg.get("batch_size"), "model.likelihood.corr_error.ess.batch_size"))
                if corr_ess_batch_size < 1:
                    raise _err("model.likelihood.corr_error.ess.batch_size", "must be >= 1")
            if "max_bracket_steps" in ess_cfg and ess_cfg.get("max_bracket_steps", None) is not None:
                corr_ess_max_bracket_steps = int(_require_num(ess_cfg.get("max_bracket_steps"), "model.likelihood.corr_error.ess.max_bracket_steps"))
                if corr_ess_max_bracket_steps < 8:
                    raise _err("model.likelihood.corr_error.ess.max_bracket_steps", "must be >= 8")
            if "block_by_station" in ess_cfg and ess_cfg.get("block_by_station", None) is not None:
                corr_ess_block_by_station = bool(_require_bool(ess_cfg.get("block_by_station"), "model.likelihood.corr_error.ess.block_by_station"))
            if "top_k_stations" in ess_cfg and ess_cfg.get("top_k_stations", None) is not None:
                corr_ess_top_k_stations = int(_require_num(ess_cfg.get("top_k_stations"), "model.likelihood.corr_error.ess.top_k_stations"))
                if corr_ess_top_k_stations < 0:
                    raise _err("model.likelihood.corr_error.ess.top_k_stations", "must be >= 0")
            if "seed" in ess_cfg and ess_cfg.get("seed", None) is not None:
                corr_ess_seed = int(_require_num(ess_cfg.get("seed"), "model.likelihood.corr_error.ess.seed"))
            if "freeze_sampler_group" in ess_cfg and ess_cfg.get("freeze_sampler_group", None) is not None:
                corr_ess_freeze_sampler_group = bool(_require_bool(ess_cfg.get("freeze_sampler_group"), "model.likelihood.corr_error.ess.freeze_sampler_group"))

    # Optional: collapsed shared-event random-effects likelihood (Gaussian, marginalized; no latent state).
    # This is intended to address "broken independence" from shared-event correlations while targeting
    # an (approximately) correct marginal posterior over (X,Y,Z,T).
    #
    # Model (per group):
    #   r_e = (b_j - b_i) + eps_e,  b_i ~ N(0, tau^2),  eps_e ~ N(0, sigma_e^2)
    # and we integrate b out exactly, yielding a correlated Gaussian likelihood on r.
    #
    # This block is intentionally OPTIONAL (absent -> disabled) to avoid breaking existing configs.
    se_cfg = lk.get("shared_event_re", None)
    se_enabled = False
    se_grouping = "phase"  # 'phase' or 'station_phase'
    se_cluster_mode = "none"  # 'none' or 'dd_khop' or 'component'
    se_cluster_k = 1
    se_tau_ps = [0.0, 0.0]  # std in seconds for [P,S]; 0 disables (iid)
    # Optional: hierarchical (cluster + event) random effects
    se_hier_enabled = False
    se_tau_event_ps = [0.0, 0.0]
    se_tau_cluster_ps = [0.0, 0.0]
    se_joint_ps = False
    se_rho_ps = 0.0
    se_max_nodes_per_group = 512
    se_max_rows_per_group = 200000
    se_fallback_to_diag = True
    se_jitter0 = 1e-8
    se_jitter_max = 1e-3
    # Optional: internal runtime controls for station_phase caching/debugging
    se_cache_max_entries = 4096
    se_cache_log_every = 0
    # Optional: GPU path controls (shared_event_re GPU batched PCG prototype)
    se_gpu_enable = None
    se_gpu_max_groups_per_batch = 64
    se_gpu_profile = False
    se_gpu_debug_max_groups = 0
    se_gpu_max_edges_per_batch = 0
    se_gpu_reuse_pcg_init = False
    # Optional: whitening operator for shared_event_re (static covariance).
    se_whiten_enabled = False
    se_whiten_edge_weighting = "uniform"
    se_whiten_edge_weight_ell_km = 1.0
    se_whiten_edge_weight_eps_km = 1e-3
    se_whiten_edge_weight_power = 1.0
    se_whiten_edge_weight_scale_km = 1.0
    se_whiten_edge_weight_global_scale = 1.0
    se_whiten_edge_weight_normalize = False
    se_whiten_cache_max_entries = 8
    se_whiten_solver = "pcg"
    se_whiten_pcg_max_iters = 200
    se_whiten_pcg_tol = 1e-6
    se_whiten_pcg_min_iters = 0
    se_whiten_precompute = False
    se_whiten_precompute_device = "gpu"
    se_whiten_pcg_batched = False
    se_whiten_pcg_bucket_nodes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    # Optional: auto-tune caps if fallbacks occur
    se_auto_tune_nodes_cap = False
    se_auto_tune_nodes_max = 20000
    se_auto_tune_rows_cap = False
    se_auto_tune_rows_max = 2000000
    # Optional: solver selection for shared_event_re
    # Phase A implements `pcg_sparse` (quadratic-only; logdet dropped). A dense/exact solver may be
    # added later for small groups (Phase B/C work).
    se_solver = "pcg_sparse"  # 'pcg_sparse' (quadratic-only; logdet dropped)
    se_drop_logdet = True
    se_pcg_max_iters = 50
    se_pcg_tol = 1e-3
    # Optional: diagnostics logging for shared_event_re (cheap subsample stats for W&B)
    se_diag_log_every_epochs = 0
    se_diag_max_groups = 8
    se_diag_max_rows_per_group = 2048
    se_diag_max_nodes = 1024
    se_diag_seed = 0
    # Optional: console stats logging cadence for shared_event_re
    se_stats_log_every_epochs = 0
    # Optional: collapsed station-phase random effects (additive)
    se_sp_enabled = False
    se_sp_tau_ps = [0.0, 0.0]
    if isinstance(se_cfg, dict):
        se_enabled = bool(se_cfg.get("enabled", False))
        if "grouping" in se_cfg and se_cfg["grouping"] is not None:
            se_grouping = str(se_cfg.get("grouping", "phase")).strip().lower()
        if se_grouping in {"stationphase", "station-phase"}:
            se_grouping = "station_phase"
        if se_grouping not in {"phase", "station_phase"}:
            raise _err("model.likelihood.shared_event_re.grouping", "supported: 'phase', 'station_phase'")
        if "cluster_mode" in se_cfg and se_cfg["cluster_mode"] is not None:
            se_cluster_mode = str(se_cfg.get("cluster_mode", se_cluster_mode)).strip().lower()
        if se_cluster_mode in {"dd_khop", "dd-khop", "khop"}:
            se_cluster_mode = "dd_khop"
        if se_cluster_mode in {"component", "connected_component", "connected-components", "connectedcomponents"}:
            se_cluster_mode = "component"
        if se_cluster_mode not in {"none", "dd_khop", "component"}:
            raise _err("model.likelihood.shared_event_re.cluster_mode", "supported: 'none', 'dd_khop', 'component'")
        if "cluster_k" in se_cfg and se_cfg["cluster_k"] is not None:
            se_cluster_k = int(_require_num(se_cfg.get("cluster_k"), "model.likelihood.shared_event_re.cluster_k"))
            if se_cluster_k < 0:
                raise _err("model.likelihood.shared_event_re.cluster_k", "must be >= 0")

        if "tau_s" in se_cfg and se_cfg["tau_s"] is not None:
            v = se_cfg["tau_s"]
            if isinstance(v, (int, float)):
                f = float(v)
                se_tau_ps = [f, f]
            elif isinstance(v, list):
                se_tau_ps = _require_float_list(v, "likelihood.shared_event_re.tau_s", length=2)
            else:
                raise _err("model.likelihood.shared_event_re.tau_s", f"expected number or [P,S] list, got {type(v).__name__}")
        if not (se_tau_ps[0] >= 0.0 and se_tau_ps[1] >= 0.0):
            raise _err("model.likelihood.shared_event_re.tau_s", "must be >= 0")

        if "hierarchical" in se_cfg and se_cfg["hierarchical"] is not None:
            se_hier_enabled = bool(_require_bool(se_cfg.get("hierarchical"), "model.likelihood.shared_event_re.hierarchical"))
        if "tau_event_s" in se_cfg and se_cfg["tau_event_s"] is not None:
            v = se_cfg["tau_event_s"]
            if isinstance(v, (int, float)):
                f = float(v)
                se_tau_event_ps = [f, f]
            elif isinstance(v, list):
                se_tau_event_ps = _require_float_list(v, "likelihood.shared_event_re.tau_event_s", length=2)
            else:
                raise _err("model.likelihood.shared_event_re.tau_event_s", f"expected number or [P,S] list, got {type(v).__name__}")
        if "tau_cluster_s" in se_cfg and se_cfg["tau_cluster_s"] is not None:
            v = se_cfg["tau_cluster_s"]
            if isinstance(v, (int, float)):
                f = float(v)
                se_tau_cluster_ps = [f, f]
            elif isinstance(v, list):
                se_tau_cluster_ps = _require_float_list(v, "likelihood.shared_event_re.tau_cluster_s", length=2)
            else:
                raise _err("model.likelihood.shared_event_re.tau_cluster_s", f"expected number or [P,S] list, got {type(v).__name__}")
        if se_hier_enabled:
            if not (se_tau_event_ps[0] >= 0.0 and se_tau_event_ps[1] >= 0.0):
                raise _err("model.likelihood.shared_event_re.tau_event_s", "must be >= 0")
            if not (se_tau_cluster_ps[0] >= 0.0 and se_tau_cluster_ps[1] >= 0.0):
                raise _err("model.likelihood.shared_event_re.tau_cluster_s", "must be >= 0")

        # Optional: station-phase random effects (collapsed)
        sp_cfg = se_cfg.get("station_phase_re", None)
        if isinstance(sp_cfg, dict):
            se_sp_enabled = bool(sp_cfg.get("enabled", False))
            if "tau_s" in sp_cfg and sp_cfg.get("tau_s", None) is not None:
                v = sp_cfg.get("tau_s")
                if isinstance(v, (int, float)):
                    f = float(v)
                    se_sp_tau_ps = [f, f]
                elif isinstance(v, list):
                    se_sp_tau_ps = _require_float_list(v, "likelihood.shared_event_re.station_phase_re.tau_s", length=2)
                else:
                    raise _err(
                        "model.likelihood.shared_event_re.station_phase_re.tau_s",
                        f"expected number or [P,S] list, got {type(v).__name__}",
                    )
            if se_sp_enabled and (not (se_sp_tau_ps[0] >= 0.0 and se_sp_tau_ps[1] >= 0.0)):
                raise _err("model.likelihood.shared_event_re.station_phase_re.tau_s", "must be >= 0")

    if lk_correlated:
        if not isinstance(se_cfg, dict):
            raise _err(
                "model.likelihood.shared_event_re",
                "required when model.likelihood.type is 'correlated'/'correlated_gaussian'",
            )
        if not se_enabled:
            raise _err(
                "model.likelihood.shared_event_re.enabled",
                "must be true when model.likelihood.type is 'correlated'/'correlated_gaussian'",
            )
        if (not isinstance(se_cfg, dict)) or ("grouping" not in se_cfg):
            se_grouping = "station_phase"

        if "joint_ps" in se_cfg and se_cfg["joint_ps"] is not None:
            se_joint_ps = bool(se_cfg.get("joint_ps", False))
        if "rho_ps" in se_cfg and se_cfg["rho_ps"] is not None:
            se_rho_ps = float(_require_num(se_cfg["rho_ps"], "model.likelihood.shared_event_re.rho_ps"))
            if not (-0.999 < se_rho_ps < 0.999):
                raise _err("model.likelihood.shared_event_re.rho_ps", "must satisfy -0.999 < rho_ps < 0.999")

        if "max_nodes_per_group" in se_cfg and se_cfg["max_nodes_per_group"] is not None:
            se_max_nodes_per_group = int(_require_num(se_cfg["max_nodes_per_group"], "model.likelihood.shared_event_re.max_nodes_per_group"))
            if se_max_nodes_per_group < 2:
                raise _err("model.likelihood.shared_event_re.max_nodes_per_group", "must be >= 2")
        if "max_rows_per_group" in se_cfg and se_cfg["max_rows_per_group"] is not None:
            se_max_rows_per_group = int(_require_num(se_cfg["max_rows_per_group"], "model.likelihood.shared_event_re.max_rows_per_group"))
            if se_max_rows_per_group < 2:
                raise _err("model.likelihood.shared_event_re.max_rows_per_group", "must be >= 2")
        if "fallback_to_diag" in se_cfg and se_cfg["fallback_to_diag"] is not None:
            se_fallback_to_diag = bool(se_cfg.get("fallback_to_diag", True))
        if "jitter0" in se_cfg and se_cfg["jitter0"] is not None:
            se_jitter0 = float(_require_num(se_cfg["jitter0"], "model.likelihood.shared_event_re.jitter0"))
            if se_jitter0 <= 0.0:
                raise _err("model.likelihood.shared_event_re.jitter0", "must be > 0")
        if "jitter_max" in se_cfg and se_cfg["jitter_max"] is not None:
            se_jitter_max = float(_require_num(se_cfg["jitter_max"], "model.likelihood.shared_event_re.jitter_max"))
            if se_jitter_max <= 0.0:
                raise _err("model.likelihood.shared_event_re.jitter_max", "must be > 0")
        if se_jitter_max < se_jitter0:
            raise _err("model.likelihood.shared_event_re.jitter_max", "must be >= jitter0")

        if "cache_max_entries" in se_cfg and se_cfg["cache_max_entries"] is not None:
            se_cache_max_entries = int(_require_num(se_cfg["cache_max_entries"], "model.likelihood.shared_event_re.cache_max_entries"))
            if se_cache_max_entries < 0:
                raise _err("model.likelihood.shared_event_re.cache_max_entries", "must be >= 0")
        if "cache_log_every" in se_cfg and se_cfg["cache_log_every"] is not None:
            se_cache_log_every = int(_require_num(se_cfg["cache_log_every"], "model.likelihood.shared_event_re.cache_log_every"))
            if se_cache_log_every < 0:
                raise _err("model.likelihood.shared_event_re.cache_log_every", "must be >= 0")

        if "gpu_enable" in se_cfg and se_cfg["gpu_enable"] is not None:
            se_gpu_enable = bool(_require_bool(se_cfg.get("gpu_enable"), "model.likelihood.shared_event_re.gpu_enable"))
        if "gpu_max_groups_per_batch" in se_cfg and se_cfg["gpu_max_groups_per_batch"] is not None:
            se_gpu_max_groups_per_batch = int(
                _require_num(se_cfg.get("gpu_max_groups_per_batch"), "model.likelihood.shared_event_re.gpu_max_groups_per_batch")
            )
            if se_gpu_max_groups_per_batch < 1:
                raise _err("model.likelihood.shared_event_re.gpu_max_groups_per_batch", "must be >= 1")
        if "gpu_profile" in se_cfg and se_cfg["gpu_profile"] is not None:
            se_gpu_profile = bool(_require_bool(se_cfg.get("gpu_profile"), "model.likelihood.shared_event_re.gpu_profile"))
        if "gpu_debug_max_groups" in se_cfg and se_cfg["gpu_debug_max_groups"] is not None:
            se_gpu_debug_max_groups = int(
                _require_num(se_cfg.get("gpu_debug_max_groups"), "model.likelihood.shared_event_re.gpu_debug_max_groups")
            )
            if se_gpu_debug_max_groups < 0:
                raise _err("model.likelihood.shared_event_re.gpu_debug_max_groups", "must be >= 0")
        if "gpu_max_edges_per_batch" in se_cfg and se_cfg["gpu_max_edges_per_batch"] is not None:
            se_gpu_max_edges_per_batch = int(
                _require_num(se_cfg.get("gpu_max_edges_per_batch"), "model.likelihood.shared_event_re.gpu_max_edges_per_batch")
            )
            if se_gpu_max_edges_per_batch < 0:
                raise _err("model.likelihood.shared_event_re.gpu_max_edges_per_batch", "must be >= 0")
        if "gpu_reuse_pcg_init" in se_cfg and se_cfg["gpu_reuse_pcg_init"] is not None:
            se_gpu_reuse_pcg_init = bool(_require_bool(se_cfg.get("gpu_reuse_pcg_init"), "model.likelihood.shared_event_re.gpu_reuse_pcg_init"))
        if "stats_log_every_epochs" in se_cfg and se_cfg["stats_log_every_epochs"] is not None:
            se_stats_log_every_epochs = int(
                _require_num(se_cfg.get("stats_log_every_epochs"), "model.likelihood.shared_event_re.stats_log_every_epochs")
            )
            if se_stats_log_every_epochs < 0:
                raise _err("model.likelihood.shared_event_re.stats_log_every_epochs", "must be >= 0")
        whiten_cfg = se_cfg.get("whitening", None)
        if isinstance(whiten_cfg, dict):
            if "enabled" in whiten_cfg and whiten_cfg.get("enabled", None) is not None:
                se_whiten_enabled = bool(_require_bool(whiten_cfg.get("enabled"), "model.likelihood.shared_event_re.whitening.enabled"))
            if "solver" in whiten_cfg and whiten_cfg.get("solver", None) is not None:
                se_whiten_solver = str(whiten_cfg.get("solver", se_whiten_solver)).strip().lower()
                if se_whiten_solver not in {"chol", "pcg"}:
                    raise _err(
                        "model.likelihood.shared_event_re.whitening.solver",
                        "supported: 'chol', 'pcg'",
                    )
            if "edge_weighting" in whiten_cfg and whiten_cfg.get("edge_weighting", None) is not None:
                se_whiten_edge_weighting = str(whiten_cfg.get("edge_weighting", se_whiten_edge_weighting)).strip().lower()
            if se_whiten_edge_weighting not in {"uniform", "distance_rbf", "distance_linear", "distance_power"}:
                raise _err(
                    "model.likelihood.shared_event_re.whitening.edge_weighting",
                    "supported: 'uniform', 'distance_rbf', 'distance_linear', 'distance_power'",
                )
            if "edge_weight_ell_km" in whiten_cfg and whiten_cfg.get("edge_weight_ell_km", None) is not None:
                se_whiten_edge_weight_ell_km = float(
                    _require_num(whiten_cfg.get("edge_weight_ell_km"), "model.likelihood.shared_event_re.whitening.edge_weight_ell_km")
                )
                if not (se_whiten_edge_weight_ell_km > 0.0):
                    raise _err("model.likelihood.shared_event_re.whitening.edge_weight_ell_km", "must be > 0")
            if "edge_weight_eps_km" in whiten_cfg and whiten_cfg.get("edge_weight_eps_km", None) is not None:
                se_whiten_edge_weight_eps_km = float(
                    _require_num(whiten_cfg.get("edge_weight_eps_km"), "model.likelihood.shared_event_re.whitening.edge_weight_eps_km")
                )
                if not (se_whiten_edge_weight_eps_km >= 0.0):
                    raise _err("model.likelihood.shared_event_re.whitening.edge_weight_eps_km", "must be >= 0")
            if "edge_weight_power" in whiten_cfg and whiten_cfg.get("edge_weight_power", None) is not None:
                se_whiten_edge_weight_power = float(
                    _require_num(whiten_cfg.get("edge_weight_power"), "model.likelihood.shared_event_re.whitening.edge_weight_power")
                )
                if not (se_whiten_edge_weight_power > 0.0):
                    raise _err("model.likelihood.shared_event_re.whitening.edge_weight_power", "must be > 0")
            if "edge_weight_scale_km" in whiten_cfg and whiten_cfg.get("edge_weight_scale_km", None) is not None:
                se_whiten_edge_weight_scale_km = float(
                    _require_num(whiten_cfg.get("edge_weight_scale_km"), "model.likelihood.shared_event_re.whitening.edge_weight_scale_km")
                )
                if not (se_whiten_edge_weight_scale_km > 0.0):
                    raise _err("model.likelihood.shared_event_re.whitening.edge_weight_scale_km", "must be > 0")
            if "edge_weight_global_scale" in whiten_cfg and whiten_cfg.get("edge_weight_global_scale", None) is not None:
                se_whiten_edge_weight_global_scale = float(
                    _require_num(whiten_cfg.get("edge_weight_global_scale"), "model.likelihood.shared_event_re.whitening.edge_weight_global_scale")
                )
                if not (se_whiten_edge_weight_global_scale > 0.0):
                    raise _err("model.likelihood.shared_event_re.whitening.edge_weight_global_scale", "must be > 0")
            if "edge_weight_normalize" in whiten_cfg and whiten_cfg.get("edge_weight_normalize", None) is not None:
                se_whiten_edge_weight_normalize = bool(
                    _require_bool(whiten_cfg.get("edge_weight_normalize"), "model.likelihood.shared_event_re.whitening.edge_weight_normalize")
                )
            if "cache_max_entries" in whiten_cfg and whiten_cfg.get("cache_max_entries", None) is not None:
                se_whiten_cache_max_entries = int(
                    _require_num(whiten_cfg.get("cache_max_entries"), "model.likelihood.shared_event_re.whitening.cache_max_entries")
                )
                if se_whiten_cache_max_entries < 0:
                    raise _err("model.likelihood.shared_event_re.whitening.cache_max_entries", "must be >= 0")
            if "precompute" in whiten_cfg and whiten_cfg.get("precompute", None) is not None:
                se_whiten_precompute = bool(
                    _require_bool(whiten_cfg.get("precompute"), "model.likelihood.shared_event_re.whitening.precompute")
                )
            if "precompute_device" in whiten_cfg and whiten_cfg.get("precompute_device", None) is not None:
                se_whiten_precompute_device = str(
                    whiten_cfg.get("precompute_device", se_whiten_precompute_device)
                ).strip().lower()
                if se_whiten_precompute_device not in {"gpu", "cpu"}:
                    raise _err("model.likelihood.shared_event_re.whitening.precompute_device", "must be 'gpu' or 'cpu'")
            if "pcg_max_iters" in whiten_cfg and whiten_cfg.get("pcg_max_iters", None) is not None:
                se_whiten_pcg_max_iters = int(
                    _require_num(whiten_cfg.get("pcg_max_iters"), "model.likelihood.shared_event_re.whitening.pcg_max_iters")
                )
                if se_whiten_pcg_max_iters <= 0:
                    raise _err("model.likelihood.shared_event_re.whitening.pcg_max_iters", "must be > 0")
            if "pcg_tol" in whiten_cfg and whiten_cfg.get("pcg_tol", None) is not None:
                se_whiten_pcg_tol = float(
                    _require_num(whiten_cfg.get("pcg_tol"), "model.likelihood.shared_event_re.whitening.pcg_tol")
                )
                if not (se_whiten_pcg_tol > 0.0):
                    raise _err("model.likelihood.shared_event_re.whitening.pcg_tol", "must be > 0")
            if "pcg_min_iters" in whiten_cfg and whiten_cfg.get("pcg_min_iters", None) is not None:
                se_whiten_pcg_min_iters = int(
                    _require_num(whiten_cfg.get("pcg_min_iters"), "model.likelihood.shared_event_re.whitening.pcg_min_iters")
                )
                if se_whiten_pcg_min_iters < 0:
                    raise _err("model.likelihood.shared_event_re.whitening.pcg_min_iters", "must be >= 0")
            if "pcg_batched" in whiten_cfg and whiten_cfg.get("pcg_batched", None) is not None:
                se_whiten_pcg_batched = bool(
                    _require_bool(whiten_cfg.get("pcg_batched"), "model.likelihood.shared_event_re.whitening.pcg_batched")
                )
            if "pcg_bucket_nodes" in whiten_cfg and whiten_cfg.get("pcg_bucket_nodes", None) is not None:
                v = whiten_cfg.get("pcg_bucket_nodes")
                if not isinstance(v, (list, tuple)) or not v:
                    raise _err("model.likelihood.shared_event_re.whitening.pcg_bucket_nodes", "must be a non-empty list of ints")
                try:
                    se_whiten_pcg_bucket_nodes = [int(x) for x in v]
                except Exception:
                    raise _err("model.likelihood.shared_event_re.whitening.pcg_bucket_nodes", "must be a list of ints")
                if any(x <= 0 for x in se_whiten_pcg_bucket_nodes):
                    raise _err("model.likelihood.shared_event_re.whitening.pcg_bucket_nodes", "all values must be > 0")
        if "auto_tune_nodes_cap" in se_cfg and se_cfg["auto_tune_nodes_cap"] is not None:
            se_auto_tune_nodes_cap = bool(_require_bool(se_cfg.get("auto_tune_nodes_cap"), "model.likelihood.shared_event_re.auto_tune_nodes_cap"))
        if "auto_tune_nodes_max" in se_cfg and se_cfg["auto_tune_nodes_max"] is not None:
            se_auto_tune_nodes_max = int(
                _require_num(se_cfg.get("auto_tune_nodes_max"), "model.likelihood.shared_event_re.auto_tune_nodes_max")
            )
            if se_auto_tune_nodes_max < 1:
                raise _err("model.likelihood.shared_event_re.auto_tune_nodes_max", "must be >= 1")
        if "auto_tune_rows_cap" in se_cfg and se_cfg["auto_tune_rows_cap"] is not None:
            se_auto_tune_rows_cap = bool(_require_bool(se_cfg.get("auto_tune_rows_cap"), "model.likelihood.shared_event_re.auto_tune_rows_cap"))
        if "auto_tune_rows_max" in se_cfg and se_cfg["auto_tune_rows_max"] is not None:
            se_auto_tune_rows_max = int(
                _require_num(se_cfg.get("auto_tune_rows_max"), "model.likelihood.shared_event_re.auto_tune_rows_max")
            )
            if se_auto_tune_rows_max < 1:
                raise _err("model.likelihood.shared_event_re.auto_tune_rows_max", "must be >= 1")

        if "solver" in se_cfg and se_cfg["solver"] is not None:
            se_solver = str(se_cfg.get("solver", se_solver)).strip().lower()
        if se_solver in {"pcg", "pcg-sparse", "pcg_sparse"}:
            se_solver = "pcg_sparse"
        if se_solver not in {"dense", "pcg_sparse"}:
            raise _err("model.likelihood.shared_event_re.solver", "supported: 'dense', 'pcg_sparse'")
        if "drop_logdet" in se_cfg and se_cfg["drop_logdet"] is not None:
            se_drop_logdet = bool(se_cfg.get("drop_logdet", True))
        if "pcg_max_iters" in se_cfg and se_cfg["pcg_max_iters"] is not None:
            se_pcg_max_iters = int(_require_num(se_cfg["pcg_max_iters"], "model.likelihood.shared_event_re.pcg_max_iters"))
            if se_pcg_max_iters < 1:
                raise _err("model.likelihood.shared_event_re.pcg_max_iters", "must be >= 1")
        if "pcg_tol" in se_cfg and se_cfg["pcg_tol"] is not None:
            se_pcg_tol = float(_require_num(se_cfg["pcg_tol"], "model.likelihood.shared_event_re.pcg_tol"))
            if not (se_pcg_tol > 0.0):
                raise _err("model.likelihood.shared_event_re.pcg_tol", "must be > 0")

        # Optional diagnostics logging controls
        if "diag_log_every_epochs" in se_cfg and se_cfg["diag_log_every_epochs"] is not None:
            se_diag_log_every_epochs = int(_require_num(se_cfg["diag_log_every_epochs"], "model.likelihood.shared_event_re.diag_log_every_epochs"))
            if se_diag_log_every_epochs < 0:
                raise _err("model.likelihood.shared_event_re.diag_log_every_epochs", "must be >= 0")
        if "diag_max_groups" in se_cfg and se_cfg["diag_max_groups"] is not None:
            se_diag_max_groups = int(_require_num(se_cfg["diag_max_groups"], "model.likelihood.shared_event_re.diag_max_groups"))
            if se_diag_max_groups < 1:
                raise _err("model.likelihood.shared_event_re.diag_max_groups", "must be >= 1")
        if "diag_max_rows_per_group" in se_cfg and se_cfg["diag_max_rows_per_group"] is not None:
            se_diag_max_rows_per_group = int(_require_num(se_cfg["diag_max_rows_per_group"], "model.likelihood.shared_event_re.diag_max_rows_per_group"))
            if se_diag_max_rows_per_group < 64:
                raise _err("model.likelihood.shared_event_re.diag_max_rows_per_group", "must be >= 64")
        if "diag_max_nodes" in se_cfg and se_cfg["diag_max_nodes"] is not None:
            se_diag_max_nodes = int(_require_num(se_cfg["diag_max_nodes"], "model.likelihood.shared_event_re.diag_max_nodes"))
            if se_diag_max_nodes < 16:
                raise _err("model.likelihood.shared_event_re.diag_max_nodes", "must be >= 16")
        if "diag_seed" in se_cfg and se_cfg["diag_seed"] is not None:
            se_diag_seed = int(_require_num(se_cfg["diag_seed"], "model.likelihood.shared_event_re.diag_seed"))
            if se_diag_seed < 0:
                raise _err("model.likelihood.shared_event_re.diag_seed", "must be >= 0")

    if se_enabled:
        # Collapsed shared-event RE is Gaussian-conjugate; require a quadratic likelihood family.
        if str(lk_type).strip().lower() not in {"gaussian", "l2", "mse"}:
            raise _err(
                "model.likelihood.type",
                "must be 'gaussian'/'l2'/'mse' when likelihood.shared_event_re.enabled=true "
                "(collapsed shared-event random effects relies on Gaussian conjugacy; "
                "'student_t'/'huber'/'laplace' are not supported in the collapsed formulation)",
            )
        # Phase-A implementation is "quadratic-only": we do not include the correlated logdet term.
        # Until we implement Phase-B (SLQ logdet), we forbid learning σ under this likelihood to
        # avoid pathological behavior / incorrect gradients.
        if bool(learn_noise_scale):
            raise _err(
                "model.likelihood.learn_noise_scale",
                "must be false when model.likelihood.shared_event_re.enabled=true "
                "(quadratic-only collapsed likelihood currently does not support learning σ; "
                "this will be supported in a future extension with logdet/SLQ)",
            )
        if bool(se_joint_ps):
            # joint_ps requires both phases to have positive tau to be meaningful; allow zeros but warn via runtime behavior.
            pass
        # pcg_sparse currently supports only scalar (per-phase) random effects (joint_ps=False).
        if str(se_solver) == "pcg_sparse" and bool(se_joint_ps):
            raise _err("model.likelihood.shared_event_re.joint_ps", "must be false when model.likelihood.shared_event_re.solver='pcg_sparse'")
        if str(se_solver) == "pcg_sparse" and (not bool(se_drop_logdet)):
            raise _err(
                "model.likelihood.shared_event_re.drop_logdet",
                "must be true when model.likelihood.shared_event_re.solver='pcg_sparse' "
                "(Phase-A implementation drops logdet; Phase-B will add SLQ logdet)",
            )

    # Optional: collapsed slowness inducing-GP covariance likelihood (Gaussian; marginalized; no latent state).
    #
    # This is a geometry-aware correlated likelihood induced by a zero-mean slowness *vector* field u(x),
    # approximated with inducing points. It is intended as a collapsed alternative to
    # `shared_event_latent.mode='slowness_inducing_gp'` that does NOT carry a latent state.
    #
    # Phase-A implementation (to be implemented in code): quadratic-only (drop logdet).
    sl_cfg = lk.get("slowness_re", None)
    sl_enabled = False
    sl_mode = "scalar_sep"  # 'scalar_sep' (event-separation slowness RE)
    sl_grouping = "station_phase"  # 'phase' | 'station_phase'
    sl_ell_km = 0.0
    sl_tau_ps = [0.0, 0.0]
    sl_tau_station_ps = [0.0, 0.0]
    sl_tau_units = "abs"  # 'abs' (s/km) | 'vel_frac' (dimensionless)
    sl_max_rows_per_group = 200000
    sl_max_nodes_per_group = 2048
    sl_fallback_to_diag = True
    sl_drop_logdet = True
    # Inducing plan (required when enabled)
    sl_plan_enabled = True
    sl_plan_cover_frac = 1.0
    sl_plan_min_m = 1
    sl_plan_max_m = 1024
    sl_plan_top_k = 10
    sl_plan_seed_strategy = "max_degree"
    sl_plan_fixed_xyz = True
    sl_plan_kernel_jitter = 1e-6
    sl_plan_selection_outfile = None
    sl_plan_interpolation_enabled = True
    sl_plan_interpolation_m = 4
    sl_plan_interpolation_outfile = None
    # FITC diagonal residual (recommended)
    sl_fitc_enabled = True
    # Optional: fixed station-geometry basis (dimension reduction across stations) for slowness_re.
    # This enables a receiver-dependent slowness formulation without per-station solves.
    sl_sta_basis_enabled = False
    sl_sta_basis_r = 0
    sl_sta_basis_ell_km = 0.0
    sl_sta_basis_jitter = 1e-6
    sl_sta_basis_method = "eigh_rbf"
    # Solver for per-group Woodbury system:
    # - "cholesky" (default): exact dense solve in 3M (fast for small M)
    # - "pcg": iterative solve using sparse neighbor matvecs (better for large M and/or huge groups)
    sl_solver = "cholesky"
    sl_pcg_max_iters = 30
    sl_pcg_tol = 1e-4
    sl_pcg_check_every = 0
    sl_pcg_min_inducing = 128
    sl_pcg_min_rows = 2000
    sl_sep_cap_km = 0.0
    sl_jitter0 = 1e-8
    sl_vp_km_s = 6.0
    sl_vs_km_s = 3.5
    # Optional ESS for explicit slowness_re latents
    sl_ess_enabled = False
    sl_ess_update_every = 1
    sl_ess_start_after = 0
    sl_ess_sweeps = 1
    sl_ess_batch_size = 50_000
    sl_ess_max_bracket_steps = 64
    sl_ess_seed = 0
    sl_ess_freeze_sampler_group = True
    # Optional: DD-graph random effects (explicit event latents with Laplacian prior).
    dd_re_enabled = False
    dd_re_tau_ps = [0.0, 0.0]
    dd_re_q_diag = 1e-3
    dd_re_weight_by_pair_count = True
    dd_re_ess_enabled = False
    dd_re_ess_update_every = 1
    dd_re_ess_start_after = 0
    dd_re_ess_sweeps = 1
    dd_re_ess_max_bracket_steps = 64
    dd_re_ess_seed = 0
    dd_re_ess_freeze_sampler_group = True
    sl_gpu_enable = None
    sl_gpu_max_groups_per_batch = 64
    sl_gpu_max_edges_per_batch = 0
    sl_gpu_profile = False
    sl_gpu_debug_max_groups = 0
    sl_freeze_g_at_map = False
    if isinstance(sl_cfg, dict):
        sl_enabled = bool(sl_cfg.get("enabled", False))
        if "mode" in sl_cfg and sl_cfg.get("mode", None) is not None:
            sl_mode = str(sl_cfg.get("mode", sl_mode)).strip().lower()
        if sl_mode in {"scalar", "scalar_sep", "scalar_separation", "event_sep"}:
            sl_mode = "scalar_sep"
        if sl_mode in {"component_station_slowness", "component_station_shared", "component_station_re"}:
            sl_mode = "component_station"
        if sl_mode in {"component_station_explicit", "component_station_uncollapsed"}:
            sl_mode = "component_station_explicit"
        if sl_mode not in {"scalar_sep", "component_station", "component_station_explicit"}:
            raise _err("model.likelihood.slowness_re.mode", "supported: 'scalar_sep', 'component_station', 'component_station_explicit'")
        if "grouping" in sl_cfg and sl_cfg.get("grouping", None) is not None:
            sl_grouping = str(sl_cfg.get("grouping", sl_grouping)).strip().lower()
        if sl_grouping in {"stationphase", "station-phase"}:
            sl_grouping = "station_phase"
        if sl_grouping not in {"phase", "station_phase"}:
            raise _err("model.likelihood.slowness_re.grouping", "supported: 'phase', 'station_phase'")

        if "ell_km" in sl_cfg and sl_cfg.get("ell_km", None) is not None:
            sl_ell_km = float(_require_num(sl_cfg.get("ell_km"), "model.likelihood.slowness_re.ell_km"))
        if "tau_units" in sl_cfg and sl_cfg.get("tau_units", None) is not None:
            sl_tau_units = str(sl_cfg.get("tau_units", sl_tau_units)).strip().lower()
            if sl_tau_units not in {"abs", "vel_frac"}:
                raise _err("model.likelihood.slowness_re.tau_units", "supported: 'abs', 'vel_frac'")
        if sl_enabled and sl_mode not in {"scalar_sep", "component_station", "component_station_explicit"}:
            if not math.isfinite(sl_ell_km) or not (sl_ell_km > 0.0):
                raise _err("model.likelihood.slowness_re.ell_km", "must be finite and > 0 when enabled")

        # tau_s: abs (s/km) or vel_frac object (dimensionless), same convention as slowness_inducing_gp latent.
        if "tau_s" in sl_cfg and sl_cfg.get("tau_s", None) is not None:
            tv = sl_cfg.get("tau_s", None)
            if isinstance(tv, dict):
                if "vel_frac" not in tv or tv.get("vel_frac", None) is None:
                    raise _err("model.likelihood.slowness_re.tau_s.vel_frac", "required when tau_s is an object")
                vv = tv.get("vel_frac", None)
                if isinstance(vv, (int, float)):
                    f = float(vv)
                    sl_tau_ps = [f, f]
                elif isinstance(vv, list):
                    sl_tau_ps = _require_float_list(vv, "model.likelihood.slowness_re.tau_s.vel_frac", length=2)
                else:
                    raise _err("model.likelihood.slowness_re.tau_s.vel_frac", f"expected number or [P,S] list, got {type(vv).__name__}")
                sl_tau_units = "vel_frac"
            elif isinstance(tv, (int, float)):
                f = float(tv)
                sl_tau_ps = [f, f]
            elif isinstance(tv, list):
                sl_tau_ps = _require_float_list(tv, "model.likelihood.slowness_re.tau_s", length=2)
            else:
                raise _err("model.likelihood.slowness_re.tau_s", f"expected number, [P,S] list, or object, got {type(tv).__name__}")
        if "tau_station_s" in sl_cfg and sl_cfg.get("tau_station_s", None) is not None:
            tv = sl_cfg.get("tau_station_s", None)
            if isinstance(tv, dict):
                if "vel_frac" not in tv or tv.get("vel_frac", None) is None:
                    raise _err("model.likelihood.slowness_re.tau_station_s.vel_frac", "required when tau_station_s is an object")
                vv = tv.get("vel_frac", None)
                if isinstance(vv, (int, float)):
                    f = float(vv)
                    sl_tau_station_ps = [f, f]
                elif isinstance(vv, list):
                    sl_tau_station_ps = _require_float_list(vv, "model.likelihood.slowness_re.tau_station_s.vel_frac", length=2)
                else:
                    raise _err(
                        "model.likelihood.slowness_re.tau_station_s.vel_frac",
                        f"expected number or [P,S] list, got {type(vv).__name__}",
                    )
                sl_tau_units = "vel_frac"
            elif isinstance(tv, (int, float)):
                f = float(tv)
                sl_tau_station_ps = [f, f]
            elif isinstance(tv, list):
                sl_tau_station_ps = _require_float_list(tv, "model.likelihood.slowness_re.tau_station_s", length=2)
            else:
                raise _err(
                    "model.likelihood.slowness_re.tau_station_s",
                    f"expected number, [P,S] list, or object, got {type(tv).__name__}",
                )
        if sl_enabled and (not (sl_tau_ps[0] >= 0.0 and sl_tau_ps[1] >= 0.0)):
            raise _err("model.likelihood.slowness_re.tau_s", "must be >= 0")
        if sl_enabled and (not (sl_tau_station_ps[0] >= 0.0 and sl_tau_station_ps[1] >= 0.0)):
            raise _err("model.likelihood.slowness_re.tau_station_s", "must be >= 0")

        if "max_rows_per_group" in sl_cfg and sl_cfg.get("max_rows_per_group", None) is not None:
            sl_max_rows_per_group = int(_require_num(sl_cfg.get("max_rows_per_group"), "model.likelihood.slowness_re.max_rows_per_group"))
            if sl_max_rows_per_group < 2:
                raise _err("model.likelihood.slowness_re.max_rows_per_group", "must be >= 2")
        if "max_nodes_per_group" in sl_cfg and sl_cfg.get("max_nodes_per_group", None) is not None:
            sl_max_nodes_per_group = int(_require_num(sl_cfg.get("max_nodes_per_group"), "model.likelihood.slowness_re.max_nodes_per_group"))
            if sl_max_nodes_per_group < 2:
                raise _err("model.likelihood.slowness_re.max_nodes_per_group", "must be >= 2")
        if "sep_cap_km" in sl_cfg and sl_cfg.get("sep_cap_km", None) is not None:
            sl_sep_cap_km = float(_require_num(sl_cfg.get("sep_cap_km"), "model.likelihood.slowness_re.sep_cap_km"))
            if sl_sep_cap_km < 0.0:
                raise _err("model.likelihood.slowness_re.sep_cap_km", "must be >= 0")
        if "jitter0" in sl_cfg and sl_cfg.get("jitter0", None) is not None:
            sl_jitter0 = float(_require_num(sl_cfg.get("jitter0"), "model.likelihood.slowness_re.jitter0"))
            if sl_jitter0 <= 0.0:
                raise _err("model.likelihood.slowness_re.jitter0", "must be > 0")
        if "vp_km_s" in sl_cfg and sl_cfg.get("vp_km_s", None) is not None:
            sl_vp_km_s = float(_require_num(sl_cfg.get("vp_km_s"), "model.likelihood.slowness_re.vp_km_s"))
            if not (math.isfinite(sl_vp_km_s) and sl_vp_km_s > 0.0):
                raise _err("model.likelihood.slowness_re.vp_km_s", "must be finite and > 0")
        if "vs_km_s" in sl_cfg and sl_cfg.get("vs_km_s", None) is not None:
            sl_vs_km_s = float(_require_num(sl_cfg.get("vs_km_s"), "model.likelihood.slowness_re.vs_km_s"))
            if not (math.isfinite(sl_vs_km_s) and sl_vs_km_s > 0.0):
                raise _err("model.likelihood.slowness_re.vs_km_s", "must be finite and > 0")
        if "gpu_enable" in sl_cfg and sl_cfg.get("gpu_enable", None) is not None:
            sl_gpu_enable = bool(_require_bool(sl_cfg.get("gpu_enable"), "model.likelihood.slowness_re.gpu_enable"))
        if "gpu_max_groups_per_batch" in sl_cfg and sl_cfg.get("gpu_max_groups_per_batch", None) is not None:
            sl_gpu_max_groups_per_batch = int(
                _require_num(sl_cfg.get("gpu_max_groups_per_batch"), "model.likelihood.slowness_re.gpu_max_groups_per_batch")
            )
            if sl_gpu_max_groups_per_batch < 1:
                raise _err("model.likelihood.slowness_re.gpu_max_groups_per_batch", "must be >= 1")
        if "gpu_max_edges_per_batch" in sl_cfg and sl_cfg.get("gpu_max_edges_per_batch", None) is not None:
            sl_gpu_max_edges_per_batch = int(
                _require_num(sl_cfg.get("gpu_max_edges_per_batch"), "model.likelihood.slowness_re.gpu_max_edges_per_batch")
            )
            if sl_gpu_max_edges_per_batch < 0:
                raise _err("model.likelihood.slowness_re.gpu_max_edges_per_batch", "must be >= 0")
        if "gpu_profile" in sl_cfg and sl_cfg.get("gpu_profile", None) is not None:
            sl_gpu_profile = bool(_require_bool(sl_cfg.get("gpu_profile"), "model.likelihood.slowness_re.gpu_profile"))
        if "gpu_debug_max_groups" in sl_cfg and sl_cfg.get("gpu_debug_max_groups", None) is not None:
            sl_gpu_debug_max_groups = int(
                _require_num(sl_cfg.get("gpu_debug_max_groups"), "model.likelihood.slowness_re.gpu_debug_max_groups")
            )
            if sl_gpu_debug_max_groups < 0:
                raise _err("model.likelihood.slowness_re.gpu_debug_max_groups", "must be >= 0")
        if "freeze_g_at_map" in sl_cfg and sl_cfg.get("freeze_g_at_map", None) is not None:
            sl_freeze_g_at_map = bool(_require_bool(sl_cfg.get("freeze_g_at_map"), "model.likelihood.slowness_re.freeze_g_at_map"))
        if "fallback_to_diag" in sl_cfg and sl_cfg.get("fallback_to_diag", None) is not None:
            sl_fallback_to_diag = bool(sl_cfg.get("fallback_to_diag", True))
        if "drop_logdet" in sl_cfg and sl_cfg.get("drop_logdet", None) is not None:
            sl_drop_logdet = bool(sl_cfg.get("drop_logdet", True))

        # Optional: station basis
        sta_basis = sl_cfg.get("station_basis", None)
        if sta_basis is not None:
            if not isinstance(sta_basis, dict):
                raise _err("model.likelihood.slowness_re.station_basis", "expected object/dict or null")
            sl_sta_basis_enabled = bool(sta_basis.get("enabled", False))
            if sl_sta_basis_enabled:
                sl_sta_basis_r = int(_require_num(_require(sta_basis, "r", "model.likelihood.slowness_re.station_basis"), "model.likelihood.slowness_re.station_basis.r"))
                if sl_sta_basis_r < 1:
                    raise _err("model.likelihood.slowness_re.station_basis.r", "must be >= 1")
                sl_sta_basis_ell_km = float(_require_num(_require(sta_basis, "ell_km", "model.likelihood.slowness_re.station_basis"), "model.likelihood.slowness_re.station_basis.ell_km"))
                if not (sl_sta_basis_ell_km > 0.0) or not math.isfinite(sl_sta_basis_ell_km):
                    raise _err("model.likelihood.slowness_re.station_basis.ell_km", "must be finite and > 0")
                if "jitter" in sta_basis and sta_basis.get("jitter", None) is not None:
                    sl_sta_basis_jitter = float(_require_num(sta_basis.get("jitter"), "model.likelihood.slowness_re.station_basis.jitter"))
                    if not math.isfinite(sl_sta_basis_jitter) or sl_sta_basis_jitter < 0.0:
                        raise _err("model.likelihood.slowness_re.station_basis.jitter", "must be finite and >= 0")
                if "method" in sta_basis and sta_basis.get("method", None) is not None:
                    sl_sta_basis_method = str(sta_basis.get("method")).strip().lower()
                    if sl_sta_basis_method not in {"eigh_rbf"}:
                        raise _err("model.likelihood.slowness_re.station_basis.method", "supported: 'eigh_rbf'")

        if "solver" in sl_cfg and sl_cfg.get("solver", None) is not None:
            sl_solver = str(sl_cfg.get("solver", sl_solver)).strip().lower()
        if sl_solver in {"chol", "cholesky"}:
            sl_solver = "cholesky"
        if sl_solver not in {"cholesky", "pcg"}:
            raise _err("model.likelihood.slowness_re.solver", "supported: 'cholesky', 'pcg'")
        pcg = sl_cfg.get("pcg", None)
        if isinstance(pcg, dict):
            if "max_iters" in pcg and pcg.get("max_iters", None) is not None:
                sl_pcg_max_iters = int(_require_num(pcg.get("max_iters"), "model.likelihood.slowness_re.pcg.max_iters"))
                if sl_pcg_max_iters < 1:
                    raise _err("model.likelihood.slowness_re.pcg.max_iters", "must be >= 1")
            if "tol" in pcg and pcg.get("tol", None) is not None:
                sl_pcg_tol = float(_require_num(pcg.get("tol"), "model.likelihood.slowness_re.pcg.tol"))
                if (not math.isfinite(sl_pcg_tol)) or (not (sl_pcg_tol > 0.0)):
                    raise _err("model.likelihood.slowness_re.pcg.tol", "must be finite and > 0")
            if "check_every" in pcg and pcg.get("check_every", None) is not None:
                sl_pcg_check_every = int(_require_num(pcg.get("check_every"), "model.likelihood.slowness_re.pcg.check_every"))
                if sl_pcg_check_every < 0:
                    raise _err("model.likelihood.slowness_re.pcg.check_every", "must be >= 0")
            if "min_inducing" in pcg and pcg.get("min_inducing", None) is not None:
                sl_pcg_min_inducing = int(_require_num(pcg.get("min_inducing"), "model.likelihood.slowness_re.pcg.min_inducing"))
                if sl_pcg_min_inducing < 1:
                    raise _err("model.likelihood.slowness_re.pcg.min_inducing", "must be >= 1")
            if "min_rows" in pcg and pcg.get("min_rows", None) is not None:
                sl_pcg_min_rows = int(_require_num(pcg.get("min_rows"), "model.likelihood.slowness_re.pcg.min_rows"))
                if sl_pcg_min_rows < 1:
                    raise _err("model.likelihood.slowness_re.pcg.min_rows", "must be >= 1")

        if sl_enabled and sl_mode in {"scalar_sep", "component_station", "component_station_explicit"}:
            sl_plan_enabled = False
            sl_plan_interpolation_enabled = False
            sl_fitc_enabled = False
            sl_sta_basis_enabled = False
            sl_drop_logdet = True
        ess_cfg = sl_cfg.get("ess", None)
        if isinstance(ess_cfg, dict):
            if "enabled" in ess_cfg and ess_cfg.get("enabled", None) is not None:
                sl_ess_enabled = bool(_require_bool(ess_cfg.get("enabled"), "model.likelihood.slowness_re.ess.enabled"))
            if "update_every_epochs" in ess_cfg and ess_cfg.get("update_every_epochs", None) is not None:
                sl_ess_update_every = int(_require_num(ess_cfg.get("update_every_epochs"), "model.likelihood.slowness_re.ess.update_every_epochs"))
                if sl_ess_update_every < 1:
                    raise _err("model.likelihood.slowness_re.ess.update_every_epochs", "must be >= 1")
            if "start_after_epochs" in ess_cfg and ess_cfg.get("start_after_epochs", None) is not None:
                sl_ess_start_after = int(_require_num(ess_cfg.get("start_after_epochs"), "model.likelihood.slowness_re.ess.start_after_epochs"))
                if sl_ess_start_after < 0:
                    raise _err("model.likelihood.slowness_re.ess.start_after_epochs", "must be >= 0")
            if "sweeps_per_update" in ess_cfg and ess_cfg.get("sweeps_per_update", None) is not None:
                sl_ess_sweeps = int(_require_num(ess_cfg.get("sweeps_per_update"), "model.likelihood.slowness_re.ess.sweeps_per_update"))
                if sl_ess_sweeps < 1:
                    raise _err("model.likelihood.slowness_re.ess.sweeps_per_update", "must be >= 1")
            if "batch_size" in ess_cfg and ess_cfg.get("batch_size", None) is not None:
                sl_ess_batch_size = int(_require_num(ess_cfg.get("batch_size"), "model.likelihood.slowness_re.ess.batch_size"))
                if sl_ess_batch_size < 1:
                    raise _err("model.likelihood.slowness_re.ess.batch_size", "must be >= 1")
            if "max_bracket_steps" in ess_cfg and ess_cfg.get("max_bracket_steps", None) is not None:
                sl_ess_max_bracket_steps = int(_require_num(ess_cfg.get("max_bracket_steps"), "model.likelihood.slowness_re.ess.max_bracket_steps"))
                if sl_ess_max_bracket_steps < 8:
                    raise _err("model.likelihood.slowness_re.ess.max_bracket_steps", "must be >= 8")
            if "seed" in ess_cfg and ess_cfg.get("seed", None) is not None:
                sl_ess_seed = int(_require_num(ess_cfg.get("seed"), "model.likelihood.slowness_re.ess.seed"))
            if "freeze_sampler_group" in ess_cfg and ess_cfg.get("freeze_sampler_group", None) is not None:
                sl_ess_freeze_sampler_group = bool(_require_bool(ess_cfg.get("freeze_sampler_group"), "model.likelihood.slowness_re.ess.freeze_sampler_group"))

        plan = sl_cfg.get("inducing_plan", None)
        if isinstance(plan, dict):
            if "enabled" in plan and plan.get("enabled", None) is not None:
                sl_plan_enabled = bool(plan.get("enabled", True))
            if "cover_frac_of_ell" in plan and plan.get("cover_frac_of_ell", None) is not None:
                sl_plan_cover_frac = float(_require_num(plan.get("cover_frac_of_ell"), "model.likelihood.slowness_re.inducing_plan.cover_frac_of_ell"))
                if not (sl_plan_cover_frac > 0.0) or not math.isfinite(sl_plan_cover_frac):
                    raise _err("model.likelihood.slowness_re.inducing_plan.cover_frac_of_ell", "must be finite and > 0")
            if "min_inducing_per_component" in plan and plan.get("min_inducing_per_component", None) is not None:
                sl_plan_min_m = int(_require_num(plan.get("min_inducing_per_component"), "model.likelihood.slowness_re.inducing_plan.min_inducing_per_component"))
                if sl_plan_min_m < 1:
                    raise _err("model.likelihood.slowness_re.inducing_plan.min_inducing_per_component", "must be >= 1")
            if "max_inducing_per_component" in plan and plan.get("max_inducing_per_component", None) is not None:
                sl_plan_max_m = int(_require_num(plan.get("max_inducing_per_component"), "model.likelihood.slowness_re.inducing_plan.max_inducing_per_component"))
                if sl_plan_max_m < 1:
                    raise _err("model.likelihood.slowness_re.inducing_plan.max_inducing_per_component", "must be >= 1")
            if sl_plan_max_m < sl_plan_min_m:
                raise _err("model.likelihood.slowness_re.inducing_plan.max_inducing_per_component", "must be >= min_inducing_per_component")
            if "top_k" in plan and plan.get("top_k", None) is not None:
                sl_plan_top_k = int(_require_num(plan.get("top_k"), "model.likelihood.slowness_re.inducing_plan.top_k"))
                if sl_plan_top_k < 1:
                    raise _err("model.likelihood.slowness_re.inducing_plan.top_k", "must be >= 1")
            if "seed_strategy" in plan and plan.get("seed_strategy", None) is not None:
                sl_plan_seed_strategy = str(plan.get("seed_strategy")).strip().lower()
                if sl_plan_seed_strategy not in {"max_degree", "random"}:
                    raise _err("model.likelihood.slowness_re.inducing_plan.seed_strategy", "supported: 'max_degree', 'random'")
            if "fixed_xyz" in plan and plan.get("fixed_xyz", None) is not None:
                sl_plan_fixed_xyz = bool(plan.get("fixed_xyz", True))
            if "kernel_jitter" in plan and plan.get("kernel_jitter", None) is not None:
                sl_plan_kernel_jitter = float(_require_num(plan.get("kernel_jitter"), "model.likelihood.slowness_re.inducing_plan.kernel_jitter"))
                if not math.isfinite(sl_plan_kernel_jitter) or sl_plan_kernel_jitter < 0.0:
                    raise _err("model.likelihood.slowness_re.inducing_plan.kernel_jitter", "must be finite and >= 0")
            if "selection_outfile" in plan:
                v = plan.get("selection_outfile", None)
                sl_plan_selection_outfile = None if v is None else _require_str(v, "model.likelihood.slowness_re.inducing_plan.selection_outfile")
            interp = plan.get("interpolation", None)
            if isinstance(interp, dict):
                if "enabled" in interp and interp.get("enabled", None) is not None:
                    sl_plan_interpolation_enabled = bool(interp.get("enabled", True))
                if "m" in interp and interp.get("m", None) is not None:
                    sl_plan_interpolation_m = int(_require_num(interp.get("m"), "model.likelihood.slowness_re.inducing_plan.interpolation.m"))
                    if sl_plan_interpolation_m < 1:
                        raise _err("model.likelihood.slowness_re.inducing_plan.interpolation.m", "must be >= 1")
                if "outfile" in interp:
                    v = interp.get("outfile", None)
                    sl_plan_interpolation_outfile = None if v is None else _require_str(v, "model.likelihood.slowness_re.inducing_plan.interpolation.outfile")

        fitc = sl_cfg.get("fitc", None)
        if isinstance(fitc, dict) and ("enabled" in fitc) and (fitc.get("enabled", None) is not None):
            sl_fitc_enabled = bool(fitc.get("enabled", True))

    if sl_enabled:
        if str(lk_type).strip().lower() not in {"gaussian", "l2", "mse"}:
            raise _err(
                "model.likelihood.type",
                "must be 'gaussian'/'l2'/'mse' when model.likelihood.slowness_re.enabled=true "
                "(collapsed slowness covariance relies on Gaussian conjugacy; "
                "'student_t'/'huber'/'laplace' are not supported in the collapsed formulation)",
            )
        if bool(learn_noise_scale):
            raise _err(
                "model.likelihood.learn_noise_scale",
                "must be false when model.likelihood.slowness_re.enabled=true (quadratic-only collapsed likelihood)",
            )
        if not bool(sl_drop_logdet):
            raise _err(
                "model.likelihood.slowness_re.drop_logdet",
                "must be true in the current Phase-A implementation (quadratic-only; logdet dropped)",
            )

    # Optional: uncollapsed shared-event latent random effects (explicit b sampled in PSG-LD/SGHMC).
    # Model hyperparameters live under `model.likelihood.shared_event_latent`.
    # Inference/sampler knobs live under `inference.sampler.overrides.shared_event_latent`.
    se_lat_cfg = lk.get("shared_event_latent", None)
    se_lat_enabled = False
    # Mode/parameterization for shared_event_latent:
    # - "inducing_gp": inducing-point GP (predictive-process mean) using Stage-2/3 inducing_plan artifacts
    # - "graph_gmrf": explicit b[s,event,phase] with DD-linked event-graph GMRF prior
    # - "full": legacy explicit b[s,event,phase] with kNN event-graph Laplacian prior (backward compatible)
    se_lat_param = "full"
    se_lat_knn = 0
    se_lat_ell_km = 0.0
    se_lat_q_diag = 0.0
    se_lat_graph_lambda = 1.0
    # Optional: graph_gmrf sparsification on the DD-edge set (performance knob for very dense DD graphs)
    se_lat_graph_max_degree = 0          # keep top-k DD neighbors per event (0 disables)
    se_lat_graph_max_edge_km = None      # optional distance cutoff on DD edges before degree pruning
    se_lat_tau_ps = [0.0, 0.0]
    # Units for tau_s in slowness_inducing_gp:
    # - "abs": tau_s is interpreted as s/km (slowness amplitude)
    # - "vel_frac": tau_s is interpreted as fractional velocity perturbation δv/v (dimensionless)
    # Must always be defined because we materialize it unconditionally later.
    tau_units = "abs"
    se_lat_rho_ps = 0.0
    # Optional: inducing-point planning diagnostics (stage-1 for future inducing GP implementation)
    se_lat_plan_enable = False
    se_lat_plan_cover_frac = 0.5  # target coverage radius r = cover_frac * ell_km (default ell/2)
    se_lat_plan_min_m = 1
    se_lat_plan_max_m = 1024
    se_lat_plan_top_k = 10
    se_lat_plan_outfile = None
    se_lat_plan_select = False
    se_lat_plan_selection_outfile = None
    se_lat_plan_seed_strategy = "max_degree"  # 'max_degree' | 'random'
    se_lat_plan_use_xyz = False
    se_lat_plan_interp = False
    se_lat_plan_interp_m = 16
    se_lat_plan_interp_outfile = None
    se_lat_plan_interp_store_dist = False
    # For slowness_inducing_gp, Option-B fixed inducing geometry: treat inducing locations as fixed XYZ points
    # (selected from MAP event cloud) rather than "moving with events" via inducing_event_idx.
    # Default: True for slowness_inducing_gp; False otherwise.
    se_lat_plan_fixed_xyz = None
    # Regularization / numerical stabilization for K_UU construction (inducing_gp only).
    # This is added to the diagonal of each per-component K_UU block: K <- K + kernel_jitter * I.
    # Default is tiny (numerical jitter); users can increase it to tame ill-conditioned inducing layouts.
    se_lat_plan_kernel_jitter = 1e-6
    # Optional: FITC-style diagonal correction for inducing-point GP (Stage 5).
    # Default: enabled for inducing_gp (to avoid DTC under-dispersion), disabled otherwise.
    se_lat_fitc_enable = None
    # Identifiability constraint (enforced): drop the station-common (constant across stations)
    # mode of b so it cannot mimic per-event origin-time shifts Δt.
    # This is not a modeling option; it is a gauge-fixing for a non-identifiable direction.
    se_lat_drop_station_common_mode = False
    # Optional: fixed station-geometry basis for shared_event_latent (dimension reduction across stations).
    # Defaults must be defined even when shared_event_latent is disabled so materialization below is safe.
    se_lat_sta_basis_enabled = False
    se_lat_sta_basis_r = 0
    se_lat_sta_basis_ell_km = 0.0
    se_lat_sta_basis_jitter = 1e-6
    se_lat_sta_basis_method = "eigh_rbf"
    if se_lat_cfg is None:
        se_lat_cfg = {}
    if not isinstance(se_lat_cfg, dict):
        raise _err("model.likelihood.shared_event_latent", "expected object/dict or null")
    se_lat_enabled = bool(se_lat_cfg.get("enabled", False))

    # Hard break: do NOT allow sampler knobs here.
    for bad_key in ("lr_mult", "temperature_mult", "eps", "include_gamma", "freeze_preconditioner_sampling"):
        if bad_key in se_lat_cfg:
            raise _err(
                f"model.likelihood.shared_event_latent.{bad_key}",
                "moved; put this under `inference.sampler.overrides.shared_event_latent`",
            )

    if se_lat_enabled:
        # Optional mode selector (preferred: `mode`; legacy alias: `parameterization`)
        if "mode" in se_lat_cfg and se_lat_cfg.get("mode", None) is not None:
            se_lat_param = str(se_lat_cfg.get("mode")).strip().lower()
        elif "parameterization" in se_lat_cfg and se_lat_cfg.get("parameterization", None) is not None:
            se_lat_param = str(se_lat_cfg.get("parameterization")).strip().lower()
        if se_lat_param not in {"full", "inducing_gp", "slowness_inducing_gp", "graph_gmrf"}:
            raise _err(
                "model.likelihood.shared_event_latent.mode",
                "supported: 'inducing_gp', 'slowness_inducing_gp', 'graph_gmrf' (legacy: 'full')",
            )

        # Shared hyperparameters
        se_lat_ell_km = float(_require_num(_require(se_lat_cfg, "ell_km", "model.likelihood.shared_event_latent"), "model.likelihood.shared_event_latent.ell_km"))
        if not (se_lat_ell_km > 0.0):
            raise _err("model.likelihood.shared_event_latent.ell_km", "must be > 0")

        # Mode-specific knobs
        if se_lat_param == "full":
            se_lat_knn = int(_require_num(_require(se_lat_cfg, "knn", "model.likelihood.shared_event_latent"), "model.likelihood.shared_event_latent.knn"))
            if se_lat_knn < 1:
                raise _err("model.likelihood.shared_event_latent.knn", "must be >= 1")
            se_lat_q_diag = float(_require_num(_require(se_lat_cfg, "q_diag", "model.likelihood.shared_event_latent"), "model.likelihood.shared_event_latent.q_diag"))
            if not (se_lat_q_diag >= 0.0):
                raise _err("model.likelihood.shared_event_latent.q_diag", "must be >= 0")
        elif se_lat_param == "graph_gmrf":
            # graph_gmrf: DD-linked event graph; no knn required.
            # Optional diagonal term (default 1.0) and Laplacian scaling lambda (default 1.0).
            try:
                if "q_diag" in se_lat_cfg and se_lat_cfg.get("q_diag", None) is not None:
                    se_lat_q_diag = float(_require_num(se_lat_cfg.get("q_diag"), "model.likelihood.shared_event_latent.q_diag"))
                else:
                    se_lat_q_diag = 1.0
            except Exception:
                se_lat_q_diag = 1.0
            if not (se_lat_q_diag >= 0.0):
                raise _err("model.likelihood.shared_event_latent.q_diag", "must be >= 0")
            vlam = None
            if "lambda" in se_lat_cfg and se_lat_cfg.get("lambda", None) is not None:
                vlam = se_lat_cfg.get("lambda", None)
            elif "graph_lambda" in se_lat_cfg and se_lat_cfg.get("graph_lambda", None) is not None:
                vlam = se_lat_cfg.get("graph_lambda", None)
            if vlam is not None:
                se_lat_graph_lambda = float(_require_num(vlam, "model.likelihood.shared_event_latent.lambda"))
            if not math.isfinite(se_lat_graph_lambda) or se_lat_graph_lambda < 0.0:
                raise _err("model.likelihood.shared_event_latent.lambda", "must be finite and >= 0")
            # Optional: prune the DD-edge set to top-k neighbors per node (and/or within a distance cutoff).
            # This is a *performance* knob; statistically it is usually safe because w_ij decays with distance.
            if "max_degree" in se_lat_cfg and se_lat_cfg.get("max_degree", None) is not None:
                se_lat_graph_max_degree = int(_require_num(se_lat_cfg.get("max_degree"), "model.likelihood.shared_event_latent.max_degree"))
                if se_lat_graph_max_degree < 0:
                    raise _err("model.likelihood.shared_event_latent.max_degree", "must be >= 0")
            if "max_edge_km" in se_lat_cfg and se_lat_cfg.get("max_edge_km", None) is not None:
                v = float(_require_num(se_lat_cfg.get("max_edge_km"), "model.likelihood.shared_event_latent.max_edge_km"))
                if not (v > 0.0) or not math.isfinite(v):
                    raise _err("model.likelihood.shared_event_latent.max_edge_km", "must be finite and > 0")
                se_lat_graph_max_edge_km = float(v)
            # Backward-compatible: allow knn key but ignore it.
            try:
                if "knn" in se_lat_cfg and se_lat_cfg.get("knn", None) is not None:
                    se_lat_knn = int(_require_num(se_lat_cfg.get("knn"), "model.likelihood.shared_event_latent.knn"))
            except Exception:
                pass
        else:
            # inducing_gp / slowness_inducing_gp: uses inducing_plan artifacts (selection + interpolation) and ignores knn/q_diag.
            # Keep backward-compatible parsing if keys are present.
            if "rank" in se_lat_cfg:
                raise _err("model.likelihood.shared_event_latent.rank", "removed")
            if "rff_seed" in se_lat_cfg:
                raise _err("model.likelihood.shared_event_latent.rff_seed", "removed")
            try:
                if "knn" in se_lat_cfg and se_lat_cfg.get("knn", None) is not None:
                    se_lat_knn = int(_require_num(se_lat_cfg.get("knn"), "model.likelihood.shared_event_latent.knn"))
                if "q_diag" in se_lat_cfg and se_lat_cfg.get("q_diag", None) is not None:
                    se_lat_q_diag = float(_require_num(se_lat_cfg.get("q_diag"), "model.likelihood.shared_event_latent.q_diag"))
            except Exception:
                pass

        # tau_s: interpretation depends on mode.
        # - For legacy modes, tau_s is in seconds (scalar nuisance amplitude).
        # - For slowness_inducing_gp, tau_s can be provided either as:
        #     (a) absolute units (s/km): number or [P,S] list (legacy behavior), OR
        #     (b) fractional velocity perturbation (dimensionless): { "vel_frac": number|[P,S] }.
        #
        # In case (b), we treat the learned field as dimensionless ε(x) (≈ δv/v) and convert to
        # slowness units at runtime using an effective 1D v(z) derived from EikoNet.
        tau_units = "abs"
        tau_v = _require(se_lat_cfg, "tau_s", "model.likelihood.shared_event_latent")
        if isinstance(tau_v, dict):
            if se_lat_param != "slowness_inducing_gp":
                raise _err("model.likelihood.shared_event_latent.tau_s", "object form is only supported when mode='slowness_inducing_gp'")
            if "vel_frac" not in tau_v or tau_v.get("vel_frac", None) is None:
                raise _err("model.likelihood.shared_event_latent.tau_s.vel_frac", "required when tau_s is an object for slowness_inducing_gp")
            vv = tau_v.get("vel_frac", None)
            if isinstance(vv, (int, float)):
                f = float(vv)
                se_lat_tau_ps = [f, f]
            elif isinstance(vv, list):
                se_lat_tau_ps = _require_float_list(vv, "model.likelihood.shared_event_latent.tau_s.vel_frac", length=2)
            else:
                raise _err("model.likelihood.shared_event_latent.tau_s.vel_frac", f"expected number or [P,S] list, got {type(vv).__name__}")
            tau_units = "vel_frac"
        elif isinstance(tau_v, (int, float)):
            f = float(tau_v)
            se_lat_tau_ps = [f, f]
        elif isinstance(tau_v, list):
            se_lat_tau_ps = _require_float_list(tau_v, "model.likelihood.shared_event_latent.tau_s", length=2)
        else:
            raise _err("model.likelihood.shared_event_latent.tau_s", f"expected number, [P,S] list, or object, got {type(tau_v).__name__}")
        if not (se_lat_tau_ps[0] >= 0.0 and se_lat_tau_ps[1] >= 0.0):
            raise _err("model.likelihood.shared_event_latent.tau_s", "must be >= 0")
        se_lat_rho_ps = float(_require_num(_require(se_lat_cfg, "rho_ps", "model.likelihood.shared_event_latent"), "model.likelihood.shared_event_latent.rho_ps"))
        if not (-0.999 < se_lat_rho_ps < 0.999):
            raise _err("model.likelihood.shared_event_latent.rho_ps", "must satisfy -0.999 < rho_ps < 0.999")
        # Uncollapsed shared-event latent b is an explicit nuisance parameter sampled/optimized alongside θ.
        # Unlike the *collapsed* shared_event_re model, this does NOT rely on Gaussian conjugacy, so robust
        # likelihood families (Huber/Laplace) are valid here.
        if str(lk_type).strip().lower() not in {"gaussian", "l2", "mse", "huber", "laplace", "l1", "mae", "student_t"}:
            raise _err(
                "model.likelihood.type",
                "must be one of: 'gaussian'/'l2'/'mse', 'huber', 'laplace' (aliases: 'l1','mae'), 'student_t' when "
                "model.likelihood.shared_event_latent.enabled=true",
            )

        # Optional: inducing planning block (no behavior change unless code consumes these keys)
        plan = se_lat_cfg.get("inducing_plan", None)
        if plan is not None:
            if not isinstance(plan, dict):
                raise _err("model.likelihood.shared_event_latent.inducing_plan", "expected object/dict or null")
            se_lat_plan_enable = bool(plan.get("enabled", False))
            if "kernel_jitter" in plan and plan.get("kernel_jitter", None) is not None:
                se_lat_plan_kernel_jitter = float(
                    _require_num(plan.get("kernel_jitter"), "model.likelihood.shared_event_latent.inducing_plan.kernel_jitter")
                )
                if not math.isfinite(se_lat_plan_kernel_jitter) or se_lat_plan_kernel_jitter < 0.0:
                    raise _err("model.likelihood.shared_event_latent.inducing_plan.kernel_jitter", "must be finite and >= 0")
            if "cover_frac_of_ell" in plan and plan.get("cover_frac_of_ell", None) is not None:
                se_lat_plan_cover_frac = float(_require_num(plan.get("cover_frac_of_ell"), "model.likelihood.shared_event_latent.inducing_plan.cover_frac_of_ell"))
                if not (se_lat_plan_cover_frac > 0.0) or not math.isfinite(se_lat_plan_cover_frac):
                    raise _err("model.likelihood.shared_event_latent.inducing_plan.cover_frac_of_ell", "must be finite and > 0")
            if "min_inducing_per_component" in plan and plan.get("min_inducing_per_component", None) is not None:
                se_lat_plan_min_m = int(_require_num(plan.get("min_inducing_per_component"), "model.likelihood.shared_event_latent.inducing_plan.min_inducing_per_component"))
                if se_lat_plan_min_m < 1:
                    raise _err("model.likelihood.shared_event_latent.inducing_plan.min_inducing_per_component", "must be >= 1")
            if "max_inducing_per_component" in plan and plan.get("max_inducing_per_component", None) is not None:
                se_lat_plan_max_m = int(_require_num(plan.get("max_inducing_per_component"), "model.likelihood.shared_event_latent.inducing_plan.max_inducing_per_component"))
                if se_lat_plan_max_m < 1:
                    raise _err("model.likelihood.shared_event_latent.inducing_plan.max_inducing_per_component", "must be >= 1")
            if se_lat_plan_max_m < se_lat_plan_min_m:
                raise _err("model.likelihood.shared_event_latent.inducing_plan.max_inducing_per_component", "must be >= min_inducing_per_component")
            if "top_k" in plan and plan.get("top_k", None) is not None:
                se_lat_plan_top_k = int(_require_num(plan.get("top_k"), "model.likelihood.shared_event_latent.inducing_plan.top_k"))
                if se_lat_plan_top_k < 1:
                    raise _err("model.likelihood.shared_event_latent.inducing_plan.top_k", "must be >= 1")
            if "outfile" in plan:
                v = plan.get("outfile", None)
                if v is None:
                    se_lat_plan_outfile = None
                else:
                    se_lat_plan_outfile = _require_str(v, "model.likelihood.shared_event_latent.inducing_plan.outfile")
            if "select" in plan and plan.get("select", None) is not None:
                se_lat_plan_select = bool(plan.get("select", False))
            if "selection_outfile" in plan:
                v = plan.get("selection_outfile", None)
                if v is None:
                    se_lat_plan_selection_outfile = None
                else:
                    se_lat_plan_selection_outfile = _require_str(v, "model.likelihood.shared_event_latent.inducing_plan.selection_outfile")
            if "seed_strategy" in plan and plan.get("seed_strategy", None) is not None:
                se_lat_plan_seed_strategy = str(plan.get("seed_strategy")).strip().lower()
                if se_lat_plan_seed_strategy not in {"max_degree", "random"}:
                    raise _err("model.likelihood.shared_event_latent.inducing_plan.seed_strategy", "supported: 'max_degree', 'random'")
            if "use_xyz" in plan and plan.get("use_xyz", None) is not None:
                se_lat_plan_use_xyz = bool(plan.get("use_xyz", False))
            if "fixed_xyz" in plan and plan.get("fixed_xyz", None) is not None:
                se_lat_plan_fixed_xyz = bool(plan.get("fixed_xyz", False))
            interp = plan.get("interpolation", None)
            if interp is not None:
                if not isinstance(interp, dict):
                    raise _err("model.likelihood.shared_event_latent.inducing_plan.interpolation", "expected object/dict or null")
                se_lat_plan_interp = bool(interp.get("enabled", False))
                if "m" in interp and interp.get("m", None) is not None:
                    se_lat_plan_interp_m = int(_require_num(interp.get("m"), "model.likelihood.shared_event_latent.inducing_plan.interpolation.m"))
                    if se_lat_plan_interp_m < 1:
                        raise _err("model.likelihood.shared_event_latent.inducing_plan.interpolation.m", "must be >= 1")
                if "outfile" in interp:
                    v = interp.get("outfile", None)
                    if v is None:
                        se_lat_plan_interp_outfile = None
                    else:
                        se_lat_plan_interp_outfile = _require_str(v, "model.likelihood.shared_event_latent.inducing_plan.interpolation.outfile")
                if "store_distances" in interp and interp.get("store_distances", None) is not None:
                    se_lat_plan_interp_store_dist = bool(interp.get("store_distances", False))

        # Optional: FITC-style diagonal correction for inducing_gp (Stage 5).
        fitc = se_lat_cfg.get("fitc", None)
        if fitc is not None:
            if not isinstance(fitc, dict):
                raise _err("model.likelihood.shared_event_latent.fitc", "expected object/dict or null")
            if "enabled" in fitc and fitc.get("enabled", None) is not None:
                se_lat_fitc_enable = bool(fitc.get("enabled", False))

        # Default FITC enablement: on for inducing_gp, off otherwise.
        if se_lat_fitc_enable is None:
            se_lat_fitc_enable = bool(se_lat_param == "inducing_gp")
        # FITC is optional for inducing-point modes. For slowness_inducing_gp, we support a FITC-style
        # diagonal correction (computed at MAP) that inflates the per-row likelihood variance.

        # Enforce identifiability: always drop station-common mode when enabled.
        # If the user provided a value, accept it but ignore it (backward compatibility / experiments).
        se_lat_drop_station_common_mode = True
        try:
            if "drop_station_common_mode" in se_lat_cfg and se_lat_cfg.get("drop_station_common_mode", None) is not None:
                if not bool(se_lat_cfg.get("drop_station_common_mode", True)):
                    print(
                        "Warning: model.likelihood.shared_event_latent.drop_station_common_mode=false is ignored; "
                        "SPIDER enforces this constraint to keep origin time (Δt) identifiable."
                    )
        except Exception:
            pass

        # Optional: fixed station-geometry basis for shared_event_latent (dimension reduction across stations).
        # This is intended to encode that nearby stations share similar latent structure.
        sta_basis = se_lat_cfg.get("station_basis", None)
        if sta_basis is not None:
            if not isinstance(sta_basis, dict):
                raise _err("model.likelihood.shared_event_latent.station_basis", "expected object/dict or null")
            se_lat_sta_basis_enabled = bool(sta_basis.get("enabled", False))
            if se_lat_sta_basis_enabled:
                se_lat_sta_basis_r = int(_require_num(_require(sta_basis, "r", "model.likelihood.shared_event_latent.station_basis"), "model.likelihood.shared_event_latent.station_basis.r"))
                if se_lat_sta_basis_r < 1:
                    raise _err("model.likelihood.shared_event_latent.station_basis.r", "must be >= 1")
                se_lat_sta_basis_ell_km = float(_require_num(_require(sta_basis, "ell_km", "model.likelihood.shared_event_latent.station_basis"), "model.likelihood.shared_event_latent.station_basis.ell_km"))
                if not (se_lat_sta_basis_ell_km > 0.0) or not math.isfinite(se_lat_sta_basis_ell_km):
                    raise _err("model.likelihood.shared_event_latent.station_basis.ell_km", "must be finite and > 0")
                if "jitter" in sta_basis and sta_basis.get("jitter", None) is not None:
                    se_lat_sta_basis_jitter = float(_require_num(sta_basis.get("jitter"), "model.likelihood.shared_event_latent.station_basis.jitter"))
                    if not math.isfinite(se_lat_sta_basis_jitter) or se_lat_sta_basis_jitter < 0.0:
                        raise _err("model.likelihood.shared_event_latent.station_basis.jitter", "must be finite and >= 0")
                if "method" in sta_basis and sta_basis.get("method", None) is not None:
                    se_lat_sta_basis_method = str(sta_basis.get("method")).strip().lower()
                    if se_lat_sta_basis_method not in {"eigh_rbf"}:
                        raise _err("model.likelihood.shared_event_latent.station_basis.method", "supported: 'eigh_rbf'")
        # else: not provided -> keep defaults (disabled)

    # Mutual exclusion: uncollapsed latent b and collapsed covariance likelihoods cannot both be active.
    # If both are enabled in config, prefer the uncollapsed model (and disable the collapsed one)
    # to avoid double-counting the same correlation mechanism.
    if bool(se_lat_enabled) and bool(se_enabled):
        se_enabled = False
    if bool(se_lat_enabled) and bool(sl_enabled):
        sl_enabled = False
    # Two collapsed likelihoods at once is ambiguous. Prefer slowness_re and disable shared_event_re.
    if bool(sl_enabled) and bool(se_enabled):
        se_enabled = False

    # sigma_inflation block was removed; keep an explicit flag for downstream guards.
    sigma_infl_enabled = False

    # sigma_inflation is currently incompatible with shared_event_re (pcg_sparse) because that code path
    # assumes a homoscedastic diagonal (no per-row variance).
    if bool(se_enabled) and bool(sigma_infl_enabled):
        raise _err(
            "model.likelihood.sigma_inflation",
            "not supported when model.likelihood.shared_event_re.enabled=true (heteroscedastic sigma not yet implemented for the PCG solver)",
        )
    if bool(sl_enabled) and bool(sigma_infl_enabled):
        raise _err(
            "model.likelihood.sigma_inflation",
            "not supported when model.likelihood.slowness_re.enabled=true (use slowness_re fitc/diag instead; sigma_inflation integration is not implemented yet)",
        )

    # Optional: latent-field model for structured residuals (NNGP + ESS).
    # Intentionally OPTIONAL (absent -> disabled) to avoid breaking existing configs.
    lf_cfg = lk.get("latent_field", None)
    lf_enabled = False
    lf_neighbor_m = 40
    lf_by_station = False
    lf_max_stations_per_update = 0
    lf_min_edges_per_station = 0
    lf_parameterization = "slowness"
    lf_slowness_amp_s_per_km = [0.0, 0.0]
    lf_update_every = 10
    lf_refresh_every = 100
    lf_edge_chunk = 200000
    lf_component_wise = True
    lf_map_iters = 3
    lf_map_damping = 1.0
    lf_map_log_every_iter = 1
    lf_map_log_n_edges = 200000
    lf_map_z_clip = 10.0
    lf_active_phases = [1, 2, 3, 4]
    # MAP fit sigma selection (to avoid Phase-1 learned noise starving latent MAP updates)
    lf_map_sigma_mode = "current"  # 'current' | 'phase_unc' | 'override' | 'clip_current'
    lf_map_sigma_override = None   # [P,S] when mode='override'
    lf_map_sigma_clip_max = None   # [P,S] when mode='clip_current'
    # GP prior hyperparameters for the event-node latent field (Matérn)
    lf_kernel = "matern"
    lf_nu = 2.5
    lf_ell_km = [1.0, 1.0]  # [P,S] allowed; implementation may share/average
    if isinstance(lf_cfg, dict):
        def _req_int_lf(v: Any, path: str, *, min_v: int = 0) -> int:
            if not isinstance(v, int):
                raise _err(path, f"expected integer, got {type(v).__name__}")
            if v < min_v:
                raise _err(path, f"must be >= {min_v}")
            return int(v)

        lf_enabled = bool(lf_cfg.get("enabled", False))
        if "parameterization" in lf_cfg and lf_cfg["parameterization"] is not None:
            lf_parameterization = str(lf_cfg.get("parameterization", "slowness")).strip().lower()
        if lf_parameterization != "slowness":
            raise _err("likelihood.latent_field.parameterization", "only 'slowness' is supported (scalar latent removed)")

        if "inference" in lf_cfg and lf_cfg.get("inference", None) is not None:
            raise _err("likelihood.latent_field.inference", "removed; latent_field uses MAP (Jacobi) z-block updates (no ESS / no phase1-only freeze)")

        if "kernel" in lf_cfg and lf_cfg["kernel"] is not None:
            lf_kernel = str(lf_cfg.get("kernel", "matern")).strip().lower()
        if lf_kernel not in {"matern"}:
            raise _err("likelihood.latent_field.kernel", "supported: 'matern'")
        if "nu" in lf_cfg and lf_cfg["nu"] is not None:
            lf_nu = float(_require_num(lf_cfg["nu"], "likelihood.latent_field.nu"))
        if lf_nu not in {0.5, 1.5, 2.5}:
            raise _err("likelihood.latent_field.nu", "supported values: 0.5, 1.5, 2.5 (closed-form Matérn)")
        if "ell_km" in lf_cfg and lf_cfg["ell_km"] is not None:
            v = lf_cfg["ell_km"]
            if isinstance(v, (int, float)):
                f = float(v)
                lf_ell_km = [f, f]
            elif isinstance(v, list):
                lf_ell_km = _require_float_list(v, "likelihood.latent_field.ell_km", length=2)
            else:
                raise _err("likelihood.latent_field.ell_km", f"expected number or [P,S] list, got {type(v).__name__}")
        if not (lf_ell_km[0] > 0.0 and lf_ell_km[1] > 0.0):
            raise _err("likelihood.latent_field.ell_km", "must be > 0")

        if "neighbor_m" in lf_cfg and lf_cfg["neighbor_m"] is not None:
            lf_neighbor_m = _req_int_lf(lf_cfg["neighbor_m"], "likelihood.latent_field.neighbor_m", min_v=1)
        if "rho_ps" in lf_cfg and lf_cfg.get("rho_ps", None) is not None:
            raise _err("likelihood.latent_field.rho_ps", "removed (scalar latent removed)")
        if "by_station" in lf_cfg and lf_cfg["by_station"] is not None:
            lf_by_station = bool(lf_cfg.get("by_station", False))
        if "max_stations_per_update" in lf_cfg and lf_cfg["max_stations_per_update"] is not None:
            lf_max_stations_per_update = _req_int_lf(
                lf_cfg["max_stations_per_update"],
                "likelihood.latent_field.max_stations_per_update",
                min_v=0,
            )
        if "min_edges_per_station" in lf_cfg and lf_cfg["min_edges_per_station"] is not None:
            lf_min_edges_per_station = _req_int_lf(
                lf_cfg["min_edges_per_station"],
                "likelihood.latent_field.min_edges_per_station",
                min_v=0,
            )
        # Slowness latent amplitude (s/km); only used when parameterization='slowness'.
        if "slowness_amp_s_per_km" in lf_cfg and lf_cfg["slowness_amp_s_per_km"] is not None:
            v = lf_cfg["slowness_amp_s_per_km"]
            if isinstance(v, (int, float)):
                f = float(v)
                lf_slowness_amp_s_per_km = [f, f]
            elif isinstance(v, list):
                lf_slowness_amp_s_per_km = _require_float_list(v, "likelihood.latent_field.slowness_amp_s_per_km", length=2)
            else:
                raise _err("likelihood.latent_field.slowness_amp_s_per_km", f"expected number or [P,S] list, got {type(v).__name__}")
        if "update_every_epochs" in lf_cfg and lf_cfg["update_every_epochs"] is not None:
            lf_update_every = _req_int_lf(lf_cfg["update_every_epochs"], "likelihood.latent_field.update_every_epochs", min_v=1)
        if "active_phases" in lf_cfg and lf_cfg["active_phases"] is not None:
            v = lf_cfg["active_phases"]
            if not isinstance(v, list) or not v:
                raise _err("likelihood.latent_field.active_phases", "expected non-empty list of ints in {1,2,3,4}")
            out = []
            for i, xi in enumerate(v):
                if not isinstance(xi, int):
                    raise _err(f"likelihood.latent_field.active_phases[{i}]", f"expected int, got {type(xi).__name__}")
                if xi not in {1, 2, 3, 4}:
                    raise _err(f"likelihood.latent_field.active_phases[{i}]", "must be one of {1,2,3,4}")
                out.append(int(xi))
            # de-dupe + keep sorted for stable downstream checks
            lf_active_phases = sorted(set(out))
        if "refresh_every_epochs" in lf_cfg and lf_cfg["refresh_every_epochs"] is not None:
            lf_refresh_every = _req_int_lf(lf_cfg["refresh_every_epochs"], "likelihood.latent_field.refresh_every_epochs", min_v=1)
        if "sweeps_per_update" in lf_cfg and lf_cfg.get("sweeps_per_update", None) is not None:
            raise _err("likelihood.latent_field.sweeps_per_update", "removed (ESS removed)")
        if "max_bracket_steps" in lf_cfg and lf_cfg.get("max_bracket_steps", None) is not None:
            raise _err("likelihood.latent_field.max_bracket_steps", "removed (ESS removed)")
        if "edge_chunk_size" in lf_cfg and lf_cfg["edge_chunk_size"] is not None:
            lf_edge_chunk = _req_int_lf(lf_cfg["edge_chunk_size"], "likelihood.latent_field.edge_chunk_size", min_v=1024)
        if "component_wise" in lf_cfg and lf_cfg["component_wise"] is not None:
            lf_component_wise = bool(lf_cfg.get("component_wise", True))
        if "map_iters" in lf_cfg and lf_cfg["map_iters"] is not None:
            lf_map_iters = _req_int_lf(lf_cfg["map_iters"], "likelihood.latent_field.map_iters", min_v=1)
        if "map_damping" in lf_cfg and lf_cfg["map_damping"] is not None:
            lf_map_damping = float(_require_num(lf_cfg["map_damping"], "likelihood.latent_field.map_damping"))
            if not (0.0 < lf_map_damping <= 1.0):
                raise _err("likelihood.latent_field.map_damping", "must satisfy 0 < map_damping <= 1")
        if "map_log_every_iter" in lf_cfg and lf_cfg["map_log_every_iter"] is not None:
            lf_map_log_every_iter = _req_int_lf(
                lf_cfg["map_log_every_iter"],
                "likelihood.latent_field.map_log_every_iter",
                min_v=0,
            )
        if "map_log_n_edges" in lf_cfg and lf_cfg["map_log_n_edges"] is not None:
            lf_map_log_n_edges = _req_int_lf(
                lf_cfg["map_log_n_edges"],
                "likelihood.latent_field.map_log_n_edges",
                min_v=0,
            )
        if "map_z_clip" in lf_cfg and lf_cfg["map_z_clip"] is not None:
            lf_map_z_clip = float(_require_num(lf_cfg["map_z_clip"], "likelihood.latent_field.map_z_clip"))
            if lf_map_z_clip <= 0.0:
                raise _err("likelihood.latent_field.map_z_clip", "must be > 0")

        if "map_sigma_mode" in lf_cfg and lf_cfg["map_sigma_mode"] is not None:
            lf_map_sigma_mode = str(lf_cfg.get("map_sigma_mode", "current")).strip().lower()
        if lf_map_sigma_mode not in {"current", "phase_unc", "override", "clip_current"}:
            raise _err("likelihood.latent_field.map_sigma_mode", "supported: 'current', 'phase_unc', 'override', 'clip_current'")

        if "map_sigma_override" in lf_cfg and lf_cfg["map_sigma_override"] is not None:
            v = lf_cfg["map_sigma_override"]
            if isinstance(v, (int, float)):
                f = float(v)
                lf_map_sigma_override = [f, f]
            elif isinstance(v, list):
                lf_map_sigma_override = _require_float_list(v, "likelihood.latent_field.map_sigma_override", length=2)
            else:
                raise _err("likelihood.latent_field.map_sigma_override", f"expected number or [P,S] list, got {type(v).__name__}")
            if not (lf_map_sigma_override[0] > 0.0 and lf_map_sigma_override[1] > 0.0):
                raise _err("likelihood.latent_field.map_sigma_override", "must be > 0")

        if "map_sigma_clip_max" in lf_cfg and lf_cfg["map_sigma_clip_max"] is not None:
            v = lf_cfg["map_sigma_clip_max"]
            if isinstance(v, (int, float)):
                f = float(v)
                lf_map_sigma_clip_max = [f, f]
            elif isinstance(v, list):
                lf_map_sigma_clip_max = _require_float_list(v, "likelihood.latent_field.map_sigma_clip_max", length=2)
            else:
                raise _err("likelihood.latent_field.map_sigma_clip_max", f"expected number or [P,S] list, got {type(v).__name__}")
            if not (lf_map_sigma_clip_max[0] > 0.0 and lf_map_sigma_clip_max[1] > 0.0):
                raise _err("likelihood.latent_field.map_sigma_clip_max", "must be > 0")

        if lf_map_sigma_mode == "override" and lf_map_sigma_override is None:
            raise _err("likelihood.latent_field.map_sigma_override", "required when map_sigma_mode='override'")
        if lf_map_sigma_mode == "clip_current" and lf_map_sigma_clip_max is None:
            raise _err("likelihood.latent_field.map_sigma_clip_max", "required when map_sigma_mode='clip_current'")

    if lf_enabled:
        # Latent-field z-block MAP update is implemented as a Jacobi / normal-equations step
        # (quadratic data term). We therefore require an L2-family likelihood for *the latent update*.
        #
        # However, the main θ-block (event locations / noise scales) can still be optimized with a robust loss.
        # We allow Huber here; note that the latent MAP update still uses an L2 approximation internally.
        if str(lk_type).strip().lower() not in {"gaussian", "l2", "mse", "huber"}:
            raise _err(
                "likelihood.type",
                "must be 'gaussian'/'l2'/'mse' (or 'huber') when likelihood.latent_field.enabled=true "
                "(latent MAP update assumes a quadratic data term)",
            )
        if not lf_by_station:
            raise _err("likelihood.latent_field.by_station", "must be true when likelihood.latent_field.enabled=true (slowness latent is station-phase)")
        if not (lf_slowness_amp_s_per_km[0] >= 0.0 and lf_slowness_amp_s_per_km[1] >= 0.0):
            raise _err("likelihood.latent_field.slowness_amp_s_per_km", "must be >= 0")

    # ---- filters ----
    flt = _require_dict(_require(model, "filters", "model"), "model.filters")
    fd = _require_dict(_require(flt, "dtimes", "model.filters"), "model.filters.dtimes")
    fe = _require_dict(_require(flt, "events", "model.filters"), "model.filters.events")
    fr = _require_dict(_require(flt, "residual", "model.filters"), "model.filters.residual")

    remove_duplicates = _require_bool(_require(fd, "remove_duplicates", "model.filters.dtimes"), "model.filters.dtimes.remove_duplicates")
    max_abs_input_dt = _require_num(_require(fd, "max_abs_input_dt", "model.filters.dtimes"), "model.filters.dtimes.max_abs_input_dt")
    dtime_thin_frac = _require_num(_require(fd, "dtime_thin_frac", "model.filters.dtimes"), "model.filters.dtimes.dtime_thin_frac")
    if not (0.0 < dtime_thin_frac <= 1.0):
        raise _err("model.filters.dtimes.dtime_thin_frac", "must satisfy 0 < frac <= 1")
    flip_dt_sign = _require_bool(_require(fd, "flip_dt_sign", "model.filters.dtimes"), "model.filters.dtimes.flip_dt_sign")
    cc_min = _require_num(_require(fd, "cc_min", "model.filters.dtimes"), "model.filters.dtimes.cc_min")

    def _req_int(v: Any, path: str, *, min_v: int = 0) -> int:
        if not isinstance(v, int):
            raise _err(path, f"expected integer, got {type(v).__name__}")
        if v < min_v:
            raise _err(path, f"must be >= {min_v}")
        return int(v)

    min_dtimes = _req_int(_require(fe, "min_dtimes", "filters.events"), "filters.events.min_dtimes", min_v=1)
    min_unique_phase_per_event = _req_int(_require(fe, "min_unique_phase_per_event", "filters.events"), "filters.events.min_unique_phase_per_event", min_v=1)
    min_dtimes_per_pair = _req_int(_require(fe, "min_dtimes_per_pair", "filters.events"), "filters.events.min_dtimes_per_pair", min_v=1)
    min_event_degree = _req_int(_require(fe, "min_event_degree", "filters.events"), "filters.events.min_event_degree", min_v=0)
    min_events_per_cluster = _req_int(_require(fe, "min_events_per_cluster", "filters.events"), "filters.events.min_events_per_cluster", min_v=0)
    max_pair_station_ratio = _require_num(_require(fe, "max_pair_station_ratio", "filters.events"), "filters.events.max_pair_station_ratio")
    ratio_filter_phase = _require_str(_require(fe, "ratio_filter_phase", "filters.events"), "filters.events.ratio_filter_phase").lower()
    if ratio_filter_phase not in {"before", "after"}:
        raise _err("filters.events.ratio_filter_phase", "must be 'before' or 'after'")

    # Optional: post-Phase1 linearization error diagnostic / filter
    # This is intentionally OPTIONAL (absent -> disabled) to avoid breaking existing configs.
    lin_cfg = fe.get("linearization_error", None)
    lin_enable = False
    lin_phase = "after_phase1"
    lin_batch_size = 50000
    lin_sample_size = 200000
    lin_log_every = 25
    lin_max_ratio = None
    if isinstance(lin_cfg, dict):
        lin_enable = bool(lin_cfg.get("enabled", False))
        lin_phase = str(lin_cfg.get("phase", "after_phase1")).strip().lower()
        if lin_phase not in {"before", "after_phase1"}:
            raise _err("filters.events.linearization_error.phase", "must be 'before' or 'after_phase1'")
        # Ratio-only linearization filter:
        # - old `max_error` (seconds) threshold was found to be hard to tune and not practically useful.
        # - we now support only a dimensionless ratio threshold `max_ratio`.
        if "mode" in lin_cfg and lin_cfg.get("mode", None) is not None:
            raise _err("filters.events.linearization_error.mode", "removed; use max_ratio only")
        # batch_size / sample_size / log_every are only used when enabled; validate if present
        if "batch_size" in lin_cfg and lin_cfg["batch_size"] is not None:
            lin_batch_size = _req_int(lin_cfg["batch_size"], "filters.events.linearization_error.batch_size", min_v=1)
        if "sample_size" in lin_cfg and lin_cfg["sample_size"] is not None:
            lin_sample_size = _req_int(lin_cfg["sample_size"], "filters.events.linearization_error.sample_size", min_v=0)
        if "log_every_batches" in lin_cfg and lin_cfg["log_every_batches"] is not None:
            lin_log_every = _req_int(lin_cfg["log_every_batches"], "filters.events.linearization_error.log_every_batches", min_v=1)
        if "max_error" in lin_cfg and lin_cfg.get("max_error", None) is not None:
            raise _err("filters.events.linearization_error.max_error", "removed; use max_ratio (dimensionless) instead")
        if "max_ratio" in lin_cfg:
            v = lin_cfg.get("max_ratio", None)
            if v is None:
                lin_max_ratio = None
            else:
                vv = _require_num(v, "filters.events.linearization_error.max_ratio")
                if not (vv > 0.0):
                    raise _err("filters.events.linearization_error.max_ratio", "must be > 0 or null")
                lin_max_ratio = float(vv)
        if lin_enable and not (lin_max_ratio is not None and lin_max_ratio > 0.0):
            raise _err("filters.events.linearization_error.max_ratio", "required and must be > 0 when enabled")

    residual_enabled = _require_bool(_require(fr, "enabled", "filters.residual"), "filters.residual.enabled")
    # Require presence of other keys; if disabled they may be null.
    method_v = _require(fr, "method", "filters.residual")
    mad_sigma_v = _require(fr, "mad_sigma", "filters.residual")
    abs_max_v = _require(fr, "abs_max", "filters.residual")
    if residual_enabled:
        residual_method = _require_str(method_v, "filters.residual.method").lower()
        residual_mad_sigma = _require_num(mad_sigma_v, "filters.residual.mad_sigma")
        residual_abs_max = _require_num(abs_max_v, "filters.residual.abs_max")
    else:
        residual_method = None
        residual_mad_sigma = None
        residual_abs_max = None

    # ---- batching ----
    bt = _require_dict(_require(inf, "batching", "inference"), "inference.batching")
    bs = _require_dict(_require(bt, "standard", "inference.batching"), "inference.batching.standard")
    eb = _require_dict(_require(bt, "event_batches", "inference.batching"), "inference.batching.event_batches")

    batch_size_warmup = _req_int(_require(bs, "warmup", "batching.standard"), "batching.standard.warmup", min_v=1)
    batch_size_sgld = _req_int(_require(bs, "sgld", "batching.standard"), "batching.standard.sgld", min_v=1)
    # Optional: allow standard batching without the per-epoch random permutation (saves huge memory traffic for big N).
    # Default to True for backward compatibility.
    batch_shuffle = True
    if isinstance(bs, dict) and ("shuffle" in bs):
        try:
            batch_shuffle = _require_bool(bs.get("shuffle"), "batching.standard.shuffle")
        except Exception:
            batch_shuffle = True

    event_batches_enabled = _require_bool(_require(eb, "enabled", "batching.event_batches"), "batching.event_batches.enabled")
    ev_size_v = _require(eb, "events_per_batch", "batching.event_batches")
    ev_edges_v = _require(eb, "max_edges_per_batch", "batching.event_batches")
    bucket_reorder_all = _require_bool(_require(eb, "bucket_reorder_all", "batching.event_batches"), "batching.event_batches.bucket_reorder_all")
    bucket_reuse_epochs_v = _require(eb, "bucket_reuse_epochs", "batching.event_batches")
    if event_batches_enabled:
        event_batch_size = _req_int(ev_size_v, "batching.event_batches.events_per_batch", min_v=1)
        event_batch_max_edges = _req_int(ev_edges_v, "batching.event_batches.max_edges_per_batch", min_v=1)
        bucket_reuse_epochs = _req_int(bucket_reuse_epochs_v, "batching.event_batches.bucket_reuse_epochs", min_v=1)
    else:
        # allow nulls when disabled
        event_batch_size = 0
        event_batch_max_edges = 0
        bucket_reuse_epochs = 0

    # ---- materialize legacy flat keys (implementation detail) ----
    params["likelihood"] = lk_type
    params["phase_unc"] = phase_unc
    params["_student_t_nu"] = float(student_t_nu)
    params["_student_t_scale_enabled"] = bool(student_t_scale_enabled)
    params["_student_t_scale_nu"] = float(student_t_scale_nu)
    params["_student_t_scale_update_every_epochs"] = int(student_t_scale_update_every)
    params["_student_t_scale_batch_size"] = int(student_t_scale_batch_size)
    params["_student_t_scale_init"] = str(student_t_scale_init)
    params["_student_t_scale_min_lambda"] = float(student_t_scale_min_lambda)
    params["_student_t_scale_max_lambda"] = float(student_t_scale_max_lambda)

    # Correlated forward-model error latent (optional)
    params["_corr_error_enabled"] = bool(corr_enabled)
    params["_corr_error_r"] = int(corr_r)
    params["_corr_error_tau_s"] = [float(corr_tau_ps[0]), float(corr_tau_ps[1])]
    params["_corr_error_rho_ps"] = float(corr_rho_ps)
    params["_corr_error_station_basis_enabled"] = bool(corr_sta_basis_enabled)
    params["_corr_error_station_basis_ell_km"] = float(corr_sta_ell_km)
    params["_corr_error_station_basis_jitter"] = float(corr_sta_jitter)
    params["_corr_error_station_basis_method"] = str(corr_sta_method)
    params["_corr_error_event_graph_enabled"] = bool(corr_graph_enabled)
    params["_corr_error_event_graph_source"] = str(corr_graph_source)
    params["_corr_error_event_graph_radius_km"] = float(corr_radius_km)
    params["_corr_error_event_graph_k"] = int(corr_k)
    params["_corr_error_event_graph_refresh_every_epochs"] = int(corr_refresh_every)
    params["_corr_error_event_graph_symmetrize"] = bool(corr_symmetrize)
    params["_corr_error_event_graph_cell_size_km"] = (float(corr_cell_size_km) if corr_cell_size_km is not None else None)
    params["_corr_error_event_graph_cell_hops"] = int(corr_cell_hops)
    params["_corr_error_event_graph_max_tries_per_neighbor"] = int(corr_max_tries_per_neighbor)
    params["_corr_error_event_graph_q_diag"] = float(corr_q_diag)
    params["_corr_error_event_graph_weighting"] = str(corr_weighting)
    params["_corr_error_event_graph_weight_ell_km"] = (float(corr_weight_ell_km) if corr_weight_ell_km is not None else None)
    params["_corr_error_event_graph_weight_eps_km"] = float(corr_weight_eps_km)
    params["_corr_error_event_graph_weight_normalize"] = bool(corr_weight_normalize)
    params["_corr_error_enable_in_phase1"] = bool(corr_enable_in_phase1)
    params["_corr_error_phase1_lr_mult"] = float(corr_phase1_lr_mult)
    params["_corr_error_hierarchical_tau_enabled"] = bool(hier_tau_enabled)
    params["_corr_error_hierarchical_tau_dof"] = float(hier_tau_dof)
    params["_corr_error_hierarchical_tau_scale_ps"] = [float(hier_tau_scale_ps[0]), float(hier_tau_scale_ps[1])]
    params["_corr_error_hierarchical_tau_update_every"] = int(hier_tau_update_every)
    params["_corr_error_hierarchical_tau_damping"] = float(hier_tau_damping)
    params["_corr_error_hierarchical_tau_start_after_epochs"] = int(hier_tau_start_after)
    params["_corr_error_hierarchical_tau_min_tau_s"] = (
        [float(hier_tau_min_ps[0]), float(hier_tau_min_ps[1])] if hier_tau_min_ps is not None else None
    )
    params["_corr_error_hierarchical_tau_max_tau_s"] = (
        [float(hier_tau_max_ps[0]), float(hier_tau_max_ps[1])] if hier_tau_max_ps is not None else None
    )
    # corr_error ESS (optional)
    params["_corr_error_ess_enabled"] = bool(corr_ess_enabled)
    params["_corr_error_ess_update_every"] = int(corr_ess_update_every)
    params["_corr_error_ess_start_after_epochs"] = int(corr_ess_start_after)
    params["_corr_error_ess_sweeps_per_update"] = int(corr_ess_sweeps)
    params["_corr_error_ess_batch_size"] = int(corr_ess_batch_size)
    params["_corr_error_ess_max_bracket_steps"] = int(corr_ess_max_bracket_steps)
    params["_corr_error_ess_block_by_station"] = bool(corr_ess_block_by_station)
    params["_corr_error_ess_top_k_stations"] = int(corr_ess_top_k_stations)
    params["_corr_error_ess_seed"] = int(corr_ess_seed)
    params["_corr_error_ess_freeze_sampler_group"] = bool(corr_ess_freeze_sampler_group)

    # Collapsed shared-event random effects (optional)
    params["_shared_event_re_enabled"] = bool(se_enabled)
    params["_shared_event_re_grouping"] = str(se_grouping)
    params["_shared_event_re_cluster_mode"] = str(se_cluster_mode)
    params["_shared_event_re_cluster_k"] = int(se_cluster_k)
    params["_shared_event_re_tau_s"] = [float(se_tau_ps[0]), float(se_tau_ps[1])]
    params["_shared_event_re_hierarchical"] = bool(se_hier_enabled)
    params["_shared_event_re_tau_event_s"] = [float(se_tau_event_ps[0]), float(se_tau_event_ps[1])]
    params["_shared_event_re_tau_cluster_s"] = [float(se_tau_cluster_ps[0]), float(se_tau_cluster_ps[1])]
    params["_shared_event_re_rho_ps"] = float(se_rho_ps)
    params["_shared_event_re_joint_ps"] = bool(se_joint_ps)
    params["_shared_event_re_max_nodes_per_group"] = int(se_max_nodes_per_group)
    params["_shared_event_re_max_rows_per_group"] = int(se_max_rows_per_group)
    params["_shared_event_re_fallback_to_diag"] = bool(se_fallback_to_diag)
    params["_shared_event_re_jitter0"] = float(se_jitter0)
    params["_shared_event_re_jitter_max"] = float(se_jitter_max)
    params["_shared_event_re_cache_max_entries"] = int(se_cache_max_entries)
    params["_shared_event_re_cache_log_every"] = int(se_cache_log_every)
    if se_gpu_enable is not None:
        params["_shared_event_re_gpu_enable"] = bool(se_gpu_enable)
    params["_shared_event_re_gpu_max_groups_per_batch"] = int(se_gpu_max_groups_per_batch)
    params["_shared_event_re_gpu_profile"] = bool(se_gpu_profile)
    params["_shared_event_re_gpu_debug_max_groups"] = int(se_gpu_debug_max_groups)
    params["_shared_event_re_gpu_max_edges_per_batch"] = int(se_gpu_max_edges_per_batch)
    params["_shared_event_re_gpu_reuse_pcg_init"] = bool(se_gpu_reuse_pcg_init)
    params["_shared_event_re_whitening_enabled"] = bool(se_whiten_enabled)
    params["_shared_event_re_whitening_edge_weighting"] = str(se_whiten_edge_weighting)
    params["_shared_event_re_whitening_edge_weight_ell_km"] = float(se_whiten_edge_weight_ell_km)
    params["_shared_event_re_whitening_edge_weight_eps_km"] = float(se_whiten_edge_weight_eps_km)
    params["_shared_event_re_whitening_edge_weight_power"] = float(se_whiten_edge_weight_power)
    params["_shared_event_re_whitening_edge_weight_scale_km"] = float(se_whiten_edge_weight_scale_km)
    params["_shared_event_re_whitening_edge_weight_global_scale"] = float(se_whiten_edge_weight_global_scale)
    params["_shared_event_re_whitening_edge_weight_normalize"] = bool(se_whiten_edge_weight_normalize)
    params["_shared_event_re_whitening_cache_max_entries"] = int(se_whiten_cache_max_entries)
    params["_shared_event_re_whitening_solver"] = str(se_whiten_solver)
    params["_shared_event_re_whitening_pcg_max_iters"] = int(se_whiten_pcg_max_iters)
    params["_shared_event_re_whitening_pcg_tol"] = float(se_whiten_pcg_tol)
    params["_shared_event_re_whitening_pcg_min_iters"] = int(se_whiten_pcg_min_iters)
    params["_shared_event_re_whitening_precompute"] = bool(se_whiten_precompute)
    params["_shared_event_re_whitening_precompute_device"] = str(se_whiten_precompute_device)
    params["_shared_event_re_whitening_pcg_batched"] = bool(se_whiten_pcg_batched)
    params["_shared_event_re_whitening_pcg_bucket_nodes"] = list(se_whiten_pcg_bucket_nodes)
    params["_shared_event_re_stats_log_every_epochs"] = int(se_stats_log_every_epochs)
    params["_shared_event_re_auto_tune_nodes_cap"] = bool(se_auto_tune_nodes_cap)
    params["_shared_event_re_auto_tune_nodes_max"] = int(se_auto_tune_nodes_max)
    params["_shared_event_re_auto_tune_rows_cap"] = bool(se_auto_tune_rows_cap)
    params["_shared_event_re_auto_tune_rows_max"] = int(se_auto_tune_rows_max)
    params["_shared_event_re_solver"] = str(se_solver)
    params["_shared_event_re_drop_logdet"] = bool(se_drop_logdet)
    params["_shared_event_re_pcg_max_iters"] = int(se_pcg_max_iters)
    params["_shared_event_re_pcg_tol"] = float(se_pcg_tol)
    params["_shared_event_re_diag_log_every_epochs"] = int(se_diag_log_every_epochs)
    params["_shared_event_re_diag_max_groups"] = int(se_diag_max_groups)
    params["_shared_event_re_diag_max_rows_per_group"] = int(se_diag_max_rows_per_group)
    params["_shared_event_re_diag_max_nodes"] = int(se_diag_max_nodes)
    params["_shared_event_re_diag_seed"] = int(se_diag_seed)
    params["_shared_event_re_station_phase_enabled"] = bool(se_sp_enabled)
    params["_shared_event_re_station_phase_tau_s"] = [float(se_sp_tau_ps[0]), float(se_sp_tau_ps[1])]

    # Slowness random effects (scalar separation mode)
    params["_slowness_re_enabled"] = bool(sl_enabled)
    params["_slowness_re_mode"] = str(sl_mode)
    params["_slowness_re_grouping"] = str(sl_grouping)
    params["_slowness_re_tau_s"] = [float(sl_tau_ps[0]), float(sl_tau_ps[1])]
    params["_slowness_re_tau_station_s"] = [float(sl_tau_station_ps[0]), float(sl_tau_station_ps[1])]
    params["_slowness_re_tau_units"] = str(sl_tau_units)
    params["_slowness_re_max_rows_per_group"] = int(sl_max_rows_per_group)
    params["_slowness_re_max_nodes_per_group"] = int(sl_max_nodes_per_group)
    params["_slowness_re_fallback_to_diag"] = bool(sl_fallback_to_diag)
    params["_slowness_re_pcg_max_iters"] = int(sl_pcg_max_iters)
    params["_slowness_re_pcg_tol"] = float(sl_pcg_tol)
    params["_slowness_re_sep_cap_km"] = float(sl_sep_cap_km)
    params["_slowness_re_jitter0"] = float(sl_jitter0)
    params["_slowness_re_freeze_g_at_map"] = bool(sl_freeze_g_at_map)
    params["_slowness_re_vp_km_s"] = float(sl_vp_km_s)
    params["_slowness_re_vs_km_s"] = float(sl_vs_km_s)
    params["_slowness_re_explicit_enabled"] = bool(sl_enabled and (sl_mode == "component_station_explicit"))
    params["_slowness_re_ess_enabled"] = bool(sl_ess_enabled)
    params["_slowness_re_ess_update_every"] = int(sl_ess_update_every)
    params["_slowness_re_ess_start_after"] = int(sl_ess_start_after)
    params["_slowness_re_ess_sweeps_per_update"] = int(sl_ess_sweeps)
    params["_slowness_re_ess_batch_size"] = int(sl_ess_batch_size)
    params["_slowness_re_ess_max_bracket_steps"] = int(sl_ess_max_bracket_steps)
    params["_slowness_re_ess_seed"] = int(sl_ess_seed)
    params["_slowness_re_ess_freeze_sampler_group"] = bool(sl_ess_freeze_sampler_group)
    if sl_gpu_enable is not None:
        params["_slowness_re_gpu_enable"] = bool(sl_gpu_enable)
    params["_slowness_re_gpu_max_groups_per_batch"] = int(sl_gpu_max_groups_per_batch)
    params["_slowness_re_gpu_max_edges_per_batch"] = int(sl_gpu_max_edges_per_batch)
    params["_slowness_re_gpu_profile"] = bool(sl_gpu_profile)
    params["_slowness_re_gpu_debug_max_groups"] = int(sl_gpu_debug_max_groups)
    params["_slowness_re_inducing_plan_enable"] = bool(sl_plan_enabled)

    # Optional: DD-graph random effects (explicit event latents)
    dd_cfg = lk.get("dd_graph_re", None)
    if isinstance(dd_cfg, dict):
        if "enabled" in dd_cfg and dd_cfg.get("enabled", None) is not None:
            dd_re_enabled = bool(_require_bool(dd_cfg.get("enabled"), "model.likelihood.dd_graph_re.enabled"))
        if "tau_s" in dd_cfg and dd_cfg.get("tau_s", None) is not None:
            dd_re_tau_ps = _require_float_list(dd_cfg.get("tau_s"), "model.likelihood.dd_graph_re.tau_s", length=2)
        if "q_diag" in dd_cfg and dd_cfg.get("q_diag", None) is not None:
            dd_re_q_diag = float(_require_num(dd_cfg.get("q_diag"), "model.likelihood.dd_graph_re.q_diag"))
        if "weight_by_pair_count" in dd_cfg and dd_cfg.get("weight_by_pair_count", None) is not None:
            dd_re_weight_by_pair_count = bool(_require_bool(dd_cfg.get("weight_by_pair_count"), "model.likelihood.dd_graph_re.weight_by_pair_count"))
        ess_cfg = dd_cfg.get("ess", None)
        if isinstance(ess_cfg, dict):
            if "enabled" in ess_cfg and ess_cfg.get("enabled", None) is not None:
                dd_re_ess_enabled = bool(_require_bool(ess_cfg.get("enabled"), "model.likelihood.dd_graph_re.ess.enabled"))
            if "update_every_epochs" in ess_cfg and ess_cfg.get("update_every_epochs", None) is not None:
                dd_re_ess_update_every = int(_require_num(ess_cfg.get("update_every_epochs"), "model.likelihood.dd_graph_re.ess.update_every_epochs"))
                if dd_re_ess_update_every < 1:
                    raise _err("model.likelihood.dd_graph_re.ess.update_every_epochs", "must be >= 1")
            if "start_after_epochs" in ess_cfg and ess_cfg.get("start_after_epochs", None) is not None:
                dd_re_ess_start_after = int(_require_num(ess_cfg.get("start_after_epochs"), "model.likelihood.dd_graph_re.ess.start_after_epochs"))
                if dd_re_ess_start_after < 0:
                    raise _err("model.likelihood.dd_graph_re.ess.start_after_epochs", "must be >= 0")
            if "sweeps_per_update" in ess_cfg and ess_cfg.get("sweeps_per_update", None) is not None:
                dd_re_ess_sweeps = int(_require_num(ess_cfg.get("sweeps_per_update"), "model.likelihood.dd_graph_re.ess.sweeps_per_update"))
                if dd_re_ess_sweeps < 1:
                    raise _err("model.likelihood.dd_graph_re.ess.sweeps_per_update", "must be >= 1")
            if "max_bracket_steps" in ess_cfg and ess_cfg.get("max_bracket_steps", None) is not None:
                dd_re_ess_max_bracket_steps = int(_require_num(ess_cfg.get("max_bracket_steps"), "model.likelihood.dd_graph_re.ess.max_bracket_steps"))
                if dd_re_ess_max_bracket_steps < 8:
                    raise _err("model.likelihood.dd_graph_re.ess.max_bracket_steps", "must be >= 8")
            if "seed" in ess_cfg and ess_cfg.get("seed", None) is not None:
                dd_re_ess_seed = int(_require_num(ess_cfg.get("seed"), "model.likelihood.dd_graph_re.ess.seed"))
            if "freeze_sampler_group" in ess_cfg and ess_cfg.get("freeze_sampler_group", None) is not None:
                dd_re_ess_freeze_sampler_group = bool(_require_bool(ess_cfg.get("freeze_sampler_group"), "model.likelihood.dd_graph_re.ess.freeze_sampler_group"))
    if dd_re_enabled:
        if dd_re_tau_ps[0] < 0.0 or dd_re_tau_ps[1] < 0.0:
            raise _err("model.likelihood.dd_graph_re.tau_s", "must be >= 0")
        if not math.isfinite(dd_re_q_diag) or dd_re_q_diag < 0.0:
            raise _err("model.likelihood.dd_graph_re.q_diag", "must be finite and >= 0")
    params["_dd_graph_re_enabled"] = bool(dd_re_enabled)
    params["_dd_graph_re_tau_s"] = [float(dd_re_tau_ps[0]), float(dd_re_tau_ps[1])]
    params["_dd_graph_re_q_diag"] = float(dd_re_q_diag)
    params["_dd_graph_re_weight_by_pair_count"] = bool(dd_re_weight_by_pair_count)
    params["_dd_graph_re_ess_enabled"] = bool(dd_re_ess_enabled)
    params["_dd_graph_re_ess_update_every"] = int(dd_re_ess_update_every)
    params["_dd_graph_re_ess_start_after"] = int(dd_re_ess_start_after)
    params["_dd_graph_re_ess_sweeps_per_update"] = int(dd_re_ess_sweeps)
    params["_dd_graph_re_ess_max_bracket_steps"] = int(dd_re_ess_max_bracket_steps)
    params["_dd_graph_re_ess_seed"] = int(dd_re_ess_seed)
    params["_dd_graph_re_ess_freeze_sampler_group"] = bool(dd_re_ess_freeze_sampler_group)

    # Latent field (NNGP + ESS) materialized keys (optional)
    params["_latent_field_enabled"] = bool(lf_enabled)
    params["_latent_field_neighbor_m"] = int(lf_neighbor_m)
    params["_latent_field_by_station"] = bool(lf_by_station)
    params["_latent_field_max_stations_per_update"] = int(lf_max_stations_per_update)
    params["_latent_field_min_edges_per_station"] = int(lf_min_edges_per_station)
    params["_latent_field_parameterization"] = str(lf_parameterization)
    params["_latent_field_slowness_amp_s_per_km"] = [float(lf_slowness_amp_s_per_km[0]), float(lf_slowness_amp_s_per_km[1])]
    params["_latent_field_update_every_epochs"] = int(lf_update_every)
    params["_latent_field_active_phases"] = [int(x) for x in lf_active_phases]
    params["_latent_field_refresh_every_epochs"] = int(lf_refresh_every)
    params["_latent_field_edge_chunk_size"] = int(lf_edge_chunk)
    params["_latent_field_component_wise"] = bool(lf_component_wise)
    params["_latent_field_map_iters"] = int(lf_map_iters)
    params["_latent_field_map_damping"] = float(lf_map_damping)
    params["_latent_field_map_log_every_iter"] = int(lf_map_log_every_iter)
    params["_latent_field_map_log_n_edges"] = int(lf_map_log_n_edges)
    params["_latent_field_map_z_clip"] = float(lf_map_z_clip)
    params["_latent_field_map_sigma_mode"] = str(lf_map_sigma_mode)
    params["_latent_field_map_sigma_override"] = (
        [float(lf_map_sigma_override[0]), float(lf_map_sigma_override[1])] if lf_map_sigma_override is not None else None
    )
    params["_latent_field_map_sigma_clip_max"] = (
        [float(lf_map_sigma_clip_max[0]), float(lf_map_sigma_clip_max[1])] if lf_map_sigma_clip_max is not None else None
    )
    params["_latent_field_kernel"] = str(lf_kernel)
    params["_latent_field_nu"] = float(lf_nu)
    params["_latent_field_ell_km"] = [float(lf_ell_km[0]), float(lf_ell_km[1])]

    params["remove_duplicates"] = bool(remove_duplicates)
    params["max_abs_input_dt"] = float(max_abs_input_dt)
    params["dtime_thin_frac"] = float(dtime_thin_frac)
    params["flip_dt_sign"] = bool(flip_dt_sign)
    params["cc_min"] = float(cc_min)

    params["min_dtimes"] = int(min_dtimes)
    params["min_unique_phase_per_event"] = int(min_unique_phase_per_event)
    params["min_dtimes_per_pair"] = int(min_dtimes_per_pair)
    params["min_event_degree"] = int(min_event_degree)
    params["min_events_per_cluster"] = int(min_events_per_cluster)
    params["max_pair_station_ratio"] = float(max_pair_station_ratio)
    params["ratio_filter_phase"] = ratio_filter_phase

    # Optional linearization error diagnostic/filter (after Phase 1)
    params["linearization_error_enable"] = bool(lin_enable)
    params["linearization_error_phase"] = lin_phase
    params["linearization_error_batch_size"] = int(lin_batch_size)
    params["linearization_error_sample_size"] = int(lin_sample_size)
    params["linearization_error_log_every_batches"] = int(lin_log_every)
    params["linearization_error_max_ratio"] = (float(lin_max_ratio) if lin_max_ratio is not None else None)

    params["residual_filter_enable"] = bool(residual_enabled)
    params["residual_filter_method"] = residual_method
    params["residual_filter_mad_sigma"] = residual_mad_sigma
    params["residual_filter_abs_max"] = residual_abs_max

    params["batch_size_warmup"] = int(batch_size_warmup)
    params["batch_size_sgld"] = int(batch_size_sgld)
    params["batch_shuffle"] = bool(batch_shuffle)
    params["event_batch_enable"] = bool(event_batches_enabled)
    params["event_batch_size"] = int(event_batch_size)
    params["event_batch_max_edges"] = int(event_batch_max_edges)
    params["event_bucket_reorder_all"] = bool(bucket_reorder_all)
    params["event_bucket_reuse_epochs"] = int(bucket_reuse_epochs)

    return params


def validate_and_materialize_block4(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Block 4 (hard-break schema): strict nested `inference.diagnostics`, `inference.runtime`, `inference.safety`.

    Required nested blocks:
      - diagnostics.{pair_count_stats_enable, sgld_log_gnoise, [optional] sgld_log_temperature, display_precond_every, ess_online.{enabled,every_n_samples,n_events,seed,window,max_lag,dims}, fim.{enable_phase2, filter_threshold}, svrg.{enabled}}
      - runtime.{phase_mads_interval.{phase2,phase3,phase4}, cuda_empty_cache_every, reset_batch_numbers, clear_samples_on_reset, min_samples_to_save, verbose, cluster_events}
      - safety.{max_abs_dX}

    No defaults: all keys must be present (some values may be null when feature disabled).
    Materializes legacy flat keys for existing code paths.

    Note: optional extra diagnostics sub-blocks may also be present and are ignored by schema materialization, e.g.:
      - diagnostics.station_basis_ell_estimate.{enabled,n_rows,seed,batch_size,n_bins,frac_of_plateau,rank_frac_var,rank_r_max}
      - diagnostics.shared_event_latent_event_ell_estimate.{enabled,n_rows,seed,batch_size,n_bins,frac_of_plateau,ridge,rtol,maxiter,variogram_pairs}
    """
    forbidden_present = [k for k in _FORBIDDEN_BLOCK4_TOPLEVEL_KEYS if k in params]
    if forbidden_present:
        raise ValueError(
            "Legacy Block-4 keys are not allowed. Move these under nested blocks:\n"
            + "\n".join(f"- {k}" for k in sorted(forbidden_present))
        )

    # SSST removed entirely (no backward compatibility).
    if "ssst" in params:
        raise _err("ssst", "removed; delete this block (no backward compatibility)")

    # Hard break: these blocks moved under `inference.*`.
    if "diagnostics" in params:
        raise _err("diagnostics", "moved; put this under `inference.diagnostics` (top-level `diagnostics` is no longer supported)")
    if "runtime" in params:
        raise _err("runtime", "moved; put this under `inference.runtime` (top-level `runtime` is no longer supported)")
    if "safety" in params:
        raise _err("safety", "moved; put this under `inference.safety` (top-level `safety` is no longer supported)")

    inf = _require_dict(_require(params, "inference", "inference"), "inference")

    dg = _require_dict(_require(inf, "diagnostics", "inference"), "inference.diagnostics")
    pair_count_stats_enable = _require_bool(_require(dg, "pair_count_stats_enable", "diagnostics"), "diagnostics.pair_count_stats_enable")
    sgld_log_gnoise = _require_bool(_require(dg, "sgld_log_gnoise", "diagnostics"), "diagnostics.sgld_log_gnoise")
    # Optional: SGHMC effective temperature diagnostic (defaults to False for backward compatibility)
    sgld_log_temperature = bool(dg.get("sgld_log_temperature", False))
    display_precond_every = int(_require_num(_require(dg, "display_precond_every", "diagnostics"), "diagnostics.display_precond_every"))
    if display_precond_every < 1:
        raise _err("diagnostics.display_precond_every", "must be >= 1")

    # Optional: W&B logging categories. Purpose:
    # - Organize metrics into logical groups
    # - Avoid computing expensive metrics unless they'll actually be logged
    #
    # This is OPTIONAL and defaults to "log everything we used to" for backward compatibility.
    # Runtime logging is additionally gated by params["_wandb_runtime_enabled"] (set in cli after init).
    wb_cfg = dg.get("wandb", None)
    wb_diag_enabled = True
    wb_groups: set[str] = {"all"}  # default: enable everything
    if isinstance(wb_cfg, dict):
        if "enabled" in wb_cfg and wb_cfg.get("enabled", None) is not None:
            wb_diag_enabled = bool(wb_cfg.get("enabled", True))
        g = wb_cfg.get("groups", None)
        if isinstance(g, dict):
            wb_groups = {str(k).strip().lower() for k, v in g.items() if bool(v)}
            if not wb_groups:
                wb_groups = set()
        elif isinstance(g, list):
            wb_groups = {str(x).strip().lower() for x in g if str(x).strip()}
            if not wb_groups:
                wb_groups = set()
        elif g is None:
            pass
        else:
            raise _err("diagnostics.wandb.groups", "expected dict[str,bool] or list[str] or null")

    # Optional: online ESS/IACT diagnostics during sampling (Phase 4).
    ess_online = dg.get("ess_online", None)
    ess_online = _require_dict(ess_online, "diagnostics.ess_online") if isinstance(ess_online, dict) else None
    ess_online_enabled = bool(ess_online.get("enabled", False)) if isinstance(ess_online, dict) else False
    if ess_online_enabled:
        ess_every = int(_require_num(_require(ess_online, "every_n_samples", "diagnostics.ess_online"), "diagnostics.ess_online.every_n_samples"))
        if ess_every < 1:
            raise _err("diagnostics.ess_online.every_n_samples", "must be >= 1")
        ess_n_events = int(_require_num(_require(ess_online, "n_events", "diagnostics.ess_online"), "diagnostics.ess_online.n_events"))
        if ess_n_events < 1:
            raise _err("diagnostics.ess_online.n_events", "must be >= 1")
        ess_seed = int(_require_num(_require(ess_online, "seed", "diagnostics.ess_online"), "diagnostics.ess_online.seed"))
        ess_window = int(_require_num(_require(ess_online, "window", "diagnostics.ess_online"), "diagnostics.ess_online.window"))
        if ess_window < 8:
            raise _err("diagnostics.ess_online.window", "must be >= 8")
        ess_max_lag = int(_require_num(_require(ess_online, "max_lag", "diagnostics.ess_online"), "diagnostics.ess_online.max_lag"))
        if ess_max_lag < 1:
            raise _err("diagnostics.ess_online.max_lag", "must be >= 1")
        dims = _require(ess_online, "dims", "diagnostics.ess_online")
        if not isinstance(dims, list) or len(dims) == 0:
            raise _err("diagnostics.ess_online.dims", "expected non-empty list of ints in {0,1,2,3}")
        for i, d in enumerate(dims):
            if not isinstance(d, int) or d not in {0, 1, 2, 3}:
                raise _err(f"diagnostics.ess_online.dims[{i}]", "must be one of {0,1,2,3}")
        ess_dims = list(dims)
    else:
        ess_every = 0
        ess_n_events = 0
        ess_seed = 0
        ess_window = 0
        ess_max_lag = 0
        ess_dims = [0, 1, 2]

    fim = _require_dict(_require(dg, "fim", "diagnostics"), "diagnostics.fim")
    fim_enable_phase2 = _require_bool(_require(fim, "enable_phase2", "diagnostics.fim"), "diagnostics.fim.enable_phase2")
    fim_filter_threshold = _require_num(_require(fim, "filter_threshold", "diagnostics.fim"), "diagnostics.fim.filter_threshold")

    svrg = _require_dict(_require(dg, "svrg", "diagnostics"), "diagnostics.svrg")
    svrg_enable = _require_bool(_require(svrg, "enabled", "diagnostics.svrg"), "diagnostics.svrg.enabled")

    rt = _require_dict(_require(inf, "runtime", "inference"), "inference.runtime")
    pm = _require_dict(_require(rt, "phase_mads_interval", "runtime"), "runtime.phase_mads_interval")
    phase2_mads_interval = int(_require_num(_require(pm, "phase2", "runtime.phase_mads_interval"), "runtime.phase_mads_interval.phase2"))
    phase3_mads_interval = int(_require_num(_require(pm, "phase3", "runtime.phase_mads_interval"), "runtime.phase_mads_interval.phase3"))
    phase4_mads_interval = int(_require_num(_require(pm, "phase4", "runtime.phase_mads_interval"), "runtime.phase_mads_interval.phase4"))
    if phase2_mads_interval < 0 or phase3_mads_interval < 0 or phase4_mads_interval < 0:
        raise _err("runtime.phase_mads_interval", "intervals must be >= 0")
    cuda_empty_cache_every = int(_require_num(_require(rt, "cuda_empty_cache_every", "runtime"), "runtime.cuda_empty_cache_every"))
    if cuda_empty_cache_every < 0:
        raise _err("runtime.cuda_empty_cache_every", "must be >= 0")
    reset_batch_numbers = _require_bool(_require(rt, "reset_batch_numbers", "runtime"), "runtime.reset_batch_numbers")
    clear_samples_on_reset = _require_bool(_require(rt, "clear_samples_on_reset", "runtime"), "runtime.clear_samples_on_reset")
    min_samples_to_save = int(_require_num(_require(rt, "min_samples_to_save", "runtime"), "runtime.min_samples_to_save"))
    if min_samples_to_save < 0:
        raise _err("runtime.min_samples_to_save", "must be >= 0")
    verbose = _require_bool(_require(rt, "verbose", "runtime"), "runtime.verbose")
    cluster_events = _require_bool(_require(rt, "cluster_events", "runtime"), "runtime.cluster_events")
    # Optional: base RNG seed offset for this run/chain (used to decorrelate multi-GPU independent chains).
    # If absent, defaults to 0 for backward compatibility.
    runtime_seed = 0
    try:
        if "seed" in rt and rt.get("seed", None) is not None:
            runtime_seed = int(_require_num(rt.get("seed"), "runtime.seed"))
    except Exception:
        runtime_seed = 0

    # Optional: gauge projection (remove translation mode by projecting out mean gradient/noise).
    # This imposes a hard constraint on the mean update.
    gp_enable = False
    gp_mode = "global"  # or "cluster" (per connected component) when cluster ids are available
    gp_dims = [0, 1, 2]  # default: spatial only; include 3 to also constrain origin-time mean
    gp_apply_noise = True
    gp_apply_momentum = True
    try:
        gp = rt.get("gauge_projection", None)
        if isinstance(gp, dict):
            if "enabled" in gp and gp.get("enabled", None) is not None:
                gp_enable = bool(gp.get("enabled", False))
            if "mode" in gp and gp.get("mode", None) is not None:
                gp_mode = str(gp.get("mode", "global")).strip().lower()
            if gp_mode not in {"global", "cluster"}:
                raise _err("runtime.gauge_projection.mode", "supported: 'global' or 'cluster'")
            if "dims" in gp and gp.get("dims", None) is not None:
                dv = gp.get("dims")
                if not isinstance(dv, list) or len(dv) == 0:
                    raise _err("runtime.gauge_projection.dims", "expected non-empty list of ints in {0,1,2,3}")
                for i, d in enumerate(dv):
                    if (not isinstance(d, int)) or d not in {0, 1, 2, 3}:
                        raise _err(f"runtime.gauge_projection.dims[{i}]", "must be one of {0,1,2,3}")
                gp_dims = list(dv)
            if "apply_noise" in gp and gp.get("apply_noise", None) is not None:
                gp_apply_noise = bool(gp.get("apply_noise", True))
            if "apply_momentum" in gp and gp.get("apply_momentum", None) is not None:
                gp_apply_momentum = bool(gp.get("apply_momentum", True))
    except Exception as e:
        # Keep backward compatibility: if user provided an invalid gauge_projection block, error out clearly.
        raise

    saf = _require_dict(_require(inf, "safety", "inference"), "inference.safety")
    max_abs_dX_v = _require(saf, "max_abs_dX", "safety")
    if max_abs_dX_v is None:
        max_abs_dX = None
    else:
        max_abs_dX = _require_float_list(max_abs_dX_v, "safety.max_abs_dX", length=4)

    # ---- materialize legacy flat keys ----
    params["pair_count_stats_enable"] = bool(pair_count_stats_enable)
    params["sgld_log_gnoise"] = bool(sgld_log_gnoise)
    params["sgld_log_temperature"] = bool(sgld_log_temperature)
    params["display_precond_every"] = int(display_precond_every)
    # W&B categories (logging + compute gating)
    params["_wandb_diag_enabled"] = bool(wb_diag_enabled)
    params["_wandb_diag_groups"] = sorted(list(wb_groups))
    params["ess_online_enabled"] = bool(ess_online_enabled)
    params["ess_online_every_n_samples"] = int(ess_every)
    params["ess_online_n_events"] = int(ess_n_events)
    params["ess_online_seed"] = int(ess_seed)
    params["ess_online_window"] = int(ess_window)
    params["ess_online_max_lag"] = int(ess_max_lag)
    params["ess_online_dims"] = list(ess_dims)
    params["fim_enable_phase2"] = bool(fim_enable_phase2)
    params["fim_filter_threshold"] = float(fim_filter_threshold)
    params["svrg_enable"] = bool(svrg_enable)

    params["phase2_mads_interval"] = int(phase2_mads_interval)
    params["phase3_mads_interval"] = int(phase3_mads_interval)
    params["phase4_mads_interval"] = int(phase4_mads_interval)
    params["cuda_empty_cache_every"] = int(cuda_empty_cache_every)
    params["reset_batch_numbers"] = bool(reset_batch_numbers)
    params["clear_samples_on_reset"] = bool(clear_samples_on_reset)
    params["min_samples_to_save"] = int(min_samples_to_save)
    params["verbose"] = bool(verbose)
    params["cluster_events"] = bool(cluster_events)
    params["runtime_seed"] = int(runtime_seed)
    params["gauge_project_enable"] = bool(gp_enable)
    params["gauge_project_mode"] = str(gp_mode)
    params["gauge_project_dims"] = list(gp_dims)
    params["gauge_project_apply_noise"] = bool(gp_apply_noise)
    params["gauge_project_apply_momentum"] = bool(gp_apply_momentum)

    if max_abs_dX is not None:
        params["max_abs_dX"] = max_abs_dX

    return params


def validate_and_materialize_block5(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Block 5 (hard-break schema): compute + dd-precision preconditioning toggle.

    Required:
      - inference.compute.devices: list of device specifiers, len>=1
        Supported entries:
          - integer CUDA device id (e.g. 0, 1, 2, ...)
          - -1 to force CPU
          - string "cpu" (case-insensitive) to force CPU
          - string "cuda:N" to select CUDA device N
          - numeric strings like "0" are accepted and treated as ints

    Also requires (if `inference.sampler` exists in params):
      - inference.sampler.dd_prec_enable: bool

    Forbids legacy top-level `devices` and `dd_prec_enable`.
    Materializes:
      - params['devices']
      - params['dd_prec_enable']
    """
    forbidden_present = [k for k in _FORBIDDEN_BLOCK5_TOPLEVEL_KEYS if k in params]
    if forbidden_present:
        raise ValueError(
            "Legacy Block-5 keys are not allowed. Move these under nested blocks:\n"
            + "\n".join(f"- {k}" for k in sorted(forbidden_present))
        )

    # Hard break: `compute` moved under `inference.compute`.
    if "compute" in params:
        raise _err("compute", "moved; put this under `inference.compute` (top-level `compute` is no longer supported)")

    inf = _require_dict(_require(params, "inference", "inference"), "inference")
    comp = _require_dict(_require(inf, "compute", "inference"), "inference.compute")
    devs_v = _require(comp, "devices", "inference.compute")
    if not isinstance(devs_v, list) or len(devs_v) < 1:
        raise _err("inference.compute.devices", "expected non-empty list of device specifiers")

    def _parse_dev(x: Any, path: str) -> int:
        # Canonical internal representation:
        # - CUDA device ids are non-negative ints
        # - CPU is encoded as -1
        if isinstance(x, int):
            return int(x)
        if isinstance(x, str):
            s = str(x).strip().lower()
            if s == "cpu":
                return -1
            if s.startswith("cuda:"):
                s2 = s.split("cuda:", 1)[1].strip()
                if not s2.isdigit():
                    raise _err(path, "expected 'cuda:<int>' or 'cpu' or integer")
                return int(s2)
            if s.isdigit():
                return int(s)
            raise _err(path, "supported: int CUDA id, -1/'cpu', 'cuda:<int>'")
        raise _err(path, f"expected int or str, got {type(x).__name__}")

    devs: List[int] = []
    for i, x in enumerate(devs_v):
        devs.append(_parse_dev(x, f"inference.compute.devices[{i}]"))

    # dd_prec_enable moved under sampler
    dd_prec_enable = None
    dd_prec_dims = [0, 1, 2, 3]  # default: apply DD degree normalization to all ΔX dims
    if "sampler" in inf:
        sampler = _require_dict(inf["sampler"], "inference.sampler")
        dd_prec_enable = _require_bool(_require(sampler, "dd_prec_enable", "inference.sampler"), "inference.sampler.dd_prec_enable")
        # Optional: restrict which ΔX dimensions use DD degree normalization in preconditioner stats.
        # Default is [0,1,2,3] for backward compatibility.
        try:
            v = sampler.get("dd_prec_dims", None)
            if v is None:
                dd_prec_dims = [0, 1, 2, 3]
            else:
                if not isinstance(v, list) or len(v) == 0:
                    raise _err("inference.sampler.dd_prec_dims", "expected a non-empty list of ints in {0,1,2,3} or null")
                out = []
                for i, x in enumerate(v):
                    if not isinstance(x, int):
                        raise _err(f"inference.sampler.dd_prec_dims[{i}]", f"expected int, got {type(x).__name__}")
                    if int(x) not in {0, 1, 2, 3}:
                        raise _err(f"inference.sampler.dd_prec_dims[{i}]", "must be one of {0,1,2,3}")
                    out.append(int(x))
                # de-dupe + stable sort
                dd_prec_dims = sorted(set(out))
        except ValueError:
            raise
        except Exception as e:
            raise _err("inference.sampler.dd_prec_dims", f"invalid: {e}")
    else:
        dd_prec_enable = False

    # materialize
    params["devices"] = devs
    params["dd_prec_enable"] = bool(dd_prec_enable)
    params["dd_prec_dims"] = list(dd_prec_dims)
    return params


