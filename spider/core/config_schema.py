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
    if "lr_mode" in sampler:
        raise _err("sampler.lr_mode", "removed; lr_mode is fixed to 'per_obs'")
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
    # lr_mode is fixed to 'per_obs' (always scale by number of observations).
    params["sampler_lr_mode"] = "per_obs"
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

    # Optional: generic overrides for parameter groups (core only).
    # For this group we only apply keys explicitly provided by the user.
    group_overrides: Dict[str, Dict[str, Any]] = {}
    any_group_override_active = False

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

    # Optional: distance-dependent sigma (linear in event-pair separation).
    sigma_dist_enable = False
    sigma_dist_slope_ps = [0.0, 0.0]  # seconds per km for [P,S]
    sigma_dist_min_ps = [0.0, 0.0]    # minimum sigma per phase (seconds)
    sigma_dist_max_km = None
    sigma_dist_cfg = lk.get("sigma_distance_linear", None)
    if isinstance(sigma_dist_cfg, dict):
        if "enabled" in sigma_dist_cfg and sigma_dist_cfg.get("enabled", None) is not None:
            sigma_dist_enable = bool(_require_bool(sigma_dist_cfg.get("enabled"), "model.likelihood.sigma_distance_linear.enabled"))
        if "slope_s_per_km" in sigma_dist_cfg and sigma_dist_cfg.get("slope_s_per_km", None) is not None:
            sigma_dist_slope_ps = _require_float_list(
                sigma_dist_cfg.get("slope_s_per_km"),
                "model.likelihood.sigma_distance_linear.slope_s_per_km",
                length=2,
            )
        if "min_sigma_s" in sigma_dist_cfg and sigma_dist_cfg.get("min_sigma_s", None) is not None:
            sigma_dist_min_ps = _require_float_list(
                sigma_dist_cfg.get("min_sigma_s"),
                "model.likelihood.sigma_distance_linear.min_sigma_s",
                length=2,
            )
        if "max_dist_km" in sigma_dist_cfg and sigma_dist_cfg.get("max_dist_km", None) is not None:
            sigma_dist_max_km = float(_require_num(sigma_dist_cfg.get("max_dist_km"), "model.likelihood.sigma_distance_linear.max_dist_km"))
            if not (sigma_dist_max_km >= 0.0):
                raise _err("model.likelihood.sigma_distance_linear.max_dist_km", "must be >= 0")
    if sigma_dist_enable:
        if not (sigma_dist_slope_ps[0] >= 0.0 and sigma_dist_slope_ps[1] >= 0.0):
            raise _err("model.likelihood.sigma_distance_linear.slope_s_per_km", "must be >= 0")
        if not (sigma_dist_min_ps[0] >= 0.0 and sigma_dist_min_ps[1] >= 0.0):
            raise _err("model.likelihood.sigma_distance_linear.min_sigma_s", "must be >= 0")

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

    if "student_t_scale" in lk:
        raise _err("model.likelihood.student_t_scale", "removed; delete this block from your config")

    # Tempering removed (start fresh; keep core residual distributions only).
    if "tempering" in lk:
        raise _err("model.likelihood.tempering", "removed; delete this block from your config")

    # Heteroscedastic sigma inflation removed (start fresh).
    if "sigma_inflation" in lk:
        raise _err("model.likelihood.sigma_inflation", "removed; delete this block from your config")

    # Residual-correlation block removed entirely (no backward compatibility).
    if "residual_correlation" in lk:
        raise _err("model.likelihood.residual_correlation", "removed; structured residual correlation models are no longer supported")

    # Structured likelihood components removed (keep only shared_event_re when enabled).
    if "shared_event_latent" in lk:
        raise _err("model.likelihood.shared_event_latent", "removed; delete this block from your config")
    if "latent_field" in lk:
        raise _err("model.likelihood.latent_field", "removed; delete this block from your config")
    if "corr_error" in lk:
        raise _err("model.likelihood.corr_error", "removed; delete this block from your config")
    if "slowness_re" in lk:
        raise _err("model.likelihood.slowness_re", "removed; delete this block from your config")
    if "dd_graph_re" in lk:
        raise _err("model.likelihood.dd_graph_re", "removed; delete this block from your config")

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
    se_abort_on_pcg_fallback = True
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
        if "abort_on_pcg_fallback" in se_cfg and se_cfg["abort_on_pcg_fallback"] is not None:
            se_abort_on_pcg_fallback = bool(se_cfg.get("abort_on_pcg_fallback", False))
        if "abort_on_fallback" in se_cfg and se_cfg["abort_on_fallback"] is not None:
            se_abort_on_pcg_fallback = bool(se_cfg.get("abort_on_fallback", False))
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

    if lk_correlated and not se_enabled:
        raise _err(
            "model.likelihood.type",
            "likelihood.type='correlated' requires model.likelihood.shared_event_re.enabled=true",
        )
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
    params["_sigma_distance_enable"] = bool(sigma_dist_enable)
    params["_sigma_distance_slope_ps"] = [float(sigma_dist_slope_ps[0]), float(sigma_dist_slope_ps[1])]
    params["_sigma_distance_min_sigma_ps"] = [float(sigma_dist_min_ps[0]), float(sigma_dist_min_ps[1])]
    params["_sigma_distance_max_dist_km"] = (float(sigma_dist_max_km) if sigma_dist_max_km is not None else None)
    params["_student_t_nu"] = float(student_t_nu)
    # Filters (materialize legacy flat keys used by data.py)
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
    params["ratio_filter_phase"] = str(ratio_filter_phase)

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
    params["_shared_event_re_abort_on_pcg_fallback"] = bool(se_abort_on_pcg_fallback)
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


