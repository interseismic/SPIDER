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
    Block 2 (hard-break schema): strict nested `inference.phases` + `inference.sampler`.

    Required:
      phases.phase1.epochs (int>=0), phases.phase1.lr (float>0)
      phases.phase2.epochs, phases.phase3.epochs, phases.phase4.epochs (int>=0)

      sampler.backend in {"psgld","sghmc"}
      sampler.lr (float>0), sampler.temperature (float>=0)
      sampler.lr_mode (optional str): "absolute" (default) or "per_obs" (psgld/sghmc/adaptive_sghmc helper; see below)
      sampler.preconditioning.enabled (bool)
      sampler.preconditioning.type (string) if enabled
        - supported (psgld): "rmsprop", "adam", "blockdiag_fisher" (alias: "matrix_ema")
        - supported (sghmc): "rmsprop", "adam"
      sampler.preconditioning.include_gamma (optional bool, default True):
        If true, include the diagonal Γ(θ) correction term in pSGLD when using diagonal
        preconditioners (rmsprop/adam). This is a low-cost correction from the pSGLD paper.
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
    """
    forbidden_present = [k for k in _FORBIDDEN_BLOCK2_TOPLEVEL_KEYS if k in params]
    if forbidden_present:
        raise ValueError(
            "Legacy Block-2 keys are not allowed. Move these under nested blocks:\n"
            + "\n".join(f"- {k}" for k in sorted(forbidden_present))
        )

    # Hard break: these blocks moved under `inference.*`.
    if "phases" in params:
        raise _err("phases", "moved; put this under `inference.phases` (top-level `phases` is no longer supported)")
    if "sampler" in params:
        raise _err("sampler", "moved; put this under `inference.sampler` (top-level `sampler` is no longer supported)")

    inf = _require_dict(_require(params, "inference", "inference"), "inference")

    phases = _require_dict(_require(inf, "phases", "inference"), "inference.phases")
    p1 = _require_dict(_require(phases, "phase1", "inference.phases"), "inference.phases.phase1")
    p2 = _require_dict(_require(phases, "phase2", "inference.phases"), "inference.phases.phase2")
    p3 = _require_dict(_require(phases, "phase3", "inference.phases"), "inference.phases.phase3")
    p4 = _require_dict(_require(phases, "phase4", "inference.phases"), "inference.phases.phase4")

    def _epochs(v: Any, path: str) -> int:
        if not isinstance(v, int):
            raise _err(path, f"expected integer, got {type(v).__name__}")
        if v < 0:
            raise _err(path, "must be >= 0")
        return int(v)

    phase1_epochs = _epochs(_require(p1, "epochs", "inference.phases.phase1"), "inference.phases.phase1.epochs")
    lr_warmup = _require_num(_require(p1, "lr", "inference.phases.phase1"), "inference.phases.phase1.lr")
    if not (lr_warmup > 0.0):
        raise _err("inference.phases.phase1.lr", "must be > 0")
    phase2_epochs = _epochs(_require(p2, "epochs", "inference.phases.phase2"), "inference.phases.phase2.epochs")
    phase3_epochs = _epochs(_require(p3, "epochs", "inference.phases.phase3"), "inference.phases.phase3.epochs")
    phase4_epochs = _epochs(_require(p4, "epochs", "inference.phases.phase4"), "inference.phases.phase4.epochs")

    sampler = _require_dict(_require(inf, "sampler", "inference"), "inference.sampler")
    backend = _require_str(_require(sampler, "backend", "inference.sampler"), "inference.sampler.backend").lower()
    if backend == "sgnht":
        raise _err("sampler.backend", "unsupported: 'sgnht' support has been removed; use 'sghmc' or 'psgld'")
    if backend not in {"psgld", "sghmc", "adaptive_sghmc"}:
        raise _err("sampler.backend", "supported: 'psgld', 'sghmc', 'adaptive_sghmc'")
    lr_sampler = _require_num(_require(sampler, "lr", "sampler"), "sampler.lr")
    if not (lr_sampler > 0.0):
        raise _err("sampler.lr", "must be > 0")
    lr_mode = str(sampler.get("lr_mode", "absolute")).strip().lower()
    if lr_mode not in {"absolute", "per_obs"}:
        raise _err("sampler.lr_mode", "supported: 'absolute', 'per_obs'")
    temperature = _require_num(_require(sampler, "temperature", "sampler"), "sampler.temperature")
    if temperature < 0.0:
        raise _err("sampler.temperature", "must be >= 0")

    # Optional: per-dimension LR multiplier for ΔT (origin time correction).
    # Implemented as a gradient scaler in the epoch runner so it works across Adam/PSGLD/SGHMC.
    dt_lr_mult = 1.0
    if "dt_lr_mult" in sampler and sampler.get("dt_lr_mult", None) is not None:
        dt_lr_mult = float(_require_num(sampler.get("dt_lr_mult"), "sampler.dt_lr_mult"))
        if not (dt_lr_mult > 0.0) or not math.isfinite(dt_lr_mult):
            raise _err("sampler.dt_lr_mult", "must be finite and > 0")

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
        precond_type = _require_str(_require(precond, "type", "sampler.preconditioning"), "sampler.preconditioning.type").lower()
        if precond_type == "matrix":
            raise _err(
                "sampler.preconditioning.type",
                "preconditioner type 'matrix' has been removed (it triggered the full-dataset FIM workflow). "
                "Use 'blockdiag_fisher' (alias: 'matrix_ema') for the online 4x4 block preconditioner, or 'rmsprop'/'adam'.",
            )
        # Backwards-compatible alias
        if precond_type == "matrix_ema":
            precond_type = "blockdiag_fisher"
        if precond_type not in {"rmsprop", "adam", "blockdiag_fisher"}:
            raise _err("sampler.preconditioning.type", "supported: 'rmsprop','adam','blockdiag_fisher' (alias: 'matrix_ema')")

        # Backend-specific support
        if backend in {"sghmc", "adaptive_sghmc"} and precond_type == "blockdiag_fisher":
            raise _err(
                "sampler.preconditioning.type",
                "'blockdiag_fisher' is currently supported for backend='psgld' only. "
                "For SGHMC/AdaptiveSGHMC use 'rmsprop'/'adam' preconditioning (AdaptiveSGHMC has its own diagonal "
                "preconditioner) unless/until a true preconditioned-SGHMC block-metric implementation is added.",
            )
    else:
        # still require key presence, but value can be null/empty
        _require(precond, "type", "sampler.preconditioning")
        precond_type = "none"

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

    # SGNHT support removed: fail fast if user still provides SGNHT-specific keys.
    if ("sgnht_diffusion" in sampler) or ("sgnht_thermostat_mass" in sampler):
        raise _err(
            "sampler",
            "contains SGNHT-specific keys (sgnht_diffusion/sgnht_thermostat_mass) but SGNHT support has been removed",
        )

    # ---- materialize legacy flat keys (implementation detail) ----
    params["phase1_epochs"] = phase1_epochs
    params["phase2_epochs"] = phase2_epochs
    params["phase3_epochs"] = phase3_epochs
    params["phase4_epochs"] = phase4_epochs
    params["lr_warmup"] = lr_warmup

    params["lr_sampler"] = lr_sampler
    # lr_mode='per_obs' is a convenience: when using pSGLD/SGHMC/AdaptiveSGHMC (which use minibatch-mean gradients
    # and internally scale the drift by N via n_obs/scale_grad), we apply lr_eff = lr_sampler / N at runtime.
    # This preserves the true posterior target but makes
    # the *user-provided* lr less sensitive to dataset size.
    params["sampler_lr_mode"] = lr_mode
    params["sampler_backend"] = backend
    params["sampler_temperature"] = temperature
    params["dt_lr_mult"] = float(dt_lr_mult)
    params["sampler_preconditioning"] = bool(precond_enabled)
    params["sampler_preconditioner"] = precond_type if precond_enabled else "none"
    params["sampler_beta"] = beta
    params["sampler_eps"] = eps
    params["freeze_preconditioner_sampling"] = bool(freeze_preconditioner_sampling)
    params["sghmc_alpha"] = alpha
    params["blockdiag_fisher_max_cluster_size"] = int(blockdiag_fisher_max_cluster_size)
    params["blockdiag_fisher_partition_method"] = str(blockdiag_fisher_partition_method)
    params["sampler_preconditioning_include_gamma"] = bool(precond_include_gamma)
    params["sampler_preconditioning_include_gamma_proxy"] = bool(precond_include_gamma_proxy)

    # ---- sampler parameter-group overrides (hard-break schema) ----
    # These are inference-only controls. Model hyperparameters for the latent itself live under:
    #   model.likelihood.shared_event_latent
    #
    # Defaults are chosen to be conservative for the high-dimensional latent b.
    se_lat_lr_mult = 0.05
    se_lat_temperature_mult = 0.25
    se_lat_eps = 1e-3
    se_lat_include_gamma = False
    se_lat_freeze_precond_sampling = False
    se_lat_overrides_active = False
    overrides = sampler.get("overrides", None)
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise _err("inference.sampler.overrides", "expected object/dict or null")
    se_lat_ov = overrides.get("shared_event_latent", None)
    if se_lat_ov is None:
        se_lat_ov = {}
    if not isinstance(se_lat_ov, dict):
        raise _err("inference.sampler.overrides.shared_event_latent", "expected object/dict or null")
    # Track whether the user explicitly provided any shared_event_latent override knobs.
    # This lets the runtime skip group-specific edits entirely when the overrides block is absent.
    for k in ("lr_mult", "temperature_mult", "eps", "include_gamma", "freeze_preconditioner_sampling"):
        if k in se_lat_ov and se_lat_ov.get(k, None) is not None:
            se_lat_overrides_active = True
            break
    if "lr_mult" in se_lat_ov and se_lat_ov.get("lr_mult", None) is not None:
        se_lat_lr_mult = float(_require_num(se_lat_ov.get("lr_mult"), "inference.sampler.overrides.shared_event_latent.lr_mult"))
        if not (se_lat_lr_mult > 0.0):
            raise _err("inference.sampler.overrides.shared_event_latent.lr_mult", "must be > 0")
    if "temperature_mult" in se_lat_ov and se_lat_ov.get("temperature_mult", None) is not None:
        se_lat_temperature_mult = float(
            _require_num(se_lat_ov.get("temperature_mult"), "inference.sampler.overrides.shared_event_latent.temperature_mult")
        )
        if not (se_lat_temperature_mult > 0.0):
            raise _err("inference.sampler.overrides.shared_event_latent.temperature_mult", "must be > 0")
    if "eps" in se_lat_ov and se_lat_ov.get("eps", None) is not None:
        se_lat_eps = float(_require_num(se_lat_ov.get("eps"), "inference.sampler.overrides.shared_event_latent.eps"))
        if not (se_lat_eps >= 0.0):
            raise _err("inference.sampler.overrides.shared_event_latent.eps", "must be >= 0")
    if "include_gamma" in se_lat_ov and se_lat_ov.get("include_gamma", None) is not None:
        se_lat_include_gamma = _require_bool(se_lat_ov.get("include_gamma"), "inference.sampler.overrides.shared_event_latent.include_gamma")
    if "freeze_preconditioner_sampling" in se_lat_ov and se_lat_ov.get("freeze_preconditioner_sampling", None) is not None:
        se_lat_freeze_precond_sampling = _require_bool(
            se_lat_ov.get("freeze_preconditioner_sampling"),
            "inference.sampler.overrides.shared_event_latent.freeze_preconditioner_sampling",
        )

    # Materialize internal keys consumed by the sampler/backend plumbing.
    params["_shared_event_latent_lr_mult"] = float(se_lat_lr_mult)
    params["_shared_event_latent_temperature_mult"] = float(se_lat_temperature_mult)
    params["_shared_event_latent_eps"] = float(se_lat_eps)
    params["_shared_event_latent_include_gamma"] = bool(se_lat_include_gamma)
    params["_shared_event_latent_freeze_preconditioner_sampling"] = bool(se_lat_freeze_precond_sampling)
    params["_shared_event_latent_sampler_overrides_active"] = bool(se_lat_overrides_active)

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
    lk_type = _require_str(_require(lk, "type", "model.likelihood"), "model.likelihood.type").lower()
    # Accept common aliases; `compute_likelihood_loss` handles the mapping.
    if lk_type not in {"huber", "l2", "gaussian", "laplace", "l1", "mae", "mse"}:
        raise _err("model.likelihood.type", "supported: 'huber', 'l2'/'gaussian' (aliases: 'mse'), 'laplace' (aliases: 'l1','mae')")
    phase_unc = _require_float_list(_require(lk, "phase_unc", "model.likelihood"), "model.likelihood.phase_unc", length=2)
    learn_noise_scale = _require_bool(_require(lk, "learn_noise_scale", "model.likelihood"), "model.likelihood.learn_noise_scale")

    # Optional: likelihood-only tempering (power posterior) for scalable uncertainty calibration.
    #
    # This multiplies the data likelihood term by alpha while leaving priors unchanged, i.e.
    #   posterior(theta | y) ∝ prior(theta) * likelihood(y | theta)^alpha
    # with alpha in (0, 1] typically (alpha<1 inflates uncertainty).
    temp_cfg = lk.get("tempering", None)
    temp_enabled = False
    temp_alpha = 1.0
    if isinstance(temp_cfg, dict):
        temp_enabled = bool(temp_cfg.get("enabled", False))
        if "alpha" in temp_cfg and temp_cfg["alpha"] is not None:
            a = float(_require_num(temp_cfg["alpha"], "model.likelihood.tempering.alpha"))
            if not (a > 0.0) or (not math.isfinite(a)):
                raise _err("model.likelihood.tempering.alpha", "must be finite and > 0")
            temp_alpha = a
    if not temp_enabled:
        temp_alpha = 1.0

    # Optional: additional heteroscedastic noise inflation for the likelihood (no latent term).
    #
    # This adds a per-observation variance term in quadrature with the base noise scale:
    #   sigma_eff^2 = sigma_phase^2 + sigma_extra_var
    #
    # Intended use: represent unmodeled path/velocity-structure uncertainty that scales with
    # event-pair separation, while still allowing `shared_event_latent` to learn coherent mean
    # corrections (nuisance_delta).
    #
    # Config:
    #   model.likelihood.sigma_inflation:
    #     enabled: bool
    #     mode: "vel_frac_linear_dd"   # sigma_struct(d) = (vel_frac / v_km_s) * d_km
    #     vel_frac: [P,S]              # fractional velocity error (e.g., 0.02 for 2%)
    #     v_km_s: [P,S]                # reference phase speeds in km/s (e.g., [6.0, 3.5])
    #     max_d_km: number|null        # optional clamp on d_km (stability / conservative cap)
    #     use_3d: bool                 # if true use sqrt(dx^2+dy^2+dz^2), else horizontal only
    #
    # Note: this is currently NOT supported with `shared_event_re` (pcg_sparse) because that
    # solver assumes a homoscedastic diagonal; we validate that below.
    sigma_infl_cfg = lk.get("sigma_inflation", None)
    sigma_infl_enabled = False
    sigma_infl_mode = "vel_frac_linear_dd"
    sigma_infl_vel_frac = [0.0, 0.0]
    sigma_infl_v_km_s = [6.0, 3.5]
    sigma_infl_max_d_km = None
    sigma_infl_use_3d = True
    if isinstance(sigma_infl_cfg, dict):
        sigma_infl_enabled = bool(sigma_infl_cfg.get("enabled", False))
        if "mode" in sigma_infl_cfg and sigma_infl_cfg.get("mode", None) is not None:
            sigma_infl_mode = str(sigma_infl_cfg.get("mode", "vel_frac_linear_dd")).strip().lower()
        if sigma_infl_mode not in {"vel_frac_linear_dd"}:
            raise _err("model.likelihood.sigma_inflation.mode", "supported: 'vel_frac_linear_dd'")
        if "vel_frac" in sigma_infl_cfg and sigma_infl_cfg.get("vel_frac", None) is not None:
            v = sigma_infl_cfg["vel_frac"]
            if isinstance(v, (int, float)):
                f = float(v)
                sigma_infl_vel_frac = [f, f]
            elif isinstance(v, list):
                sigma_infl_vel_frac = _require_float_list(v, "model.likelihood.sigma_inflation.vel_frac", length=2)
            else:
                raise _err("model.likelihood.sigma_inflation.vel_frac", f"expected number or [P,S] list, got {type(v).__name__}")
        if not (sigma_infl_vel_frac[0] >= 0.0 and sigma_infl_vel_frac[1] >= 0.0):
            raise _err("model.likelihood.sigma_inflation.vel_frac", "must be >= 0")
        if "v_km_s" in sigma_infl_cfg and sigma_infl_cfg.get("v_km_s", None) is not None:
            v = sigma_infl_cfg["v_km_s"]
            if isinstance(v, (int, float)):
                f = float(v)
                sigma_infl_v_km_s = [f, f]
            elif isinstance(v, list):
                sigma_infl_v_km_s = _require_float_list(v, "model.likelihood.sigma_inflation.v_km_s", length=2)
            else:
                raise _err("model.likelihood.sigma_inflation.v_km_s", f"expected number or [P,S] list, got {type(v).__name__}")
        if not (sigma_infl_v_km_s[0] > 0.0 and sigma_infl_v_km_s[1] > 0.0):
            raise _err("model.likelihood.sigma_inflation.v_km_s", "must be > 0")
        if "max_d_km" in sigma_infl_cfg and sigma_infl_cfg.get("max_d_km", None) is not None:
            sigma_infl_max_d_km = float(_require_num(sigma_infl_cfg["max_d_km"], "model.likelihood.sigma_inflation.max_d_km"))
            if not (sigma_infl_max_d_km > 0.0) or (not math.isfinite(sigma_infl_max_d_km)):
                raise _err("model.likelihood.sigma_inflation.max_d_km", "must be finite and > 0 (or null)")
        if "use_3d" in sigma_infl_cfg and sigma_infl_cfg.get("use_3d", None) is not None:
            sigma_infl_use_3d = _require_bool(sigma_infl_cfg.get("use_3d"), "model.likelihood.sigma_inflation.use_3d")
    if not sigma_infl_enabled:
        # Materialize safe defaults (disabled)
        sigma_infl_mode = "vel_frac_linear_dd"
        sigma_infl_vel_frac = [0.0, 0.0]
        sigma_infl_max_d_km = None
        sigma_infl_use_3d = True

    # Residual-correlation block removed entirely (no backward compatibility).
    if "residual_correlation" in lk:
        raise _err("model.likelihood.residual_correlation", "removed; structured residual correlation models are no longer supported")

    # The following likelihood extensions have been removed from SPIDER.
    # Keep the config surface area focused on `likelihood.shared_event_latent`.
    #
    # NOTE: `shared_event_re` is supported again as an OPTIONAL *collapsed* (marginalized) Gaussian
    # likelihood. The uncollapsed/latent version remains implemented under `shared_event_latent`.
    if "latent_field" in lk:
        raise _err("model.likelihood.latent_field", "removed; use model.likelihood.shared_event_latent instead")

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
    se_tau_ps = [0.0, 0.0]  # std in seconds for [P,S]; 0 disables (iid)
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
    if isinstance(se_cfg, dict):
        se_enabled = bool(se_cfg.get("enabled", False))
        if "grouping" in se_cfg and se_cfg["grouping"] is not None:
            se_grouping = str(se_cfg.get("grouping", "phase")).strip().lower()
        if se_grouping in {"stationphase", "station-phase"}:
            se_grouping = "station_phase"
        if se_grouping not in {"phase", "station_phase"}:
            raise _err("model.likelihood.shared_event_re.grouping", "supported: 'phase', 'station_phase'")

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
                "(collapsed shared-event random effects is implemented for Gaussian likelihood only)",
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
    sl_grouping = "station_phase"  # 'phase' | 'station_phase'
    sl_ell_km = 0.0
    sl_tau_ps = [0.0, 0.0]
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
    if isinstance(sl_cfg, dict):
        sl_enabled = bool(sl_cfg.get("enabled", False))
        if "grouping" in sl_cfg and sl_cfg.get("grouping", None) is not None:
            sl_grouping = str(sl_cfg.get("grouping", sl_grouping)).strip().lower()
        if sl_grouping in {"stationphase", "station-phase"}:
            sl_grouping = "station_phase"
        if sl_grouping not in {"phase", "station_phase"}:
            raise _err("model.likelihood.slowness_re.grouping", "supported: 'phase', 'station_phase'")

        if "ell_km" in sl_cfg and sl_cfg.get("ell_km", None) is not None:
            sl_ell_km = float(_require_num(sl_cfg.get("ell_km"), "model.likelihood.slowness_re.ell_km"))
        if sl_enabled:
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
                sl_tau_units = "abs"
            elif isinstance(tv, list):
                sl_tau_ps = _require_float_list(tv, "model.likelihood.slowness_re.tau_s", length=2)
                sl_tau_units = "abs"
            else:
                raise _err("model.likelihood.slowness_re.tau_s", f"expected number, [P,S] list, or object, got {type(tv).__name__}")
        if sl_enabled and (not (sl_tau_ps[0] >= 0.0 and sl_tau_ps[1] >= 0.0)):
            raise _err("model.likelihood.slowness_re.tau_s", "must be >= 0")

        if "max_rows_per_group" in sl_cfg and sl_cfg.get("max_rows_per_group", None) is not None:
            sl_max_rows_per_group = int(_require_num(sl_cfg.get("max_rows_per_group"), "model.likelihood.slowness_re.max_rows_per_group"))
            if sl_max_rows_per_group < 2:
                raise _err("model.likelihood.slowness_re.max_rows_per_group", "must be >= 2")
        if "max_nodes_per_group" in sl_cfg and sl_cfg.get("max_nodes_per_group", None) is not None:
            sl_max_nodes_per_group = int(_require_num(sl_cfg.get("max_nodes_per_group"), "model.likelihood.slowness_re.max_nodes_per_group"))
            if sl_max_nodes_per_group < 2:
                raise _err("model.likelihood.slowness_re.max_nodes_per_group", "must be >= 2")
        if "fallback_to_diag" in sl_cfg and sl_cfg.get("fallback_to_diag", None) is not None:
            sl_fallback_to_diag = bool(sl_cfg.get("fallback_to_diag", True))
        if "drop_logdet" in sl_cfg and sl_cfg.get("drop_logdet", None) is not None:
            sl_drop_logdet = bool(sl_cfg.get("drop_logdet", True))

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
                "(collapsed slowness covariance is implemented for Gaussian likelihood only)",
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
        if str(lk_type).strip().lower() not in {"gaussian", "l2", "mse", "huber", "laplace", "l1", "mae"}:
            raise _err(
                "model.likelihood.type",
                "must be one of: 'gaussian'/'l2'/'mse', 'huber', 'laplace' (aliases: 'l1','mae') when "
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
    params["learn_noise_scale"] = bool(learn_noise_scale)
    params["_likelihood_tempering_enabled"] = bool(temp_enabled)
    params["_likelihood_tempering_alpha"] = float(temp_alpha)
    params["_likelihood_sigma_inflation_enabled"] = bool(sigma_infl_enabled)
    params["_likelihood_sigma_inflation_mode"] = str(sigma_infl_mode)
    params["_likelihood_sigma_inflation_vel_frac"] = [float(sigma_infl_vel_frac[0]), float(sigma_infl_vel_frac[1])]
    params["_likelihood_sigma_inflation_v_km_s"] = [float(sigma_infl_v_km_s[0]), float(sigma_infl_v_km_s[1])]
    params["_likelihood_sigma_inflation_max_d_km"] = (float(sigma_infl_max_d_km) if sigma_infl_max_d_km is not None else None)
    params["_likelihood_sigma_inflation_use_3d"] = bool(sigma_infl_use_3d)

    # Collapsed shared-event random effects (optional)
    params["_shared_event_re_enabled"] = bool(se_enabled)
    params["_shared_event_re_grouping"] = str(se_grouping)
    params["_shared_event_re_tau_s"] = [float(se_tau_ps[0]), float(se_tau_ps[1])]
    params["_shared_event_re_joint_ps"] = bool(se_joint_ps)
    params["_shared_event_re_rho_ps"] = float(se_rho_ps)
    params["_shared_event_re_max_nodes_per_group"] = int(se_max_nodes_per_group)
    params["_shared_event_re_max_rows_per_group"] = int(se_max_rows_per_group)
    params["_shared_event_re_fallback_to_diag"] = bool(se_fallback_to_diag)
    params["_shared_event_re_jitter0"] = float(se_jitter0)
    params["_shared_event_re_jitter_max"] = float(se_jitter_max)
    params["_shared_event_re_cache_max_entries"] = int(se_cache_max_entries)
    params["_shared_event_re_cache_log_every"] = int(se_cache_log_every)
    params["_shared_event_re_solver"] = str(se_solver)
    params["_shared_event_re_drop_logdet"] = bool(se_drop_logdet)
    params["_shared_event_re_pcg_max_iters"] = int(se_pcg_max_iters)
    params["_shared_event_re_pcg_tol"] = float(se_pcg_tol)
    params["_shared_event_re_diag_log_every_epochs"] = int(se_diag_log_every_epochs)
    params["_shared_event_re_diag_max_groups"] = int(se_diag_max_groups)
    params["_shared_event_re_diag_max_rows_per_group"] = int(se_diag_max_rows_per_group)
    params["_shared_event_re_diag_max_nodes"] = int(se_diag_max_nodes)
    params["_shared_event_re_diag_seed"] = int(se_diag_seed)

    # Collapsed slowness covariance likelihood (optional)
    params["_slowness_re_enabled"] = bool(sl_enabled)
    params["_slowness_re_grouping"] = str(sl_grouping)
    params["_slowness_re_ell_km"] = float(sl_ell_km)
    params["_slowness_re_tau_s"] = [float(sl_tau_ps[0]), float(sl_tau_ps[1])]
    params["_slowness_re_tau_units"] = str(sl_tau_units)
    params["_slowness_re_max_nodes_per_group"] = int(sl_max_nodes_per_group)
    params["_slowness_re_max_rows_per_group"] = int(sl_max_rows_per_group)
    params["_slowness_re_fallback_to_diag"] = bool(sl_fallback_to_diag)
    params["_slowness_re_drop_logdet"] = bool(sl_drop_logdet)
    # Inducing plan (required for slowness_re)
    params["_slowness_re_inducing_plan_enable"] = bool(sl_plan_enabled)
    params["_slowness_re_inducing_plan_cover_frac_of_ell"] = float(sl_plan_cover_frac)
    params["_slowness_re_inducing_plan_min_inducing_per_component"] = int(sl_plan_min_m)
    params["_slowness_re_inducing_plan_max_inducing_per_component"] = int(sl_plan_max_m)
    params["_slowness_re_inducing_plan_top_k"] = int(sl_plan_top_k)
    params["_slowness_re_inducing_plan_seed_strategy"] = str(sl_plan_seed_strategy)
    params["_slowness_re_inducing_fixed_xyz"] = bool(sl_plan_fixed_xyz)
    params["_slowness_re_inducing_jitter"] = float(sl_plan_kernel_jitter)
    params["_slowness_re_inducing_plan_selection_outfile"] = sl_plan_selection_outfile
    params["_slowness_re_inducing_plan_interpolation_enable"] = bool(sl_plan_interpolation_enabled)
    params["_slowness_re_inducing_plan_interpolation_m"] = int(sl_plan_interpolation_m)
    params["_slowness_re_inducing_plan_interpolation_outfile"] = sl_plan_interpolation_outfile
    params["_slowness_re_inducing_fitc_enable"] = bool(sl_fitc_enabled)

    # Uncollapsed shared-event latent random effects (optional)
    params["_shared_event_latent_enabled"] = bool(se_lat_enabled)
    params["_shared_event_latent_parameterization"] = str(se_lat_param)
    params["_shared_event_latent_knn"] = int(se_lat_knn)
    params["_shared_event_latent_ell_km"] = float(se_lat_ell_km)
    params["_shared_event_latent_q_diag"] = float(se_lat_q_diag)
    params["_shared_event_latent_graph_lambda"] = float(se_lat_graph_lambda)
    params["_shared_event_latent_graph_max_degree"] = int(se_lat_graph_max_degree)
    params["_shared_event_latent_graph_max_edge_km"] = (float(se_lat_graph_max_edge_km) if se_lat_graph_max_edge_km is not None else None)
    params["_shared_event_latent_tau_s"] = [float(se_lat_tau_ps[0]), float(se_lat_tau_ps[1])]
    # Units for tau_s in slowness_inducing_gp:
    # - "abs": tau_s is interpreted as s/km (slowness amplitude)
    # - "vel_frac": tau_s is interpreted as fractional velocity perturbation δv/v (dimensionless)
    params["_shared_event_latent_slowness_tau_units"] = str(tau_units)
    params["_shared_event_latent_rho_ps"] = float(se_lat_rho_ps)
    params["_shared_event_latent_drop_station_common_mode"] = bool(se_lat_drop_station_common_mode)
    # Inducing plan diagnostics (optional)
    params["_shared_event_latent_inducing_plan_enable"] = bool(se_lat_plan_enable)
    params["_shared_event_latent_inducing_plan_cover_frac_of_ell"] = float(se_lat_plan_cover_frac)
    params["_shared_event_latent_inducing_plan_min_inducing_per_component"] = int(se_lat_plan_min_m)
    params["_shared_event_latent_inducing_plan_max_inducing_per_component"] = int(se_lat_plan_max_m)
    params["_shared_event_latent_inducing_plan_top_k"] = int(se_lat_plan_top_k)
    params["_shared_event_latent_inducing_plan_outfile"] = se_lat_plan_outfile
    params["_shared_event_latent_inducing_plan_select"] = bool(se_lat_plan_select)
    params["_shared_event_latent_inducing_plan_selection_outfile"] = se_lat_plan_selection_outfile
    params["_shared_event_latent_inducing_plan_seed_strategy"] = str(se_lat_plan_seed_strategy)
    params["_shared_event_latent_inducing_plan_use_xyz"] = bool(se_lat_plan_use_xyz)
    # Fixed inducing geometry (Option-B) defaulting:
    # - slowness_inducing_gp: True unless explicitly overridden
    # - other modes: False unless explicitly overridden
    if se_lat_plan_fixed_xyz is None:
        se_lat_plan_fixed_xyz = bool(se_lat_param == "slowness_inducing_gp")
    params["_shared_event_latent_inducing_fixed_xyz"] = bool(se_lat_plan_fixed_xyz)
    params["_shared_event_latent_inducing_plan_interpolation_enable"] = bool(se_lat_plan_interp)
    params["_shared_event_latent_inducing_plan_interpolation_m"] = int(se_lat_plan_interp_m)
    params["_shared_event_latent_inducing_plan_interpolation_outfile"] = se_lat_plan_interp_outfile
    params["_shared_event_latent_inducing_plan_interpolation_store_distances"] = bool(se_lat_plan_interp_store_dist)
    # K_UU diagonal jitter (regularization) for inducing_gp prior construction
    params["_shared_event_latent_inducing_jitter"] = float(se_lat_plan_kernel_jitter)
    params["_shared_event_latent_inducing_fitc_enable"] = bool(se_lat_fitc_enable)

    # Station-geometry basis (fixed), optional
    params["_shared_event_latent_station_basis_enabled"] = bool(se_lat_sta_basis_enabled)
    params["_shared_event_latent_station_basis_r"] = int(se_lat_sta_basis_r)
    params["_shared_event_latent_station_basis_ell_km"] = float(se_lat_sta_basis_ell_km)
    params["_shared_event_latent_station_basis_jitter"] = float(se_lat_sta_basis_jitter)
    params["_shared_event_latent_station_basis_method"] = str(se_lat_sta_basis_method)

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
    ess_online = _require_dict(_require(dg, "ess_online", "diagnostics"), "diagnostics.ess_online")
    ess_online_enabled = _require_bool(_require(ess_online, "enabled", "diagnostics.ess_online"), "diagnostics.ess_online.enabled")
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
    # This is an alternative to the centroid prior: it imposes a hard constraint on the mean update.
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


