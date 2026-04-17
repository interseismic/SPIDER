"""
Transitional bridge from canonical config_v2 to current legacy runtime params.

This module exists only during migration from config_v2 to legacy runtime
parameter access patterns. It intentionally does NOT depend on config_schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List

from spider.core.priors_config import validate_and_materialize_priors

from .types import ResolvedConfig


def _require_dict(v: Any, path: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise ValueError(f"Invalid config at `{path}`: expected object/dict, got {type(v).__name__}")
    return v


def _require_list(v: Any, path: str, *, length: int | None = None) -> list:
    if not isinstance(v, list):
        raise ValueError(f"Invalid config at `{path}`: expected list, got {type(v).__name__}")
    if length is not None and len(v) != int(length):
        raise ValueError(f"Invalid config at `{path}`: expected list length {length}, got {len(v)}")
    return v


def _parse_device_entry(x: Any) -> int:
    if x is None:
        raise ValueError("Invalid device entry: null")
    if isinstance(x, int):
        return int(x)
    s = str(x).strip().lower()
    if s == "cpu":
        return -1
    if s.startswith("cuda:"):
        tail = s.split("cuda:", 1)[1].strip()
        if not tail.isdigit():
            raise ValueError(f"Invalid device specifier {x!r} (expected 'cuda:<int>')")
        return int(tail)
    if s.lstrip("-").isdigit():
        return int(s)
    raise ValueError(f"Invalid device specifier {x!r} (supported: int, -1/'cpu', 'cuda:<int>')")


def _canonical_to_legacy_nested(canonical_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map canonical config_v2 top-level shape to the legacy nested shape expected by
    current block validators.
    """
    params = json.loads(json.dumps(canonical_params))
    obs = params.pop("observability", None)
    if not isinstance(obs, dict):
        raise ValueError("Invalid config_v2 payload: missing required object `observability`")

    wb = obs.get("wandb", None)
    diag = obs.get("diagnostics", {})
    if not isinstance(wb, dict):
        raise ValueError("Invalid config_v2 payload: `observability.wandb` must be an object/dict")
    if not isinstance(diag, dict):
        raise ValueError("Invalid config_v2 payload: `observability.diagnostics` must be an object/dict")

    params["wandb"] = wb
    inf = params.get("inference", None)
    if not isinstance(inf, dict):
        raise ValueError("Invalid config_v2 payload: missing required object `inference`")
    if "diagnostics" in inf:
        raise ValueError(
            "Invalid config_v2 payload: `inference.diagnostics` is not allowed; "
            "use `observability.diagnostics`"
        )
    inf["diagnostics"] = diag

    return params


def _materialize_block1(params: Dict[str, Any]) -> Dict[str, Any]:
    io = _require_dict(params.get("io"), "io")
    model = _require_dict(params.get("model"), "model")
    domain = _require_dict(model.get("domain"), "model.domain")
    wb = _require_dict(params.get("wandb"), "wandb")

    checkpoint_interval = int(io.get("checkpoint_interval"))
    sample_write_interval = io.get("sample_write_interval", None)
    if sample_write_interval is None:
        sample_write_interval = int(checkpoint_interval)
    sample_write_interval = int(sample_write_interval)

    params["dtime_file"] = str(io.get("dtime_file"))
    params["station_file"] = str(io.get("station_file"))
    params["catalog_infile"] = str(io.get("catalog_infile"))
    params["catalog_outfile"] = str(io.get("catalog_outfile"))
    params["samples_outfile"] = str(io.get("samples_outfile"))
    params["checkpoint_dir"] = str(io.get("checkpoint_dir"))
    params["checkpoint_interval"] = int(checkpoint_interval)
    params["sample_write_interval"] = int(sample_write_interval)
    params["save_every_n"] = int(io.get("save_every_n"))
    params["write_samples"] = bool(io.get("write_samples"))

    params["model_file"] = str(model.get("model_file"))
    params["lon_min"] = float(domain.get("lon_min"))
    params["lat_min"] = float(domain.get("lat_min"))
    params["z_min"] = float(domain.get("z_min"))
    params["z_max"] = float(domain.get("z_max"))
    params["scale"] = float(domain.get("scale"))

    params["use_wandb"] = bool(wb.get("enabled", False))
    params["wandb_project_name"] = wb.get("project_name", None)
    params["wandb_run_name"] = wb.get("run_name", None)
    return params


def _materialize_block2(params: Dict[str, Any]) -> Dict[str, Any]:
    sampler = _require_dict(_require_dict(params.get("inference"), "inference").get("sampler"), "inference.sampler")
    epochs = [int(x) for x in _require_list(sampler.get("epochs_per_phase"), "inference.sampler.epochs_per_phase", length=4)]
    lrs = [float(x) for x in _require_list(sampler.get("lr"), "inference.sampler.lr", length=4)]
    backend = str(sampler.get("backend", "")).strip().lower()
    if backend not in {"psgld", "sghmc"}:
        raise ValueError("Invalid config at `inference.sampler.backend`: supported values are 'psgld' or 'sghmc'")

    precond = _require_dict(sampler.get("preconditioning"), "inference.sampler.preconditioning")
    precond_enabled = bool(precond.get("enabled", False))
    precond_type = str(precond.get("type", "none")).strip().lower()
    if not precond_enabled:
        precond_type = "none"
    if precond_type not in {"none", "rmsprop", "lrd"}:
        raise ValueError(
            "Invalid config at `inference.sampler.preconditioning.type`: "
            "supported values are 'rmsprop' and 'lrd' (or disable preconditioning)."
        )
    lrd_cfg = precond.get("lrd", {})
    if not isinstance(lrd_cfg, dict):
        lrd_cfg = {}
    lrd_rank = int(lrd_cfg.get("rank", 16))
    lrd_mode = str(lrd_cfg.get("mode", "svd")).strip().lower()
    if lrd_mode in {"randomized_svd", "stochastic_svd"}:
        lrd_mode = "svd"
    if lrd_mode not in {"svd", "oja"}:
        lrd_mode = "svd"
    lrd_update_every = int(lrd_cfg.get("update_every", 20))
    lrd_buffer_size = int(lrd_cfg.get("buffer_size", 64))
    lrd_oja_eta = float(lrd_cfg.get("eta", lrd_cfg.get("oja_eta", 0.02)))
    lrd_diag_floor = float(lrd_cfg.get("diag_floor", sampler.get("eps", 1e-8)))
    lrd_target = str(lrd_cfg.get("target", "dX_src_only")).strip().lower()

    params["phase1_epochs"] = int(epochs[0])
    params["phase2_epochs"] = int(epochs[1])
    params["phase3_epochs"] = int(epochs[2])
    params["phase4_epochs"] = int(epochs[3])
    params["lr_warmup"] = float(lrs[0])
    params["lr_sampler"] = float(lrs[1])
    params["sampler_lr_mode"] = "per_obs"
    params["sampler_backend"] = str(backend)
    params["sampler_temperature"] = float(sampler.get("temperature", 1.0))
    noise_scale_mult = sampler.get("noise_scale_mult", 1.0)
    dt_lr_mult = sampler.get("dt_lr_mult", 1.0)
    grad_clip_norm = sampler.get("grad_clip_norm", 0.0)
    params["sampler_noise_scale_mult"] = float(1.0 if noise_scale_mult is None else noise_scale_mult)
    params["dt_lr_mult"] = float(1.0 if dt_lr_mult is None else dt_lr_mult)
    params["sampler_grad_clip_norm"] = float(0.0 if grad_clip_norm is None else grad_clip_norm)
    params["sampler_preconditioning"] = bool(precond_enabled)
    params["sampler_preconditioner"] = str(precond_type)
    params["sampler_beta"] = float(sampler.get("beta", 0.99))
    params["sampler_eps"] = float(sampler.get("eps", 1e-8))
    params["freeze_preconditioner_sampling"] = bool(sampler.get("freeze_preconditioner_sampling", False))
    params["sghmc_alpha"] = float(sampler.get("sghmc_alpha", 0.0) or 0.0)
    params["sampler_preconditioning_include_gamma"] = bool(precond.get("include_gamma", True))
    params["_sampler_lr_per_phase"] = list(lrs)
    params["sampler_preconditioning_lrd_rank"] = int(lrd_rank)
    params["sampler_preconditioning_lrd_mode"] = str(lrd_mode)
    params["sampler_preconditioning_lrd_update_every"] = int(lrd_update_every)
    params["sampler_preconditioning_lrd_buffer_size"] = int(lrd_buffer_size)
    params["sampler_preconditioning_lrd_oja_eta"] = float(lrd_oja_eta)
    params["sampler_preconditioning_lrd_diag_floor"] = float(lrd_diag_floor)
    params["sampler_preconditioning_lrd_target"] = str(lrd_target)

    overrides = sampler.get("overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}
    group_overrides: Dict[str, Dict[str, Any]] = {}
    core = overrides.get("core", None)
    if isinstance(core, dict):
        out: Dict[str, Any] = {}
        if core.get("lr_mult", None) is not None:
            out["lr_mult"] = float(core.get("lr_mult"))
        if core.get("temperature_mult", None) is not None:
            out["temperature_mult"] = float(core.get("temperature_mult"))
        if core.get("eps", None) is not None:
            out["eps"] = float(core.get("eps"))
        if core.get("freeze_preconditioner_sampling", None) is not None:
            out["freeze_preconditioner_sampling"] = bool(core.get("freeze_preconditioner_sampling"))
        if out:
            group_overrides["core"] = out
    params["_sampler_group_overrides"] = dict(group_overrides)
    params["_sampler_group_overrides_active"] = bool(group_overrides)
    return params


def _normalize_likelihood_type(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"student-t", "studentt"}:
        return "student_t"
    if s in {"mse"}:
        return "l2"
    if s in {"mae", "l1"}:
        return "laplace"
    return s


def _materialize_block3(params: Dict[str, Any]) -> Dict[str, Any]:
    model = _require_dict(params.get("model"), "model")
    inf = _require_dict(params.get("inference"), "inference")
    likelihoods = _require_dict(model.get("likelihoods"), "model.likelihoods")
    lk_loc = _require_dict(likelihoods.get("locate_map"), "model.likelihoods.locate_map")
    lk_smp = _require_dict(likelihoods.get("sample"), "model.likelihoods.sample")

    locate_type = _normalize_likelihood_type(lk_loc.get("type", "huber"))
    locate_phase_unc = [float(x) for x in _require_list(lk_loc.get("phase_unc"), "model.likelihoods.locate_map.phase_unc", length=2)]
    locate_student_t = _require_dict(lk_loc.get("student_t", {}), "model.likelihoods.locate_map.student_t")
    locate_student_t_nu = float(locate_student_t.get("nu", 4.0))
    locate_huber_delta = float(lk_loc.get("huber_delta", 1.0))
    locate_group = {
        "likelihood": locate_type,
        "phase_unc": locate_phase_unc,
        "_student_t_nu": float(locate_student_t_nu),
        "_huber_delta": float(locate_huber_delta),
        "_shared_event_re_enabled": False,
    }

    sample_type_raw = str(lk_smp.get("type", "")).strip().lower()
    lk_correlated = sample_type_raw in {"correlated", "correlated_gaussian"}
    if not lk_correlated:
        raise ValueError("Invalid config at `model.likelihoods.sample.type`: must be 'correlated_gaussian'")
    sample_type = "gaussian"
    sample_phase_unc = [float(x) for x in _require_list(lk_smp.get("phase_unc"), "model.likelihoods.sample.phase_unc", length=2)]
    sample_student_t = _require_dict(lk_smp.get("student_t", {}), "model.likelihoods.sample.student_t")
    sample_student_t_nu = float(sample_student_t.get("nu", 4.0))
    sample_huber_delta = float(lk_smp.get("huber_delta", 1.0))

    se_cfg = _require_dict(lk_smp.get("shared_event_re", {}), "model.likelihoods.sample.shared_event_re")
    legacy_top_keys = {
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
    legacy_found = sorted([k for k in legacy_top_keys if k in se_cfg])
    if legacy_found:
        raise ValueError(
            "Invalid config at `model.likelihoods.sample.shared_event_re`: "
            f"legacy keys not allowed: {', '.join(legacy_found)}. "
            "Use sub-blocks: model, limits, fallback, numerics, station_phase_term."
        )

    se_enabled = bool(se_cfg.get("enabled", False))
    se_model = _require_dict(se_cfg.get("model", {}), "model.likelihoods.sample.shared_event_re.model")
    se_grouping = str(se_model.get("group_by", "station_phase")).strip().lower().replace("-", "_")
    se_cluster = _require_dict(se_model.get("cluster", {}), "model.likelihoods.sample.shared_event_re.model.cluster")
    se_cluster_mode = str(se_cluster.get("mode", "none")).strip().lower().replace("-", "_")
    se_cluster_k = int(se_cluster.get("k", 1))
    tau_s = se_model.get("tau_s", [0.0, 0.0])
    if isinstance(tau_s, (int, float)):
        se_tau_ps = [float(tau_s), float(tau_s)]
    else:
        se_tau_ps = [float(x) for x in _require_list(tau_s, "model.likelihoods.sample.shared_event_re.tau_s", length=2)]
    se_limits = _require_dict(se_cfg.get("limits", {}), "model.likelihoods.sample.shared_event_re.limits")
    se_max_nodes_per_group = int(se_limits.get("max_nodes", 512))
    se_max_rows_per_group = int(se_limits.get("max_rows", 200000))
    se_fallback = _require_dict(se_cfg.get("fallback", {}), "model.likelihoods.sample.shared_event_re.fallback")
    se_fallback_to_diag = bool(se_fallback.get("to_diag", True))
    se_abort_on_pcg_fallback = bool(se_fallback.get("abort_on_pcg_fallback", False))
    se_numerics = _require_dict(se_cfg.get("numerics", {}), "model.likelihoods.sample.shared_event_re.numerics")
    se_jitter0 = float(se_numerics.get("jitter0", 1e-8))
    se_jitter_max = float(se_numerics.get("jitter_max", 1e-3))

    sp_cfg = se_cfg.get("station_phase_term", {})
    if not isinstance(sp_cfg, dict):
        sp_cfg = {}
    sp_enabled = bool(sp_cfg.get("enabled", False))
    sp_tau = sp_cfg.get("tau_s", [0.0, 0.0])
    if isinstance(sp_tau, (int, float)):
        sp_tau_ps = [float(sp_tau), float(sp_tau)]
    else:
        sp_tau_ps = [float(x) for x in _require_list(sp_tau, "model.likelihoods.sample.shared_event_re.station_phase_re.tau_s", length=2)]

    solver_cfg = se_cfg.get("solver", {})
    if not isinstance(solver_cfg, dict):
        solver_cfg = {}
    legacy_solver_keys = {
        "bucket_nodes",
        "merge_sparse_edge_bins",
        "min_groups_per_edge_bin",
        "max_edge_bins_per_node",
    }
    legacy_solver_found = sorted([k for k in legacy_solver_keys if k in solver_cfg])
    if legacy_solver_found:
        raise ValueError(
            "Invalid config at `model.likelihoods.sample.shared_event_re.solver`: "
            f"legacy keys not allowed: {', '.join(legacy_solver_found)}. "
            "Use node_bin_edges, merge_sparse_node_bins, min_groups_per_node_bin, max_node_bins_per_node."
        )
    se_solver_kind = str(solver_cfg.get("kind", "pcg")).strip().lower()
    se_solver_max_iters = int(solver_cfg.get("max_iters", 50))
    se_solver_min_iters = int(solver_cfg.get("min_iters", 2))
    se_solver_tol = float(solver_cfg.get("tol", 1e-3))
    se_solver_batched = bool(solver_cfg.get("batched", True))
    se_solver_bucket_nodes = [int(x) for x in solver_cfg.get("node_bin_edges", [512, 1024, 2048, 4096, 8192, 16384, 32768])]
    se_solver_warm_start = bool(solver_cfg.get("warm_start", False))
    se_solver_cache_max_entries = int(solver_cfg.get("cache_max_entries", 8))
    se_solver_prefetch_grouping = bool(solver_cfg.get("prefetch_grouping", False))
    se_solver_profile_micro_steps = bool(solver_cfg.get("profile_micro_steps", False))
    se_solver_merge_sparse_edge_bins = bool(solver_cfg.get("merge_sparse_node_bins", True))
    se_solver_min_groups_per_edge_bin = int(solver_cfg.get("min_groups_per_node_bin", 32))
    se_solver_max_edge_bins_per_node = int(solver_cfg.get("max_node_bins_per_node", 4))
    precompute_cfg = solver_cfg.get("precompute", {})
    if not isinstance(precompute_cfg, dict):
        precompute_cfg = {}
    se_solver_precompute_enabled = bool(precompute_cfg.get("enabled", False))
    se_solver_precompute_device = str(precompute_cfg.get("device", "gpu")).strip().lower()

    ew_cfg = se_cfg.get("edge_weights", {})
    if not isinstance(ew_cfg, dict):
        ew_cfg = {}
    se_edge_weight_mode = str(ew_cfg.get("mode", "uniform")).strip().lower()
    se_edge_weight_ell_km = float(ew_cfg.get("ell_km", 1.0))
    se_edge_weight_eps_km = float(ew_cfg.get("eps_km", 1e-3))
    se_edge_weight_power = float(ew_cfg.get("power", 1.0))
    se_edge_weight_scale_km = float(ew_cfg.get("scale_km", 1.0))
    se_edge_weight_global_scale = float(ew_cfg.get("global_scale", 1.0))
    se_edge_weight_normalize = bool(ew_cfg.get("normalize", False))

    at_cfg = se_cfg.get("autotune", {})
    if not isinstance(at_cfg, dict):
        at_cfg = {}
    legacy_autotune_keys = {"max_bins", "min_bin_groups", "min_bucket_node"}
    legacy_autotune_found = sorted([k for k in legacy_autotune_keys if k in at_cfg])
    if legacy_autotune_found:
        raise ValueError(
            "Invalid config at `model.likelihoods.sample.shared_event_re.autotune`: "
            f"legacy keys not allowed: {', '.join(legacy_autotune_found)}. "
            "Use max_node_bins, min_groups_per_node_bin, min_node_bin."
        )
    se_autotune_enabled = bool(at_cfg.get("enabled", True))
    se_autotune_observe_epochs = int(at_cfg.get("observe_epochs", 1))
    se_autotune_latest_epoch = int(at_cfg.get("latest_epoch", 2))
    se_autotune_min_groups = int(at_cfg.get("min_groups", 128))
    se_autotune_max_bins = int(at_cfg.get("max_node_bins", 8))
    se_autotune_min_bin_groups = int(at_cfg.get("min_groups_per_node_bin", 24))
    se_autotune_min_bucket_node = int(at_cfg.get("min_node_bin", 512))
    se_autotune_min_gain = float(at_cfg.get("min_gain", 0.08))
    se_autotune_raise_nodes_cap = bool(at_cfg.get("raise_nodes_cap", True))
    se_autotune_nodes_cap_max = int(at_cfg.get("nodes_cap_max", 65536))

    lg_cfg = se_cfg.get("logging", {})
    if not isinstance(lg_cfg, dict):
        lg_cfg = {}
    se_logging_quiet = bool(lg_cfg.get("quiet", True))
    se_stats_log_every_epochs = int(lg_cfg.get("stats_log_every_epochs", 0))

    # Filters
    filters = _require_dict(model.get("filters"), "model.filters")
    fd = _require_dict(filters.get("dtimes"), "model.filters.dtimes")
    fe = _require_dict(filters.get("events"), "model.filters.events")
    fr = _require_dict(filters.get("residual"), "model.filters.residual")
    remove_duplicates = bool(fd.get("remove_duplicates", False))
    max_abs_input_dt = float(fd.get("max_abs_input_dt", 99999.0))
    dtime_thin_frac = float(fd.get("dtime_thin_frac", 1.0))
    flip_dt_sign = bool(fd.get("flip_dt_sign", False))
    cc_min = float(fd.get("cc_min", 0.0))
    min_dtimes = int(fe.get("min_dtimes", 1))
    min_unique_phase_per_event = int(fe.get("min_unique_phase_per_event", 1))
    min_dtimes_per_pair = int(fe.get("min_dtimes_per_pair", 1))
    min_event_degree = int(fe.get("min_event_degree", 0))
    min_events_per_cluster = int(fe.get("min_events_per_cluster", 0))
    max_pair_station_ratio = float(fe.get("max_pair_station_ratio", 1.0))
    ratio_filter_phase = str(fe.get("ratio_filter_phase", "before")).strip().lower()

    lin_cfg = fe.get("linearization_error", {})
    if not isinstance(lin_cfg, dict):
        lin_cfg = {}
    lin_enable = bool(lin_cfg.get("enabled", False))
    lin_phase = str(lin_cfg.get("phase", "after_phase1")).strip().lower()
    lin_batch_size = int(lin_cfg.get("batch_size", 50000))
    lin_sample_size = int(lin_cfg.get("sample_size", 200000))
    lin_log_every = int(lin_cfg.get("log_every_batches", 25))
    lin_max_ratio = lin_cfg.get("max_ratio", None)
    lin_max_ratio = None if lin_max_ratio is None else float(lin_max_ratio)

    residual_enabled = bool(fr.get("enabled", False))
    residual_method = str(fr.get("method", "mad")).lower() if residual_enabled else None
    residual_mad_sigma = float(fr.get("mad_sigma", 6.0)) if residual_enabled else None
    residual_abs_max = float(fr.get("abs_max", 99999.0)) if residual_enabled else None

    # Batching
    bt = _require_dict(inf.get("batching"), "inference.batching")
    bs = _require_dict(bt.get("standard"), "inference.batching.standard")
    eb = _require_dict(bt.get("event_batches"), "inference.batching.event_batches")
    batch_size_warmup = int(bs.get("warmup", 10000))
    batch_size_sgld = int(bs.get("sgld", 10000))
    batch_shuffle = bool(bs.get("shuffle", True))
    event_batch_enable = bool(eb.get("enabled", False))
    event_batch_size = int(eb.get("events_per_batch", 0)) if event_batch_enable else 0
    event_batch_max_edges = int(eb.get("max_edges_per_batch", 0)) if event_batch_enable else 0
    event_bucket_reorder_all = bool(eb.get("bucket_reorder_all", False))
    event_bucket_reuse_epochs = int(eb.get("bucket_reuse_epochs", 0)) if event_batch_enable else 0

    params["likelihood"] = str(sample_type)
    params["phase_unc"] = [float(sample_phase_unc[0]), float(sample_phase_unc[1])]
    params["_student_t_nu"] = float(sample_student_t_nu)
    params["_huber_delta"] = float(sample_huber_delta)
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

    params["_shared_event_re_enabled"] = bool(se_enabled)
    params["_shared_event_re_grouping"] = str(se_grouping)
    params["_shared_event_re_cluster_mode"] = str(se_cluster_mode)
    params["_shared_event_re_cluster_k"] = int(se_cluster_k)
    params["_shared_event_re_tau_s"] = [float(se_tau_ps[0]), float(se_tau_ps[1])]
    params["_shared_event_re_hierarchical"] = False
    params["_shared_event_re_tau_event_s"] = [0.0, 0.0]
    params["_shared_event_re_tau_cluster_s"] = [0.0, 0.0]
    params["_shared_event_re_rho_ps"] = 0.0
    params["_shared_event_re_joint_ps"] = False
    params["_shared_event_re_max_nodes_per_group"] = int(se_max_nodes_per_group)
    params["_shared_event_re_max_rows_per_group"] = int(se_max_rows_per_group)
    params["_shared_event_re_fallback_to_diag"] = bool(se_fallback_to_diag)
    params["_shared_event_re_abort_on_pcg_fallback"] = bool(se_abort_on_pcg_fallback)
    params["_shared_event_re_jitter0"] = float(se_jitter0)
    params["_shared_event_re_jitter_max"] = float(se_jitter_max)
    params["_shared_event_re_solver_kind"] = str(se_solver_kind)
    params["_shared_event_re_solver_max_iters"] = int(se_solver_max_iters)
    params["_shared_event_re_solver_min_iters"] = int(se_solver_min_iters)
    params["_shared_event_re_solver_tol"] = float(se_solver_tol)
    params["_shared_event_re_solver_batched"] = bool(se_solver_batched)
    params["_shared_event_re_solver_bucket_nodes"] = list(se_solver_bucket_nodes)
    params["_shared_event_re_solver_warm_start"] = bool(se_solver_warm_start)
    params["_shared_event_re_solver_cache_max_entries"] = int(se_solver_cache_max_entries)
    params["_shared_event_re_solver_prefetch_grouping"] = bool(se_solver_prefetch_grouping)
    params["_shared_event_re_solver_profile_micro_steps"] = bool(se_solver_profile_micro_steps)
    params["_shared_event_re_solver_merge_sparse_edge_bins"] = bool(se_solver_merge_sparse_edge_bins)
    params["_shared_event_re_solver_min_groups_per_edge_bin"] = int(se_solver_min_groups_per_edge_bin)
    params["_shared_event_re_solver_max_edge_bins_per_node"] = int(se_solver_max_edge_bins_per_node)
    params["_shared_event_re_solver_precompute_enabled"] = bool(se_solver_precompute_enabled)
    params["_shared_event_re_solver_precompute_device"] = str(se_solver_precompute_device)
    params["_shared_event_re_edge_weight_mode"] = str(se_edge_weight_mode)
    params["_shared_event_re_edge_weight_ell_km"] = float(se_edge_weight_ell_km)
    params["_shared_event_re_edge_weight_eps_km"] = float(se_edge_weight_eps_km)
    params["_shared_event_re_edge_weight_power"] = float(se_edge_weight_power)
    params["_shared_event_re_edge_weight_scale_km"] = float(se_edge_weight_scale_km)
    params["_shared_event_re_edge_weight_global_scale"] = float(se_edge_weight_global_scale)
    params["_shared_event_re_edge_weight_normalize"] = bool(se_edge_weight_normalize)
    params["_shared_event_re_autotune_enabled"] = bool(se_autotune_enabled)
    params["_shared_event_re_autotune_observe_epochs"] = int(se_autotune_observe_epochs)
    params["_shared_event_re_autotune_latest_epoch"] = int(se_autotune_latest_epoch)
    params["_shared_event_re_autotune_min_groups"] = int(se_autotune_min_groups)
    params["_shared_event_re_autotune_max_bins"] = int(se_autotune_max_bins)
    params["_shared_event_re_autotune_min_bin_groups"] = int(se_autotune_min_bin_groups)
    params["_shared_event_re_autotune_min_bucket_node"] = int(se_autotune_min_bucket_node)
    params["_shared_event_re_autotune_min_gain"] = float(se_autotune_min_gain)
    params["_shared_event_re_autotune_raise_nodes_cap"] = bool(se_autotune_raise_nodes_cap)
    params["_shared_event_re_autotune_nodes_cap_max"] = int(se_autotune_nodes_cap_max)
    params["_shared_event_re_logging_quiet"] = bool(se_logging_quiet)
    params["_shared_event_re_stats_log_every_epochs"] = int(se_stats_log_every_epochs)
    params["_shared_event_re_station_phase_enabled"] = bool(sp_enabled)
    params["_shared_event_re_station_phase_tau_s"] = [float(sp_tau_ps[0]), float(sp_tau_ps[1])]

    sample_group: Dict[str, Any] = {}
    for k, v in params.items():
        if k in {"likelihood", "phase_unc", "_student_t_nu", "_huber_delta"} or k.startswith("_shared_event_re_"):
            sample_group[k] = v
    params["_likelihood_groups"] = {"locate_map": locate_group, "sample": sample_group}
    params["_likelihood_groups_raw"] = {"locate_map": lk_loc, "sample": lk_smp}
    params["_likelihood_group_active"] = "sample"
    params.setdefault("model", {})["likelihood"] = lk_smp

    params["linearization_error_enable"] = bool(lin_enable)
    params["linearization_error_phase"] = str(lin_phase)
    params["linearization_error_batch_size"] = int(lin_batch_size)
    params["linearization_error_sample_size"] = int(lin_sample_size)
    params["linearization_error_log_every_batches"] = int(lin_log_every)
    params["linearization_error_max_ratio"] = lin_max_ratio
    params["residual_filter_enable"] = bool(residual_enabled)
    params["residual_filter_method"] = residual_method
    params["residual_filter_mad_sigma"] = residual_mad_sigma
    params["residual_filter_abs_max"] = residual_abs_max
    params["batch_size_warmup"] = int(batch_size_warmup)
    params["batch_size_sgld"] = int(batch_size_sgld)
    params["batch_shuffle"] = bool(batch_shuffle)
    params["event_batch_enable"] = bool(event_batch_enable)
    params["event_batch_size"] = int(event_batch_size)
    params["event_batch_max_edges"] = int(event_batch_max_edges)
    params["event_bucket_reorder_all"] = bool(event_bucket_reorder_all)
    params["event_bucket_reuse_epochs"] = int(event_bucket_reuse_epochs)
    return params


def _materialize_block4(params: Dict[str, Any]) -> Dict[str, Any]:
    inf = _require_dict(params.get("inference"), "inference")
    dg = _require_dict(inf.get("diagnostics", {}), "inference.diagnostics")
    rt = _require_dict(inf.get("runtime"), "inference.runtime")
    saf = _require_dict(inf.get("safety"), "inference.safety")

    pair_count_stats_enable = bool(dg.get("pair_count_stats_enable", False))
    sgld_log_gnoise = bool(dg.get("sgld_log_gnoise", False))
    sgld_log_temperature = bool(dg.get("sgld_log_temperature", False))
    display_precond_every = int(dg.get("display_precond_every", 50))

    wb_cfg = dg.get("wandb", {})
    wb_diag_enabled = True
    wb_groups = {"all"}
    if isinstance(wb_cfg, dict):
        if wb_cfg.get("enabled", None) is not None:
            wb_diag_enabled = bool(wb_cfg.get("enabled"))
        g = wb_cfg.get("groups", None)
        if isinstance(g, dict):
            wb_groups = {str(k).strip().lower() for k, v in g.items() if bool(v)}
        elif isinstance(g, list):
            wb_groups = {str(x).strip().lower() for x in g if str(x).strip()}

    ess_online = dg.get("ess_online", {})
    if not isinstance(ess_online, dict):
        ess_online = {}
    ess_online_enabled = bool(ess_online.get("enabled", False))
    ess_every = int(ess_online.get("every_n_samples", 0)) if ess_online_enabled else 0
    ess_n_events = int(ess_online.get("n_events", 0)) if ess_online_enabled else 0
    ess_seed = int(ess_online.get("seed", 0)) if ess_online_enabled else 0
    ess_window = int(ess_online.get("window", 0)) if ess_online_enabled else 0
    ess_max_lag = int(ess_online.get("max_lag", 0)) if ess_online_enabled else 0
    dims = ess_online.get("dims", [0, 1, 2])
    if not isinstance(dims, list) or not dims:
        dims = [0, 1, 2]

    runtime_seed = int(rt.get("seed", 0) or 0)
    gp_cfg = rt.get("gauge_projection", {})
    if not isinstance(gp_cfg, dict):
        gp_cfg = {}
    gp_enable = bool(gp_cfg.get("enabled", False))
    gp_mode = str(gp_cfg.get("mode", "global")).strip().lower()
    gp_dims = gp_cfg.get("dims", [0, 1, 2])
    if not isinstance(gp_dims, list) or not gp_dims:
        gp_dims = [0, 1, 2]
    gp_apply_noise = bool(gp_cfg.get("apply_noise", True))
    gp_apply_momentum = bool(gp_cfg.get("apply_momentum", True))

    max_abs_dX = saf.get("max_abs_dX", None)
    if max_abs_dX is not None:
        if not isinstance(max_abs_dX, list) or len(max_abs_dX) != 4:
            raise ValueError("Invalid config at `inference.safety.max_abs_dX`: expected null or list[4]")
        max_abs_dX = [float(x) for x in max_abs_dX]

    params["pair_count_stats_enable"] = bool(pair_count_stats_enable)
    params["sgld_log_gnoise"] = bool(sgld_log_gnoise)
    params["sgld_log_temperature"] = bool(sgld_log_temperature)
    params["display_precond_every"] = int(display_precond_every)
    params["_wandb_diag_enabled"] = bool(wb_diag_enabled)
    params["_wandb_diag_groups"] = sorted(list(wb_groups))
    params["ess_online_enabled"] = bool(ess_online_enabled)
    params["ess_online_every_n_samples"] = int(ess_every)
    params["ess_online_n_events"] = int(ess_n_events)
    params["ess_online_seed"] = int(ess_seed)
    params["ess_online_window"] = int(ess_window)
    params["ess_online_max_lag"] = int(ess_max_lag)
    params["ess_online_dims"] = [int(x) for x in dims]
    params["cuda_empty_cache_every"] = int(rt.get("cuda_empty_cache_every", 0))
    params["reset_batch_numbers"] = bool(rt.get("reset_batch_numbers", True))
    params["clear_samples_on_reset"] = bool(rt.get("clear_samples_on_reset", True))
    params["min_samples_to_save"] = int(rt.get("min_samples_to_save", 0))
    params["verbose"] = bool(rt.get("verbose", True))
    params["cluster_events"] = bool(rt.get("cluster_events", False))
    params["runtime_seed"] = int(runtime_seed)
    params["gauge_project_enable"] = bool(gp_enable)
    params["gauge_project_mode"] = str(gp_mode)
    params["gauge_project_dims"] = [int(x) for x in gp_dims]
    params["gauge_project_apply_noise"] = bool(gp_apply_noise)
    params["gauge_project_apply_momentum"] = bool(gp_apply_momentum)
    if max_abs_dX is not None:
        params["max_abs_dX"] = list(max_abs_dX)
    return params


def _materialize_block5(params: Dict[str, Any]) -> Dict[str, Any]:
    inf = _require_dict(params.get("inference"), "inference")
    compute = inf.get("compute", None)
    if compute is None:
        params["devices"] = []
        return params
    compute = _require_dict(compute, "inference.compute")
    devices = compute.get("devices", None)
    if devices is None:
        params["devices"] = []
        return params
    devices = _require_list(devices, "inference.compute.devices")
    if len(devices) == 0:
        params["devices"] = []
        return params
    parsed: List[int] = [_parse_device_entry(x) for x in devices]
    params["devices"] = parsed
    return params


def to_legacy_runtime_params(
    resolved: ResolvedConfig,
    *,
    profile: str = "all",
    require_priors: bool = True,
) -> Dict[str, Any]:
    """
    Convert resolved canonical config to legacy runtime params dictionary.

    Parameters
    ----------
    profile:
      - "all": full runtime materialization
      - "synth": synth-specific materialization (blocks 1,3,5 analog)
    require_priors:
      If True, run priors materialization.
    """
    if profile not in {"all", "synth"}:
        raise ValueError(f"Unsupported legacy bridge profile: {profile!r}")

    params = _canonical_to_legacy_nested(asdict(resolved.canonical))
    params = _materialize_block1(params)
    if profile == "all":
        params = _materialize_block2(params)
    params = _materialize_block3(params)
    if profile == "all":
        params = _materialize_block4(params)
    params = _materialize_block5(params)

    if require_priors:
        params = validate_and_materialize_priors(params)

    params["_config_v2_defaults_applied"] = list(resolved.defaults_applied)
    return params

