from __future__ import annotations

import json
from typing import Any, Dict, List


def load_params(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


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


def _require_list(v: Any, path: str, *, length: int | None = None) -> List[Any]:
    if not isinstance(v, list):
        raise _err(path, f"expected list, got {type(v).__name__}")
    if length is not None and len(v) != length:
        raise _err(path, f"expected list of length {length}, got {len(v)}")
    return list(v)


def _require_float_list(v: Any, path: str, *, length: int) -> List[float]:
    xs = _require_list(v, path, length=length)
    out: List[float] = []
    for i, xi in enumerate(xs):
        if not isinstance(xi, (int, float)):
            raise _err(f"{path}[{i}]", f"expected number, got {type(xi).__name__}")
        out.append(float(xi))
    return out


def validate_and_materialize(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the clean SPIDER schema and materialize internal flat keys.

    This schema is intentionally narrow and only supports the Cahuilla feature set.
    """
    # --- io ---
    io = _require_dict(_require(params, "io", "io"), "io")
    params["dtime_file"] = _require_str(_require(io, "dtime_file", "io"), "io.dtime_file")
    params["station_file"] = _require_str(_require(io, "station_file", "io"), "io.station_file")
    params["catalog_infile"] = _require_str(_require(io, "catalog_infile", "io"), "io.catalog_infile")
    params["catalog_outfile"] = _require_str(_require(io, "catalog_outfile", "io"), "io.catalog_outfile")
    params["samples_outfile"] = _require_str(_require(io, "samples_outfile", "io"), "io.samples_outfile")
    params["checkpoint_dir"] = _require_str(_require(io, "checkpoint_dir", "io"), "io.checkpoint_dir")
    params["checkpoint_interval"] = int(_require_num(_require(io, "checkpoint_interval", "io"), "io.checkpoint_interval"))
    params["save_every_n"] = int(_require_num(_require(io, "save_every_n", "io"), "io.save_every_n"))
    params["sample_write_interval"] = int(_require_num(io.get("sample_write_interval", params["checkpoint_interval"]), "io.sample_write_interval"))
    params["write_samples"] = _require_bool(_require(io, "write_samples", "io"), "io.write_samples")

    # --- wandb (optional) ---
    wb = params.get("wandb", None)
    if wb is None:
        params["use_wandb"] = False
        params["wandb_project_name"] = None
        params["wandb_run_name"] = None
        params["wandb_log_every_batches"] = 0
        params["wandb_max_median_samples"] = 200_000
        params["wandb_residual_sample_size"] = 200_000
        params["wandb_precond_sample_size"] = 200_000
    else:
        wb = _require_dict(wb, "wandb")
        params["use_wandb"] = _require_bool(_require(wb, "enabled", "wandb"), "wandb.enabled")
        project_name_v = _require(wb, "project_name", "wandb")
        run_name_v = _require(wb, "run_name", "wandb")
        if params["use_wandb"]:
            params["wandb_project_name"] = _require_str(project_name_v, "wandb.project_name")
            params["wandb_run_name"] = None if run_name_v is None else _require_str(run_name_v, "wandb.run_name")
        else:
            params["wandb_project_name"] = None if project_name_v is None else str(project_name_v)
            params["wandb_run_name"] = None if run_name_v is None else str(run_name_v)
        params["wandb_log_every_batches"] = int(wb.get("log_every_batches", 0) or 0)
        params["wandb_max_median_samples"] = int(wb.get("max_median_samples", 200_000) or 200_000)
        params["wandb_residual_sample_size"] = int(wb.get("residual_sample_size", 200_000) or 200_000)
        params["wandb_precond_sample_size"] = int(wb.get("precond_sample_size", 200_000) or 200_000)

    # --- model ---
    model = _require_dict(_require(params, "model", "model"), "model")
    params["model_file"] = _require_str(_require(model, "model_file", "model"), "model.model_file")
    dom = _require_dict(_require(model, "domain", "model"), "model.domain")
    params["lon_min"] = _require_num(_require(dom, "lon_min", "model.domain"), "model.domain.lon_min")
    params["lat_min"] = _require_num(_require(dom, "lat_min", "model.domain"), "model.domain.lat_min")
    params["z_min"] = _require_num(_require(dom, "z_min", "model.domain"), "model.domain.z_min")
    params["z_max"] = _require_num(_require(dom, "z_max", "model.domain"), "model.domain.z_max")
    params["scale"] = _require_num(_require(dom, "scale", "model.domain"), "model.domain.scale")

    priors = _require_dict(_require(model, "priors", "model"), "model.priors")
    ev = _require_dict(_require(priors, "event", "model.priors"), "model.priors.event")
    params["prior_event_enable"] = _require_bool(_require(ev, "enabled", "model.priors.event"), "model.priors.event.enabled")
    ev_params = _require_dict(_require(ev, "params", "model.priors.event"), "model.priors.event.params")
    params["prior_event_std"] = _require_float_list(_require(ev_params, "std", "model.priors.event.params"), "model.priors.event.params.std", length=4)

    # --- likelihood ---
    lk = _require_dict(_require(model, "likelihood", "model"), "model.likelihood")
    lk_type = _require_str(_require(lk, "type", "model.likelihood"), "model.likelihood.type").lower()
    if lk_type not in {"correlated_gaussian"}:
        raise _err("model.likelihood.type", "supported: 'correlated_gaussian'")
    params["likelihood_type"] = lk_type
    params["phase_unc"] = _require_float_list(_require(lk, "phase_unc", "model.likelihood"), "model.likelihood.phase_unc", length=2)

    # Shared-event random effects (required in this clean build)
    se = _require_dict(_require(lk, "shared_event_re", "model.likelihood"), "model.likelihood.shared_event_re")
    params["_shared_event_re_enabled"] = _require_bool(_require(se, "enabled", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.enabled")
    params["_shared_event_re_grouping"] = _require_str(_require(se, "grouping", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.grouping")
    params["_shared_event_re_tau_s"] = _require_float_list(_require(se, "tau_s", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.tau_s", length=2)
    params["_shared_event_re_max_nodes_per_group"] = int(_require_num(_require(se, "max_nodes_per_group", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.max_nodes_per_group"))
    params["_shared_event_re_max_rows_per_group"] = int(_require_num(_require(se, "max_rows_per_group", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.max_rows_per_group"))
    params["_shared_event_re_solver"] = _require_str(_require(se, "solver", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.solver")
    params["_shared_event_re_pcg_max_iters"] = int(_require_num(_require(se, "pcg_max_iters", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.pcg_max_iters"))
    params["_shared_event_re_pcg_tol"] = float(_require_num(_require(se, "pcg_tol", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.pcg_tol"))

    # Whitening sub-block (required when shared_event_re is enabled)
    w = _require_dict(_require(se, "whitening", "model.likelihood.shared_event_re"), "model.likelihood.shared_event_re.whitening")
    if "enabled" in w:
        enabled_val = _require_bool(w.get("enabled"), "model.likelihood.shared_event_re.whitening.enabled")
        if params["_shared_event_re_enabled"] and not enabled_val:
            raise _err("model.likelihood.shared_event_re.whitening.enabled", "whitening cannot be disabled when shared_event_re is enabled")
    params["_shared_event_re_whitening_enabled"] = bool(params["_shared_event_re_enabled"])
    params["_shared_event_re_whitening_solver"] = _require_str(_require(w, "solver", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.solver")
    params["_shared_event_re_whitening_pcg_max_iters"] = int(_require_num(_require(w, "pcg_max_iters", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.pcg_max_iters"))
    params["_shared_event_re_whitening_pcg_tol"] = float(_require_num(_require(w, "pcg_tol", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.pcg_tol"))
    params["_shared_event_re_whitening_pcg_min_iters"] = int(_require_num(_require(w, "pcg_min_iters", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.pcg_min_iters"))
    params["_shared_event_re_whitening_edge_weighting"] = _require_str(_require(w, "edge_weighting", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.edge_weighting")
    params["_shared_event_re_whitening_edge_weight_power"] = float(_require_num(_require(w, "edge_weight_power", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.edge_weight_power"))
    params["_shared_event_re_whitening_edge_weight_scale_km"] = float(_require_num(_require(w, "edge_weight_scale_km", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.edge_weight_scale_km"))
    params["_shared_event_re_whitening_edge_weight_eps_km"] = float(_require_num(_require(w, "edge_weight_eps_km", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.edge_weight_eps_km"))
    params["_shared_event_re_whitening_edge_weight_global_scale"] = float(_require_num(_require(w, "edge_weight_global_scale", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.edge_weight_global_scale"))
    params["_shared_event_re_whitening_edge_weight_normalize"] = _require_bool(_require(w, "edge_weight_normalize", "model.likelihood.shared_event_re.whitening"), "model.likelihood.shared_event_re.whitening.edge_weight_normalize")

    # --- filters ---
    flt = _require_dict(_require(model, "filters", "model"), "model.filters")
    dtf = _require_dict(_require(flt, "dtimes", "model.filters"), "model.filters.dtimes")
    params["remove_duplicates"] = _require_bool(_require(dtf, "remove_duplicates", "model.filters.dtimes"), "model.filters.dtimes.remove_duplicates")
    params["max_abs_input_dt"] = float(_require_num(_require(dtf, "max_abs_input_dt", "model.filters.dtimes"), "model.filters.dtimes.max_abs_input_dt"))
    params["dtime_thin_frac"] = float(_require_num(_require(dtf, "dtime_thin_frac", "model.filters.dtimes"), "model.filters.dtimes.dtime_thin_frac"))
    params["flip_dt_sign"] = _require_bool(_require(dtf, "flip_dt_sign", "model.filters.dtimes"), "model.filters.dtimes.flip_dt_sign")
    params["cc_min"] = float(_require_num(_require(dtf, "cc_min", "model.filters.dtimes"), "model.filters.dtimes.cc_min"))

    ef = _require_dict(_require(flt, "events", "model.filters"), "model.filters.events")
    params["min_dtimes"] = int(_require_num(_require(ef, "min_dtimes", "model.filters.events"), "model.filters.events.min_dtimes"))
    params["min_unique_phase_per_event"] = int(_require_num(_require(ef, "min_unique_phase_per_event", "model.filters.events"), "model.filters.events.min_unique_phase_per_event"))
    params["min_dtimes_per_pair"] = int(_require_num(_require(ef, "min_dtimes_per_pair", "model.filters.events"), "model.filters.events.min_dtimes_per_pair"))
    params["min_event_degree"] = int(_require_num(_require(ef, "min_event_degree", "model.filters.events"), "model.filters.events.min_event_degree"))
    params["min_events_per_cluster"] = int(_require_num(_require(ef, "min_events_per_cluster", "model.filters.events"), "model.filters.events.min_events_per_cluster"))
    params["max_pair_station_ratio"] = float(_require_num(_require(ef, "max_pair_station_ratio", "model.filters.events"), "model.filters.events.max_pair_station_ratio"))
    params["ratio_filter_phase"] = _require_str(_require(ef, "ratio_filter_phase", "model.filters.events"), "model.filters.events.ratio_filter_phase").lower()
    lin = _require_dict(_require(ef, "linearization_error", "model.filters.events"), "model.filters.events.linearization_error")
    params["linearization_error_enable"] = _require_bool(_require(lin, "enabled", "model.filters.events.linearization_error"), "model.filters.events.linearization_error.enabled")
    params["linearization_error_phase"] = _require_str(_require(lin, "phase", "model.filters.events.linearization_error"), "model.filters.events.linearization_error.phase").lower()
    params["linearization_error_max_ratio"] = float(_require_num(_require(lin, "max_ratio", "model.filters.events.linearization_error"), "model.filters.events.linearization_error.max_ratio"))
    params["linearization_error_batch_size"] = int(_require_num(_require(lin, "batch_size", "model.filters.events.linearization_error"), "model.filters.events.linearization_error.batch_size"))
    params["linearization_error_sample_size"] = int(_require_num(_require(lin, "sample_size", "model.filters.events.linearization_error"), "model.filters.events.linearization_error.sample_size"))
    params["linearization_error_log_every_batches"] = int(_require_num(_require(lin, "log_every_batches", "model.filters.events.linearization_error"), "model.filters.events.linearization_error.log_every_batches"))

    rf = _require_dict(_require(flt, "residual", "model.filters"), "model.filters.residual")
    params["residual_filter_enable"] = _require_bool(_require(rf, "enabled", "model.filters.residual"), "model.filters.residual.enabled")
    params["residual_filter_method"] = _require_str(_require(rf, "method", "model.filters.residual"), "model.filters.residual.method").lower()
    params["residual_filter_mad_sigma"] = float(_require_num(_require(rf, "mad_sigma", "model.filters.residual"), "model.filters.residual.mad_sigma"))
    params["residual_filter_abs_max"] = float(_require_num(_require(rf, "abs_max", "model.filters.residual"), "model.filters.residual.abs_max"))

    # --- inference ---
    inf = _require_dict(_require(params, "inference", "inference"), "inference")
    comp = _require_dict(_require(inf, "compute", "inference"), "inference.compute")
    devs = _require_list(_require(comp, "devices", "inference.compute"), "inference.compute.devices")
    params["devices"] = [int(x) for x in devs]
    if not params["devices"]:
        raise _err("inference.compute.devices", "must contain at least one device id")

    samp = _require_dict(_require(inf, "sampler", "inference"), "inference.sampler")
    backend = _require_str(_require(samp, "backend", "inference.sampler"), "inference.sampler.backend").lower()
    if backend not in {"psgld", "monge"}:
        raise _err("inference.sampler.backend", "supported: 'psgld' or 'monge'")
    params["sampler_backend"] = backend
    if "lr_mode" in samp:
        raise _err("inference.sampler.lr_mode", "removed; lr_mode is always per_obs in spider")
    params["sampler_lr_mode"] = "per_obs"
    params["sampler_epochs_per_phase"] = _require_list(_require(samp, "epochs_per_phase", "inference.sampler"), "inference.sampler.epochs_per_phase", length=4)
    params["sampler_lr_per_phase"] = _require_float_list(_require(samp, "lr", "inference.sampler"), "inference.sampler.lr", length=4)
    params["lr_warmup"] = float(params["sampler_lr_per_phase"][0])
    params["lr_sampler"] = float(params["sampler_lr_per_phase"][3])
    params["sampler_dt_lr_mult"] = float(_require_num(_require(samp, "dt_lr_mult", "inference.sampler"), "inference.sampler.dt_lr_mult"))
    params["sampler_temperature"] = float(_require_num(_require(samp, "temperature", "inference.sampler"), "inference.sampler.temperature"))
    if backend == "psgld":
        params["sampler_eps"] = float(_require_num(_require(samp, "eps", "inference.sampler"), "inference.sampler.eps"))
        params["sampler_beta"] = float(_require_num(_require(samp, "beta", "inference.sampler"), "inference.sampler.beta"))
        pre = _require_dict(_require(samp, "preconditioning", "inference.sampler"), "inference.sampler.preconditioning")
        params["sampler_preconditioning"] = _require_bool(_require(pre, "enabled", "inference.sampler.preconditioning"), "inference.sampler.preconditioning.enabled")
        params["sampler_preconditioner"] = _require_str(_require(pre, "type", "inference.sampler.preconditioning"), "inference.sampler.preconditioning.type")
        params["sampler_preconditioning_include_gamma"] = _require_bool(_require(pre, "include_gamma", "inference.sampler.preconditioning"), "inference.sampler.preconditioning.include_gamma")
    else:
        if "preconditioning" in samp:
            raise _err("inference.sampler.preconditioning", "not used for backend='monge'")
        if "beta" in samp:
            raise _err("inference.sampler.beta", "not used for backend='monge'; use inference.sampler.monge.ema_beta")
        if "eps" in samp:
            raise _err("inference.sampler.eps", "not used for backend='monge'; use inference.sampler.monge.eps")
        monge = _require_dict(_require(samp, "monge", "inference.sampler"), "inference.sampler.monge")
        params["sampler_monge_alpha"] = float(_require_num(_require(monge, "alpha", "inference.sampler.monge"), "inference.sampler.monge.alpha"))
        params["sampler_monge_ema_beta"] = float(_require_num(_require(monge, "ema_beta", "inference.sampler.monge"), "inference.sampler.monge.ema_beta"))
        params["sampler_monge_eps"] = float(_require_num(_require(monge, "eps", "inference.sampler.monge"), "inference.sampler.monge.eps"))
    params["freeze_preconditioner_sampling"] = _require_bool(_require(samp, "freeze_preconditioner_sampling", "inference.sampler"), "inference.sampler.freeze_preconditioner_sampling")

    # batching
    batching = _require_dict(_require(inf, "batching", "inference"), "inference.batching")
    bstd = _require_dict(_require(batching, "standard", "inference.batching"), "inference.batching.standard")
    params["batch_size_warmup"] = int(_require_num(_require(bstd, "warmup", "inference.batching.standard"), "inference.batching.standard.warmup"))
    params["batch_size_sgld"] = int(_require_num(_require(bstd, "sgld", "inference.batching.standard"), "inference.batching.standard.sgld"))
    params["batching_shuffle"] = _require_bool(_require(bstd, "shuffle", "inference.batching.standard"), "inference.batching.standard.shuffle")

    # runtime / safety
    rt = _require_dict(_require(params, "runtime", "runtime"), "runtime")
    safety = _require_dict(_require(rt, "safety", "runtime"), "runtime.safety")
    params["max_abs_dX"] = _require_float_list(_require(safety, "max_abs_dX", "runtime.safety"), "runtime.safety.max_abs_dX", length=4)
    params["runtime_seed"] = int(rt.get("seed", 0) or 0)
    params["runtime_verbose"] = bool(rt.get("verbose", True))
    params["clear_samples_on_reset"] = bool(rt.get("clear_samples_on_reset", False))
    gp = _require_dict(_require(rt, "gauge_projection", "runtime"), "runtime.gauge_projection")
    params["gauge_projection_enabled"] = _require_bool(_require(gp, "enabled", "runtime.gauge_projection"), "runtime.gauge_projection.enabled")
    params["gauge_projection_mode"] = _require_str(_require(gp, "mode", "runtime.gauge_projection"), "runtime.gauge_projection.mode").lower()
    params["gauge_projection_dims"] = [int(x) for x in _require_list(_require(gp, "dims", "runtime.gauge_projection"), "runtime.gauge_projection.dims")]
    params["gauge_projection_apply_noise"] = _require_bool(_require(gp, "apply_noise", "runtime.gauge_projection"), "runtime.gauge_projection.apply_noise")
    params["gauge_projection_apply_momentum"] = _require_bool(_require(gp, "apply_momentum", "runtime.gauge_projection"), "runtime.gauge_projection.apply_momentum")

    # Phase epochs
    params["phase1_epochs"] = int(params["sampler_epochs_per_phase"][0])
    params["phase2_epochs"] = int(params["sampler_epochs_per_phase"][1])
    params["phase3_epochs"] = int(params["sampler_epochs_per_phase"][2])
    params["phase4_epochs"] = int(params["sampler_epochs_per_phase"][3])

    return params
