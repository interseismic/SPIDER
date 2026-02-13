"""Sampler backend factory for SPIDER inference phases (2–4)."""

import torch
from typing import Tuple, List

from .sgld import pSGLD  # default backend
from .sghmc import SGHMC


def _attach_set_lr(opt: torch.optim.Optimizer) -> None:
    if not hasattr(opt, "set_lr"):
        def set_lr(self, new_lr: float):
            for g in self.param_groups:
                g["lr"] = float(new_lr)
        opt.set_lr = set_lr.__get__(opt, opt.__class__)  # type: ignore[attr-defined]
    if not hasattr(opt, "preconditioner_stats"):
        def preconditioner_stats(self):
            return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
        opt.preconditioner_stats = preconditioner_stats.__get__(opt, opt.__class__)  # type: ignore[attr-defined]
    if not hasattr(opt, "grad_vs_noise_geomean"):
        def grad_vs_noise_geomean(self) -> float:
            if hasattr(self, "grad_vs_noise_stats"):
                return float(self.grad_vs_noise_stats().get("gm", float("nan")))
            return float("nan")
        opt.grad_vs_noise_geomean = grad_vs_noise_geomean.__get__(opt, opt.__class__)  # type: ignore[attr-defined]
    if not hasattr(opt, "grad_vs_noise_stats"):
        def grad_vs_noise_stats(self) -> dict:
            return {"gm": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
        opt.grad_vs_noise_stats = grad_vs_noise_stats.__get__(opt, opt.__class__)  # type: ignore[attr-defined]


def _ensure_common_group_keys(opt: torch.optim.Optimizer, *, params: dict, n_obs: int) -> None:
    for g in opt.param_groups:
        g.setdefault("beta", float(params["sampler_beta"]))
        g.setdefault("eps", float(params["sampler_eps"]))
        g.setdefault("preconditioning", False)
        g.setdefault("preconditioner", "none")
        g.setdefault("add_noise", False)
        g.setdefault("noise_scale", 0.0)
        # Force update temperature as backend defaults might set it to 1.0
        g["temperature"] = float(params["sampler_temperature"])
        g.setdefault("n_obs", int(n_obs))
        g.setdefault("freeze_preconditioner", bool(params["freeze_preconditioner_sampling"]))
        g.setdefault("is_burnin", True)


def create_sampler_backend(params: dict, state) -> Tuple[str, torch.optim.Optimizer]:
    """
    Factory for sampler backends used in Phases 2–4.
    Returns (backend_name, optimizer_instance).

    Supported backends:
      - 'psgld' (default)
      - 'sghmc'

    Legacy aliases are no longer supported; use the canonical backend names above.
    """
    backend = str(params["sampler_backend"]).strip().lower()
    # Parameter list: ΔX_src only (noise learning removed; fixed phase_unc only)
    base_params_list: List[torch.nn.Parameter] = [state.dX_src]

    lr = float(params["lr_sampler"])
    n_obs = int(getattr(state, "N", params.get("n_obs", 1)))
    if n_obs <= 0:
        n_obs = 1

    if backend == "psgld":
        # pSGLD multiplies minibatch-mean grads by n_obs internally; scale lr down by n_obs.
        lr_eff = lr / float(n_obs)
        base_group = {"params": base_params_list, "group_name": "core"}
        param_groups = [base_group]
        precond_arg = str(params["sampler_preconditioner"]).strip().lower()
        if bool(params.get("sampler_preconditioning", False)) and precond_arg in {"none", "false", ""}:
            precond_arg = "rmsprop"
        opt = pSGLD(
            params=param_groups,
            n_obs=state.N,
            lr=lr_eff,
            beta=float(params["sampler_beta"]),
            eps=float(params["sampler_eps"]),
            preconditioning=bool(params["sampler_preconditioning"]),
            preconditioner=precond_arg,
            include_gamma=bool(params.get("sampler_preconditioning_include_gamma", True)),
            add_noise=False,
        )
        _ensure_common_group_keys(opt, params=params, n_obs=state.N)

        # Ensure group keys reflect config (avoid accidental "none")
        precond = str(params["sampler_preconditioner"]).strip().lower()
        if bool(params.get("sampler_preconditioning", False)) and precond in {"none", "false", ""}:
            precond = "rmsprop"
        # Backwards-compatible alias
        if precond == "matrix_ema":
            precond = "blockdiag_fisher"
        include_gamma_proxy_bdf = bool(params.get("sampler_preconditioning_include_gamma_proxy", False))
        for g in opt.param_groups:
            g["preconditioner"] = precond
            g["preconditioning"] = bool(params["sampler_preconditioning"])
            if precond == "monge":
                g["monge_alpha"] = float(params.get("sampler_preconditioning_monge_alpha", 1.0))
            if precond == "shampoo":
                g["shampoo_beta"] = float(params.get("sampler_preconditioning_shampoo_beta", 0.99))
                g["shampoo_eps"] = float(params.get("sampler_preconditioning_shampoo_eps", 1e-6))
                g["shampoo_update_every"] = int(params.get("sampler_preconditioning_shampoo_update_every", 10))
                g["shampoo_max_dim"] = int(params.get("sampler_preconditioning_shampoo_max_dim", 512))
            # Optional: attach static disjoint blocks for blockdiag_fisher (computed in LocateState).
            if precond == "blockdiag_fisher":
                # Optional: cheap diagonal Γ proxy for blockdiag_fisher.
                g["blockdiag_fisher_include_gamma_proxy"] = include_gamma_proxy_bdf
                bm = getattr(state, "precond_block_members", None)
                bs = getattr(state, "precond_block_sizes", None)
                if bm is not None and bs is not None:
                    g["blockdiag_fisher_block_members"] = bm
                    g["blockdiag_fisher_block_sizes"] = bs
                    g["blockdiag_fisher_max_cluster_size"] = int(params.get("blockdiag_fisher_max_cluster_size", bm.shape[1]))

        _maybe_attach_gauge_projection(opt, params=params, state=state)
        return "psgld", opt

    if backend == "sghmc":
        # SGHMC drift uses n_obs * (minibatch-mean grad) internally; scale lr down by n_obs.
        lr_eff = lr / float(n_obs)
        base_group = {"params": base_params_list, "group_name": "core"}
        param_groups = [base_group]
        opt = SGHMC(
            params=param_groups,
            n_obs=state.N,
            lr=lr_eff,
            beta=float(params["sampler_beta"]),  # reuse beta for RMSprop stats
            eps=float(params["sampler_eps"]),
            alpha=float(params.get("sghmc_alpha", 0.01)),
            preconditioning=bool(params["sampler_preconditioning"]),
            add_noise=False,  # noise off in phase 2; enabled later
        )
        _ensure_common_group_keys(opt, params=params, n_obs=state.N)

        # Force preconditioner mode from config (avoid default "none")
        precond = str(params["sampler_preconditioner"]).strip().lower()
        for g in opt.param_groups:
            g["preconditioner"] = precond
            g["preconditioning"] = bool(params["sampler_preconditioning"])

        # Ensure alpha exists in groups
        for g in opt.param_groups:
            g.setdefault("alpha", float(params.get("sghmc_alpha", 0.01)))
        # Attach optional gauge-projection config (hard constraint on translation mode) to optimizer instance.
        _maybe_attach_gauge_projection(opt, params=params, state=state)
        return "sghmc", opt

    raise ValueError(
        f"Unknown sampler backend '{backend}'. Supported: psgld, sghmc."
    )


def _maybe_attach_gauge_projection(opt: torch.optim.Optimizer, *, params: dict, state) -> None:
    """
    Attach gauge-projection configuration to the optimizer instance (not stored in state_dict).

    This allows sampler backends to:
    - project out the mean gradient before preconditioner updates, and
    - project injected noise / momentum so the translation mode cannot random-walk.
    """
    try:
        enable = bool(params.get("gauge_project_enable", False))
    except Exception:
        enable = False
    if not enable:
        return
    try:
        dims = params.get("gauge_project_dims", [0, 1, 2])
        if not isinstance(dims, list) or len(dims) == 0:
            dims = [0, 1, 2]
        dims = tuple(int(d) for d in dims)
    except Exception:
        dims = (0, 1, 2)
    try:
        mode = str(params.get("gauge_project_mode", "global")).strip().lower()
    except Exception:
        mode = "global"
    if mode not in {"global", "cluster"}:
        mode = "global"
    try:
        apply_noise = bool(params.get("gauge_project_apply_noise", True))
    except Exception:
        apply_noise = True
    try:
        apply_momentum = bool(params.get("gauge_project_apply_momentum", True))
    except Exception:
        apply_momentum = True

    # Store on optimizer instance; samplers check these attributes in step().
    setattr(opt, "_gauge_project_enable", True)
    setattr(opt, "_gauge_project_dims", dims)
    setattr(opt, "_gauge_project_mode", mode)
    setattr(opt, "_gauge_project_apply_noise", apply_noise)
    setattr(opt, "_gauge_project_apply_momentum", apply_momentum)
    dX_param = getattr(state, "dX_src", None)
    cid = getattr(state, "cluster_ids", None)
    cc = getattr(state, "cluster_counts", None)
    # Enforce cluster mode if requested: do not silently fall back to global.
    if mode == "cluster":
        if not isinstance(cid, torch.Tensor):
            raise RuntimeError("gauge_projection.mode='cluster' requires state.cluster_ids (Tensor).")
        if not isinstance(cc, torch.Tensor):
            raise RuntimeError("gauge_projection.mode='cluster' requires state.cluster_counts (Tensor).")
        if not isinstance(dX_param, torch.Tensor):
            raise RuntimeError("gauge_projection requires state.dX_src (Tensor).")
        if cid.ndim != 1 or int(cid.numel()) != int(dX_param.shape[0]):
            raise RuntimeError("gauge_projection.mode='cluster': cluster_ids must have shape (n_events,) matching dX_src.")
    setattr(opt, "_gauge_project_param", dX_param)
    setattr(opt, "_gauge_cluster_ids", cid)
    setattr(opt, "_gauge_cluster_counts", cc)
    return


def transplant_from_adam_if_supported(adam_opt: torch.optim.Optimizer, sampler: torch.optim.Optimizer) -> None:
    """
    If the sampler backend supports transplanting Adam's exp_avg_sq (RMSprop stats),
    perform the transplant. Currently supported for pSGLD (and used for SGHMC as well).
    """
    try:
        # Only pSGLD exposes a direct transplant function; SGHMC uses the same state key.
        from .sgld import transplant_v_from_adam as _tx
        _tx(adam_opt, sampler)
    except Exception:
        # No-op if unsupported
        pass


