from typing import Dict, List
import time
import torch
import math
import numpy as np
import torch.distributed as dist
from spider.core.hierarchy import (
    update_precision_hyperparameter,
    update_corr_error_tau_hyperparameter,
)
from spider.utils.console import info, warn
from spider.core.state import (
    LocateState,
    _current_noise_scales,
    _clamp_dX_inplace,
    _apply_shared_event_latent_constraints_inplace,
    _apply_dd_graph_re_constraints_inplace,
)
from spider.core.batching import _ensure_owner_buckets, _iter_event_batches
from spider.core.modeling import (
    posterior_loss,
    compute_likelihood_loss,
    compute_prior_loss,
    compute_residuals,
    write_output,
)
from spider.utils.wandb_gates import want_wandb_group as _want_wandb_group
from spider.optim.gauge import project_event_mean_inplace


def _corr_error_nll_sum_from_resid(
    *,
    resid: torch.Tensor,
    sigma: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """
    Sum of per-row negative log-likelihood terms for residuals, matching compute_likelihood_loss.

    NOTE: This is used for corr_error ESS where we precompute base residuals once (no model forward)
    and then evaluate many likelihoods under different nuisance deltas.
    """
    # Clamp sigma to avoid division by zero
    sigma = sigma.clamp_min(1e-12)
    scaled_resid = resid / sigma

    loss_type = str(params.get("likelihood", "huber")).strip().lower()
    if loss_type in {"gaussian", "mse", "l2"}:
        data_loss = 0.5 * (scaled_resid ** 2)
    elif loss_type in {"student_t", "student-t", "studentt"}:
        try:
            nu_f = float(params.get("_student_t_nu", 4.0))
        except Exception:
            nu_f = 4.0
        if not (nu_f > 0.0):
            nu_f = 4.0
        nu = scaled_resid.new_tensor(nu_f)
        pi = scaled_resid.new_tensor(float(np.pi))
        t_const = 0.5 * torch.log(nu * pi) + torch.lgamma(0.5 * nu) - torch.lgamma(0.5 * (nu + 1.0))
        data_loss = 0.5 * (nu + 1.0) * torch.log1p((scaled_resid ** 2) / nu) + t_const
    elif loss_type in {"laplace", "l1", "mae"}:
        data_loss = torch.abs(scaled_resid)
    else:
        # Huber (Smooth L1)
        huber_delta = float(params["model"]["likelihood"].get("huber_delta", 1.0))
        data_loss = torch.nn.functional.huber_loss(
            scaled_resid,
            torch.zeros_like(scaled_resid),
            reduction="none",
            delta=huber_delta,
        )

    total_nll = data_loss + torch.log(sigma)
    return total_nll.sum()


def _corr_error_nll_terms_from_resid(
    *,
    resid: torch.Tensor,
    sigma: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """
    Per-row negative log-likelihood terms for residuals, matching compute_likelihood_loss.

    Returns a tensor of shape (B,) on the same device.
    """
    sigma = sigma.clamp_min(1e-12)
    scaled_resid = resid / sigma

    loss_type = str(params.get("likelihood", "huber")).strip().lower()
    if loss_type in {"gaussian", "mse", "l2"}:
        data_loss = 0.5 * (scaled_resid ** 2)
    elif loss_type in {"student_t", "student-t", "studentt"}:
        try:
            nu_f = float(params.get("_student_t_nu", 4.0))
        except Exception:
            nu_f = 4.0
        if not (nu_f > 0.0):
            nu_f = 4.0
        nu = scaled_resid.new_tensor(nu_f)
        pi = scaled_resid.new_tensor(float(np.pi))
        t_const = 0.5 * torch.log(nu * pi) + torch.lgamma(0.5 * nu) - torch.lgamma(0.5 * (nu + 1.0))
        data_loss = 0.5 * (nu + 1.0) * torch.log1p((scaled_resid ** 2) / nu) + t_const
    elif loss_type in {"laplace", "l1", "mae"}:
        data_loss = torch.abs(scaled_resid)
    else:
        huber_delta = float(params["model"]["likelihood"].get("huber_delta", 1.0))
        data_loss = torch.nn.functional.huber_loss(
            scaled_resid,
            torch.zeros_like(scaled_resid),
            reduction="none",
            delta=huber_delta,
        )
    return data_loss + torch.log(sigma)


@torch.no_grad()
def _student_t_scale_update_full(*, state: LocateState, epoch_index: int) -> dict | None:
    """
    Full-dataset Gibbs update for Student-t scale-mixture per-row precision (lambda).
    """
    try:
        enabled = bool(state.params.get("_student_t_scale_enabled", False))
    except Exception:
        enabled = False
    try:
        update_every = int(state.params.get("_student_t_scale_update_every_epochs", 1))
    except Exception:
        update_every = 1
    lam = getattr(state, "student_t_lambda", None)
    try:
        nu = float(state.params.get("_student_t_scale_nu", 4.0))
    except Exception:
        nu = 4.0
    try:
        bs = int(state.params.get("_student_t_scale_batch_size", 200_000))
    except Exception:
        bs = 200_000
    N = int(state.N)
    if (not enabled) or (update_every <= 0) or ((int(epoch_index) % int(update_every)) != 0) or (not isinstance(lam, torch.Tensor)) or (not (nu > 0.0)) or (N <= 0):
        return None
    bs = max(1024, int(bs))
    lam_min = float(state.params.get("_student_t_scale_min_lambda", 1e-6))
    lam_max = float(state.params.get("_student_t_scale_max_lambda", 1e6))

    σp, σs = _current_noise_scales(state)
    shape = (nu + 1.0) * 0.5
    for i0 in range(0, N, bs):
        i1 = min(i0 + bs, N)
        II_b = state.II[i0:i1]
        YY_b = state.YY[i0:i1]
        resid_b = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().to(torch.float32)
        ph = YY_b[:, 4].detach()
        is_p = (ph < 0.5)
        sigma_b = torch.where(is_p, σp, σs).clamp_min(1e-12)
        z = resid_b / sigma_b
        rate = (nu + (z * z)) * 0.5
        rate = rate.clamp_min(1e-12)
        gamma = torch.distributions.Gamma(concentration=shape, rate=rate)
        lam_b = gamma.sample().to(dtype=lam.dtype, device=lam.device)
        lam_b = lam_b.clamp_min(lam_min).clamp_max(lam_max)
        lam[i0:i1] = lam_b
    state.student_t_lambda = lam
    state.params["_student_t_lambda"] = lam

    try:
        lam_det = lam.detach().float()
        lam_mean = float(lam_det.mean().item())
        # Quantiles: subsample if too large to avoid quantile() size limits.
        n = int(lam_det.numel())
        q_src = lam_det
        if n > 5_000_000:
            try:
                gen = torch.Generator(device=lam_det.device)
                seed0 = int(state.params.get("runtime_seed", 0) or 0)
                gen.manual_seed(int(seed0 + 1000003 * int(epoch_index)))
            except Exception:
                gen = None
            k = 5_000_000
            idx = torch.randperm(n, device=lam_det.device, generator=gen)[:k]
            q_src = lam_det.index_select(0, idx)
        q_cpu = q_src.cpu()
        stats = {
            "student_t_scale/lambda_mean": float(lam_mean),
            "student_t_scale/lambda_p50": float(torch.quantile(q_cpu, 0.50).item()),
            "student_t_scale/lambda_p90": float(torch.quantile(q_cpu, 0.90).item()),
            "student_t_scale/lambda_p99": float(torch.quantile(q_cpu, 0.99).item()),
        }
        return stats
    except Exception:
        return None


@torch.no_grad()
def _maybe_freeze_corr_error_group_for_sampler(state: LocateState, optimizer: torch.optim.Optimizer) -> None:
    """
    If corr_error ESS is enabled, we treat corr_error_b as a blocked latent and do NOT
    update it via the sampler backend (pSGLD/SGHMC). We instead update it via ESS.
    """
    try:
        if not bool(state.params.get("_corr_error_enabled", False)):
            return
        if not bool(state.params.get("_corr_error_ess_enabled", False)):
            return
        if not bool(state.params.get("_corr_error_ess_freeze_sampler_group", True)):
            return
    except Exception:
        return
    try:
        for g in optimizer.param_groups:
            if str(g.get("group_name", "")).strip().lower() != "corr_error":
                continue
            # Preserve original lr so other code can still read it (no compounding).
            if "base_lr" not in g:
                g["base_lr"] = float(g.get("lr", 0.0))
            g["lr"] = 0.0
            # Force noise off for this group.
            g["add_noise"] = False
            g["noise_scale"] = 0.0
    except Exception:
        return

@torch.no_grad()
def _maybe_freeze_dd_graph_re_group_for_sampler(state: LocateState, optimizer: torch.optim.Optimizer) -> None:
    """
    If dd_graph_re ESS is enabled, freeze the dd_graph_re param group in the sampler backend.
    """
    try:
        if not bool(state.params.get("_dd_graph_re_enabled", False)):
            return
        if not bool(state.params.get("_dd_graph_re_ess_enabled", False)):
            return
        if not bool(state.params.get("_dd_graph_re_ess_freeze_sampler_group", True)):
            return
    except Exception:
        return
    try:
        for g in optimizer.param_groups:
            if str(g.get("group_name", "")).strip().lower() != "dd_graph_re":
                continue
            if "base_lr" not in g:
                g["base_lr"] = float(g.get("lr", 0.0))
            g["lr"] = 0.0
            g["add_noise"] = False
            g["noise_scale"] = 0.0
    except Exception:
        return


@torch.no_grad()
def _maybe_freeze_slowness_re_group_for_sampler(state: LocateState, optimizer: torch.optim.Optimizer) -> None:
    """
    If slowness_re explicit ESS is enabled, freeze the slowness_re param group in the sampler backend.
    """
    try:
        if not bool(state.params.get("_slowness_re_explicit_enabled", False)):
            return
        if not bool(state.params.get("_slowness_re_ess_enabled", False)):
            return
        if not bool(state.params.get("_slowness_re_ess_freeze_sampler_group", True)):
            return
    except Exception:
        return
    try:
        for g in optimizer.param_groups:
            if str(g.get("group_name", "")).strip().lower() != "slowness_re":
                continue
            if "base_lr" not in g:
                g["base_lr"] = float(g.get("lr", 0.0))
            g["lr"] = 0.0
            g["add_noise"] = False
            g["noise_scale"] = 0.0
    except Exception:
        return

@torch.no_grad()
def _corr_error_ess_update(
    *,
    state: LocateState,
    epoch_index: int,
) -> Dict[str, float]:
    """
    Exact elliptical slice sampling update for corr_error_b using full-data log-likelihood.

    Currently supported:
      - event_graph.enabled=false (IID prior; Q=I)
      - station_basis.enabled=false (per-station coefficients) with block_by_station=True (recommended)
        OR block_by_station=False (global ESS over b)
      - station_basis.enabled=true is supported only in global mode (can be expensive)

    This update holds ΔX_src fixed and updates corr_error_b conditional on current ΔX_src.
    """
    # Always return a (possibly empty) metrics dict so callers can log to W&B.
    ess_metrics: Dict[str, float] = {}

    # Gate
    try:
        if not bool(state.params.get("_corr_error_enabled", False)):
            return ess_metrics
        if not bool(state.params.get("_corr_error_ess_enabled", False)):
            return ess_metrics
    except Exception:
        return ess_metrics

    ess_metrics["corr_error_ess/enabled"] = float(1.0)
    ess_metrics["corr_error_ess/updated"] = float(0.0)

    # Avoid DDP for now (would require consistent global loglik and synchronized accept/reject).
    ddp_enabled, ddp_rank, ddp_world_size, ddp_is_main = _ddp_info(state.params)

    # Ensure shared_event_re whitening cache exists (static operator).
    try:
        if getattr(state, "shared_event_re_whitening_cache", None) is None:
            state.shared_event_re_whitening_cache = {}
        state.params["_shared_event_re_whitening_cache"] = state.shared_event_re_whitening_cache
    except Exception:
        pass
    if ddp_enabled:
        if ddp_is_main:
            warn("corr_error ESS is not supported under torchrun/DDP yet; skipping.", section="ESS")
        return ess_metrics

    # Cadence
    try:
        every = int(state.params.get("_corr_error_ess_update_every", 1))
        start_after = int(state.params.get("_corr_error_ess_start_after_epochs", 0))
        if every < 1:
            every = 1
        if start_after < 0:
            start_after = 0
    except Exception:
        every = 1
        start_after = 0
    if int(epoch_index) < int(start_after):
        return ess_metrics
    if (int(epoch_index) % int(every)) != 0:
        return ess_metrics

    # Currently only IID prior is implemented (event_graph.disabled => Q=I).
    try:
        if bool(state.params.get("_corr_error_event_graph_enabled", True)):
            warn("corr_error ESS currently requires event_graph.enabled=false (IID prior); skipping.", section="ESS")
            return ess_metrics
    except Exception:
        pass

    b_param = getattr(state, "corr_error_b", None)
    if not isinstance(b_param, torch.nn.Parameter):
        return ess_metrics

    # Precompute base residuals once (no corr_error), and sigma per row.
    N = int(getattr(state, "N", 0) or 0)
    if N <= 0:
        return ess_metrics
    try:
        bs = int(state.params.get("_corr_error_ess_batch_size", 50_000))
        bs = max(1, int(bs))
    except Exception:
        bs = 50_000

    t0 = time.perf_counter()

    σp, σs = _current_noise_scales(state)
    ph = state.YY[:, 4]
    is_p = (ph < 0.5)
    sigma_all = torch.where(is_p, σp, σs).to(device=state.device, dtype=torch.float32).clamp_min(1e-12)

    # Base residuals: dt_obs - dt_pred (no corr_error)
    resid0 = torch.empty((N,), device=state.device, dtype=torch.float32)
    for i0 in range(0, N, bs):
        i1 = min(i0 + bs, N)
        II_b = state.II[i0:i1]
        YY_b = state.YY[i0:i1]
        r = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().to(torch.float32)
        resid0[i0:i1] = r

    # Prior covariance across (P,S) for each latent element.
    try:
        tau_ps = state.params.get("_corr_error_tau_s", [0.0, 0.0])
        tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
        rho = float(state.params.get("_corr_error_rho_ps", 0.0))
    except Exception:
        tau_p, tau_s, rho = 0.0, 0.0, 0.0
    if not (tau_p > 0.0 and tau_s > 0.0):
        warn("corr_error ESS: tau_p/tau_s must be > 0; skipping.", section="ESS")
        return ess_metrics
    rho = max(-0.999, min(0.999, rho))
    cov = torch.tensor(
        [[tau_p * tau_p, rho * tau_p * tau_s], [rho * tau_p * tau_s, tau_s * tau_s]],
        device=state.device,
        dtype=torch.float32,
    )
    # Cholesky with jitter for safety
    try:
        L = torch.linalg.cholesky(cov + 1e-12 * torch.eye(2, device=state.device, dtype=torch.float32))
    except Exception:
        warn("corr_error ESS: covariance not PD; skipping.", section="ESS")
        return ess_metrics

    # ESS settings
    try:
        sweeps = int(state.params.get("_corr_error_ess_sweeps_per_update", 1))
        sweeps = max(1, int(sweeps))
        max_steps = int(state.params.get("_corr_error_ess_max_bracket_steps", 64))
        max_steps = max(8, int(max_steps))
        block_by_station = bool(state.params.get("_corr_error_ess_block_by_station", True))
        top_k = int(state.params.get("_corr_error_ess_top_k_stations", 0))
        top_k = max(0, int(top_k))
        seed = int(state.params.get("_corr_error_ess_seed", 0))
        seed0 = int(state.params.get("runtime_seed", 0))
    except Exception:
        sweeps, max_steps, block_by_station, top_k, seed, seed0 = 1, 64, True, 0, 0, 0

    # If per-station coefficients, we can do exact station-factorized ESS blocks.
    W = getattr(state, "corr_error_station_basis_W", None)
    per_station = (not isinstance(W, torch.Tensor))
    if (not per_station) and bool(block_by_station):
        # Can't factorize when W is present; fallback to global.
        block_by_station = False

    rng = np.random.default_rng(int(seed0 + 10000019 * int(epoch_index) + int(seed)))

    # Stats accumulators
    n_blocks = 0
    n_blocks_accepted = 0
    n_ll_evals = 0
    n_bracket_steps = 0  # number of proposal evaluations (excluding the initial ll0)
    ll_delta_sum = 0.0

    def _ess_update_tensor(
        b0: torch.Tensor,
        loglike_fn,
        *,
        sample_nu_fn,
    ) -> tuple[torch.Tensor, int, int, float]:
        # Draw ellipse direction from prior
        nu = sample_nu_fn()
        ll0 = float(loglike_fn(b0))
        # Slice threshold
        logy = ll0 + float(np.log(max(rng.random(), 1e-30)))
        # Draw initial angle and bracket
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        theta_min = theta - 2.0 * np.pi
        theta_max = theta
        evals = 1
        for _ in range(int(max_steps)):
            ct = math.cos(theta)
            st = math.sin(theta)
            b_prop = (b0 * ct + nu * st).to(dtype=b0.dtype, device=b0.device)
            ll = float(loglike_fn(b_prop))
            evals += 1
            if ll >= logy:
                # accepted
                return b_prop, evals, 1, float(ll - ll0)
            if theta < 0.0:
                theta_min = theta
            else:
                theta_max = theta
            theta = float(rng.uniform(theta_min, theta_max))
        # reject (max bracket steps exhausted)
        return b0, evals, 0, 0.0

    # GLOBAL ESS (works for both W present and per-station mode)
    if not block_by_station:
        sta_idx_all = getattr(state, "row_station_index", None)
        if not isinstance(sta_idx_all, torch.Tensor):
            warn("corr_error ESS: missing row_station_index; skipping.", section="ESS")
            return
        sta_idx_all = sta_idx_all.to(device=state.device, dtype=torch.int64)
        e1_all = state.II[:, 0].to(torch.int64)
        e2_all = state.II[:, 1].to(torch.int64)
        is_s_all = (state.YY[:, 4] >= 0.5)

        def loglike_global(b_all: torch.Tensor) -> torch.Tensor:
            # b_all: (Ne,R,2)
            nll_sum = torch.zeros((), device=state.device, dtype=torch.float32)
            for j0 in range(0, N, bs):
                j1 = min(j0 + bs, N)
                e1 = e1_all[j0:j1]; e2 = e2_all[j0:j1]
                bi = b_all.index_select(0, e1)
                bj = b_all.index_select(0, e2)
                db = (bj - bi).to(torch.float32)  # (B,R,2)
                is_s = is_s_all[j0:j1]
                db_phase = torch.where(is_s.view(-1, 1), db[:, :, 1], db[:, :, 0])  # (B,R)
                if isinstance(W, torch.Tensor):
                    Wr = W.index_select(0, sta_idx_all[j0:j1]).to(torch.float32)  # (B,R)
                    delta = (Wr * db_phase).sum(dim=1)
                else:
                    sidx = sta_idx_all[j0:j1].view(-1, 1)
                    delta = db_phase.gather(1, sidx).squeeze(1)
                resid = resid0[j0:j1] - delta
                nll_sum = nll_sum + _corr_error_nll_sum_from_resid(resid=resid, sigma=sigma_all[j0:j1], params=state.params)
            return -nll_sum

        def sample_nu_global() -> torch.Tensor:
            # IID prior across events and basis dims; only 2x2 coupling across phases.
            z = torch.randn_like(b_param.detach())
            # einsum over last dim (phase)
            return torch.einsum("nrc,cd->nrd", z, L.T)

        b0 = b_param.detach()
        for _ in range(int(sweeps)):
            n_blocks += 1
            b1, evals, acc, dll = _ess_update_tensor(b0, loglike_global, sample_nu_fn=sample_nu_global)
            n_ll_evals += int(evals)
            n_bracket_steps += int(max(0, evals - 1))
            n_blocks_accepted += int(acc)
            ll_delta_sum += float(dll)
            b0 = b1
        b_param.copy_(b0)
        dt_ms = 1000.0 * float(time.perf_counter() - t0)
        ess_metrics["corr_error_ess/updated"] = float(1.0)
        ess_metrics["corr_error_ess/mode"] = float(0.0)  # 0=global, 1=per_station
        ess_metrics["corr_error_ess/sweeps"] = float(int(sweeps))
        ess_metrics["corr_error_ess/blocks"] = float(int(n_blocks))
        ess_metrics["corr_error_ess/blocks_accepted"] = float(int(n_blocks_accepted))
        ess_metrics["corr_error_ess/ll_evals"] = float(int(n_ll_evals))
        ess_metrics["corr_error_ess/mean_bracket_steps"] = float(n_bracket_steps / max(1, n_blocks))
        ess_metrics["corr_error_ess/mean_ll_delta"] = float(ll_delta_sum / max(1, n_blocks))
        ess_metrics["corr_error_ess/time_ms"] = float(dt_ms)
        info(
            f"corr_error ESS(global): sweeps={sweeps} blocks={n_blocks} "
            f"mean_steps={ess_metrics['corr_error_ess/mean_bracket_steps']:.2f} "
            f"time_ms={dt_ms:.1f}",
            section="ESS",
        )
        return ess_metrics

    # PER-STATION blocked ESS (station_basis.enabled=false) — optimized parallel implementation.
    #
    # Key optimization vs the naive per-station loop:
    # - We evaluate log-likelihoods for *all stations at once* using a single pass over rows:
    #     ll_s(b) = -sum_{rows with sta=s} nll(resid0 - delta_corr(b))
    #   and we use index_add_ to reduce per-row nll terms into per-station totals.
    # - This avoids thousands of tiny GPU kernels + .item() synchronizations that look like CPU-only time.
    n_stations = int(getattr(state, "n_stations", 0) or 0)
    if n_stations <= 0:
        return ess_metrics
    sta_idx_all = getattr(state, "row_station_index", None)
    if not isinstance(sta_idx_all, torch.Tensor):
        warn("corr_error ESS: missing row_station_index; skipping.", section="ESS")
        return ess_metrics
    sta_idx_all = sta_idx_all.to(device=state.device, dtype=torch.int64)
    S = int(n_stations)
    # Sanity: b must be per-station coefficients (R == n_stations) in this mode.
    if not (b_param.ndim == 3 and int(b_param.shape[1]) == int(S) and int(b_param.shape[2]) == 2):
        warn("corr_error ESS: expected b shape (n_events,n_stations,2) for per-station ESS; skipping.", section="ESS")
        return ess_metrics

    # Select stations to update (top-K by row count), if requested.
    # We compute counts cheaply on GPU once per update.
    try:
        counts = torch.bincount(sta_idx_all.clamp_min(0), minlength=S).to(device="cpu")
    except Exception:
        counts = None
    if isinstance(counts, torch.Tensor) and int(counts.numel()) == int(S):
        if top_k > 0 and top_k < S:
            # choose top-k stations by count
            _, sel = torch.topk(counts.to(torch.int64), k=int(top_k), largest=True, sorted=False)
            sta_sel = sel.to(device=state.device, dtype=torch.int64)
        else:
            sta_sel = torch.arange(S, device=state.device, dtype=torch.int64)
    else:
        sta_sel = torch.arange(S, device=state.device, dtype=torch.int64)
    K = int(sta_sel.numel())
    if K <= 0:
        return ess_metrics

    # Precompute e1/e2 and phase mask once (global)
    e1_all = state.II[:, 0].to(torch.int64)
    e2_all = state.II[:, 1].to(torch.int64)
    is_s_all = (state.YY[:, 4] >= 0.5)

    # Helper: compute per-station log-likelihoods for a candidate b (Ne,S,2).
    def _loglike_per_station(b_all: torch.Tensor) -> torch.Tensor:
        # Returns ll_by_station: (S,) float32 on device
        ll = torch.zeros((S,), device=state.device, dtype=torch.float32)
        bP_flat = b_all[:, :, 0].reshape(-1).to(torch.float32)
        bS_flat = b_all[:, :, 1].reshape(-1).to(torch.float32)
        for j0 in range(0, N, bs):
            j1 = min(j0 + bs, N)
            sta_b = sta_idx_all[j0:j1]
            # linear indices into flattened (event,station) grid
            idx1 = (e1_all[j0:j1] * S + sta_b).to(torch.int64)
            idx2 = (e2_all[j0:j1] * S + sta_b).to(torch.int64)
            dp = bP_flat.index_select(0, idx2) - bP_flat.index_select(0, idx1)
            ds = bS_flat.index_select(0, idx2) - bS_flat.index_select(0, idx1)
            delta = torch.where(is_s_all[j0:j1], ds, dp).to(torch.float32)
            resid = resid0[j0:j1] - delta
            nll_terms = _corr_error_nll_terms_from_resid(resid=resid, sigma=sigma_all[j0:j1], params=state.params)
            # Accumulate -NLL into station loglike
            ll.index_add_(0, sta_b, -nll_terms.to(torch.float32))
        return ll

    # Initial state (we update only selected stations but evaluate ll for all stations)
    b_base = b_param.detach().to(torch.float32)
    ll0_all = _loglike_per_station(b_base)  # (S,)
    ll0 = ll0_all.index_select(0, sta_sel)  # (K,)

    # Slice thresholds per station
    u = torch.rand((K,), device=state.device, dtype=torch.float32).clamp_min(1e-30)
    logy = ll0 + torch.log(u)

    # Draw ellipse directions nu for selected stations
    z = torch.randn((int(b_base.shape[0]), K, 2), device=state.device, dtype=torch.float32)
    nu = torch.matmul(z, L.T)  # (Ne,K,2)

    # Bracket init
    theta = (2.0 * float(np.pi)) * torch.rand((K,), device=state.device, dtype=torch.float32)
    theta_min = theta - 2.0 * float(np.pi)
    theta_max = theta.clone()
    active = torch.ones((K,), device=state.device, dtype=torch.bool)

    # Current b for selected stations
    b_cur_sel = b_base.index_select(1, sta_sel).contiguous()  # (Ne,K,2)

    # ESS loop (vectorized across stations; one sync per bracket step to check completion)
    for step in range(int(max_steps)):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        # Keep accepted stations fixed
        ct = torch.where(active, ct, torch.ones_like(ct))
        st = torch.where(active, st, torch.zeros_like(st))
        b_prop_sel = b_cur_sel * ct.view(1, K, 1) + nu * st.view(1, K, 1)
        # Assemble proposed full b (only selected columns changed)
        b_prop = b_base.clone()
        b_prop.index_copy_(1, sta_sel, b_prop_sel)
        ll_prop_all = _loglike_per_station(b_prop)
        ll_prop = ll_prop_all.index_select(0, sta_sel)

        n_blocks += int(active.sum().item())
        n_ll_evals += 1  # one batched ll evaluation

        accept = active & (ll_prop >= logy)
        if accept.any():
            n_blocks_accepted += int(accept.sum().item())
            ll_delta_sum += float((ll_prop[accept] - ll0[accept]).sum().item())
            # Update accepted stations
            b_cur_sel = torch.where(accept.view(1, K, 1), b_prop_sel, b_cur_sel)
            ll0 = torch.where(accept, ll_prop, ll0)
            active = active & (~accept)

        if (not active.any()):
            n_bracket_steps += int(step + 1)
            break

        # Shrink brackets and resample theta for active stations
        neg = (theta < 0.0) & active
        pos = (theta >= 0.0) & active
        theta_min = torch.where(neg, theta, theta_min)
        theta_max = torch.where(pos, theta, theta_max)
        r = torch.rand((K,), device=state.device, dtype=torch.float32)
        theta = torch.where(active, theta_min + r * (theta_max - theta_min), theta)
        if step == int(max_steps) - 1:
            n_bracket_steps += int(max_steps)

    # Write back updated b for selected stations
    b_out = b_base.clone()
    b_out.index_copy_(1, sta_sel, b_cur_sel)
    b_param.copy_(b_out.to(dtype=b_param.dtype))

    dt_ms = 1000.0 * float(time.perf_counter() - t0)
    ess_metrics["corr_error_ess/updated"] = float(1.0)
    ess_metrics["corr_error_ess/mode"] = float(1.0)  # 0=global, 1=per_station
    ess_metrics["corr_error_ess/sweeps"] = float(int(sweeps))
    ess_metrics["corr_error_ess/stations_updated"] = float(int(K))
    ess_metrics["corr_error_ess/stations_target"] = float(int(K))
    ess_metrics["corr_error_ess/blocks"] = float(int(n_blocks))
    ess_metrics["corr_error_ess/blocks_accepted"] = float(int(n_blocks_accepted))
    ess_metrics["corr_error_ess/ll_evals"] = float(int(n_ll_evals))
    ess_metrics["corr_error_ess/mean_bracket_steps"] = float(n_bracket_steps / max(1, n_ll_evals))
    ess_metrics["corr_error_ess/mean_ll_delta"] = float(ll_delta_sum / max(1, max(n_blocks_accepted, 1)))
    ess_metrics["corr_error_ess/time_ms"] = float(dt_ms)
    info(
        f"corr_error ESS(per-station, batched): stations={K}/{K} "
        f"ll_evals={n_ll_evals} mean_steps={ess_metrics['corr_error_ess/mean_bracket_steps']:.2f} "
        f"time_ms={dt_ms:.1f}",
        section="ESS",
    )
    return ess_metrics


@torch.no_grad()
def _slowness_re_ess_update(
    *,
    state: LocateState,
    epoch_index: int,
) -> Dict[str, float]:
    """
    Exact ESS update for explicit slowness_re latents (component + station).
    """
    ess_metrics: Dict[str, float] = {}
    try:
        if not bool(state.params.get("_slowness_re_explicit_enabled", False)):
            return ess_metrics
        if not bool(state.params.get("_slowness_re_ess_enabled", False)):
            return ess_metrics
    except Exception:
        return ess_metrics

    ess_metrics["slowness_re_ess/enabled"] = float(1.0)
    ess_metrics["slowness_re_ess/updated"] = float(0.0)

    ddp_enabled, _, _, ddp_is_main = _ddp_info(state.params)
    if ddp_enabled:
        if ddp_is_main:
            warn("slowness_re ESS is not supported under torchrun/DDP yet; skipping.", section="ESS")
        return ess_metrics

    try:
        every = int(state.params.get("_slowness_re_ess_update_every", 1))
        start_after = int(state.params.get("_slowness_re_ess_start_after", 0))
        if every < 1:
            every = 1
        if start_after < 0:
            start_after = 0
    except Exception:
        every, start_after = 1, 0
    if int(epoch_index) < int(start_after):
        return ess_metrics
    if (int(epoch_index) % int(every)) != 0:
        return ess_metrics

    # Latents
    s_comp_p = getattr(state, "slowness_re_comp_p", None)
    s_comp_s = getattr(state, "slowness_re_comp_s", None)
    a_sta_p = getattr(state, "slowness_re_station_p", None)
    a_sta_s = getattr(state, "slowness_re_station_s", None)
    if not all(isinstance(x, torch.nn.Parameter) for x in (s_comp_p, s_comp_s, a_sta_p, a_sta_s)):
        return ess_metrics

    n_comp = int(s_comp_p.numel())
    n_sta = int(a_sta_p.numel())
    if n_comp <= 0 or n_sta <= 0:
        return ess_metrics

    N = int(getattr(state, "N", 0) or 0)
    if N <= 0:
        return ess_metrics

    sta_idx_all = getattr(state, "row_station_index", None)
    if not isinstance(sta_idx_all, torch.Tensor) or int(sta_idx_all.numel()) != int(N):
        warn("slowness_re ESS requires row_station_index; skipping.", section="ESS")
        return ess_metrics

    cid_ev = getattr(state, "cluster_ids", None)
    if not isinstance(cid_ev, torch.Tensor):
        warn("slowness_re ESS requires event cluster_ids; skipping.", section="ESS")
        return ess_metrics

    try:
        bs = int(state.params.get("_slowness_re_ess_batch_size", 50_000))
        bs = max(1, int(bs))
    except Exception:
        bs = 50_000

    t0 = time.perf_counter()

    σp, σs = _current_noise_scales(state)
    ph = state.YY[:, 4]
    is_p = (ph < 0.5)
    sigma_all = torch.where(is_p, σp, σs).to(device=state.device, dtype=torch.float32).clamp_min(1e-12)

    # Base residuals: dt_obs - dt_pred (no slowness correction)
    resid0 = torch.empty((N,), device=state.device, dtype=torch.float32)
    for i0 in range(0, N, bs):
        i1 = min(i0 + bs, N)
        II_b = state.II[i0:i1]
        YY_b = state.YY[i0:i1]
        r = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().to(torch.float32)
        resid0[i0:i1] = r

    # Separation magnitude g_ij (same as component_station).
    g_all = torch.empty((N,), device=state.device, dtype=torch.float32)
    sep_cap_km = float(state.params.get("_slowness_re_sep_cap_km", 0.0))
    freeze_g = bool(state.params.get("_slowness_re_freeze_g_at_map", False))
    if freeze_g:
        X_map = state.params.get("_slowness_re_g_x_map", None)
        if not isinstance(X_map, torch.Tensor) or int(X_map.shape[0]) != int(state.X_src.shape[0]):
            X_map = (state.X_src + state.dX_src)[:, :3].detach().to(device=state.device, dtype=torch.float32)
            state.params["_slowness_re_g_x_map"] = X_map
    for i0 in range(0, N, bs):
        i1 = min(i0 + bs, N)
        idx_b = state.II[i0:i1].to(torch.int64)
        if freeze_g:
            x1 = X_map.index_select(0, idx_b[:, 0])
            x2 = X_map.index_select(0, idx_b[:, 1])
        else:
            x1 = state.X_src.index_select(0, idx_b[:, 0])[:, :3] + state.dX_src.index_select(0, idx_b[:, 0])[:, :3]
            x2 = state.X_src.index_select(0, idx_b[:, 1])[:, :3] + state.dX_src.index_select(0, idx_b[:, 1])[:, :3]
        g = torch.linalg.norm(x1 - x2, dim=1).clamp_min(1e-6)
        if sep_cap_km > 0.0 and math.isfinite(sep_cap_km):
            g = g.clamp_max(float(sep_cap_km))
        g_all[i0:i1] = g

    comp_row_all = cid_ev.index_select(0, state.II[:, 0].to(torch.int64))

    tau_ps = state.params.get("_slowness_re_tau_s", [0.0, 0.0])
    tau_sta_ps = state.params.get("_slowness_re_tau_station_s", [0.0, 0.0])
    tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
    tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
    tau_sta_p = float(tau_sta_ps[0]) if isinstance(tau_sta_ps, (list, tuple)) and len(tau_sta_ps) >= 2 else float(tau_sta_ps)
    tau_sta_s = float(tau_sta_ps[1]) if isinstance(tau_sta_ps, (list, tuple)) and len(tau_sta_ps) >= 2 else float(tau_sta_ps)
    tau_units = str(state.params.get("_slowness_re_tau_units", "abs")).strip().lower()
    if tau_units == "vel_frac":
        vp = float(state.params.get("_slowness_re_vp_km_s", 6.0))
        vs = float(state.params.get("_slowness_re_vs_km_s", 3.5))
        tau_p = tau_p / max(vp, 1e-6)
        tau_s = tau_s / max(vs, 1e-6)
        tau_sta_p = tau_sta_p / max(vp, 1e-6)
        tau_sta_s = tau_sta_s / max(vs, 1e-6)

    sweeps = int(state.params.get("_slowness_re_ess_sweeps_per_update", 1))
    sweeps = max(1, int(sweeps))
    max_steps = int(state.params.get("_slowness_re_ess_max_bracket_steps", 64))
    max_steps = max(8, int(max_steps))
    seed = int(state.params.get("_slowness_re_ess_seed", 0))
    seed0 = int(state.params.get("runtime_seed", 0))
    rng = np.random.default_rng(int(seed0 + 10000079 * int(epoch_index) + int(seed)))

    n_ll_evals = 0
    n_bracket_steps = 0
    n_blocks = 0
    n_blocks_accepted = 0
    ll_delta_sum = 0.0

    def _ess_update_vector(x0: torch.Tensor, loglike_fn, *, sample_nu_fn) -> tuple[torch.Tensor, int, int, float]:
        nu = sample_nu_fn()
        ll0 = float(loglike_fn(x0))
        logy = ll0 + float(np.log(max(rng.random(), 1e-30)))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        theta_min = theta - 2.0 * np.pi
        theta_max = theta
        evals = 1
        for _ in range(int(max_steps)):
            ct = math.cos(theta)
            st = math.sin(theta)
            x_prop = (x0 * ct + nu * st).to(dtype=x0.dtype, device=x0.device)
            ll = float(loglike_fn(x_prop))
            evals += 1
            if ll >= logy:
                return x_prop, evals, 1, float(ll - ll0)
            if theta < 0.0:
                theta_min = theta
            else:
                theta_max = theta
            theta = float(rng.uniform(theta_min, theta_max))
        return x0, evals, 0, 0.0

    def _update_phase(
        *,
        phase_mask: torch.Tensor,
        s_comp: torch.nn.Parameter,
        a_sta: torch.nn.Parameter,
        tau_comp: float,
        tau_sta: float,
        label: str,
    ) -> None:
        nonlocal n_ll_evals, n_bracket_steps, n_blocks, n_blocks_accepted, ll_delta_sum
        if int(phase_mask.numel()) <= 0:
            return
        if not (tau_comp > 0.0 and tau_sta > 0.0):
            return
        resid_p = resid0.index_select(0, phase_mask)
        g_p = g_all.index_select(0, phase_mask)
        comp_p = comp_row_all.index_select(0, phase_mask)
        sta_p = sta_idx_all.index_select(0, phase_mask)
        sigma_p = sigma_all.index_select(0, phase_mask)

        std = torch.cat(
            [
                torch.full((n_comp,), float(tau_comp), device=state.device, dtype=torch.float32),
                torch.full((n_sta,), float(tau_sta), device=state.device, dtype=torch.float32),
            ],
            dim=0,
        )

        def loglike_fn(x: torch.Tensor) -> torch.Tensor:
            s_c = x[:n_comp]
            a_k = x[n_comp:]
            pred = g_p * (s_c.index_select(0, comp_p) + a_k.index_select(0, sta_p))
            r = resid_p - pred
            return -0.5 * ((r / sigma_p).square().sum())

        def sample_nu_fn() -> torch.Tensor:
            return torch.randn_like(std) * std

        x0 = torch.cat([s_comp.detach(), a_sta.detach()], dim=0)
        for _ in range(int(sweeps)):
            x1, evals, acc, dll = _ess_update_vector(x0, loglike_fn, sample_nu_fn=sample_nu_fn)
            n_blocks += 1
            n_ll_evals += int(evals)
            n_bracket_steps += int(max(0, evals - 1))
            n_blocks_accepted += int(acc)
            ll_delta_sum += float(dll)
            x0 = x1
        s_comp.data.copy_(x0[:n_comp])
        a_sta.data.copy_(x0[n_comp:])
        ess_metrics[f"slowness_re_ess/{label}_updated"] = float(1.0)

    # Precompute indices for P/S masks.
    idx_p = torch.nonzero(is_p, as_tuple=False).flatten()
    idx_s = torch.nonzero(~is_p, as_tuple=False).flatten()
    _update_phase(phase_mask=idx_p, s_comp=s_comp_p, a_sta=a_sta_p, tau_comp=tau_p, tau_sta=tau_sta_p, label="p")
    _update_phase(phase_mask=idx_s, s_comp=s_comp_s, a_sta=a_sta_s, tau_comp=tau_s, tau_sta=tau_sta_s, label="s")

    dt_ms = (time.perf_counter() - t0) * 1000.0
    ess_metrics["slowness_re_ess/updated"] = float(1.0 if (n_blocks > 0) else 0.0)
    ess_metrics["slowness_re_ess/sweeps"] = float(int(sweeps))
    ess_metrics["slowness_re_ess/blocks"] = float(int(n_blocks))
    ess_metrics["slowness_re_ess/blocks_accepted"] = float(int(n_blocks_accepted))
    ess_metrics["slowness_re_ess/ll_evals"] = float(int(n_ll_evals))
    ess_metrics["slowness_re_ess/mean_bracket_steps"] = float(n_bracket_steps / max(1, n_blocks))
    ess_metrics["slowness_re_ess/mean_ll_delta"] = float(ll_delta_sum / max(1, n_blocks_accepted))
    ess_metrics["slowness_re_ess/time_ms"] = float(dt_ms)

    return ess_metrics


@torch.no_grad()
def _dd_graph_re_ess_update(
    *,
    state: LocateState,
    epoch_index: int,
) -> Dict[str, float]:
    """
    Exact ESS update for dd_graph_re event latents (per phase), blocked by connected components.
    """
    ess_metrics: Dict[str, float] = {}
    try:
        if not bool(state.params.get("_dd_graph_re_enabled", False)):
            return ess_metrics
        if not bool(state.params.get("_dd_graph_re_ess_enabled", False)):
            return ess_metrics
    except Exception:
        return ess_metrics

    ess_metrics["dd_graph_re_ess/enabled"] = float(1.0)
    ess_metrics["dd_graph_re_ess/updated"] = float(0.0)

    ddp_enabled, _, _, ddp_is_main = _ddp_info(state.params)
    if ddp_enabled:
        if ddp_is_main:
            warn("dd_graph_re ESS is not supported under torchrun/DDP yet; skipping.", section="ESS")
        return ess_metrics

    try:
        every = int(state.params.get("_dd_graph_re_ess_update_every", 1))
        start_after = int(state.params.get("_dd_graph_re_ess_start_after", 0))
        if every < 1:
            every = 1
        if start_after < 0:
            start_after = 0
    except Exception:
        every, start_after = 1, 0
    if int(epoch_index) < int(start_after):
        return ess_metrics
    if (int(epoch_index) % int(every)) != 0:
        return ess_metrics

    b_p = getattr(state, "dd_graph_re_b_p", None)
    b_s = getattr(state, "dd_graph_re_b_s", None)
    if not (isinstance(b_p, torch.nn.Parameter) and isinstance(b_s, torch.nn.Parameter)):
        return ess_metrics

    comp_id = getattr(state, "cluster_ids", None)
    if not isinstance(comp_id, torch.Tensor):
        return ess_metrics

    u = state.params.get("_dd_graph_re_u", None)
    v = state.params.get("_dd_graph_re_v", None)
    w = state.params.get("_dd_graph_re_w", None)
    if not (isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor)):
        return ess_metrics

    tau_ps = state.params.get("_dd_graph_re_tau_s", [0.0, 0.0])
    tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
    tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
    q_diag = float(state.params.get("_dd_graph_re_q_diag", 0.0))
    max_steps = int(state.params.get("_dd_graph_re_ess_max_bracket_steps", 64))
    max_steps = max(8, int(max_steps))
    sweeps = int(state.params.get("_dd_graph_re_ess_sweeps_per_update", 1))
    sweeps = max(1, int(sweeps))
    seed = int(state.params.get("_dd_graph_re_ess_seed", 0))
    seed0 = int(state.params.get("runtime_seed", 0))

    # Residuals and sigma (full data)
    resid0 = compute_residuals(state.II, state.YY, state.X_src, state.dX_src, state.model).detach().to(torch.float32)
    σp, σs = _current_noise_scales(state)
    ph = state.YY[:, 4]
    is_p = (ph < 0.5)
    sigma_all = torch.where(is_p, σp, σs).to(device=state.device, dtype=torch.float32).clamp_min(1e-12)

    II = state.II.to(torch.int64)
    ev1 = II[:, 0]
    ev2 = II[:, 1]
    comp_id_cpu = comp_id.detach().cpu().numpy()
    ev1_cpu = ev1.detach().cpu().numpy()
    ev2_cpu = ev2.detach().cpu().numpy()
    is_p_cpu = is_p.detach().cpu().numpy()

    u_cpu = u.detach().cpu().numpy()
    v_cpu = v.detach().cpu().numpy()
    w_cpu = w.detach().cpu().numpy()

    rng = np.random.default_rng(int(seed0 + 10000091 * int(epoch_index) + int(seed)))

    n_blocks = 0
    n_blocks_accepted = 0
    n_ll_evals = 0
    n_bracket_steps = 0
    ll_delta_sum = 0.0
    t0 = time.perf_counter()

    def _ess_update_vector(x0: torch.Tensor, loglike_fn, *, sample_nu_fn) -> tuple[torch.Tensor, int, int, float]:
        nu = sample_nu_fn()
        ll0 = float(loglike_fn(x0))
        logy = ll0 + float(np.log(max(rng.random(), 1e-30)))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        theta_min = theta - 2.0 * np.pi
        theta_max = theta
        evals = 1
        for _ in range(int(max_steps)):
            ct = math.cos(theta)
            st = math.sin(theta)
            x_prop = (x0 * ct + nu * st).to(dtype=x0.dtype, device=x0.device)
            ll = float(loglike_fn(x_prop))
            evals += 1
            if ll >= logy:
                return x_prop, evals, 1, float(ll - ll0)
            if theta < 0.0:
                theta_min = theta
            else:
                theta_max = theta
            theta = float(rng.uniform(theta_min, theta_max))
        return x0, evals, 0, 0.0

    def _sample_nu(chol: torch.Tensor, n: int, tau: float) -> torch.Tensor:
        if n <= 0 or not (tau > 0.0):
            return torch.zeros((n,), device=state.device, dtype=torch.float32)
        z = torch.randn((n, 1), device=chol.device, dtype=chol.dtype)
        # Solve chol @ y = z, then chol.T @ x = y  -> x ~ N(0, Q^{-1})
        y = torch.linalg.solve_triangular(chol, z, upper=False)
        x = torch.linalg.solve_triangular(chol.T, y, upper=True)
        return x.squeeze(1).to(device=state.device, dtype=torch.float32) * float(tau)

    # Build component blocks
    n_comp = int(comp_id.max().item()) + 1 if comp_id.numel() > 0 else 0
    for c in range(int(n_comp)):
        ev_mask = (comp_id_cpu == c)
        ev_ids = np.nonzero(ev_mask)[0]
        if ev_ids.size <= 1:
            continue
        # Map global event -> local index
        map_local = -np.ones((int(comp_id_cpu.size),), dtype=np.int64)
        map_local[ev_ids] = np.arange(ev_ids.size, dtype=np.int64)

        row_mask = ev_mask[ev1_cpu] & ev_mask[ev2_cpu]
        if not np.any(row_mask):
            continue
        row_idx = np.nonzero(row_mask)[0]
        row_idx_p = row_idx[is_p_cpu[row_idx]]
        row_idx_s = row_idx[~is_p_cpu[row_idx]]

        # Edges for prior
        edge_mask = ev_mask[u_cpu] & ev_mask[v_cpu]
        u_loc_np = map_local[u_cpu[edge_mask]]
        v_loc_np = map_local[v_cpu[edge_mask]]
        w_loc_np = w_cpu[edge_mask].astype(np.float32, copy=False)
        u_loc = torch.tensor(u_loc_np, device="cpu", dtype=torch.int64)
        v_loc = torch.tensor(v_loc_np, device="cpu", dtype=torch.int64)
        w_loc = torch.tensor(w_loc_np, device="cpu", dtype=torch.float64)
        # Build dense Laplacian for this component (CPU) and factor once.
        n_loc = int(ev_ids.size)
        if n_loc <= 1:
            continue
        L = torch.zeros((n_loc, n_loc), device="cpu", dtype=torch.float64)
        if u_loc.numel() > 0:
            L[u_loc, u_loc] += w_loc
            L[v_loc, v_loc] += w_loc
            L[u_loc, v_loc] -= w_loc
            L[v_loc, u_loc] -= w_loc
        if float(q_diag) > 0.0:
            L = L + float(q_diag) * torch.eye(n_loc, device="cpu", dtype=torch.float64)
        # Enforce symmetry and add a data-driven diagonal shift to ensure PD.
        L = 0.5 * (L + L.T)
        try:
            eig_min = float(torch.linalg.eigvalsh(L).min().item()) if n_loc > 0 else 0.0
        except Exception:
            eig_min = float("nan")
        shift = 0.0
        if not math.isfinite(eig_min):
            shift = 1e-3
        elif eig_min < 1e-8:
            shift = float(-eig_min + 1e-6)
        try:
            chol = torch.linalg.cholesky(L + float(shift) * torch.eye(n_loc, device="cpu", dtype=torch.float64))
        except Exception:
            warn(f"dd_graph_re ESS: chol failed for component size={n_loc}; skipping.", section="ESS")
            continue

        def _update_phase(*, row_idx_phase: np.ndarray, b_param: torch.nn.Parameter, tau: float, label: str) -> None:
            nonlocal n_blocks, n_blocks_accepted, n_ll_evals, n_bracket_steps, ll_delta_sum
            if row_idx_phase.size == 0 or not (tau > 0.0):
                return
            ev1_loc = torch.tensor(map_local[ev1_cpu[row_idx_phase]], device=state.device, dtype=torch.int64)
            ev2_loc = torch.tensor(map_local[ev2_cpu[row_idx_phase]], device=state.device, dtype=torch.int64)
            r0 = resid0.index_select(0, torch.tensor(row_idx_phase, device=state.device, dtype=torch.int64))
            sig = sigma_all.index_select(0, torch.tensor(row_idx_phase, device=state.device, dtype=torch.int64))
            # current block
            b0 = b_param.index_select(0, torch.tensor(ev_ids, device=state.device, dtype=torch.int64))

            def loglike_fn(x: torch.Tensor) -> torch.Tensor:
                pred = x.index_select(0, ev1_loc) - x.index_select(0, ev2_loc)
                rr = r0 - pred
                return -0.5 * ((rr / sig).square().sum())

            def sample_nu_fn() -> torch.Tensor:
                return _sample_nu(chol, int(ev_ids.size), tau)

            x0 = b0
            for _ in range(int(sweeps)):
                x1, evals, acc, dll = _ess_update_vector(x0, loglike_fn, sample_nu_fn=sample_nu_fn)
                n_blocks += 1
                n_ll_evals += int(evals)
                n_bracket_steps += int(max(0, evals - 1))
                n_blocks_accepted += int(acc)
                ll_delta_sum += float(dll)
                x0 = x1
            b_param.data.index_copy_(0, torch.tensor(ev_ids, device=state.device, dtype=torch.int64), x0)
            ess_metrics[f"dd_graph_re_ess/{label}_updated"] = float(1.0)

        _update_phase(row_idx_phase=row_idx_p, b_param=b_p, tau=tau_p, label="p")
        _update_phase(row_idx_phase=row_idx_s, b_param=b_s, tau=tau_s, label="s")

    dt_ms = (time.perf_counter() - t0) * 1000.0
    ess_metrics["dd_graph_re_ess/updated"] = float(1.0 if (n_blocks > 0) else 0.0)
    ess_metrics["dd_graph_re_ess/sweeps"] = float(int(sweeps))
    ess_metrics["dd_graph_re_ess/blocks"] = float(int(n_blocks))
    ess_metrics["dd_graph_re_ess/blocks_accepted"] = float(int(n_blocks_accepted))
    ess_metrics["dd_graph_re_ess/ll_evals"] = float(int(n_ll_evals))
    ess_metrics["dd_graph_re_ess/mean_bracket_steps"] = float(n_bracket_steps / max(1, n_blocks))
    ess_metrics["dd_graph_re_ess/mean_ll_delta"] = float(ll_delta_sum / max(1, n_blocks_accepted))
    ess_metrics["dd_graph_re_ess/time_ms"] = float(dt_ms)

    return ess_metrics

def _maybe_apply_gauge_projection_for_optimizer(state: LocateState, optimizer: torch.optim.Optimizer) -> None:
    """
    Apply gauge projection (remove translation mode) for optimizers that do NOT implement it internally.

    Sampler backends in `spider.optim.*` already apply gauge projection inside their `step()` method
    (to gradients and optionally to injected noise/momentum). Phase-1 MAP uses torch.optim.Adam,
    which does not. This hook makes `runtime.gauge_projection` apply to locate-map as well.
    """
    try:
        if not bool(state.params.get("gauge_project_enable", False)):
            return
    except Exception:
        return

    # Avoid double-projecting for SPIDER samplers which already do this in optimizer.step().
    try:
        mod = str(getattr(optimizer.__class__, "__module__", "") or "")
        if mod.startswith("spider.optim"):
            return
    except Exception:
        pass

    try:
        dims_v = state.params.get("gauge_project_dims", [0, 1, 2])
        if not isinstance(dims_v, list) or len(dims_v) == 0:
            dims_v = [0, 1, 2]
        dims = tuple(int(d) for d in dims_v)
    except Exception:
        dims = (0, 1, 2)
    try:
        mode = str(state.params.get("gauge_project_mode", "global")).strip().lower()
    except Exception:
        mode = "global"
    if mode not in {"global", "cluster"}:
        mode = "global"

    cid = state.cluster_ids if mode == "cluster" else None
    cc = state.cluster_counts if mode == "cluster" else None

    # Project the gradient of dX_src in-place.
    try:
        g = getattr(state.dX_src, "grad", None)
        if isinstance(g, torch.Tensor) and g.ndim == 2 and int(g.shape[0]) == int(state.dX_src.shape[0]):
            project_event_mean_inplace(g, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
    except Exception:
        pass


def _maybe_apply_gauge_projection_to_momentum_buffers(state: LocateState, optimizer: torch.optim.Optimizer) -> None:
    """
    Optionally project optimizer momentum buffers for non-sampler optimizers (e.g., Adam).

    This prevents the gauge (translation) mode from accumulating in exp_avg / momentum buffers.
    """
    try:
        if not bool(state.params.get("gauge_project_enable", False)):
            return
        if not bool(state.params.get("gauge_project_apply_momentum", True)):
            return
    except Exception:
        return
    # SPIDER samplers handle momentum/noise internally.
    try:
        mod = str(getattr(optimizer.__class__, "__module__", "") or "")
        if mod.startswith("spider.optim"):
            return
    except Exception:
        pass
    try:
        dims_v = state.params.get("gauge_project_dims", [0, 1, 2])
        if not isinstance(dims_v, list) or len(dims_v) == 0:
            dims_v = [0, 1, 2]
        dims = tuple(int(d) for d in dims_v)
    except Exception:
        dims = (0, 1, 2)
    try:
        mode = str(state.params.get("gauge_project_mode", "global")).strip().lower()
    except Exception:
        mode = "global"
    if mode not in {"global", "cluster"}:
        mode = "global"
    cid = state.cluster_ids if mode == "cluster" else None
    cc = state.cluster_counts if mode == "cluster" else None

    p = getattr(state, "dX_src", None)
    if p is None:
        return
    st = getattr(optimizer, "state", {}).get(p, None)  # type: ignore[call-arg]
    if not isinstance(st, dict):
        return
    # Common momentum buffer keys:
    # - Adam/AdamW: 'exp_avg' (first moment)
    # - SGD: 'momentum_buffer'
    for k in ("exp_avg", "momentum_buffer"):
        try:
            buf = st.get(k, None)
            if isinstance(buf, torch.Tensor) and buf.ndim == 2 and int(buf.shape[0]) == int(p.shape[0]):
                project_event_mean_inplace(buf, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
        except Exception:
            pass


def _ddp_info(params: dict) -> tuple[bool, int, int, bool]:
    """
    Returns (enabled, rank, world_size, is_main) for torchrun/DDP mode.
    We treat DDP as enabled only when torch.distributed is initialized and world_size>1.
    """
    try:
        ws = int(params.get("_ddp_world_size", 1) or 1)
        rk = int(params.get("_ddp_rank", 0) or 0)
    except Exception:
        ws, rk = 1, 0
    enabled = bool(ws > 1) and bool(dist.is_available()) and bool(dist.is_initialized())
    if enabled:
        try:
            # Trust the runtime communicator if available.
            ws = int(dist.get_world_size())
            rk = int(dist.get_rank())
        except Exception:
            pass
    is_main = (int(rk) == 0)
    return enabled, int(rk), int(ws), bool(is_main)


def _ddp_allreduce_grads(optimizer: torch.optim.Optimizer) -> None:
    """All-reduce gradients (SUM) across ranks. Assumes dist is initialized."""
    grads = []
    # Control tensor device for tiny collectives (e.g., all_gather on totals).
    # IMPORTANT: even if a rank has no grads this step, it must still participate
    # in the same collectives as other ranks, otherwise NCCL will deadlock.
    ctrl_dev = None
    for g in optimizer.param_groups:  # type: ignore[attr-defined]
        for p in g.get("params", []):
            if p is None:
                continue
            if ctrl_dev is None and isinstance(p, torch.Tensor):
                ctrl_dev = p.device
            gg = getattr(p, "grad", None)
            if gg is None:
                continue
            if not isinstance(gg, torch.Tensor) or gg.numel() <= 0:
                continue
            grads.append(gg)

    if ctrl_dev is None:
        # Extremely defensive: if optimizer has no tensor params, fall back to CPU.
        ctrl_dev = torch.device("cpu")

    # Coalesce into a single buffer to reduce per-parameter allreduce overhead.
    # This matters a lot on systems without fast GPU interconnect, where many small allreduces
    # can dominate the step time.
    dev0 = grads[0].device if grads else ctrl_dev
    dt0 = grads[0].dtype if grads else torch.float32
    same = True
    total = 0
    for gg in grads:
        total += int(gg.numel())
        if gg.device != dev0 or gg.dtype != dt0:
            same = False
            break

    # Fail fast if different ranks have different parameter sets / grad sizes.
    # IMPORTANT: avoid ReduceOp.MIN/MAX on int64 here — some NCCL stacks return garbage for those.
    # Instead, all_gather the scalar totals and compare exactly.
    total_t = torch.tensor([int(total)], device=dev0, dtype=torch.int64)
    try:
        ws = int(dist.get_world_size())
    except Exception:
        ws = 0
    if ws > 1:
        totals = [torch.empty_like(total_t) for _ in range(ws)]
        dist.all_gather(totals, total_t)
        vals = [int(t.item()) for t in totals]
        if any(v != vals[0] for v in vals[1:]):
            raise RuntimeError(
                "DDP grad buffer size mismatch across ranks: "
                + ", ".join(f"rank{i}={v}" for i, v in enumerate(vals))
                + ". This usually means some rank is missing a Parameter (e.g., optional latent disabled) or "
                "a rank hit an exception and skipped initializing part of the model."
            )

    # If *all* ranks have total==0, there's nothing to reduce this step.
    # (But we still did the all_gather above to keep the collective schedule aligned.)
    if (not grads) or total <= 0:
        return

    if (not same):
        # Fallback: allreduce each grad separately (after the mismatch check above).
        for gg in grads:
            dist.all_reduce(gg, op=dist.ReduceOp.SUM)
        return

    # Reuse a persistent buffer attached to the optimizer to avoid allocating every step.
    buf = getattr(optimizer, "_ddp_grad_buffer", None)
    try:
        if not isinstance(buf, torch.Tensor) or int(buf.numel()) != int(total) or buf.device != dev0 or buf.dtype != dt0:
            buf = torch.empty((int(total),), device=dev0, dtype=dt0)
            setattr(optimizer, "_ddp_grad_buffer", buf)
    except Exception:
        buf = torch.empty((int(total),), device=dev0, dtype=dt0)
        try:
            setattr(optimizer, "_ddp_grad_buffer", buf)
        except Exception:
            pass

    # Pack
    off = 0
    for gg in grads:
        n = int(gg.numel())
        buf[off : off + n].copy_(gg.reshape(-1))
        off += n

    # Allreduce once
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    # Unpack
    off = 0
    for gg in grads:
        n = int(gg.numel())
        gg.copy_(buf[off : off + n].view_as(gg))
        off += n


def _ddp_set_step_seed(params: dict, step: int, *, device: torch.device) -> None:
    """
    Make sampler noise deterministic across ranks by resetting RNG state per step.
    This is critical for SGHMC/pSGLD because the optimizer step injects random noise.
    """
    try:
        base = int(params.get("runtime_seed", 0))
    except Exception:
        base = 0
    s = int(base + 10000019 * int(step))
    try:
        torch.manual_seed(s)
    except Exception:
        pass
    try:
        if device.type == "cuda":
            # IMPORTANT: do NOT call manual_seed_all() here.
            # That can initialize CUDA contexts on *all* visible GPUs, including devices not used by this rank,
            # which can cause major slowdowns and can violate user-intended device selection.
            torch.cuda.manual_seed(s)
    except Exception:
        pass


def _maybe_write_map_csv(state: LocateState, epoch: int) -> None:
    """
    Periodically write a MAP CSV snapshot during Phase 1.

    NOTE: This is intentionally a lightweight side-effect used only when `write_map_csv=True`
    is passed to `_run_epoch` (Phase 1).
    """
    # In torchrun/DDP mode, only rank0 should emit files.
    try:
        ddp_enabled, _rk, _ws, ddp_is_main = _ddp_info(state.params)
        if ddp_enabled and not ddp_is_main:
            return
    except Exception:
        pass
    if epoch % 100 != 0:
        return
    try:
        X_src1 = (state.X_src + state.dX_src).detach().cpu().numpy()
        origins = write_output(
            state.origins0,
            X_src1,
            state.X_src.detach().cpu().numpy(),
            state.projector,
        )
        origins.write_csv(f"{state.params['catalog_outfile']}_MAP.csv")
    except Exception:
        # Best-effort only; never fail training for CSV output.
        return


def _get_diagnostics_cfg(params: dict) -> dict:
    """Return the diagnostics config dict (new schema only).

    Diagnostics are configured under `inference.diagnostics`.
    Legacy top-level `diagnostics` is no longer supported.
    """
    inf = params.get("inference", None)
    if not isinstance(inf, dict):
        raise KeyError("Missing required config block: inference")
    d = inf.get("diagnostics", None)
    if not isinstance(d, dict):
        raise KeyError("Missing required config block: inference.diagnostics")
    return d

def _update_svrg_snapshot(state: LocateState, batch_size: int, optimizer: torch.optim.Optimizer) -> None:
    """
    Compute full gradient at current parameters and store as snapshot.
    This is an expensive O(N) operation.
    """
    print("SVRG: Updating full gradient snapshot...")
    
    # Store snapshot of parameters
    state.svrg_dX_snapshot = state.dX_src.detach().clone()
    
    # Compute full gradient as an *average-gradient* consistent with our minibatch convention.
    # IMPORTANT:
    # - Do NOT call posterior_loss in a per-batch loop: it includes the prior term scaled by 1/N,
    #   which would get counted multiple times.
    # - Instead: add the prior ONCE, and add likelihood contributions as a weighted sum of batch means.
    from spider.core.modeling import compute_likelihood_loss, compute_prior_loss
    
    # Clear any existing grads
    optimizer.zero_grad(set_to_none=True)
    if state.dX_src.grad is not None:
        state.dX_src.grad.zero_()
    
    N = int(state.N)
    bs = max(int(batch_size), 10000)  # use a large batch for efficiency if possible
    
    # Current noise scales (constant during this snapshot computation)
    σp, σs = _current_noise_scales(state)
    
    # 1. Prior
    l_prior = compute_prior_loss(
        ΔX_src=state.dX_src,
        prior_event=state.prior_event,
        σ_p=σp,
        σ_s=σs,
        N_total=state.N,
        params=state.params,
        cluster_ids=state.cluster_ids,
        cluster_counts=state.cluster_counts,
        event_precision_matrix=state.event_precision_matrix,
    )
    l_prior.backward()
    
    # 2. Likelihood
    for i in range(0, N, bs):
        i_end = min(i + bs, N)
        bs_curr = i_end - i
        
        II_b = state.II[i:i_end]
        YY_b = state.YY[i:i_end]
             
        # Compute mean NLL for this batch
        nll_batch = compute_likelihood_loss(
            idx=II_b,
            y=YY_b,
            X_src=state.X_src,
            ΔX_src=state.dX_src,
            model=state.model,
            σ_p=σp,
            σ_s=σs,
            params=state.params,
            nuisance_delta=None
        )
        
        # Scale by fraction of total data to contribute to global mean
        loss_chunk = nll_batch * (bs_curr / N)
        loss_chunk.backward()
        
    # Store result
    state.svrg_grad_full = state.dX_src.grad.detach().clone()
    
    # Zero out again to leave clean state
    state.dX_src.grad.zero_()
    
    # Scale adjustment if user wants Total Gradient logic
    # (Check if optimizer scales by N_obs)
    # The optimizer wrapper usually handles scaling.
    # However, if we feed this into the SVRG correction formula:
    # v = g_batch - g_snap + g_full
    # All terms must be consistently scaled.
    # Our batches in _run_epoch produce gradients from posterior_loss().
    # posterior_loss() returns Average Posterior NLL.
    # So g_batch is Average Gradient.
    # Our g_full computation above produces Average Gradient.
    # So they match! 
    # (The optimizer might multiply by N later, but that applies to the sum v, which is fine).
    
    print(f"SVRG: Snapshot updated. Grad norm: {state.svrg_grad_full.norm().item():.3e}")

def _set_backend_noise(optimizer: torch.optim.Optimizer, *, enabled: bool, scale: float) -> None:
    """Set noise flags consistently for any sampler backend."""
    if not hasattr(optimizer, "param_groups"):
        return
    # AdaptiveSGHMC uses a BOHAMIANN-style update which *can* inject noise, but in SPIDER we
    # still want consistent Phase semantics:
    # - Phase 2: noise_scale_factor=0 => effectively deterministic drift (no injected noise)
    # - Phase 3: ramp noise_scale_factor up
    # - Phase 4: full sampling noise
    #
    # We detect AdaptiveSGHMC via the param-group preconditioner tag set by the backend factory.
    force_on = False
    try:
        if len(optimizer.param_groups) > 0:  # type: ignore[attr-defined]
            p0 = optimizer.param_groups[0]  # type: ignore[index]
            if str(p0.get("preconditioner", "")).strip().lower() == "adaptive_sghmc":
                force_on = True
    except Exception:
        force_on = False

    for g in optimizer.param_groups:  # type: ignore[attr-defined]
        if force_on:
            g["add_noise"] = True
            # For AdaptiveSGHMC, keep `add_noise=True` but allow `noise_scale` to be 0.0 to
            # match Phase 2 (deterministic) behavior.
            try:
                s = float(scale)
                if not math.isfinite(s) or s < 0.0:
                    s = 0.0
            except Exception:
                s = 0.0
            g["noise_scale"] = s
        else:
            g["add_noise"] = bool(enabled and (scale > 0.0))
            g["noise_scale"] = float(scale if enabled else 0.0)

def _run_epoch(
    state: LocateState,
    epoch_index: int,
    optimizer: torch.optim.Optimizer,
    *,
    is_sampling: bool = False,
    noise_scale_factor: float = 0.0,
    grad_clip_norm: float = 0.0,
    write_map_csv: bool = False,
) -> Dict[str, float]:
    """
    Unified epoch execution for all phases.
    
    Args:
        state: The LocateState object
        epoch_index: Current epoch index (for logging/seeding)
        optimizer: The optimizer to step (Adam or SGLD)
        is_sampling: If True, collect samples (Phase 4)
        noise_scale_factor: Factor to scale Langevin noise (0.0-1.0).
                           Handled generically for any sampler backend.
        grad_clip_norm: If > 0, clip gradients to this norm.
        write_map_csv: If True, check/write MAP CSV (Phase 1 behavior)
    """
    if bool(is_sampling):
        phase_id = 4
    elif float(noise_scale_factor) > 0.0:
        phase_id = 3
    elif isinstance(optimizer, torch.optim.Adam):
        phase_id = 1
    else:
        phase_id = 2

    # --- Per-prior runtime enables (materialized by validate_and_materialize_priors) ---
    # Hard break: priors are active in all phases when enabled (no per-phase scheduling).
    state.params["_prior_event_runtime_enable"] = bool(state.params.get("prior_event_enable", True))
    # Noise prior removed (fixed phase_unc only).

    ddp_enabled, ddp_rank, ddp_world_size, ddp_is_main = _ddp_info(state.params)

    # One-time diagnostics: residual correlation tests across candidate groupings.
    if (
        bool(state.params.get("_slowness_re_explicit_enabled", False))
        and not bool(state.params.get("_slowness_re_ddhop_corr_logged", False))
        and (not ddp_enabled)
    ):
        try:
            state.params["_slowness_re_ddhop_corr_logged"] = True
            resid_all = compute_residuals(state.II, state.YY, state.X_src, state.dX_src, state.model)
            resid_np = resid_all.detach().float().cpu().numpy()
            ph_np = state.YY[:, 4].detach().cpu().numpy().astype(np.int64, copy=False)
            sta_idx = getattr(state, "row_station_index", None)
            if isinstance(sta_idx, torch.Tensor):
                sta_np = sta_idx.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                sta_np = None
            II_np = state.II.detach().cpu().numpy().astype(np.int64, copy=False)
            if sta_np is None or sta_np.size != resid_np.size:
                warn("corr tests: missing station index; skipping station-based tests.", section="DIAG")

            ev1 = II_np[:, 0]
            ev2 = II_np[:, 1]

            def _corr(a, b) -> float:
                if a.size < 2:
                    return float("nan")
                a0 = a - a.mean()
                b0 = b - b.mean()
                denom = np.sqrt((a0 * a0).mean() * (b0 * b0).mean())
                if denom <= 0.0:
                    return float("nan")
                return float((a0 * b0).mean() / denom)

            def _loo_corr_from_keys(keys: np.ndarray, r: np.ndarray) -> tuple[float, int, int]:
                if keys.size == 0:
                    return float("nan"), 0, 0
                uniq, inv = np.unique(keys, axis=0, return_inverse=True)
                counts = np.bincount(inv)
                sums = np.bincount(inv, weights=r)
                cnt = counts[inv]
                s = sums[inv]
                valid = cnt > 1
                if not np.any(valid):
                    return float("nan"), int(uniq.shape[0]), 0
                r_v = r[valid]
                loo = (s[valid] - r_v) / (cnt[valid] - 1.0)
                return _corr(r_v, loo), int(uniq.shape[0]), int(r_v.size)

            # Same phase (global)
            try:
                keys_phase = ph_np.reshape(-1, 1)
                corr_phase, g_phase, n_phase = _loo_corr_from_keys(keys_phase, resid_np)
                print(f"[corr phase] corr={corr_phase:.4f} groups={g_phase} rows={n_phase}", flush=True)
            except Exception:
                pass

            # Same station (regardless of phase)
            if sta_np is not None:
                try:
                    keys_sta = sta_np.reshape(-1, 1)
                    corr_sta, g_sta, n_sta = _loo_corr_from_keys(keys_sta, resid_np)
                    print(f"[corr station] corr={corr_sta:.4f} groups={g_sta} rows={n_sta}", flush=True)
                except Exception:
                    pass

            # Same station-phase
            if sta_np is not None:
                try:
                    keys_sp = np.stack([sta_np, ph_np], axis=1)
                    corr_sp, g_sp, n_sp = _loo_corr_from_keys(keys_sp, resid_np)
                    print(f"[corr station-phase] corr={corr_sp:.4f} groups={g_sp} rows={n_sp}", flush=True)
                except Exception:
                    pass

            # Same event (event-only, and event+phase)
            try:
                ev_all = np.concatenate([ev1, ev2], axis=0)
                r_all = np.concatenate([resid_np, resid_np], axis=0)
                if ev_all.size > 0:
                    keys_e = ev_all.reshape(-1, 1)
                    corr_e, g_e, n_e = _loo_corr_from_keys(keys_e, r_all)
                    print(f"[corr event] corr={corr_e:.4f} groups={g_e} rows={n_e}", flush=True)
                if ev_all.size > 0:
                    ph_all = np.concatenate([ph_np, ph_np], axis=0)
                    keys_ep = np.stack([ev_all, ph_all], axis=1)
                    corr_ep, g_ep, n_ep = _loo_corr_from_keys(keys_ep, r_all)
                    print(f"[corr event-phase] corr={corr_ep:.4f} groups={g_ep} rows={n_ep}", flush=True)
            except Exception:
                pass

            # One-hop DD correlation: same event + station + phase
            if sta_np is not None:
                try:
                    ev_all = np.concatenate([ev1, ev2], axis=0)
                    sta_all = np.concatenate([sta_np, sta_np], axis=0)
                    ph_all = np.concatenate([ph_np, ph_np], axis=0)
                    r_all = np.concatenate([resid_np, resid_np], axis=0)
                    msk = np.isfinite(r_all) & (sta_all >= 0) & (ph_all >= 0)
                    ev_all = ev_all[msk]
                    sta_all = sta_all[msk]
                    ph_all = ph_all[msk]
                    r_all = r_all[msk]
                    keys_esp = np.stack([ev_all, sta_all, ph_all], axis=1)
                    corr_esp, g_esp, n_esp = _loo_corr_from_keys(keys_esp, r_all)
                    print(f"[dd-hop corr] corr={corr_esp:.4f} groups={g_esp} rows={n_esp}", flush=True)
                except Exception:
                    pass

            # k-hop DD correlation proxy (k=1,2) using neighbor event means for same station-phase
            if sta_np is not None:
                try:
                    n_events = int(state.X_src.shape[0])
                    n_sta = int(getattr(state, "n_stations", 0) or 0)
                    if n_events > 0 and n_sta > 0:
                        # Build adjacency
                        adj = [[] for _ in range(n_events)]
                        for a, b in zip(ev1, ev2):
                            if int(a) >= 0 and int(b) >= 0 and int(a) < n_events and int(b) < n_events:
                                adj[int(a)].append(int(b))
                                adj[int(b)].append(int(a))
                        # Per-event, station, phase mean residual
                        ev_all = np.concatenate([ev1, ev2], axis=0)
                        sta_all = np.concatenate([sta_np, sta_np], axis=0)
                        ph_all = np.concatenate([ph_np, ph_np], axis=0)
                        r_all = np.concatenate([resid_np, resid_np], axis=0)
                        msk = np.isfinite(r_all) & (sta_all >= 0) & (ph_all >= 0)
                        ev_all = ev_all[msk]
                        sta_all = sta_all[msk]
                        ph_all = ph_all[msk]
                        r_all = r_all[msk]
                        key = (ev_all.astype(np.int64) * n_sta + sta_all.astype(np.int64)) * 2 + ph_all.astype(np.int64)
                        uniq, inv = np.unique(key, return_inverse=True)
                        sums = np.bincount(inv, weights=r_all)
                        counts = np.bincount(inv)
                        mu = sums / np.maximum(1, counts)
                        mu_map = dict(zip(uniq.tolist(), mu.tolist()))
                        # Duplicate rows for ev1/ev2 entries
                        ev_du = np.concatenate([ev1, ev2], axis=0)
                        sta_du = np.concatenate([sta_np, sta_np], axis=0)
                        ph_du = np.concatenate([ph_np, ph_np], axis=0)
                        r_du = np.concatenate([resid_np, resid_np], axis=0)
                        def _neighbor_mean(ev_id: int, sta_id: int, ph_id: int, k: int) -> float:
                            if ev_id < 0 or ev_id >= n_events:
                                return float("nan")
                            if k == 1:
                                neigh = adj[ev_id]
                            else:
                                s = set(adj[ev_id])
                                for nb in list(s):
                                    s.update(adj[nb])
                                neigh = list(s)
                            if not neigh:
                                return float("nan")
                            vals = []
                            for nb in neigh:
                                kk = (nb * n_sta + int(sta_id)) * 2 + int(ph_id)
                                vv = mu_map.get(kk, None)
                                if vv is not None and np.isfinite(vv):
                                    vals.append(float(vv))
                            if not vals:
                                return float("nan")
                            return float(np.mean(vals))
                        for k in (1, 2):
                            loo_vals = []
                            r_vals = []
                            for ev_id, sta_id, ph_id, r0 in zip(ev_du, sta_du, ph_du, r_du):
                                if sta_id < 0 or ph_id < 0 or not np.isfinite(r0):
                                    continue
                                m = _neighbor_mean(int(ev_id), int(sta_id), int(ph_id), k)
                                if np.isfinite(m):
                                    loo_vals.append(m)
                                    r_vals.append(r0)
                            if len(r_vals) > 1:
                                corr_k = _corr(np.asarray(r_vals), np.asarray(loo_vals))
                                print(f"[dd-hop{k} corr] corr={corr_k:.4f} rows={len(r_vals)}", flush=True)
                except Exception:
                    pass

            # Spatial proximity (event-pair midpoint bins)
            try:
                X_cur = (state.X_src + state.dX_src).detach().cpu().numpy()
                x1 = X_cur[ev1, :2]
                x2 = X_cur[ev2, :2]
                mid = 0.5 * (x1 + x2)
                bin_km = 10.0
                bx = np.floor(mid[:, 0] / bin_km).astype(np.int64)
                by = np.floor(mid[:, 1] / bin_km).astype(np.int64)
                keys_xy = np.stack([bx, by], axis=1)
                corr_xy, g_xy, n_xy = _loo_corr_from_keys(keys_xy, resid_np)
                print(f"[corr midpoint_xy] corr={corr_xy:.4f} groups={g_xy} rows={n_xy}", flush=True)
            except Exception:
                pass

            # Path similarity (event-pair azimuth + distance bins)
            try:
                X_cur = (state.X_src + state.dX_src).detach().cpu().numpy()
                dx = X_cur[ev2, 0] - X_cur[ev1, 0]
                dy = X_cur[ev2, 1] - X_cur[ev1, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                az = np.arctan2(dy, dx)  # [-pi, pi]
                az = (az + 2.0 * np.pi) % (2.0 * np.pi)
                az_bin = np.floor(az / (np.pi / 6.0)).astype(np.int64)  # 30 deg
                dist_bin = np.floor(dist / 10.0).astype(np.int64)       # 10 km
                keys_ad = np.stack([az_bin, dist_bin], axis=1)
                corr_ad, g_ad, n_ad = _loo_corr_from_keys(keys_ad, resid_np)
                print(f"[corr az_dist] corr={corr_ad:.4f} groups={g_ad} rows={n_ad}", flush=True)
            except Exception:
                pass

            # Time drift (event time bins) if available
            try:
                if isinstance(state.origins0, pl.DataFrame):
                    for col in ("origin_time", "time", "t0", "event_time"):
                        if col in state.origins0.columns:
                            t_ev = state.origins0[col].to_numpy()
                            if t_ev.size == int(state.X_src.shape[0]):
                                t1 = t_ev[ev1]
                                t2 = t_ev[ev2]
                                tmid = 0.5 * (t1 + t2)
                                # Bin by 1-day intervals (seconds)
                                bin_t = np.floor(np.asarray(tmid, dtype=np.float64) / 86400.0).astype(np.int64)
                                corr_t, g_t, n_t = _loo_corr_from_keys(bin_t.reshape(-1, 1), resid_np)
                                print(f"[corr time_bin] corr={corr_t:.4f} groups={g_t} rows={n_t}", flush=True)
                            break
            except Exception:
                pass

            # Cross-phase correlation: P vs S residuals within shared groups.
            try:
                if sta_np is not None:
                    # Build per-event, per-station, per-phase mean residuals
                    ev_all = np.concatenate([ev1, ev2], axis=0)
                    sta_all = np.concatenate([sta_np, sta_np], axis=0)
                    ph_all = np.concatenate([ph_np, ph_np], axis=0)
                    r_all = np.concatenate([resid_np, resid_np], axis=0)
                    msk = np.isfinite(r_all) & (sta_all >= 0) & (ph_all >= 0)
                    ev_all = ev_all[msk]
                    sta_all = sta_all[msk]
                    ph_all = ph_all[msk]
                    r_all = r_all[msk]
                    # Mean residual per (event,station,phase)
                    keys_esp = np.stack([ev_all, sta_all, ph_all], axis=1)
                    uniq_esp, inv_esp = np.unique(keys_esp, axis=0, return_inverse=True)
                    sums = np.bincount(inv_esp, weights=r_all)
                    counts = np.bincount(inv_esp)
                    means = sums / np.maximum(1, counts)
                    esp_map = {tuple(k): means[i] for i, k in enumerate(uniq_esp)}

                    # P/S pairs within same event+station
                    ps_pairs = []
                    uniq_es = np.unique(np.stack([ev_all, sta_all], axis=1), axis=0)
                    for ev_id, sta_id in uniq_es:
                        mp = esp_map.get((int(ev_id), int(sta_id), 0), None)
                        ms = esp_map.get((int(ev_id), int(sta_id), 1), None)
                        if (mp is not None) and (ms is not None) and np.isfinite(mp) and np.isfinite(ms):
                            ps_pairs.append((mp, ms))
                    if ps_pairs:
                        p_arr = np.asarray([x[0] for x in ps_pairs])
                        s_arr = np.asarray([x[1] for x in ps_pairs])
                        corr_ps_es = _corr(p_arr, s_arr)
                        print(f"[corr P-S | event+station] corr={corr_ps_es:.4f} pairs={len(ps_pairs)}", flush=True)

                    # P/S pairs within same event (aggregate across stations)
                    uniq_e = np.unique(ev_all)
                    ps_e = []
                    for ev_id in uniq_e:
                        mp = np.mean([v for k, v in esp_map.items() if k[0] == int(ev_id) and k[2] == 0], dtype=np.float64) if True else float("nan")
                        ms = np.mean([v for k, v in esp_map.items() if k[0] == int(ev_id) and k[2] == 1], dtype=np.float64) if True else float("nan")
                        if np.isfinite(mp) and np.isfinite(ms):
                            ps_e.append((mp, ms))
                    if ps_e:
                        p_arr = np.asarray([x[0] for x in ps_e])
                        s_arr = np.asarray([x[1] for x in ps_e])
                        corr_ps_e = _corr(p_arr, s_arr)
                        print(f"[corr P-S | event] corr={corr_ps_e:.4f} pairs={len(ps_e)}", flush=True)

                    # P/S pairs within same station (aggregate across events)
                    uniq_s = np.unique(sta_all)
                    ps_s = []
                    for sta_id in uniq_s:
                        mp = np.mean([v for k, v in esp_map.items() if k[1] == int(sta_id) and k[2] == 0], dtype=np.float64) if True else float("nan")
                        ms = np.mean([v for k, v in esp_map.items() if k[1] == int(sta_id) and k[2] == 1], dtype=np.float64) if True else float("nan")
                        if np.isfinite(mp) and np.isfinite(ms):
                            ps_s.append((mp, ms))
                    if ps_s:
                        p_arr = np.asarray([x[0] for x in ps_s])
                        s_arr = np.asarray([x[1] for x in ps_s])
                        corr_ps_s = _corr(p_arr, s_arr)
                        print(f"[corr P-S | station] corr={corr_ps_s:.4f} pairs={len(ps_s)}", flush=True)
            except Exception:
                pass
        except Exception as e:
            warn(f"dd-hop corr failed: {e}", section="DIAG")
    
    use_event_batches = bool(state.params.get("event_batch_enable", False))
    if ddp_enabled and use_event_batches:
        raise ValueError(
            "torchrun/DDP mode for `spider sample` currently supports only standard batching "
            "(inference.batching.standard.*). Disable inference.batching.event_batches.enabled."
        )
    permute_time_s = 0.0
    if not use_event_batches:
        # Allow a per-run seed offset (e.g. for multi-GPU independent chains).
        seed0 = int(state.params.get("runtime_seed", 0))
        shuffle = bool(state.params.get("batch_shuffle", True))
        # Optional: timing + one-line confirmation (helps diagnose big-N performance).
        t_perm0 = time.time()
        state.begin_epoch_rr(seed=int(seed0 + epoch_index), shuffle=shuffle)
        permute_time_s = float(time.time() - t_perm0)
        # Expose whether this epoch is shuffled to lower-level code (e.g. caching in likelihoods).
        state.params["_runtime_batch_shuffle"] = bool(shuffle)

    # Expose epoch index to lower-level code paths (e.g., likelihood caching / refresh schedules).
    # This is an internal implementation detail, not a user-facing config key.
    state.params["_runtime_epoch_index"] = int(epoch_index)
        
    epoch_start_time = time.time()
    total_loss_vals: List[float] = []
    # Weighted loss aggregation (by number of rows/edges in each batch) so epoch "loss"
    # is comparable across batch sizes / event-batch partitioning.
    total_loss_weighted_sum: float = 0.0
    total_loss_weighted_denom: int = 0

    # Optional: uncollapsed shared-event latent diagnostics (accumulated across minibatches; one sync at end).
    want_lat_diag = bool(state.params.get("_shared_event_latent_enabled", False)) and _want_wandb_group(state.params, "shared_event_latent")
    b_delta_sum = None
    b_delta_sumsq = None
    b_delta_maxabs = None
    b_delta_count = None
    # Also track RMS of *event-level* latent endpoints b(e1), b(e2) used by DD rows.
    # This is more interpretable than RMS of the underlying parameter tensor in inducing_gp mode
    # (where `shared_event_latent_b` stores inducing coefficients rather than event values).
    b_end_sumsq_p = None
    b_end_sumsq_s = None
    b_end_count_p = None
    b_end_count_s = None
    # slowness_inducing_gp: track RMS of the *event-level u vectors* at endpoints (more interpretable than dual coeffs).
    u_end_sumsq_p = None
    u_end_sumsq_s = None
    u_end_count_p = None
    u_end_count_s = None
    u_end_maxnorm_p = None
    u_end_maxnorm_s = None
    # corr_error: track RMS of the *actual per-row correction* delta_corr = W(sta)·(b_e2 - b_e1) (split P/S).
    corr_dc_sumsq_p = None
    corr_dc_sumsq_s = None
    corr_dc_count_p = None
    corr_dc_count_s = None
    corr_dc_maxabs_p = None
    corr_dc_maxabs_s = None
    # Online ESS metrics are computed only when enough *saved samples* exist and the cadence triggers.
    # To avoid gaps in W&B time series (epochs where ESS isn't recomputed), we cache the last metrics
    # on the state object and re-log them each epoch.
    ess_online_updated_this_epoch = False
    if not hasattr(state, "_ess_online_last_metrics"):
        setattr(state, "_ess_online_last_metrics", None)
    if not hasattr(state, "_ess_online_error_count"):
        setattr(state, "_ess_online_error_count", 0)

    # Batching parameters
    if use_event_batches:
        seed0 = int(state.params.get("runtime_seed", 0))
        epoch_seed = int(seed0 + epoch_index)
        # Use sgld params by default, fallback to warmup if key missing (Phase 1 uses warmup key)
        # But to simplify, we can look at the phase. Phase 1 sets its own max_edges logic.
        # For unified logic, we'll try sgld key first, then warmup key.
        # Actually, looking at original code, Phase 1 used 'event_batch_max_edges_warmup', others 'sgld'.
        # We will rely on the caller to set these in params or just pick one logic.
        # Let's stick to the logic: if optim is Adam (Phase 1), use warmup key?
        # Or just use 'event_batch_max_edges' generic key if specific one is missing.
        # To be safe and simple:
        max_edges_key = "event_batch_max_edges_sgld"
        if isinstance(optimizer, torch.optim.Adam):
             max_edges_key = "event_batch_max_edges_warmup"
             
        max_edges = int(state.params.get(max_edges_key, state.params.get("event_batch_max_edges", 0)))
        events_per_batch = int(state.params.get("event_batch_size", 256))
        reorder_all = bool(state.params.get("event_bucket_reorder_all", False))
        reuse_epochs = int(state.params.get("event_bucket_reuse_epochs", 1))
        
        _ensure_owner_buckets(
            state,
            epoch_index=epoch_seed,
            events_per_batch=events_per_batch,
            max_edges_per_batch=max_edges,
            reorder_all=reorder_all,
            reuse_epochs=reuse_epochs,
        )
        
        # Determine iterator
        if state._bucket_offsets is None or (reorder_all and (state._bucket_II is None or state._bucket_YY is None)):
            batch_iter = _iter_event_batches(state, epoch_seed=epoch_seed, max_edges_per_batch=max_edges, events_per_batch=events_per_batch)
            use_buckets = False
        else:
            batch_iter = range(int(state._bucket_offsets.numel()) - 1)
            use_buckets = True
            
    else:
        # Standard batching
        bs_key = "batch_size_sgld"
        if isinstance(optimizer, torch.optim.Adam):
            bs_key = "batch_size_warmup"
        batch_size = int(state.params.get(bs_key, 10000))
        # IMPORTANT: avoid an extra empty final batch when N is exactly divisible by batch_size.
        # `range(0, N//bs + 1)` yields a last iteration with i_start==i_end==N (no data),
        # which can create rank-divergent control flow and pointless DDP collectives.
        if batch_size <= 0:
            batch_iter = range(0)
        else:
            n_batches = int((int(state.N) + int(batch_size) - 1) // int(batch_size))
            batch_iter = range(int(n_batches))
        use_buckets = False
        # Expose standard batching parameters for lower-level caching.
        state.params["_runtime_batch_size"] = int(batch_size)
        state.params["_runtime_batching_mode"] = "standard"

    # Noise setup (generic sampler backend)
    # If noise_scale_factor > 0, enable noise; else disable (e.g., Phase 2). Phase 3 ramps it.
    _set_backend_noise(optimizer, enabled=(noise_scale_factor > 0.0), scale=float(noise_scale_factor))
    # If we are using corr_error ESS (blocked update for corr_error_b), freeze the corr_error
    # param group in the sampler backend so only ΔX_src is updated by Langevin.
    if bool(is_sampling):
        _maybe_freeze_corr_error_group_for_sampler(state, optimizer)
        _maybe_freeze_slowness_re_group_for_sampler(state, optimizer)
        _maybe_freeze_dd_graph_re_group_for_sampler(state, optimizer)

    # For AdaptiveSGHMC we want burn-in driven by *epochs* (Phase 3) rather than optimizer step counts.
    # Expose this via `param_group['is_burnin']`, which the optimizer reads.
    try:
        if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
            p0 = optimizer.param_groups[0]
            if str(p0.get("preconditioner", "")).strip().lower() == "adaptive_sghmc":
                for g in optimizer.param_groups:
                    g["is_burnin"] = bool(int(phase_id) == 3)
    except Exception:
        pass

    # SVRG Snapshot Update (only if SVRG enabled and we are sampling)
    svrg_enabled = state.svrg_enable and is_sampling
    if ddp_enabled and svrg_enabled:
        raise ValueError("SVRG is not supported in torchrun/DDP mode (it requires extra full-gradient bookkeeping). Disable inference.diagnostics.svrg.enabled.")
    if svrg_enabled:
        # Check if we need to update snapshot (e.g. every epoch)
        # For simplicity, update at start of every epoch for now if enabled
        _update_svrg_snapshot(state, batch_size=batch_size, optimizer=optimizer)

    # Optional profiling accumulators (only active when inference.diagnostics.profile_shared_event_latent=true)
    try:
        diag0 = _get_diagnostics_cfg(state.params)
        if isinstance(diag0, dict) and bool(diag0.get("profile_shared_event_latent", False)):
            state.params["_se_lat_time_ms_sum"] = 0.0
            state.params["_se_lat_time_ms_count"] = 0
        # Optional profiling: collapsed shared_event_re likelihood (PCG) cost + workload stats.
        # Expected config location: inference.diagnostics.profile_shared_event_re (bool).
        if isinstance(diag0, dict) and bool(diag0.get("profile_shared_event_re", False)):
            state.params["_se_re_time_ms_sum"] = 0.0
            state.params["_se_re_time_ms_count"] = 0
            state.params["_se_re_groups_sum"] = 0
            state.params["_se_re_groups_pcg_sum"] = 0
            state.params["_se_re_groups_fallback_sum"] = 0
            state.params["_se_re_max_rows_max"] = 0
            state.params["_se_re_max_nodes_max"] = 0
        # Optional profiling: collapsed slowness_re likelihood (inducing GP) cost + breakdown.
        # Expected config location: inference.diagnostics.profile_slowness_re (bool).
        if isinstance(diag0, dict) and bool(diag0.get("profile_slowness_re", False)):
            # Runtime gate read by modeling.py
            state.params["_profile_slowness_re"] = True
            # Profiling knobs (optional)
            try:
                state.params["_profile_slowness_re_max_groups"] = int(diag0.get("profile_slowness_re_max_groups", 2))
            except Exception:
                state.params["_profile_slowness_re_max_groups"] = 2
            try:
                state.params["_profile_slowness_re_use_cuda_events"] = bool(diag0.get("profile_slowness_re_use_cuda_events", True))
            except Exception:
                state.params["_profile_slowness_re_use_cuda_events"] = True
            # Aggregates (per-epoch)
            state.params["_sl_re_time_ms_sum"] = 0.0
            state.params["_sl_re_time_ms_count"] = 0
            state.params["_sl_re_grouping_ms_sum"] = 0.0
            state.params["_sl_re_kernel_ms_sum"] = 0.0
            state.params["_sl_re_assemble_ms_sum"] = 0.0
            state.params["_sl_re_solve_ms_sum"] = 0.0
            state.params["_sl_re_profiled_groups_sum"] = 0
            state.params["_sl_re_pcg_iters_sum"] = 0
            # Workload (per batch)
            state.params["_sl_re_groups_sum"] = 0
            state.params["_sl_re_groups_woodbury_sum"] = 0
            state.params["_sl_re_groups_fallback_sum"] = 0
            state.params["_sl_re_max_rows_max"] = 0
            state.params["_sl_re_max_nodes_max"] = 0
            # Also compute a static upper bound for M from stored inducing artifacts (cheap, one-time).
            M_static = 0
            try:
                K_full = state.params.get("_slowness_re_inducing_K_full", None)
                if isinstance(K_full, torch.Tensor) and K_full.ndim == 2:
                    M_static = int(K_full.shape[0])
                else:
                    K_blocks = state.params.get("_slowness_re_inducing_K_blocks", None)
                    if isinstance(K_blocks, list) and K_blocks:
                        ms = []
                        for K in K_blocks:
                            if isinstance(K, torch.Tensor) and K.ndim == 2:
                                ms.append(int(K.shape[0]))
                        if ms:
                            M_static = int(max(ms))
            except Exception:
                M_static = 0
            state.params["_sl_re_max_M_static"] = int(M_static)
            state.params["_sl_re_max_M_max"] = int(M_static)
    except Exception:
        pass

    # Best-effort: total batches (works for standard batching / ranges)
    total_batches = None
    try:
        if isinstance(batch_iter, range):
            total_batches = int(len(batch_iter))
    except Exception:
        total_batches = None

    # Inner Loop
    for bi, batch_item in enumerate(batch_iter):
        optimizer.zero_grad(set_to_none=True)
        
        # Prepare batch data
        if use_event_batches:
            if use_buckets:
                # batch_item is index in offsets
                i0 = int(state._bucket_offsets[batch_item].item())
                i1 = int(state._bucket_offsets[batch_item + 1].item())
                if i1 <= i0: continue
                
                if reorder_all and state._bucket_II is not None:
                    II_b = state._bucket_II[i0:i1, :]
                    YY_b = state._bucket_YY[i0:i1, :]
                    rows = None
                else:
                    rows = state._bucket_rows_order[i0:i1]
                    # Fail fast with a clear error if bucket row indices are invalid.
                    try:
                        N_rt = int(state.II.shape[0])
                        if isinstance(rows, torch.Tensor) and rows.numel() > 0:
                            mn = int(rows.min().item())
                            mx = int(rows.max().item())
                            if mn < 0 or mx >= N_rt:
                                raise ValueError(
                                    f"owner_buckets produced out-of-range row indices: min={mn} max={mx} N={N_rt} "
                                    f"(bucket={int(batch_item)} slice=[{i0}:{i1}])"
                                )
                    except Exception:
                        raise
                    II_b = state.II.index_select(0, rows)
                    YY_b = state.YY.index_select(0, rows)
                # Expose stable bucket id to lower-level code (e.g., correlated likelihood caching).
                # This is an internal implementation detail, not a user-facing config key.
                state.params["_runtime_bucket_id"] = int(batch_item)
                # Also expose a bucket "generation" token that changes whenever buckets are rebuilt.
                # Without this, caches keyed only by bucket_id will collide across rebuilds because
                # bucket ids are reused from 0..num_buckets-1 each rebuild.
                try:
                    state.params["_runtime_bucket_gen"] = int(getattr(state, "_bucket_last_epoch", -1))
                except Exception:
                    state.params["_runtime_bucket_gen"] = -1
                try:
                    if reorder_all and getattr(state, "_bucket_p_counts", None) is not None:
                        state.params["_runtime_bucket_p_count"] = int(state._bucket_p_counts[batch_item].item())
                    else:
                        state.params["_runtime_bucket_p_count"] = -1
                except Exception:
                    state.params["_runtime_bucket_p_count"] = -1

                # If available, provide precomputed per-bucket phase graphs (nodes/u/v) to the likelihood.
                # This avoids per-batch torch.unique/remapping when grouping='phase'.
                try:
                    if reorder_all and getattr(state, "_bucket_nodes_p", None) is not None:
                        bi = int(batch_item)
                        state.params["_runtime_bucket_nodes_p"] = state._bucket_nodes_p[bi]
                        state.params["_runtime_bucket_u_p"] = state._bucket_u_p[bi]
                        state.params["_runtime_bucket_v_p"] = state._bucket_v_p[bi]
                        state.params["_runtime_bucket_nodes_s"] = state._bucket_nodes_s[bi]
                        state.params["_runtime_bucket_u_s"] = state._bucket_u_s[bi]
                        state.params["_runtime_bucket_v_s"] = state._bucket_v_s[bi]
                    else:
                        state.params["_runtime_bucket_nodes_p"] = None
                        state.params["_runtime_bucket_u_p"] = None
                        state.params["_runtime_bucket_v_p"] = None
                        state.params["_runtime_bucket_nodes_s"] = None
                        state.params["_runtime_bucket_u_s"] = None
                        state.params["_runtime_bucket_v_s"] = None
                except Exception:
                    state.params["_runtime_bucket_nodes_p"] = None
                    state.params["_runtime_bucket_u_p"] = None
                    state.params["_runtime_bucket_v_p"] = None
                    state.params["_runtime_bucket_nodes_s"] = None
                    state.params["_runtime_bucket_u_s"] = None
                    state.params["_runtime_bucket_v_s"] = None

                # Preferred: pre-chunked phase blocks (each chunk <= max_nodes), fully deterministic.
                try:
                    if reorder_all and getattr(state, "_bucket_chunks_p", None) is not None:
                        bi = int(batch_item)
                        state.params["_runtime_bucket_chunks_p"] = state._bucket_chunks_p[bi]
                        state.params["_runtime_bucket_chunks_s"] = state._bucket_chunks_s[bi]
                    else:
                        state.params["_runtime_bucket_chunks_p"] = None
                        state.params["_runtime_bucket_chunks_s"] = None
                except Exception:
                    state.params["_runtime_bucket_chunks_p"] = None
                    state.params["_runtime_bucket_chunks_s"] = None

                # Provide per-row station index (stable int id for (network,station)) when available.
                # This enables station-dependent nuisance models (e.g. shared_event_latent) without relying
                # on float receiver coordinates.
                try:
                    sta_b = None
                    if reorder_all and getattr(state, "_bucket_station_index", None) is not None:
                        sta_b = state._bucket_station_index[i0:i1]
                    elif rows is not None and getattr(state, "row_station_index", None) is not None:
                        sta_b = state.row_station_index.index_select(0, rows)
                    state.params["_runtime_bucket_station_index"] = sta_b
                except Exception:
                    state.params["_runtime_bucket_station_index"] = None
                # Provide per-row component ids aligned to this bucket slice when available.
                try:
                    comp_b = None
                    if reorder_all and getattr(state, "_bucket_comp_index", None) is not None:
                        comp_b = state._bucket_comp_index[i0:i1]
                    state.params["_runtime_bucket_comp_index"] = comp_b
                except Exception:
                    state.params["_runtime_bucket_comp_index"] = None
                # Provide per-row original indices for Student-t scale-mixture (lambda).
                try:
                    if reorder_all and getattr(state, "_bucket_rows_order", None) is not None:
                        state.params["_runtime_batch_rows"] = state._bucket_rows_order[i0:i1]
                    elif isinstance(rows, torch.Tensor):
                        state.params["_runtime_batch_rows"] = rows
                    else:
                        state.params["_runtime_batch_rows"] = None
                except Exception:
                    state.params["_runtime_batch_rows"] = None
                # Not a standard batch; clear standard ids to avoid accidental cache hits.
                state.params["_runtime_batch_id"] = -1
                state.params["_runtime_batch_i0"] = -1
                state.params["_runtime_batch_i1"] = -1
            else:
                # batch_item is batch_idx tensor
                # Defensive: ensure batch row indices are in-range before index_select.
                try:
                    N_rt = int(state.II.shape[0])
                    if isinstance(batch_item, torch.Tensor) and batch_item.numel() > 0:
                        mn = int(batch_item.min().item())
                        mx = int(batch_item.max().item())
                        if mn < 0 or mx >= N_rt:
                            raise ValueError(
                                f"event_batches produced out-of-range row indices: min={mn} max={mx} N={N_rt}"
                            )
                except Exception:
                    raise
                II_b = state.II.index_select(0, batch_item)
                YY_b = state.YY.index_select(0, batch_item)
                rows = batch_item # for SSST index select if needed logic? 
                # Actually SSST logic uses batch_idx directly in fallback loop
                state.params["_runtime_bucket_id"] = -1
                state.params["_runtime_bucket_gen"] = -1
                state.params["_runtime_bucket_p_count"] = -1
                state.params["_runtime_bucket_nodes_p"] = None
                state.params["_runtime_bucket_u_p"] = None
                state.params["_runtime_bucket_v_p"] = None
                state.params["_runtime_bucket_nodes_s"] = None
                state.params["_runtime_bucket_u_s"] = None
                state.params["_runtime_bucket_v_s"] = None
                state.params["_runtime_bucket_chunks_p"] = None
                state.params["_runtime_bucket_chunks_s"] = None
                try:
                    if getattr(state, "row_station_index", None) is not None:
                        state.params["_runtime_bucket_station_index"] = state.row_station_index.index_select(0, batch_item)
                    else:
                        state.params["_runtime_bucket_station_index"] = None
                except Exception:
                    state.params["_runtime_bucket_station_index"] = None
                state.params["_runtime_bucket_comp_index"] = None
                # Provide per-row original indices for Student-t scale-mixture (lambda).
                state.params["_runtime_batch_rows"] = rows
                # Not a standard batch; clear standard ids to avoid accidental cache hits.
                state.params["_runtime_batch_id"] = -1
                state.params["_runtime_batch_i0"] = -1
                state.params["_runtime_batch_i1"] = -1
        else:
            # batch_item is j (index of batch)
            i_start = batch_item * batch_size
            i_end = min(i_start + batch_size, state.N)
            II_b = state.II_epoch[i_start:i_end, :]
            YY_b = state.YY_epoch[i_start:i_end]
            global_bsz = int(i_end - i_start)
            # indices for SSST
            # If using standard batching, we assume contiguous indices in epoch permutation
            rows = None 
            state.params["_runtime_bucket_id"] = -1
            state.params["_runtime_bucket_gen"] = -1
            state.params["_runtime_bucket_p_count"] = -1
            state.params["_runtime_bucket_nodes_p"] = None
            state.params["_runtime_bucket_u_p"] = None
            state.params["_runtime_bucket_v_p"] = None
            state.params["_runtime_bucket_nodes_s"] = None
            state.params["_runtime_bucket_u_s"] = None
            state.params["_runtime_bucket_v_s"] = None
            # Provide per-row original indices for Student-t scale-mixture (lambda).
            try:
                if getattr(state, "_perm_epoch", None) is not None:
                    state.params["_runtime_batch_rows"] = state._perm_epoch[i_start:i_end]
                else:
                    state.params["_runtime_batch_rows"] = torch.arange(
                        int(i_start), int(i_end), device=II_b.device, dtype=torch.int64
                    )
            except Exception:
                state.params["_runtime_batch_rows"] = None
            state.params["_runtime_bucket_chunks_p"] = None
            state.params["_runtime_bucket_chunks_s"] = None
            # Stable standard batch id (only meaningful when _runtime_batch_shuffle is false).
            state.params["_runtime_batch_id"] = int(batch_item)
            state.params["_runtime_batch_i0"] = int(i_start)
            state.params["_runtime_batch_i1"] = int(i_end)
            try:
                if getattr(state, "row_station_index_epoch", None) is not None:
                    state.params["_runtime_bucket_station_index"] = state.row_station_index_epoch[i_start:i_end]
                elif getattr(state, "row_station_index", None) is not None:
                    state.params["_runtime_bucket_station_index"] = state.row_station_index[i_start:i_end]
                else:
                    state.params["_runtime_bucket_station_index"] = None
            except Exception:
                state.params["_runtime_bucket_station_index"] = None
            state.params["_runtime_bucket_comp_index"] = None

        # --- DDP shard: split *this batch* across ranks (global batch size is the configured batch size) ---
        if ddp_enabled:
            try:
                # We only support standard batching here (event-batch path returns earlier).
                B = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
                if B != int(global_bsz):
                    global_bsz = int(B)
                # IMPORTANT:
                # Use a *strided* shard (interleaved rows) rather than a contiguous slice.
                #
                # Rationale: if rows are partially sorted (e.g., by phase/component/station),
                # contiguous sharding can give different ranks different data *types* within the same
                # global batch. That can lead to rank-specific unused Parameters (grad=None) and
                # NCCL deadlocks in our manual gradient allreduce.
                #
                # Strided sharding is deterministic and tends to preserve mixture across ranks.
                if global_bsz > 0:
                    sel = torch.arange(int(ddp_rank), int(global_bsz), int(ddp_world_size), device=II_b.device)
                    II_b = II_b.index_select(0, sel)
                    YY_b = YY_b.index_select(0, sel)
                sta_rt = state.params.get("_runtime_bucket_station_index", None)
                if isinstance(sta_rt, torch.Tensor) and int(sta_rt.shape[0]) == int(global_bsz):
                    if global_bsz > 0:
                        state.params["_runtime_bucket_station_index"] = sta_rt.index_select(0, sel)
            except Exception:
                # Leave batch unsharded if something goes wrong; better than crashing mid-run.
                pass

        # Noise scales for loss
        σp, σs = _current_noise_scales(state)

        # ---- Fail-fast index validation (prevents opaque CUDA IndexKernel asserts) ----
        # Validate event indices in II_b are within [0, n_events).
        # This MUST run outside any try/except that might swallow the error.
        try:
            Ne_rt = int(state.X_src.shape[0])
            if Ne_rt > 0 and isinstance(II_b, torch.Tensor) and II_b.numel() > 0:
                e1_rt = II_b[:, 0].to(torch.int64)
                e2_rt = II_b[:, 1].to(torch.int64)
                mn = int(torch.minimum(e1_rt.min(), e2_rt.min()).item())
                mx = int(torch.maximum(e1_rt.max(), e2_rt.max()).item())
                if mn < 0 or mx >= Ne_rt:
                    raise ValueError(f"Batch II has out-of-range event indices: min={mn} max={mx} n_events={Ne_rt}")
        except Exception:
            raise

        # Validate station indices (if present) are within [0, n_stations).
        try:
            sta_rt = state.params.get("_runtime_bucket_station_index", None)
            n_stations_rt = int(getattr(state, "n_stations", 0))
            if isinstance(sta_rt, torch.Tensor) and sta_rt.numel() > 0 and n_stations_rt > 0:
                mn = int(sta_rt.min().item())
                mx = int(sta_rt.max().item())
                if mn < 0 or mx >= n_stations_rt:
                    raise ValueError(f"Batch station index out of range: min={mn} max={mx} n_stations={n_stations_rt}")
        except Exception:
            raise
        
        nuisance_delta = None
        sigma_extra_var = None
        prof_se_lat = False
        se_lat_t0 = None
        try:
            # Optional runtime profiling: time the shared_event_latent nuisance term reconstruction.
            # Expected config location: inference.diagnostics.profile_shared_event_latent (bool).
            diag = _get_diagnostics_cfg(state.params)
            prof_se_lat = bool(diag.get("profile_shared_event_latent", False)) if isinstance(diag, dict) else False
        except Exception:
            prof_se_lat = False

        # Optional: uncollapsed shared-event latent random effects b[s,event,phase] (sampled).
        # Adds nuisance term: (b_{s,e2,phase} - b_{s,e1,phase}) to predicted differential time.
        try:
            if prof_se_lat:
                import time as _time
                if state.device.type == "cuda":
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                se_lat_t0 = _time.perf_counter()
            if bool(state.params.get("_shared_event_latent_enabled", False)):
                mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                b_lat = getattr(state, "shared_event_latent_b", None)
                sta_b = state.params.get("_runtime_bucket_station_index", None)
                if isinstance(sta_b, torch.Tensor) and int(sta_b.shape[0]) == int(II_b.shape[0]) and isinstance(b_lat, torch.Tensor):
                    e1 = II_b[:, 0].to(torch.int64)
                    e2 = II_b[:, 1].to(torch.int64)
                    sta_bi = sta_b.to(torch.int64)
                    ph = YY_b[:, 4]
                    is_s = (ph >= 0.5)

                    # Fail fast if event indices are out of range for any event-indexed tensors.
                    # This prevents CUDA device-side asserts inside index_select on per-event arrays.
                    try:
                        Ne_rt = int(state.X_src.shape[0])
                        if Ne_rt > 0 and e1.numel() > 0:
                            mn = int(torch.minimum(e1.min(), e2.min()).item())
                            mx = int(torch.maximum(e1.max(), e2.max()).item())
                            if mn < 0 or mx >= Ne_rt:
                                raise ValueError(f"II contains out-of-range event indices in batch: min={mn} max={mx} n_events={Ne_rt}")
                    except Exception:
                        raise

                    if mode == "inducing_gp":
                        # Inducing-point GP (predictive-process mean):
                        # b(e) ≈ Σ_j k(e,u_j) * c_j, where c are inducing coefficients and k is an RBF kernel.
                        # We store per-event neighbor inducing indices + kernel values from Stage 3.
                        # Optional Stage 5 (FITC): inflate per-observation noise by the diagonal residual variance
                        #   Var[ε_e] = (1 - Q_ee) * τ^2, so for a differential (e1,e2): Var[ε_e2 - ε_e1] = (r2 + r1) * τ^2.
                        try:
                            if bool(state.params.get("_shared_event_latent_inducing_fitc_enable", False)):
                                resid = getattr(state, "shared_event_latent_inducing_fitc_resid", None)
                                if isinstance(resid, torch.Tensor) and resid.ndim == 1:
                                    r1 = resid.index_select(0, e1).to(torch.float32)
                                    r2 = resid.index_select(0, e2).to(torch.float32)
                                    rsum = (r1 + r2).clamp_min(0.0)
                                    tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                                    tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                                    extra = torch.where(is_s, rsum * (tau_s * tau_s), rsum * (tau_p * tau_p)).to(torch.float32)
                                    sigma_extra_var = extra if sigma_extra_var is None else (sigma_extra_var + extra)
                        except Exception:
                            pass
                        nei_idx = getattr(state, "shared_event_latent_inducing_neighbor_idx", None)
                        nei_k = getattr(state, "shared_event_latent_inducing_neighbor_k", None)
                        if isinstance(nei_idx, torch.Tensor) and isinstance(nei_k, torch.Tensor) and b_lat.ndim == 3 and int(b_lat.shape[2]) == 2:
                            # Two modes:
                            # - per-station coefficients (b_lat shape (n_stations, M, 2))
                            # - station-basis coefficients (b_lat shape (R, M, 2) + W_sta shape (n_stations, R))
                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                            # Robust mode selection to avoid OOB indexing:
                            # - If b_lat[0] matches n_stations -> per-station
                            # - Else if W exists and b_lat[0] matches W.shape[1] -> basis
                            # - Else -> disable this contribution (better than crashing CUDA)
                            n_stations_rt = int(getattr(state, "n_stations", 0))
                            is_per_station = (n_stations_rt > 0) and (int(b_lat.shape[0]) == int(n_stations_rt))
                            is_basis = (
                                isinstance(W_sta, torch.Tensor)
                                and W_sta.ndim == 2
                                and int(b_lat.shape[0]) == int(W_sta.shape[1])
                                and int(W_sta.shape[0]) == int(n_stations_rt)
                            )

                            # Defensive: ensure station indices are in range before any index_select.
                            # Avoid CUDA device-side asserts; if violated, skip this nuisance term.
                            ok_sta = True
                            try:
                                if n_stations_rt <= 0:
                                    ok_sta = False
                                elif sta_bi.numel() > 0:
                                    mn = int(sta_bi.min().item())
                                    mx = int(sta_bi.max().item())
                                    if mn < 0 or mx >= n_stations_rt:
                                        ok_sta = False
                            except Exception:
                                ok_sta = False
                            if not ok_sta:
                                delta_b = None
                            else:
                                # Also defensively check neighbor inducing indices against M_total for this parameterization.
                                # This catches mismatches between interpolation NPZ and coefficient tensor shape without crashing CUDA.
                                ok_nei = True
                                try:
                                    M_total_rt = int(b_lat.shape[1])
                                    if M_total_rt <= 0:
                                        ok_nei = False
                                    else:
                                        # sample check on the two endpoint sets (cheap relative to failing later)
                                        idx1 = nei_idx.index_select(0, e1)
                                        idx2 = nei_idx.index_select(0, e2)
                                        mx1 = int(idx1.max().item()) if idx1.numel() > 0 else -1
                                        mx2 = int(idx2.max().item()) if idx2.numel() > 0 else -1
                                        mn1 = int(idx1.min().item()) if idx1.numel() > 0 else 0
                                        mn2 = int(idx2.min().item()) if idx2.numel() > 0 else 0
                                        mx_all = max(mx1, mx2)
                                        mn_all = min(mn1, mn2)
                                        # -1 is allowed padding; anything >= M_total is invalid
                                        if mx_all >= M_total_rt or mn_all < -1:
                                            ok_nei = False
                                except Exception:
                                    ok_nei = False
                                if not ok_nei:
                                    delta_b = None
                                elif is_basis:
                                    # Basis mode: b(s,e,phase) = Σ_r W[s,r] * a_r(e,phase)
                                    #
                                    # Performance note: the naive implementation gathers per-rank coefficients and then
                                    # weights by W, which costs ~O(R) extra work. Since n_stations is small (e.g. ~52),
                                    # it is faster to first compute per-station inducing coefficients via matmul:
                                    #   C_sta[:, j, phase] = W[:, :] @ A[:, j, phase]   (U x M)
                                    # and then do the same gather+weighted-sum as the per-station path.
                                    A = b_lat.to(torch.float32)  # (R,M,2)
                                    R = int(A.shape[0])
                                    M = int(A.shape[1])
                                    # Compress stations in this minibatch
                                    sta_u, sta_inv = torch.unique(sta_bi, sorted=False, return_inverse=True)  # (U,), (B,)
                                    W_u = W_sta.index_select(0, sta_u).to(torch.float32)  # (U,R)
                                    # (U,M) per phase
                                    C_u_P = torch.matmul(W_u, A[:, :, 0])  # (U,M)
                                    C_u_S = torch.matmul(W_u, A[:, :, 1])  # (U,M)

                                    def _recon_endpoint_scalar(C_u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
                                        # C_u: (U,M), returns (B,)
                                        idx = nei_idx.index_select(0, e)  # (B,m)
                                        kk = nei_k.index_select(0, e).to(torch.float32)  # (B,m)
                                        msk = (idx >= 0)
                                        idxc = idx.clamp_min(0)
                                        w = kk * msk.to(torch.float32)  # (B,m)
                                        # Gather per-row station-specific coefficients at neighbor indices:
                                        # (B,m) = C_u[sta_inv[:,None], idxc]
                                        vals = C_u[sta_inv.unsqueeze(1), idxc]
                                        return (vals * w).sum(dim=1)  # (B,)

                                    b1P = _recon_endpoint_scalar(C_u_P, e1)
                                    b2P = _recon_endpoint_scalar(C_u_P, e2)
                                    b1S = _recon_endpoint_scalar(C_u_S, e1)
                                    b2S = _recon_endpoint_scalar(C_u_S, e2)
                                    # Keep endpoint tensors for diagnostics below
                                    b1 = torch.stack([b1P, b1S], dim=1)
                                    b2 = torch.stack([b2P, b2S], dim=1)
                                    dP = (b2P - b1P).to(torch.float32)
                                    dS = (b2S - b1S).to(torch.float32)
                                    delta_b = torch.where(is_s, dS, dP)
                                elif is_per_station:
                                    # Per-station coefficients: compute both channels for all rows (single fused gather).
                                    ev12 = torch.cat([e1, e2], dim=0)
                                    idx12 = nei_idx.index_select(0, ev12)  # (2B,m)
                                    k12 = nei_k.index_select(0, ev12)      # (2B,m)
                                    idx12c = idx12.clamp_min(0)
                                    w12 = k12.to(torch.float32) * (idx12 >= 0).to(torch.float32)

                                    b_lat_sta = b_lat.index_select(0, sta_bi).to(torch.float32)  # (B,M,2)
                                    m = int(idx12c.shape[1])
                                    idx12c3 = idx12c.view(2, -1, m).unsqueeze(-1).expand(-1, -1, -1, 2)  # (2,B,m,2)
                                    g12 = torch.gather(b_lat_sta.unsqueeze(0).expand(2, -1, -1, -1), 2, idx12c3)  # (2,B,m,2)
                                    b12 = (g12 * w12.view(2, -1, m).unsqueeze(-1)).sum(dim=2)  # (2,B,2)
                                    b1 = b12[0]
                                    b2 = b12[1]

                                    d = (b2 - b1)
                                    delta_b = torch.where(is_s, d[:, 1], d[:, 0])
                                else:
                                    delta_b = None
                        else:
                            delta_b = None
                    elif mode == "slowness_inducing_gp":
                        # Station×phase slowness-vector inducing GP (pair-shared approximation):
                        #
                        # For each station×phase, we model a 3D slowness perturbation vector field u(x) (units s/km)
                        # with a Matérn(3/2) kernel in XYZ (km). We store inducing coefficients c such that:
                        #   u(e) ≈ Σ_j k(||x_e - x_u||) * c_u_j
                        # where c has prior c ~ N(0, K_UU^{-1}) and K_UU is the Matérn kernel matrix.
                        #
                        # Under the "pair-shared" approximation for DD pairs separated by <=~1 km, the DD correction is:
                        #   δt_ij ≈ 0.5*(u(e1)+u(e2)) · (x1 - x2)
                        #
                        # IMPORTANT: we use current locations (X_src + ΔX_src) so gradients flow through geometry.
                        try:
                            nei_idx = getattr(state, "shared_event_latent_inducing_neighbor_idx", None)
                            ind_ev = getattr(state, "shared_event_latent_inducing_event_idx", None)
                            U_xyz = getattr(state, "shared_event_latent_inducing_xyz_km", None)
                            fixed_xyz = bool(state.params.get("_shared_event_latent_inducing_fixed_xyz", False))
                            if not isinstance(nei_idx, torch.Tensor):
                                raise RuntimeError("missing inducing_gp neighbor tensors")
                            if fixed_xyz:
                                if not (isinstance(U_xyz, torch.Tensor) and U_xyz.ndim == 2 and int(U_xyz.shape[1]) >= 3):
                                    raise RuntimeError("slowness_inducing_gp fixed_xyz requires shared_event_latent_inducing_xyz_km")
                            else:
                                if not isinstance(ind_ev, torch.Tensor):
                                    raise RuntimeError("missing inducing_gp inducing_event_idx tensor")
                            if not (isinstance(b_lat, torch.Tensor) and b_lat.ndim == 4 and int(b_lat.shape[2]) == 2 and int(b_lat.shape[3]) == 3):
                                raise RuntimeError("invalid slowness_inducing_gp shared_event_latent_b shape")

                            # Current event coords (km) for endpoints and inducing points (subset of events).
                            Xcur = (state.X_src + state.dX_src)[:, :3].to(torch.float32)
                            x1 = Xcur.index_select(0, e1)  # (B,3)
                            x2 = Xcur.index_select(0, e2)  # (B,3)
                            dx = (x1 - x2).to(torch.float32)  # (B,3)
                            dx_norm2 = (dx * dx).sum(dim=1).to(torch.float32)  # (B,)

                            # Neighbor lists per endpoint
                            idx1 = nei_idx.index_select(0, e1)  # (B,m)
                            idx2 = nei_idx.index_select(0, e2)  # (B,m)
                            m1 = (idx1 >= 0)
                            m2 = (idx2 >= 0)
                            idx1c = idx1.clamp_min(0)
                            idx2c = idx2.clamp_min(0)

                            # Kernel weights: Matérn ν=3/2 in XYZ at current positions.
                            ell = float(state.params.get("_shared_event_latent_ell_km", 0.0))
                            if not (ell > 0.0):
                                raise RuntimeError("slowness_inducing_gp requires ell_km > 0")
                            a = float(np.sqrt(3.0) / float(ell))

                            def _weights(endpoint_x: torch.Tensor, idxc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                                # endpoint_x: (B,3), idxc: (B,m) global inducing idx, mask: (B,m)
                                if fixed_xyz and isinstance(U_xyz, torch.Tensor):
                                    u_xyz = U_xyz.index_select(0, idxc.reshape(-1)).reshape(idxc.shape[0], idxc.shape[1], 3)  # (B,m,3)
                                else:
                                    u_ev = ind_ev.index_select(0, idxc.reshape(-1)).reshape(idxc.shape)  # (B,m) event ids
                                    u_xyz = Xcur.index_select(0, u_ev.reshape(-1)).reshape(idxc.shape[0], idxc.shape[1], 3)  # (B,m,3)
                                d = torch.linalg.norm(endpoint_x.unsqueeze(1) - u_xyz, dim=2).to(torch.float32)  # (B,m)
                                x = (a * d).to(torch.float32)
                                w = (1.0 + x) * torch.exp(-x)
                                return w * mask.to(torch.float32)

                            w1 = _weights(x1, idx1c, m1)  # (B,m)
                            w2 = _weights(x2, idx2c, m2)  # (B,m)

                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                            n_stations_rt = int(getattr(state, "n_stations", 0))
                            is_per_station = (n_stations_rt > 0) and (int(b_lat.shape[0]) == int(n_stations_rt))
                            is_basis = (
                                isinstance(W_sta, torch.Tensor)
                                and W_sta.ndim == 2
                                and int(W_sta.shape[0]) == int(n_stations_rt)
                                and int(b_lat.shape[0]) == int(W_sta.shape[1])
                            )
                            if (not is_per_station) and (not is_basis):
                                raise RuntimeError("slowness_inducing_gp: expected per-station or station-basis coefficients")

                            # Gather coefficients at neighbor indices, returning (B,m,2,3) for each endpoint.
                            if is_per_station:
                                # IMPORTANT: avoid advanced indexing b_lat[sta, idx] here.
                                # On large batches this can be extremely slow due to non-coalesced gather kernels.
                                # Instead, flatten (station, inducing_idx) into a single axis and use index_select.
                                try:
                                    S = int(b_lat.shape[0])
                                    M_total = int(b_lat.shape[1])
                                    flat = b_lat.reshape(S * M_total, 2, 3).to(torch.float32)  # (S*M,2,3)
                                    sta0 = sta_bi.to(torch.int64).clamp_min(0).clamp_max(max(S - 1, 0))
                                    lin1 = (sta0.unsqueeze(1) * M_total + idx1c.to(torch.int64)).reshape(-1)
                                    lin2 = (sta0.unsqueeze(1) * M_total + idx2c.to(torch.int64)).reshape(-1)
                                    C1 = flat.index_select(0, lin1).reshape(idx1c.shape[0], idx1c.shape[1], 2, 3)  # (B,m,2,3)
                                    C2 = flat.index_select(0, lin2).reshape(idx2c.shape[0], idx2c.shape[1], 2, 3)  # (B,m,2,3)
                                except Exception:
                                    # Fallback (should be rare): keep old behavior.
                                    C1 = b_lat[sta_bi.unsqueeze(1), idx1c].to(torch.float32)  # (B,m,2,3)
                                    C2 = b_lat[sta_bi.unsqueeze(1), idx2c].to(torch.float32)  # (B,m,2,3)
                            else:
                                # Basis: reconstruct station-specific coefficients by weighting basis ranks.
                                # A6: (R,M,6) where last dim packs phase×xyz.
                                A = b_lat.to(torch.float32)
                                R = int(A.shape[0]); M = int(A.shape[1])
                                A6 = A.reshape(R, M, 6)
                                Wb = W_sta.index_select(0, sta_bi).to(torch.float32)  # (B,R)
                                Wr = Wb.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # (R,B,1,1)

                                def _gather_station_coeffs(idxc: torch.Tensor) -> torch.Tensor:
                                    # idxc: (B,m) -> (B,m,2,3)
                                    B0, m0 = int(idxc.shape[0]), int(idxc.shape[1])
                                    Aexp = A6.unsqueeze(1).expand(R, B0, M, 6)  # (R,B,M,6)
                                    idxe = idxc.unsqueeze(0).unsqueeze(-1).expand(R, B0, m0, 6)  # (R,B,m,6)
                                    g = torch.gather(Aexp, 2, idxe)  # (R,B,m,6)
                                    g = g.reshape(R, B0, m0, 2, 3)
                                    return (g * Wr.unsqueeze(-1)).sum(dim=0)  # (B,m,2,3)

                                C1 = _gather_station_coeffs(idx1c)
                                C2 = _gather_station_coeffs(idx2c)

                            # Interpolate endpoint u vectors (B,2,3)
                            u1 = (C1 * w1.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (B,2,3)
                            u2 = (C2 * w2.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (B,2,3)
                            u_avg = 0.5 * (u1 + u2)  # (B,2,3)

                            # Convert to time using configured units for tau:
                            # - abs: u_avg is interpreted as slowness vector (s/km) -> seconds = u·dx
                            # - vel_frac: u_avg is interpreted as dimensionless ε (≈ δv/v) -> seconds ≈ (ε·dx) / v(z)
                            units = str(state.params.get("_shared_event_latent_slowness_tau_units", "abs")).strip().lower()
                            if units == "vel_frac":
                                # Lookup vP(z), vS(z) via *GPU* linear interpolation over depth centers.
                                #
                                # IMPORTANT: avoid GPU->CPU numpy round-trips here; with large batches this
                                # can dominate runtime and cause DDP/NCCL timeouts (one rank finishes much later).
                                #
                                # We cache torch tensors in `state.params` so this setup happens once per process.
                                zavg = (0.5 * (x1[:, 2] + x2[:, 2])).to(torch.float32)  # (B,)

                                def _get_v1d_tensors() -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
                                    # Cached tensors
                                    zc_t = state.params.get("_runtime_eikonet_v1d_z_cent_km_t", None)
                                    vp_t = state.params.get("_runtime_eikonet_v1d_vp_km_s_t", None)
                                    vs_t = state.params.get("_runtime_eikonet_v1d_vs_km_s_t", None)
                                    dev0 = dx.device
                                    try:
                                        if (
                                            isinstance(zc_t, torch.Tensor)
                                            and isinstance(vp_t, torch.Tensor)
                                            and isinstance(vs_t, torch.Tensor)
                                            and zc_t.ndim == 1
                                            and vp_t.ndim == 1
                                            and vs_t.ndim == 1
                                            and int(zc_t.numel()) == int(vp_t.numel()) == int(vs_t.numel())
                                            and int(zc_t.numel()) >= 2
                                            and zc_t.device == dev0
                                            and vp_t.device == dev0
                                            and vs_t.device == dev0
                                        ):
                                            return zc_t, vp_t, vs_t
                                    except Exception:
                                        pass

                                    # Build from config arrays (usually stored by EikoNet loading)
                                    try:
                                        z_cent = np.asarray(state.params.get("_eikonet_v1d_depth_centers_km", []), dtype=np.float32).reshape(-1)
                                        vp = np.asarray(state.params.get("_eikonet_v1d_vp_km_s", []), dtype=np.float32).reshape(-1)
                                        vs = np.asarray(state.params.get("_eikonet_v1d_vs_km_s", []), dtype=np.float32).reshape(-1)
                                    except Exception:
                                        z_cent = np.zeros((0,), dtype=np.float32)
                                        vp = np.zeros((0,), dtype=np.float32)
                                        vs = np.zeros((0,), dtype=np.float32)
                                    if z_cent.size < 2 or vp.size != z_cent.size or vs.size != z_cent.size:
                                        return None, None, None

                                    zc = torch.from_numpy(z_cent).to(device=dev0, dtype=torch.float32)
                                    vp0 = torch.from_numpy(vp).to(device=dev0, dtype=torch.float32)
                                    vs0 = torch.from_numpy(vs).to(device=dev0, dtype=torch.float32)
                                    # Enforce sorted z-centers for bucketize/searchsorted
                                    try:
                                        if not bool(torch.all(zc[1:] >= zc[:-1]).item()):
                                            perm = torch.argsort(zc)
                                            zc = zc.index_select(0, perm)
                                            vp0 = vp0.index_select(0, perm)
                                            vs0 = vs0.index_select(0, perm)
                                    except Exception:
                                        pass
                                    # Replace any invalid values with reasonable fallbacks (keeps interpolation stable)
                                    vp0 = torch.where(torch.isfinite(vp0) & (vp0 > 0.0), vp0, torch.full_like(vp0, 6.0))
                                    vs0 = torch.where(torch.isfinite(vs0) & (vs0 > 0.0), vs0, torch.full_like(vs0, 3.5))

                                    # Cache for future batches
                                    state.params["_runtime_eikonet_v1d_z_cent_km_t"] = zc
                                    state.params["_runtime_eikonet_v1d_vp_km_s_t"] = vp0
                                    state.params["_runtime_eikonet_v1d_vs_km_s_t"] = vs0
                                    return zc, vp0, vs0

                                def _interp_torch(zq: torch.Tensor, zc: torch.Tensor, vv: torch.Tensor, v_fallback: float) -> torch.Tensor:
                                    # zq: (B,), zc/vv: (K,) sorted ascending. Clamp to endpoints outside range.
                                    K = int(zc.numel())
                                    if K < 2:
                                        return torch.full_like(zq, float(v_fallback), dtype=torch.float32)
                                    idx = torch.bucketize(zq, zc)  # 0..K
                                    idx0 = (idx - 1).clamp(min=0, max=K - 2)
                                    idx1 = idx0 + 1
                                    z0 = zc.index_select(0, idx0)
                                    z1 = zc.index_select(0, idx1)
                                    v0 = vv.index_select(0, idx0)
                                    v1 = vv.index_select(0, idx1)
                                    denom = (z1 - z0).clamp_min(1e-12)
                                    w = ((zq - z0) / denom).clamp(0.0, 1.0)
                                    out = v0 + w * (v1 - v0)
                                    out = torch.where(torch.isfinite(out) & (out > 0.0), out, torch.full_like(out, float(v_fallback)))
                                    return out.to(torch.float32)

                                zc, vp0, vs0 = _get_v1d_tensors()
                                if zc is None or vp0 is None or vs0 is None:
                                    vP = torch.full_like(zavg, 6.0, dtype=torch.float32)
                                    vS = torch.full_like(zavg, 3.5, dtype=torch.float32)
                                else:
                                    vP = _interp_torch(zavg, zc, vp0, 6.0)
                                    vS = _interp_torch(zavg, zc, vs0, 3.5)
                                vP = vP.clamp_min(1e-6)
                                vS = vS.clamp_min(1e-6)

                                # Optional Stage-5 FITC diagonal correction (variance inflation).
                                # Var(0.5(u1+u2)·dx / v)^approx ≈ 0.25*(r1+r2) * ||dx||^2 * tau^2 / v^2
                                try:
                                    if bool(state.params.get("_shared_event_latent_inducing_fitc_enable", False)):
                                        resid = getattr(state, "shared_event_latent_inducing_fitc_resid", None)
                                        if isinstance(resid, torch.Tensor) and resid.ndim == 1:
                                            r1 = resid.index_select(0, e1).to(torch.float32)
                                            r2 = resid.index_select(0, e2).to(torch.float32)
                                            rsum = (0.25 * (r1 + r2)).clamp_min(0.0)
                                            tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                                            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                                            extraP = rsum * dx_norm2 * (tau_p * tau_p) / (vP * vP)
                                            extraS = rsum * dx_norm2 * (tau_s * tau_s) / (vS * vS)
                                            extra = torch.where(is_s, extraS, extraP).to(torch.float32)
                                            sigma_extra_var = extra if sigma_extra_var is None else (sigma_extra_var + extra)
                                except Exception:
                                    pass

                                dP = (u_avg[:, 0, :] * dx).sum(dim=1) / vP
                                dS = (u_avg[:, 1, :] * dx).sum(dim=1) / vS
                                delta_b = torch.where(is_s, dS, dP)
                            else:
                                # Dot with event separation vector: seconds = (s/km)·(km)
                                # Optional Stage-5 FITC diagonal correction (variance inflation).
                                # Var(0.5(u1+u2)·dx)^approx ≈ 0.25*(r1+r2) * ||dx||^2 * tau^2
                                try:
                                    if bool(state.params.get("_shared_event_latent_inducing_fitc_enable", False)):
                                        resid = getattr(state, "shared_event_latent_inducing_fitc_resid", None)
                                        if isinstance(resid, torch.Tensor) and resid.ndim == 1:
                                            r1 = resid.index_select(0, e1).to(torch.float32)
                                            r2 = resid.index_select(0, e2).to(torch.float32)
                                            rsum = (0.25 * (r1 + r2)).clamp_min(0.0)
                                            tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                                            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                                            extraP = rsum * dx_norm2 * (tau_p * tau_p)
                                            extraS = rsum * dx_norm2 * (tau_s * tau_s)
                                            extra = torch.where(is_s, extraS, extraP).to(torch.float32)
                                            sigma_extra_var = extra if sigma_extra_var is None else (sigma_extra_var + extra)
                                except Exception:
                                    pass
                                dP = (u_avg[:, 0, :] * dx).sum(dim=1)
                                dS = (u_avg[:, 1, :] * dx).sum(dim=1)
                                delta_b = torch.where(is_s, dS, dP)

                            # Diagnostics: log physical u-field magnitudes at endpoints (per-phase).
                            # This is much more interpretable than raw inducing coefficients, especially in the dual/K^{-1} parameterization.
                            try:
                                if want_lat_diag:
                                    u1P = u1[:, 0, :]
                                    u2P = u2[:, 0, :]
                                    u1S = u1[:, 1, :]
                                    u2S = u2[:, 1, :]
                                    # per-row endpoint norms
                                    n1P = torch.linalg.norm(u1P, dim=1).to(torch.float32)
                                    n2P = torch.linalg.norm(u2P, dim=1).to(torch.float32)
                                    n1S = torch.linalg.norm(u1S, dim=1).to(torch.float32)
                                    n2S = torch.linalg.norm(u2S, dim=1).to(torch.float32)
                                    is_p_f = (~is_s).to(torch.float32)
                                    is_s_f = is_s.to(torch.float32)

                                    if u_end_sumsq_p is None:
                                        u_end_sumsq_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_sumsq_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_count_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_count_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_maxnorm_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_maxnorm_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)

                                    u_end_sumsq_p = u_end_sumsq_p + ((n1P * n1P + n2P * n2P) * is_p_f).sum()
                                    u_end_sumsq_s = u_end_sumsq_s + ((n1S * n1S + n2S * n2S) * is_s_f).sum()
                                    u_end_count_p = u_end_count_p + (2.0 * is_p_f.sum())
                                    u_end_count_s = u_end_count_s + (2.0 * is_s_f.sum())
                                    # Max endpoint norm (helps detect a small number of pathological events)
                                    u_end_maxnorm_p = torch.maximum(u_end_maxnorm_p, torch.maximum(n1P, n2P).max())
                                    u_end_maxnorm_s = torch.maximum(u_end_maxnorm_s, torch.maximum(n1S, n2S).max())
                            except Exception:
                                pass
                        except Exception:
                            delta_b = None
                    else:
                        # Explicit event-latent parameterization (full / graph_gmrf):
                        # - per-station: b_lat shape (n_stations, n_events, 2)
                        # - station-basis: b_lat shape (R, n_events, 2) with W_sta (n_stations, R)
                        if b_lat.ndim == 3 and int(b_lat.shape[2]) == 2:
                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                            n_stations_rt = int(getattr(state, "n_stations", 0))
                            is_per_station = (n_stations_rt > 0) and (int(b_lat.shape[0]) == int(n_stations_rt))
                            is_basis = (
                                isinstance(W_sta, torch.Tensor)
                                and W_sta.ndim == 2
                                and int(W_sta.shape[0]) == int(n_stations_rt)
                                and int(b_lat.shape[0]) == int(W_sta.shape[1])
                            )

                            if is_basis:
                                # b(s,e,phase) = Σ_r W[s,r] * a_r(e,phase)
                                A = b_lat.to(torch.float32)  # (R,Ne,2)
                                W = W_sta.index_select(0, sta_bi).to(torch.float32)  # (B,R)
                                # Gather event endpoints in basis space: (R,B,2) -> (B,R,2)
                                A1 = A.index_select(1, e1).transpose(0, 1).contiguous()
                                A2 = A.index_select(1, e2).transpose(0, 1).contiguous()
                                b1 = (W.unsqueeze(-1) * A1).sum(dim=1)  # (B,2)
                                b2 = (W.unsqueeze(-1) * A2).sum(dim=1)  # (B,2)
                                d = (b2 - b1).to(torch.float32)
                                delta_b = torch.where(is_s, d[:, 1], d[:, 0])
                            elif is_per_station:
                                # Direct per-station coefficients
                                b1 = b_lat[sta_bi, e1]  # (B,2)
                                b2 = b_lat[sta_bi, e2]  # (B,2)
                                d = (b2 - b1).to(torch.float32)
                                delta_b = torch.where(is_s, d[:, 1], d[:, 0])
                            else:
                                delta_b = None
                        else:
                            delta_b = None

                    if isinstance(delta_b, torch.Tensor):
                        nuisance_delta = delta_b if nuisance_delta is None else (nuisance_delta + delta_b)
                        if want_lat_diag:
                            if b_delta_sum is None:
                                b_delta_sum = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_delta_sumsq = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_delta_maxabs = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_delta_count = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_end_sumsq_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_end_sumsq_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_end_count_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_end_count_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                            b_delta_sum = b_delta_sum + delta_b.sum()
                            b_delta_sumsq = b_delta_sumsq + (delta_b * delta_b).sum()
                            b_delta_maxabs = torch.maximum(b_delta_maxabs, delta_b.abs().max())
                            b_delta_count = b_delta_count + float(delta_b.numel())
                            # Endpoint RMS diagnostics:
                            # For each DD row, we can also track the RMS of the *event-level* latent values at both endpoints.
                            # This remains meaningful for both parameterizations:
                            #   - full: bP1/bP2/bS1/bS2 are direct event-level values
                            #   - inducing_gp: b1P/b2P/b1S/b2S are reconstructed event-level values via interpolation
                            try:
                                _bP1 = b1[:, 0].to(torch.float32)
                                _bP2 = b2[:, 0].to(torch.float32)
                                _bS1 = b1[:, 1].to(torch.float32)
                                _bS2 = b2[:, 1].to(torch.float32)
                                is_p_f = (~is_s).to(torch.float32)
                                is_s_f = is_s.to(torch.float32)
                                b_end_sumsq_p = b_end_sumsq_p + ((_bP1 * _bP1 + _bP2 * _bP2) * is_p_f).sum()
                                b_end_sumsq_s = b_end_sumsq_s + ((_bS1 * _bS1 + _bS2 * _bS2) * is_s_f).sum()
                                # Each row contributes two endpoints
                                b_end_count_p = b_end_count_p + (2.0 * is_p_f.sum())
                                b_end_count_s = b_end_count_s + (2.0 * is_s_f.sum())
                            except Exception:
                                pass
        except Exception:
            pass
        finally:
            if prof_se_lat and (se_lat_t0 is not None):
                try:
                    import time as _time
                    if state.device.type == "cuda":
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    dt_ms = 1000.0 * float(_time.perf_counter() - se_lat_t0)
                    # Accumulate per-epoch; emit once at end like other per-epoch metrics.
                    if "_se_lat_time_ms_sum" not in state.params:
                        state.params["_se_lat_time_ms_sum"] = 0.0
                        state.params["_se_lat_time_ms_count"] = 0
                    state.params["_se_lat_time_ms_sum"] = float(state.params["_se_lat_time_ms_sum"]) + float(dt_ms)
                    state.params["_se_lat_time_ms_count"] = int(state.params["_se_lat_time_ms_count"]) + 1
                except Exception:
                    pass

        # sigma_inflation removed (start fresh).

        # Optional: corr_error latent contribution to likelihood.
        # Supports:
        #  - low-rank station basis W (n_stations,R) with b (n_events,R,2)
        #  - per-station coefficients (station_basis.enabled=false): b (n_events,n_stations,2) and we index by sta_idx
        try:
            if bool(state.params.get("_corr_error_enabled", False)):
                W = getattr(state, "corr_error_station_basis_W", None)
                b = getattr(state, "corr_error_b", None)
                sta_b = state.params.get("_runtime_bucket_station_index", None)
                if isinstance(b, torch.Tensor) and isinstance(sta_b, torch.Tensor):
                    e1 = II_b[:, 0].to(torch.int64)
                    e2 = II_b[:, 1].to(torch.int64)
                    bi = b.index_select(0, e1)  # (B,R,2)
                    bj = b.index_select(0, e2)
                    db = (bj - bi).to(torch.float32)
                    ph = YY_b[:, 4]
                    is_s = (ph >= 0.5)
                    db_phase = torch.where(is_s.view(-1, 1), db[:, :, 1], db[:, :, 0])  # (B,R)
                    if isinstance(W, torch.Tensor):
                        Wr = W.index_select(0, sta_b.to(torch.int64))  # (B,R)
                        delta_corr = (Wr * db_phase).sum(dim=1)  # (B,)
                    else:
                        # Per-station coefficients: pick the coefficient corresponding to sta_idx.
                        sidx = sta_b.to(torch.int64).view(-1, 1)
                        delta_corr = db_phase.gather(1, sidx).squeeze(1)
                    nuisance_delta = delta_corr if nuisance_delta is None else (nuisance_delta + delta_corr)
                    # Per-epoch stats for the *actual* correction delta_corr (split by phase).
                    # Keep these as scalar tensors so we can all-reduce in DDP.
                    dc = delta_corr.detach().to(torch.float32)
                    if corr_dc_sumsq_p is None:
                        z = torch.zeros((), device=dc.device, dtype=torch.float32)
                        corr_dc_sumsq_p = z.clone()
                        corr_dc_sumsq_s = z.clone()
                        corr_dc_count_p = z.clone()
                        corr_dc_count_s = z.clone()
                        corr_dc_maxabs_p = z.clone()
                        corr_dc_maxabs_s = z.clone()
                    try:
                        if (~is_s).any():
                            dc_p = dc[~is_s]
                            corr_dc_sumsq_p = corr_dc_sumsq_p + (dc_p * dc_p).sum()
                            corr_dc_count_p = corr_dc_count_p + float(dc_p.numel())
                            corr_dc_maxabs_p = torch.maximum(corr_dc_maxabs_p, torch.max(torch.abs(dc_p)))
                        if is_s.any():
                            dc_s = dc[is_s]
                            corr_dc_sumsq_s = corr_dc_sumsq_s + (dc_s * dc_s).sum()
                            corr_dc_count_s = corr_dc_count_s + float(dc_s.numel())
                            corr_dc_maxabs_s = torch.maximum(corr_dc_maxabs_s, torch.max(torch.abs(dc_s)))
                    except Exception:
                        pass
        except Exception:
            pass

        b_corr_for_prior = (getattr(state, "corr_error_b", None) if bool(state.params.get("_corr_error_enabled", False)) else None)
        if ddp_enabled:
            # DDP-safe loss construction:
            # - Likelihood term is the *global batch mean* across all ranks: (1/B) Σ_i NLL_i.
            #   Each rank computes a local mean and scales by (B_r / B).
            # - Prior term is already scaled by 1/N_total in compute_prior_loss(); we want it included once,
            #   so we add it as (1/world_size) per rank.
            try:
                local_bsz = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
            except Exception:
                local_bsz = 0
            try:
                B_global = int(global_bsz) if "global_bsz" in locals() else int(local_bsz)
            except Exception:
                B_global = int(local_bsz)
            B_global = max(1, int(B_global))

            if local_bsz > 0:
                # Optional profiling: time the shared_event_re likelihood computation.
                try:
                    diag = _get_diagnostics_cfg(state.params)
                    prof_se_re = bool(diag.get("profile_shared_event_re", False)) if isinstance(diag, dict) else False
                except Exception:
                    prof_se_re = False
                se_re_t0 = None
                sl_re_t0 = None
                try:
                    diag = _get_diagnostics_cfg(state.params)
                    prof_sl_re = bool(diag.get("profile_slowness_re", False)) if isinstance(diag, dict) else False
                except Exception:
                    prof_sl_re = False
                if prof_se_re and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        se_re_t0 = _time.perf_counter()
                    except Exception:
                        se_re_t0 = None
                if prof_sl_re and bool(state.params.get("_slowness_re_enabled", False)):
                    try:
                        import time as _time
                        sl_re_t0 = _time.perf_counter()
                    except Exception:
                        sl_re_t0 = None
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_prelog", False)):
                    state.params["_shared_event_re_prelog"] = True
                    try:
                        n_rows = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
                        solver = str(state.params.get("_shared_event_re_solver", ""))
                        grouping = str(state.params.get("_shared_event_re_grouping", ""))
                        # Sentinels to verify shared_event_re block executed.
                        state.params["_shared_event_re_runtime_last_groups"] = -1
                        state.params["_shared_event_re_runtime_last_groups_pcg"] = -1
                        state.params["_shared_event_re_runtime_last_groups_fallback_diag"] = -1
                        info(
                            f"shared_event_re prelog rows={n_rows} solver={solver} grouping={grouping}",
                            section="LIKELIHOOD",
                        )
                    except Exception:
                        pass
                loss_like = compute_likelihood_loss(
                    idx=II_b,
                    y=YY_b,
                    X_src=state.X_src,
                    ΔX_src=state.dX_src,
                    model=state.model,
                    σ_p=σp,
                    σ_s=σs,
                    params=state.params,
                    nuisance_delta=nuisance_delta,
                    sigma_extra_var=sigma_extra_var,
                )
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_postlog", False)):
                    state.params["_shared_event_re_postlog"] = True
                    g = int(state.params.get("_shared_event_re_runtime_last_groups", -1))
                    g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", -1))
                    g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", -1))
                    info(
                        f"shared_event_re postlog groups={g} pcg={g_pcg} fallback={g_fb}",
                        section="LIKELIHOOD",
                    )
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_batch_logged", False)):
                    state.params["_shared_event_re_batch_logged"] = True
                    try:
                        n_rows = int(II_b.shape[0])
                        info(f"shared_event_re batch rows={n_rows}", section="LIKELIHOOD")
                        if n_rows > 0:
                            ph_id = torch.where(
                                YY_b[:, 4] < 0.5,
                                torch.zeros_like(YY_b[:, 4], dtype=torch.int64),
                                torch.ones_like(YY_b[:, 4], dtype=torch.int64),
                            )
                            grouping = str(state.params.get("_shared_event_re_grouping", "phase")).strip().lower()
                            if grouping in {"stationphase", "station-phase"}:
                                grouping = "station_phase"
                            if grouping == "station_phase":
                                sta_idx = state.params.get("_runtime_bucket_station_index", None)
                                if isinstance(sta_idx, torch.Tensor) and int(sta_idx.numel()) == int(ph_id.numel()):
                                    keys = (sta_idx.to(dtype=torch.int64) * 2) + ph_id
                                else:
                                    keys = ph_id
                                    grouping = "phase"
                            else:
                                keys = ph_id
                            keys_sorted, _ = torch.sort(keys)
                            if keys_sorted.numel() > 0:
                                is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
                                is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
                                n_groups = int(torch.nonzero(is_new, as_tuple=False).shape[0])
                            else:
                                n_groups = 0
                            info(f"shared_event_re batch grouping={grouping} groups={n_groups}", section="LIKELIHOOD")
                    except Exception as e:
                        info(f"shared_event_re batch log failed: {e}", section="LIKELIHOOD")
                if not bool(state.params.get("_shared_event_re_flag_logged", False)):
                    state.params["_shared_event_re_flag_logged"] = True
                    info(
                        f"shared_event_re flag={bool(state.params.get('_shared_event_re_enabled', False))}",
                        section="LIKELIHOOD",
                    )
                # One-time shared_event_re runtime report (from modeling.py stats).
                if bool(state.params.get("_shared_event_re_enabled", False)):
                    g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                    g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                    g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                    g_rows = int(state.params.get("_shared_event_re_runtime_last_groups_rows_cap", 0) or 0)
                    g_nodes = int(state.params.get("_shared_event_re_runtime_last_groups_nodes_cap", 0) or 0)
                    g_tau0 = int(state.params.get("_shared_event_re_runtime_last_groups_tau_zero", 0) or 0)
                    mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                    mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                    mr_all = int(state.params.get("_shared_event_re_runtime_last_max_rows_all", 0) or 0)
                    mn_all = int(state.params.get("_shared_event_re_runtime_last_max_nodes_all", 0) or 0)
                    tg = state.params.get("_shared_event_re_tau_s", [0.0, 0.0])
                    log_every = int(state.params.get("_shared_event_re_stats_log_every_epochs", 0) or 0)
                    if log_every > 0 and (epoch_index % log_every == 0):
                        grouping = str(state.params.get("_shared_event_re_runtime_last_grouping", "phase"))
                        info(
                            f"shared_event_re stats epoch={epoch_index} grouping={grouping} "
                            f"groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"rows_cap={g_rows} nodes_cap={g_nodes} tau0={g_tau0} "
                            f"max_rows={mr} max_nodes={mn} max_rows_all={mr_all} max_nodes_all={mn_all}",
                            section="LIKELIHOOD",
                        )
                    if not bool(state.params.get("_shared_event_re_reported", False)):
                        state.params["_shared_event_re_reported"] = True
                        info(
                            f"shared_event_re stats groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"max_rows={mr} max_nodes={mn} tau_s={tg}",
                            section="LIKELIHOOD",
                        )
                    if g_fb > 0:
                        info(
                            f"shared_event_re fallback reasons rows_cap={g_rows} nodes_cap={g_nodes} tau_zero={g_tau0}",
                            section="LIKELIHOOD",
                        )
                        if (not ddp_enabled) or ddp_is_main:
                            if bool(state.params.get("_shared_event_re_auto_tune_nodes_cap", False)) and (g_nodes > 0):
                                cur = int(state.params.get("_shared_event_re_max_nodes_per_group", 0) or 0)
                                cap = int(state.params.get("_shared_event_re_auto_tune_nodes_max", 0) or 0)
                                target = int(min(max(mn_all, cur), cap)) if cap > 0 else int(max(mn_all, cur))
                                if target > cur and mn_all > 0:
                                    state.params["_shared_event_re_max_nodes_per_group"] = target
                                    info(
                                        f"shared_event_re auto-tune: max_nodes_per_group -> {target}",
                                        section="LIKELIHOOD",
                                    )
                            if bool(state.params.get("_shared_event_re_auto_tune_rows_cap", False)) and (g_rows > 0):
                                cur = int(state.params.get("_shared_event_re_max_rows_per_group", 0) or 0)
                                cap = int(state.params.get("_shared_event_re_auto_tune_rows_max", 0) or 0)
                                target = int(min(max(mr_all, cur), cap)) if cap > 0 else int(max(mr_all, cur))
                                if target > cur and mr_all > 0:
                                    state.params["_shared_event_re_max_rows_per_group"] = target
                                    info(
                                        f"shared_event_re auto-tune: max_rows_per_group -> {target}",
                                        section="LIKELIHOOD",
                                    )
                    if g > 0 and g_pcg == 0 and g_fb >= g:
                        info(
                            "shared_event_re warning: all groups fell back to diagonal; "
                            "increase shared_event_re.max_nodes_per_group or max_rows_per_group",
                            section="LIKELIHOOD",
                        )
                if bool(state.params.get("_shared_event_re_whitening_enabled", False)) and not bool(state.params.get("_shared_event_re_whitening_reported", False)):
                    state.params["_shared_event_re_whitening_reported"] = True
                    wg = int(state.params.get("_shared_event_re_whitening_last_groups", 0) or 0)
                    wchol = int(state.params.get("_shared_event_re_whitening_last_groups_chol", 0) or 0)
                    wfb = int(state.params.get("_shared_event_re_whitening_last_groups_fallback_diag", 0) or 0)
                    wrows = int(state.params.get("_shared_event_re_whitening_last_groups_rows_cap", 0) or 0)
                    wnodes = int(state.params.get("_shared_event_re_whitening_last_groups_nodes_cap", 0) or 0)
                    wtau0 = int(state.params.get("_shared_event_re_whitening_last_groups_tau_zero", 0) or 0)
                    wmr = int(state.params.get("_shared_event_re_whitening_last_max_rows", 0) or 0)
                    wmn = int(state.params.get("_shared_event_re_whitening_last_max_nodes", 0) or 0)
                    info(
                        f"shared_event_re whitening groups={wg} chol={wchol} fallback={wfb} max_rows={wmr} max_nodes={wmn}",
                        section="LIKELIHOOD",
                    )
                    if wfb > 0:
                        info(
                            f"shared_event_re whitening fallback reasons rows_cap={wrows} nodes_cap={wnodes} tau_zero={wtau0}",
                            section="LIKELIHOOD",
                        )
                # One-time debug: compare shared_event_re vs baseline loss on this batch.
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_debug_compare_logged", False)):
                    state.params["_shared_event_re_debug_compare_logged"] = True
                    try:
                        with torch.no_grad():
                            params_dbg = dict(state.params)
                            params_dbg["_shared_event_re_enabled"] = False
                            loss_base = compute_likelihood_loss(
                                idx=II_b,
                                y=YY_b,
                                X_src=state.X_src,
                                ΔX_src=state.dX_src,
                                model=state.model,
                                σ_p=σp,
                                σ_s=σs,
                                params=params_dbg,
                                nuisance_delta=nuisance_delta,
                                sigma_extra_var=sigma_extra_var,
                            )
                        delta = float((loss_like - loss_base).detach().item())
                        print(f"[shared_event_re] loss delta vs baseline = {delta:.6e}", flush=True)
                    except Exception as e:
                        print(f"[shared_event_re] debug compare failed: {e}", flush=True)
                # One-time debug: compare slowness_re vs baseline loss on this batch.
                if bool(state.params.get("_slowness_re_enabled", False)) and not bool(state.params.get("_slowness_re_debug_compare_logged", False)):
                    state.params["_slowness_re_debug_compare_logged"] = True
                    try:
                        with torch.no_grad():
                            params_dbg = dict(state.params)
                            params_dbg["_slowness_re_enabled"] = False
                            loss_base = compute_likelihood_loss(
                                idx=II_b,
                                y=YY_b,
                                X_src=state.X_src,
                                ΔX_src=state.dX_src,
                                model=state.model,
                                σ_p=σp,
                                σ_s=σs,
                                params=params_dbg,
                                nuisance_delta=nuisance_delta,
                                sigma_extra_var=sigma_extra_var,
                            )
                        delta = float((loss_like - loss_base).detach().item())
                        print(f"[slowness_re] loss delta vs baseline = {delta:.6e}", flush=True)
                    except Exception as e:
                        print(f"[slowness_re] debug compare failed: {e}", flush=True)
                # One-time shared_event_re runtime summary after the first loss call.
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_runtime_logged", False)):
                    state.params["_shared_event_re_runtime_logged"] = True
                    try:
                        g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                        g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                        g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                        print(
                            f"[shared_event_re] runtime groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"max_rows={mr} max_nodes={mn}",
                            flush=True,
                        )
                    except Exception:
                        pass
                # One-time slowness_re runtime summary after the first loss call.
                if bool(state.params.get("_slowness_re_enabled", False)) and not bool(state.params.get("_slowness_re_runtime_logged", False)):
                    state.params["_slowness_re_runtime_logged"] = True
                    try:
                        g = int(state.params.get("_slowness_re_runtime_last_groups", 0) or 0)
                        g_pcg = int(state.params.get("_slowness_re_runtime_last_groups_woodbury", 0) or 0)
                        g_fb = int(state.params.get("_slowness_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_slowness_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_slowness_re_runtime_last_max_nodes", 0) or 0)
                        print(
                            f"[slowness_re] runtime groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"max_rows={mr} max_nodes={mn}",
                            flush=True,
                        )
                    except Exception:
                        pass
                if prof_se_re and (se_re_t0 is not None) and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - se_re_t0)
                        state.params["_se_re_time_ms_sum"] = float(state.params.get("_se_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        state.params["_se_re_time_ms_count"] = int(state.params.get("_se_re_time_ms_count", 0) or 0) + 1
                        # Workload stats from modeling.py (set per-call)
                        g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                        g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                        g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                        state.params["_se_re_groups_sum"] = int(state.params.get("_se_re_groups_sum", 0) or 0) + g
                        state.params["_se_re_groups_pcg_sum"] = int(state.params.get("_se_re_groups_pcg_sum", 0) or 0) + g_pcg
                        state.params["_se_re_groups_fallback_sum"] = int(state.params.get("_se_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_se_re_max_rows_max"] = max(int(state.params.get("_se_re_max_rows_max", 0) or 0), mr)
                        state.params["_se_re_max_nodes_max"] = max(int(state.params.get("_se_re_max_nodes_max", 0) or 0), mn)
                    except Exception:
                        pass
                if prof_sl_re and (sl_re_t0 is not None) and bool(state.params.get("_slowness_re_enabled", False)):
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - sl_re_t0)
                        state.params["_slowness_re_time_ms_sum"] = float(state.params.get("_slowness_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        state.params["_slowness_re_time_ms_count"] = int(state.params.get("_slowness_re_time_ms_count", 0) or 0) + 1
                        # Workload stats from modeling.py (set per-call)
                        g = int(state.params.get("_slowness_re_runtime_last_groups", 0) or 0)
                        g_w = int(state.params.get("_slowness_re_runtime_last_groups_woodbury", 0) or 0)
                        g_fb = int(state.params.get("_slowness_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_slowness_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_slowness_re_runtime_last_max_nodes", 0) or 0)
                        state.params["_slowness_re_groups_sum"] = int(state.params.get("_slowness_re_groups_sum", 0) or 0) + g
                        state.params["_slowness_re_groups_woodbury_sum"] = int(state.params.get("_slowness_re_groups_woodbury_sum", 0) or 0) + g_w
                        state.params["_slowness_re_groups_fallback_sum"] = int(state.params.get("_slowness_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_slowness_re_max_rows_max"] = max(int(state.params.get("_slowness_re_max_rows_max", 0) or 0), mr)
                        state.params["_slowness_re_max_nodes_max"] = max(int(state.params.get("_slowness_re_max_nodes_max", 0) or 0), mn)
                        # Also update new slowness_re profiler workload counters so W&B plots are consistent.
                        state.params["_sl_re_groups_sum"] = int(state.params.get("_sl_re_groups_sum", 0) or 0) + g
                        state.params["_sl_re_groups_woodbury_sum"] = int(state.params.get("_sl_re_groups_woodbury_sum", 0) or 0) + g_w
                        state.params["_sl_re_groups_fallback_sum"] = int(state.params.get("_sl_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_sl_re_max_rows_max"] = max(int(state.params.get("_sl_re_max_rows_max", 0) or 0), mr)
                        state.params["_sl_re_max_nodes_max"] = max(int(state.params.get("_sl_re_max_nodes_max", 0) or 0), mn)
                        state.params["_sl_re_max_M_max"] = max(int(state.params.get("_sl_re_max_M_max", 0) or 0), int(state.params.get("_sl_re_max_M_static", 0) or 0))
                    except Exception:
                        pass
            else:
                loss_like = torch.tensor(0.0, device=state.device, dtype=torch.float32)

            loss_prior = compute_prior_loss(
                ΔX_src=state.dX_src,
                prior_event=state.prior_event,
                σ_p=σp,
                σ_s=σs,
                N_total=state.N,
                params=state.params,
                cluster_ids=state.cluster_ids,
                cluster_counts=state.cluster_counts,
                event_precision_matrix=state.event_precision_matrix,
                corr_error_b=b_corr_for_prior,
            )

            loss = loss_like * (float(local_bsz) / float(B_global)) + (loss_prior / float(ddp_world_size))
        else:
            # Non-DDP path: compute likelihood + prior explicitly (same math as posterior_loss),
            # so we can reuse shared_event_re profiling/timing.
            try:
                local_bsz = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
            except Exception:
                local_bsz = 0

            if local_bsz > 0:
                # Optional profiling: time the shared_event_re likelihood computation.
                try:
                    diag = _get_diagnostics_cfg(state.params)
                    prof_se_re = bool(diag.get("profile_shared_event_re", False)) if isinstance(diag, dict) else False
                    prof_sl_re = bool(diag.get("profile_slowness_re", False)) if isinstance(diag, dict) else False
                except Exception:
                    prof_se_re = False
                    prof_sl_re = False
                se_re_t0 = None
                sl_re_t0 = None
                if prof_se_re and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        se_re_t0 = _time.perf_counter()
                    except Exception:
                        se_re_t0 = None
                if prof_sl_re and bool(state.params.get("_slowness_re_enabled", False)):
                    try:
                        import time as _time
                        sl_re_t0 = _time.perf_counter()
                    except Exception:
                        sl_re_t0 = None
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_prelog", False)):
                    state.params["_shared_event_re_prelog"] = True
                    try:
                        n_rows = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
                        solver = str(state.params.get("_shared_event_re_solver", ""))
                        grouping = str(state.params.get("_shared_event_re_grouping", ""))
                        # Sentinels to verify shared_event_re block executed.
                        state.params["_shared_event_re_runtime_last_groups"] = -1
                        state.params["_shared_event_re_runtime_last_groups_pcg"] = -1
                        state.params["_shared_event_re_runtime_last_groups_fallback_diag"] = -1
                        info(
                            f"shared_event_re prelog rows={n_rows} solver={solver} grouping={grouping}",
                            section="LIKELIHOOD",
                        )
                    except Exception:
                        pass

                loss_like = compute_likelihood_loss(
                    idx=II_b,
                    y=YY_b,
                    X_src=state.X_src,
                    ΔX_src=state.dX_src,
                    model=state.model,
                    σ_p=σp,
                    σ_s=σs,
                    params=state.params,
                    nuisance_delta=nuisance_delta,
                    sigma_extra_var=sigma_extra_var,
                )
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_postlog", False)):
                    state.params["_shared_event_re_postlog"] = True
                    g = int(state.params.get("_shared_event_re_runtime_last_groups", -1))
                    g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", -1))
                    g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", -1))
                    info(
                        f"shared_event_re postlog groups={g} pcg={g_pcg} fallback={g_fb}",
                        section="LIKELIHOOD",
                    )
                if bool(state.params.get("_shared_event_re_enabled", False)):
                    g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                    g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                    g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                    g_rows = int(state.params.get("_shared_event_re_runtime_last_groups_rows_cap", 0) or 0)
                    g_nodes = int(state.params.get("_shared_event_re_runtime_last_groups_nodes_cap", 0) or 0)
                    g_tau0 = int(state.params.get("_shared_event_re_runtime_last_groups_tau_zero", 0) or 0)
                    mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                    mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                    tg = state.params.get("_shared_event_re_tau_s", [0.0, 0.0])
                    if not bool(state.params.get("_shared_event_re_reported", False)):
                        state.params["_shared_event_re_reported"] = True
                        info(
                            f"shared_event_re stats groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"max_rows={mr} max_nodes={mn} tau_s={tg}",
                            section="LIKELIHOOD",
                        )
                    if g_fb > 0:
                        info(
                            f"shared_event_re fallback reasons rows_cap={g_rows} nodes_cap={g_nodes} tau_zero={g_tau0}",
                            section="LIKELIHOOD",
                        )
                        if (not ddp_enabled) or ddp_is_main:
                            if bool(state.params.get("_shared_event_re_auto_tune_nodes_cap", False)) and (g_nodes > 0):
                                cur = int(state.params.get("_shared_event_re_max_nodes_per_group", 0) or 0)
                                cap = int(state.params.get("_shared_event_re_auto_tune_nodes_max", 0) or 0)
                                target = int(min(max(mn_all, cur), cap)) if cap > 0 else int(max(mn_all, cur))
                                if target > cur and mn_all > 0:
                                    state.params["_shared_event_re_max_nodes_per_group"] = target
                                    info(
                                        f"shared_event_re auto-tune: max_nodes_per_group -> {target}",
                                        section="LIKELIHOOD",
                                    )
                            if bool(state.params.get("_shared_event_re_auto_tune_rows_cap", False)) and (g_rows > 0):
                                cur = int(state.params.get("_shared_event_re_max_rows_per_group", 0) or 0)
                                cap = int(state.params.get("_shared_event_re_auto_tune_rows_max", 0) or 0)
                                target = int(min(max(mr_all, cur), cap)) if cap > 0 else int(max(mr_all, cur))
                                if target > cur and mr_all > 0:
                                    state.params["_shared_event_re_max_rows_per_group"] = target
                                    info(
                                        f"shared_event_re auto-tune: max_rows_per_group -> {target}",
                                        section="LIKELIHOOD",
                                    )
                    if g > 0 and g_pcg == 0 and g_fb >= g:
                        info(
                            "shared_event_re warning: all groups fell back to diagonal; "
                            "increase shared_event_re.max_nodes_per_group or max_rows_per_group",
                            section="LIKELIHOOD",
                        )
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_debug_compare_logged", False)):
                    state.params["_shared_event_re_debug_compare_logged"] = True
                    try:
                        with torch.no_grad():
                            params_dbg = dict(state.params)
                            params_dbg["_shared_event_re_enabled"] = False
                            loss_base = compute_likelihood_loss(
                                idx=II_b,
                                y=YY_b,
                                X_src=state.X_src,
                                ΔX_src=state.dX_src,
                                model=state.model,
                                σ_p=σp,
                                σ_s=σs,
                                params=params_dbg,
                                nuisance_delta=nuisance_delta,
                                sigma_extra_var=sigma_extra_var,
                            )
                        delta = float((loss_like - loss_base).detach().item())
                        info(f"shared_event_re loss delta vs baseline = {delta:.6e}", section="LIKELIHOOD")
                    except Exception as e:
                        info(f"shared_event_re debug compare failed: {e}", section="LIKELIHOOD")

                if prof_se_re and (se_re_t0 is not None) and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - se_re_t0)
                        state.params["_se_re_time_ms_sum"] = float(state.params.get("_se_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        state.params["_se_re_time_ms_count"] = int(state.params.get("_se_re_time_ms_count", 0) or 0) + 1
                        # Workload stats from modeling.py (set per-call)
                        g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                        g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                        g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                        state.params["_se_re_groups_sum"] = int(state.params.get("_se_re_groups_sum", 0) or 0) + g
                        state.params["_se_re_groups_pcg_sum"] = int(state.params.get("_se_re_groups_pcg_sum", 0) or 0) + g_pcg
                        state.params["_se_re_groups_fallback_sum"] = int(state.params.get("_se_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_se_re_max_rows_max"] = max(int(state.params.get("_se_re_max_rows_max", 0) or 0), mr)
                        state.params["_se_re_max_nodes_max"] = max(int(state.params.get("_se_re_max_nodes_max", 0) or 0), mn)
                    except Exception:
                        pass
                if prof_sl_re and (sl_re_t0 is not None) and bool(state.params.get("_slowness_re_enabled", False)):
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - sl_re_t0)
                        state.params["_slowness_re_time_ms_sum"] = float(state.params.get("_slowness_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        state.params["_slowness_re_time_ms_count"] = int(state.params.get("_slowness_re_time_ms_count", 0) or 0) + 1
                        g = int(state.params.get("_slowness_re_runtime_last_groups", 0) or 0)
                        g_w = int(state.params.get("_slowness_re_runtime_last_groups_woodbury", 0) or 0)
                        g_fb = int(state.params.get("_slowness_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_slowness_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_slowness_re_runtime_last_max_nodes", 0) or 0)
                        state.params["_slowness_re_groups_sum"] = int(state.params.get("_slowness_re_groups_sum", 0) or 0) + g
                        state.params["_slowness_re_groups_woodbury_sum"] = int(state.params.get("_slowness_re_groups_woodbury_sum", 0) or 0) + g_w
                        state.params["_slowness_re_groups_fallback_sum"] = int(state.params.get("_slowness_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_slowness_re_max_rows_max"] = max(int(state.params.get("_slowness_re_max_rows_max", 0) or 0), mr)
                        state.params["_slowness_re_max_nodes_max"] = max(int(state.params.get("_slowness_re_max_nodes_max", 0) or 0), mn)
                        # Also update new slowness_re profiler workload counters so W&B plots are consistent.
                        state.params["_sl_re_groups_sum"] = int(state.params.get("_sl_re_groups_sum", 0) or 0) + g
                        state.params["_sl_re_groups_woodbury_sum"] = int(state.params.get("_sl_re_groups_woodbury_sum", 0) or 0) + g_w
                        state.params["_sl_re_groups_fallback_sum"] = int(state.params.get("_sl_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_sl_re_max_rows_max"] = max(int(state.params.get("_sl_re_max_rows_max", 0) or 0), mr)
                        state.params["_sl_re_max_nodes_max"] = max(int(state.params.get("_sl_re_max_nodes_max", 0) or 0), mn)
                        state.params["_sl_re_max_M_max"] = max(int(state.params.get("_sl_re_max_M_max", 0) or 0), int(state.params.get("_sl_re_max_M_static", 0) or 0))
                    except Exception:
                        pass
            else:
                loss_like = torch.tensor(0.0, device=state.device, dtype=torch.float32)

            loss_prior = compute_prior_loss(
                ΔX_src=state.dX_src,
                prior_event=state.prior_event,
                σ_p=σp,
                σ_s=σs,
                N_total=state.N,
                params=state.params,
                cluster_ids=state.cluster_ids,
                cluster_counts=state.cluster_counts,
                event_precision_matrix=state.event_precision_matrix,
                corr_error_b=b_corr_for_prior,
            )
            loss = loss_like + loss_prior

        # L2 Reg (Phase 1/General)
        if state.nuisance_enable and state.nuisance_alpha is not None:
             lam = float(state.params.get("nuisance_alpha_l2", 0.0))
             if lam > 0.0:
                 # In DDP mode, treat this like a prior/regularizer (include once).
                 reg = lam * (state.nuisance_alpha.square().mean())
                 loss = loss + (reg / float(ddp_world_size) if ddp_enabled else reg)

        # Minibatch graph Laplacian penalty (suppresses internal floppy deformation modes)
        # Uses the same observed event pairs (II_b) that define the DD graph.
        # if lap_w > 0.0 and II_b is not None and II_b.numel() > 0:
        #     e1 = II_b[:, 0]
        #     e2 = II_b[:, 1]
        #     dx1 = state.dX_src.index_select(0, e1)
        #     dx2 = state.dX_src.index_select(0, e2)
        #     diff = (dx1 - dx2) * lap_dims
        #     lap_term = (diff * diff).sum(dim=1).mean()
        #     loss = loss + lap_w * lap_term

        # IMPORTANT DDP invariant:
        # All ranks must execute the same sequence of collectives.
        #
        # Previously, a rank could hit a non-finite loss and `continue` here, while other ranks
        # proceeded into gradient allreduce, causing NCCL to deadlock until watchdog timeout.
        #
        # Fix: in DDP mode, detect non-finite loss, synchronize a `bad_loss` flag, and if any
        # rank is bad, run a *dummy* backward that touches all parameters (zero gradients) so
        # the collective schedule stays aligned. Then skip the optimizer step on all ranks.
        bad_loss = 0
        try:
            bad_loss = 0 if bool(torch.isfinite(loss).item()) else 1
        except Exception:
            bad_loss = 1
        if ddp_enabled:
            try:
                bad_t0 = torch.tensor([bad_loss], device=state.device, dtype=torch.int32)
                dist.all_reduce(bad_t0, op=dist.ReduceOp.MAX)
                bad_loss = int(bad_t0.item())
            except Exception:
                bad_loss = 1
        if (not ddp_enabled) and bad_loss:
            # Non-DDP: keep historical behavior (skip this batch).
            continue

        if ddp_enabled and bad_loss:
            # Dummy loss: depends on parameters but yields zero gradients.
            # This ensures p.grad tensors exist and `_ddp_allreduce_grads` can run safely.
            loss0 = torch.zeros((), device=state.device, dtype=torch.float32)
            try:
                for g in optimizer.param_groups:  # type: ignore[attr-defined]
                    for p in g.get("params", []):
                        if isinstance(p, torch.Tensor) and bool(getattr(p, "requires_grad", False)):
                            loss0 = loss0 + (p.sum() * 0.0)
            except Exception:
                pass
            loss = loss0

        loss.backward()

        # DDP: all-reduce gradients (SUM) so all ranks take identical optimizer steps.
        if ddp_enabled:
            _ddp_allreduce_grads(optimizer)
            # Abort step if any rank produced non-finite gradients
            bad = 0
            try:
                for g in optimizer.param_groups:  # type: ignore[attr-defined]
                    for p in g.get("params", []):
                        if p is None or getattr(p, "grad", None) is None:
                            continue
                        if not torch.isfinite(p.grad).all():
                            bad = 1
                            break
                    if bad:
                        break
            except Exception:
                bad = 1
            try:
                bad_t = torch.tensor([bad], device=state.device, dtype=torch.int32)
                dist.all_reduce(bad_t, op=dist.ReduceOp.MAX)
                bad = int(bad_t.item())
            except Exception:
                bad = 1
            if bad:
                optimizer.zero_grad(set_to_none=True)
                continue
            # Also skip the step if any rank had a non-finite loss (handled via dummy backward above).
            if bad_loss:
                optimizer.zero_grad(set_to_none=True)
                continue

        # Optional: log explicit slowness_re latent RMS + grad RMS (debugging).
        if bool(state.params.get("_slowness_re_explicit_enabled", False)) and not ddp_enabled:
            try:
                if int(state.params.get("_slowness_re_explicit_grad_logged_epoch", -1)) != int(epoch_index):
                    state.params["_slowness_re_explicit_grad_logged_epoch"] = int(epoch_index)
                    s_cp = getattr(state, "slowness_re_comp_p", None)
                    s_cs = getattr(state, "slowness_re_comp_s", None)
                    s_sp = getattr(state, "slowness_re_station_p", None)
                    s_ss = getattr(state, "slowness_re_station_s", None)
                    def _rms(t: torch.Tensor | None) -> float:
                        if not isinstance(t, torch.Tensor) or t.numel() == 0:
                            return float("nan")
                        return float(t.detach().square().mean().sqrt().item())
                    def _grms(t: torch.Tensor | None) -> float:
                        if not isinstance(t, torch.Tensor):
                            return float("nan")
                        g = getattr(t, "grad", None)
                        if not isinstance(g, torch.Tensor) or g.numel() == 0:
                            return float("nan")
                        return float(g.detach().square().mean().sqrt().item())
                    print(
                        f"[slowness_re] explicit latents rms/grad: "
                        f"comp_p={_rms(s_cp):.3g}/{_grms(s_cp):.3g} "
                        f"comp_s={_rms(s_cs):.3g}/{_grms(s_cs):.3g} "
                        f"sta_p={_rms(s_sp):.3g}/{_grms(s_sp):.3g} "
                        f"sta_s={_rms(s_ss):.3g}/{_grms(s_ss):.3g}",
                        flush=True,
                    )
            except Exception:
                pass

        # SVRG Correction
        if svrg_enabled and state.svrg_grad_full is not None and state.svrg_dX_snapshot is not None:
            # 1. Compute grad at snapshot location for THIS batch
            # We need to temporarily swap dX_src to snapshot
            # Detach current grad first
            grad_batch_current = state.dX_src.grad.clone()
            
            # Swap params
            dX_current_data = state.dX_src.data.clone()
            state.dX_src.data.copy_(state.svrg_dX_snapshot)
            if state.dX_src.grad is not None:
                state.dX_src.grad.zero_()
                
            # Compute loss at snapshot
            loss_snap = posterior_loss(
                idx=II_b,
                y=YY_b,
                X_src=state.X_src,
                ΔX_src=state.dX_src, # Now holding snapshot
                model=state.model,
                prior_event=state.prior_event,
                σ_p=σp,
                σ_s=σs,
                N_total=state.N,
                params=state.params,
                nuisance_delta=nuisance_delta,
                sigma_extra_var=sigma_extra_var,
                cluster_ids=state.cluster_ids,
                cluster_counts=state.cluster_counts,
                event_precision_matrix=state.event_precision_matrix,
                corr_error_b=(getattr(state, "corr_error_b", None) if bool(state.params.get("_corr_error_enabled", False)) else None),
            )
            loss_snap.backward()
            grad_batch_snapshot = state.dX_src.grad
            
            # Restore current params
            state.dX_src.data.copy_(dX_current_data)
            
            # Apply Correction: g = g_curr - g_snap + g_full
            # Note: We must handle potential None grads or shape mismatches carefully
            if grad_batch_snapshot is not None:
                # Corrected gradient
                corrected_grad = grad_batch_current - grad_batch_snapshot + state.svrg_grad_full
                state.dX_src.grad.copy_(corrected_grad)
            else:
                # Fallback: restore original
                state.dX_src.grad.copy_(grad_batch_current)

        # Grad Checks
        if state.dX_src.grad is not None and not torch.isfinite(state.dX_src.grad).all():
            optimizer.zero_grad(set_to_none=True)
            continue

        # Optional: per-dimension learning-rate multiplier for ΔT (origin time correction).
        # This is implemented as a gradient scaler so it works for Adam and our SGLD/SGHMC backends.
        try:
            dt_lr_mult = float(state.params.get("dt_lr_mult", 1.0))
            if math.isfinite(dt_lr_mult) and dt_lr_mult > 0.0 and (dt_lr_mult != 1.0):
                g = state.dX_src.grad
                if isinstance(g, torch.Tensor) and g.ndim == 2 and int(g.shape[1]) >= 4:
                    g[:, 3].mul_(dt_lr_mult)
        except Exception:
            pass

        # Gauge projection (translation-mode removal): apply to MAP/Adam as well (sampler backends handle internally).
        _maybe_apply_gauge_projection_for_optimizer(state, optimizer)
            
        # Clipping
        if grad_clip_norm > 0.0:
            params_to_clip = [state.dX_src]
            if state.nuisance_enable and state.nuisance_alpha is not None and state.nuisance_alpha.grad is not None:
                params_to_clip.append(state.nuisance_alpha)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=grad_clip_norm)
            
        # Ensure sampler noise is identical across ranks (SGHMC/pSGLD inject noise in optimizer.step()).
        if ddp_enabled:
            _ddp_set_step_seed(state.params, int(state.global_step_count), device=state.device)
        optimizer.step()
        _maybe_apply_gauge_projection_to_momentum_buffers(state, optimizer)
        
        # Safety Check
        if not torch.isfinite(state.dX_src).all():
             with torch.no_grad():
                 state.dX_src.data = torch.nan_to_num(state.dX_src.data, nan=0.0, posinf=0.0, neginf=0.0)
        
        _clamp_dX_inplace(state)
        _apply_shared_event_latent_constraints_inplace(state)
        _apply_dd_graph_re_constraints_inplace(state)
        
        # Sampling (Phase 4)
        if is_sampling:
            with torch.no_grad():
                # Save every N steps (to in-memory buffer; flushing to disk is handled elsewhere).
                write_samples = bool(state.params.get("write_samples", True))
                save_every_n = int(state.params.get("save_every_n", 1))
                if write_samples and save_every_n > 0 and ddp_is_main and (state.global_step_count % save_every_n == 0):
                    # Initialize sampling wall-clock start time on first saved sample.
                    # This is used for online ESS/sec diagnostics (independent of GPU timings).
                    try:
                        if not hasattr(state, "_sampling_wall_t0") or getattr(state, "_sampling_wall_t0") is None:
                            setattr(state, "_sampling_wall_t0", float(time.time()))
                    except Exception:
                        pass

                    state.samples.append(state.dX_src.detach().cpu().clone())
                    # Noise learning removed (fixed phase_unc only).

                    # Optional online ESS/IACT diagnostic using the in-memory samples buffer.
                    try:
                        if bool(state.params.get("ess_online_enabled", False)) and _want_wandb_group(state.params, "ess_online"):
                            every = int(state.params.get("ess_online_every_n_samples", 0))
                            window = int(state.params.get("ess_online_window", 0))
                            if window <= 0:
                                window = 512
                            min_needed = max(8, int(window))
                            if every > 0 and len(state.samples) >= min_needed and (len(state.samples) % every) == 0:
                                n_events_total = int(state.dX_src.shape[0])
                                k = int(state.params.get("ess_online_n_events", 0))
                                k = min(k, n_events_total)
                                if k > 0:
                                    seed = int(state.params.get("ess_online_seed", 0))
                                    rng = np.random.default_rng(seed)
                                    # Cache the chosen subset so it's stable.
                                    if "_ess_online_event_idx" not in state.params:
                                        idx = rng.choice(n_events_total, size=k, replace=False).astype(np.int64)
                                        state.params["_ess_online_event_idx"] = idx.tolist()
                                    idx_np = np.asarray(state.params.get("_ess_online_event_idx", []), dtype=np.int64)
                                    if idx_np.size > 0:
                                        dims = state.params.get("ess_online_dims", [0, 1, 2])
                                        if not isinstance(dims, list) or len(dims) == 0:
                                            dims = [0, 1, 2]
                                        max_lag_v = int(state.params.get("ess_online_max_lag", 0))
                                        from spider.diagnostics.online_ess import compute_online_ess
                                        res = compute_online_ess(
                                            samples=state.samples,
                                            event_idx=torch.as_tensor(idx_np, dtype=torch.int64),
                                            dims=[int(d) for d in dims],
                                            window=int(window),
                                            max_lag=(max_lag_v if max_lag_v > 0 else None),
                                        )
                                        ess_online_last = res.to_metrics(prefix="ess_online")

                                        # Derive ESS/sec style metrics so users can tune batch size empirically.
                                        # Use sampling wall-clock time since the first saved sample and also the
                                        # delta rate since the previous ESS update.
                                        try:
                                            now_t = float(time.time())
                                            t0 = float(getattr(state, "_sampling_wall_t0", now_t))
                                            elapsed_s = max(1e-9, now_t - t0)
                                            ess_med = float(ess_online_last.get("ess_online/ess_median", float("nan")))
                                            n_samp = float(ess_online_last.get("ess_online/n_samples", float("nan")))
                                            ess_online_last["ess_online/elapsed_s"] = float(elapsed_s)
                                            if np.isfinite(ess_med):
                                                ess_online_last["ess_online/ess_per_s_median"] = float(ess_med / elapsed_s)
                                            if np.isfinite(n_samp):
                                                ess_online_last["ess_online/samples_per_s"] = float(n_samp / elapsed_s)

                                            prev_t = getattr(state, "_ess_online_last_wall_time", None)
                                            prev_ess = getattr(state, "_ess_online_last_ess_median", None)
                                            if prev_t is not None and prev_ess is not None:
                                                dt = max(1e-9, float(now_t) - float(prev_t))
                                                dess = float(ess_med) - float(prev_ess) if np.isfinite(ess_med) else float("nan")
                                                if np.isfinite(dess):
                                                    ess_online_last["ess_online/ess_per_s_delta_median"] = float(dess / dt)
                                                ess_online_last["ess_online/delta_s"] = float(dt)

                                            setattr(state, "_ess_online_last_wall_time", float(now_t))
                                            setattr(state, "_ess_online_last_ess_median", float(ess_med))
                                        except Exception:
                                            pass

                                        setattr(state, "_ess_online_last_metrics", ess_online_last)
                                        setattr(state, "_ess_online_last_n_samples", int(ess_online_last.get("ess_online/n_samples", 0)))
                                        ess_online_updated_this_epoch = True
                                        # Terminal output intentionally suppressed; metrics are still logged.
                    except Exception:
                        # Don't crash sampling for diagnostics; count failures so users can see if it's flaky.
                        try:
                            setattr(state, "_ess_online_error_count", int(getattr(state, "_ess_online_error_count", 0)) + 1)
                        except Exception:
                            pass
                        
                # Periodic cache clear
                ec_every = int(state.params.get("cuda_empty_cache_every", 0))
                if ec_every > 0 and (state.global_step_count % ec_every == 0):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Stats
        # In DDP mode, the per-rank loss is constructed so that SUM across ranks equals the true global loss.
        # Only rank0 logs/aggregates to avoid duplicated histories.
        if ddp_enabled:
            try:
                loss_sum = loss.detach().clone()
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                loss_f = float(loss_sum.item())
            except Exception:
                loss_f = float("nan")
        else:
            loss_f = float(loss.item())
        if (not ddp_enabled) or ddp_is_main:
            total_loss_vals.append(loss_f)
        try:
            # Weight by number of rows/edges in this batch.
            # NOTE: must use the current batch tensor (II_b); other locals named `idx` exist in this function
            # for ESS diagnostics and are unrelated to batch size.
            if ddp_enabled:
                bsz = int(global_bsz) if "global_bsz" in locals() else int(II_b.shape[0])
            else:
                bsz = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
        except Exception:
            bsz = 0
        if bsz > 0 and ((not ddp_enabled) or ddp_is_main):
            total_loss_weighted_sum += loss_f * float(bsz)
            total_loss_weighted_denom += int(bsz)
        state.global_step_count += 1

    # End of Epoch Stats & Logging
    with torch.no_grad():
        # Optional: Gibbs update for hierarchical event prior precision (Wishart hyperprior).
        # This is allowed in any phase; updates are controlled only by `every_epochs` cadence.
        try:
            hier_enable = bool(getattr(state, "hierarchical_prior_enable", False)) and bool(state.params.get("hierarchical_event_prior", False))
        except Exception:
            hier_enable = False
        if hier_enable:
            # Hard break: per-phase scheduling removed; hyper-updates are always allowed when enabled.
            state.params["_hierarchical_hyper_runtime_enable"] = True
            try:
                if "hierarchical_prior_dof" not in state.params or "_hierarchical_scale_std" not in state.params:
                    raise KeyError("Hierarchical event hyperprior requires hierarchical_prior_dof and _hierarchical_scale_std (from priors.event.hyper.params)")
                if "_hierarchical_update_every_epochs" not in state.params:
                    raise KeyError("Hierarchical event hyperprior requires priors.event.hyper.update.every_epochs")
                update_every = int(state.params["_hierarchical_update_every_epochs"])
                update_every = max(1, update_every)
                do_hier_update = (epoch_index % update_every) == 0
            except Exception:
                do_hier_update = False
            if do_hier_update:
                try:
                    nu = float(state.params["hierarchical_prior_dof"])
                    p_std = torch.tensor(state.params["_hierarchical_scale_std"], device=state.device, dtype=torch.float32)
                    V_inv_common = nu * torch.diag(p_std ** 2)

                    if state.cluster_ids is not None and state.cluster_counts is not None:
                        K = int(state.cluster_counts.shape[0])
                        new_P0s = []
                        for k in range(K):
                            mask_k = (state.cluster_ids == k)
                            dX_k = state.dX_src[mask_k]
                            if dX_k.shape[0] == 0:
                                new_P0s.append(V_inv_common * (1.0 / max(nu, 1e-12)))
                                continue
                            P0_k = update_precision_hyperparameter(dX_k.detach(), nu, V_inv_common, mode="sample")
                            new_P0s.append(P0_k)
                        state.event_precision_matrix = torch.stack(new_P0s, dim=0)
                    else:
                        new_P0 = update_precision_hyperparameter(state.dX_src.detach(), nu, V_inv_common, mode="sample")
                        state.event_precision_matrix = new_P0
                    # Lightweight visibility: report implied stds periodically (or every update if update_every is large).
                    try:
                        # Prefer the existing diagnostics cadence knob if present.
                        # Default to something visible (10) so users can confirm updates are happening.
                        log_every = int(state.params.get("display_precond_every", state.params.get("hierarchical_log_every_epochs", 10)))
                        log_every = max(1, log_every)
                        if (epoch_index % log_every) == 0:
                            P0_all = state.event_precision_matrix
                            P0_mean = P0_all.mean(dim=0) if isinstance(P0_all, torch.Tensor) and P0_all.ndim == 3 else P0_all
                            Cov = torch.linalg.inv(P0_mean)
                            stds = torch.sqrt(torch.diag(Cov)).clamp_min(torch.tensor(0.0, device=Cov.device, dtype=Cov.dtype))
                            corr_zt = float(Cov[2, 3].item()) / (float(stds[2].item()) * float(stds[3].item()) + 1e-12)
                            state.params["_hierarchical_event_std_est"] = [float(stds[0].item()), float(stds[1].item()), float(stds[2].item()), float(stds[3].item())]
                            info(
                                f"Hierarchical event prior updated: std≈[{stds[0]:.3g},{stds[1]:.3g},{stds[2]:.3g},{stds[3]:.3g}] corr_zt≈{corr_zt:.3g}",
                                section="PRIORS",
                            )
                    except Exception:
                        pass
                except Exception:
                    # Avoid silent failures; warn once per epoch.
                    warn("Hierarchical event prior update failed; leaving P0 unchanged for this epoch.", section="PRIORS")

        # Optional gauge-fixing: center corr_error_b within each connected component.
        #
        # This is only needed for an *intrinsic* GMRF prior (Laplacian-only, q_diag=0), where the
        # precision has a constant-nullspace per graph component and the likelihood is difference-only.
        # If q_diag > 0 (proper prior) or the event graph is disabled (IID prior), centering would
        # bias the posterior by enforcing a hard mean-zero constraint, so we skip it.
        need_gauge_fix = False
        try:
            graph_enabled = bool(state.params.get("_corr_error_event_graph_enabled", True))
            qd = float(state.params.get("_corr_error_q_diag", 0.0))
            u = state.params.get("_corr_error_u", None)
            has_edges = bool(isinstance(u, torch.Tensor) and int(u.numel()) > 0)
            need_gauge_fix = bool(graph_enabled and has_edges and (qd <= 0.0))
        except Exception:
            need_gauge_fix = False

        if need_gauge_fix and isinstance(getattr(state, "corr_error_b", None), torch.nn.Parameter):
            with torch.no_grad():
                b_param = getattr(state, "corr_error_b")  # (n_events,R,2)
                comp = getattr(state, "corr_error_component_id", None)
                n_comp = int(getattr(state, "corr_error_n_components", 0) or 0)
                # Fallback (should not happen): treat as single component.
                if (not isinstance(comp, torch.Tensor)) or (comp.ndim != 1) or (int(comp.shape[0]) != int(b_param.shape[0])) or (n_comp <= 0):
                    comp = torch.zeros((int(b_param.shape[0]),), dtype=torch.int64, device=b_param.device)
                    n_comp = 1
                comp = comp.to(device=b_param.device, dtype=torch.int64)

                x = b_param.reshape(int(b_param.shape[0]), -1)  # (N, K) where K=R*2
                K = int(x.shape[1])
                sums = torch.zeros((n_comp, K), device=x.device, dtype=x.dtype)
                sums.index_add_(0, comp, x)
                counts = torch.bincount(comp, minlength=n_comp).to(device=x.device, dtype=x.dtype).clamp_min(1.0)
                means = sums / counts.view(-1, 1)
                x = x - means.index_select(0, comp)
                b_param.copy_(x.view_as(b_param))

        # Optional: Gibbs update for correlated-error tau (P/S covariance).
        try:
            hier_tau_enable = (
                bool(state.params.get("_corr_error_enabled", False))
                and bool(state.params.get("_corr_error_hierarchical_tau_enabled", False))
                and isinstance(getattr(state, "corr_error_b", None), torch.nn.Parameter)
            )
        except Exception:
            hier_tau_enable = False
        if hier_tau_enable:
            try:
                update_every = int(state.params.get("_corr_error_hierarchical_tau_update_every", 5))
                start_after = int(state.params.get("_corr_error_hierarchical_tau_start_after_epochs", 0) or 0)
                if start_after < 0:
                    start_after = 0
                do_update = (epoch_index >= start_after) and ((epoch_index % max(1, update_every)) == 0)
                if do_update:
                    nu = float(state.params.get("_corr_error_hierarchical_tau_dof", 10.0))
                    p_std = torch.tensor(state.params["_corr_error_hierarchical_tau_scale_ps"], device=state.device, dtype=torch.float32)
                    V_inv = nu * torch.diag(p_std ** 2)
                    
                    b = getattr(state, "corr_error_b").detach()
                    u = getattr(state, "corr_error_u")
                    v = getattr(state, "corr_error_v")
                    w = getattr(state, "corr_error_w")
                    q_diag = float(state.params.get("_corr_error_q_diag", 1e-3))
                    
                    # Compute Gibbs update for covariance matrix
                    cov_sample = update_corr_error_tau_hyperparameter(
                        b, u, v, w, q_diag, nu, V_inv, mode="sample"
                    )
                    
                    # Convert covariance to tau_p, tau_s, rho_ps
                    t2p = float(cov_sample[0, 0].item())
                    t2s = float(cov_sample[1, 1].item())
                    tps = float(cov_sample[0, 1].item())
                    
                    tp = math.sqrt(max(1e-12, t2p))
                    ts = math.sqrt(max(1e-12, t2s))
                    rho = tps / (tp * ts + 1e-12)
                    rho = max(-0.999, min(0.999, rho))
                    
                    # Damping: move only 20% toward the new sample per update.
                    # This prevents sudden massive jumps in the prior energy that can destabilize the Langevin sampler.
                    old_tau_ps = state.params.get("_corr_error_tau_s", [tp, ts])
                    old_rho = float(state.params.get("_corr_error_rho_ps", rho))
                    alpha = float(state.params.get("_corr_error_hierarchical_tau_damping", 0.2))
                    tp = (1.0 - alpha) * old_tau_ps[0] + alpha * tp
                    ts = (1.0 - alpha) * old_tau_ps[1] + alpha * ts
                    rho = (1.0 - alpha) * old_rho + alpha * rho

                    # Optional safety clamps (seconds).
                    try:
                        mn = state.params.get("_corr_error_hierarchical_tau_min_tau_s", None)
                        mx = state.params.get("_corr_error_hierarchical_tau_max_tau_s", None)
                        if isinstance(mn, list) and len(mn) == 2:
                            tp = max(float(mn[0]), float(tp))
                            ts = max(float(mn[1]), float(ts))
                        if isinstance(mx, list) and len(mx) == 2:
                            tp = min(float(mx[0]), float(tp))
                            ts = min(float(mx[1]), float(ts))
                    except Exception:
                        pass
                    
                    state.params["_corr_error_tau_s"] = [tp, ts]
                    state.params["_corr_error_rho_ps"] = rho
                    
                    # Log to console periodically
                    log_every = int(state.params.get("display_precond_every", 10))
                    if (epoch_index % max(1, log_every)) == 0:
                        info(
                            f"Hierarchical corr_error tau updated: tau_p={tp:.3g}s tau_s={ts:.3g}s rho_ps={rho:.3g}",
                            section="PRIORS"
                        )
            except Exception as e:
                warn(f"Hierarchical corr_error tau update failed: {e}", section="PRIORS")

        # Prefer weighted mean over edges; fallback to unweighted mean if something went wrong.
        if total_loss_weighted_denom > 0:
            total_loss_mean = total_loss_weighted_sum / float(total_loss_weighted_denom)
        else:
            total_loss_mean = (sum(total_loss_vals) / len(total_loss_vals)) if total_loss_vals else 0.0
        epoch_time = time.time() - epoch_start_time
        
        # Phase 1 MAP CSV
        if write_map_csv:
            _maybe_write_map_csv(state, epoch_index)
            
        # Compute MADs occasionally
        # Logic differs slightly per phase in original code, but can be standardized
        # Phase 1: % 10
        # Phase 2/3/4: customized interval
        # We can pass a flag or check params here.
        # Simplifying: check standardized keys or default
        phase_name = "phase1" if isinstance(optimizer, torch.optim.Adam) else "phase4" # simplified
        # Actually rely on param keys directly
        # ... (omitted for brevity, logic handled in return values)

        # Update stats tensor
        st = state.stats_tensor
        st[0] = state.dX_src[:, 0].mean()
        st[1] = state.dX_src[:, 1].mean()
        st[2] = state.dX_src[:, 2].mean()
        st[3] = torch.abs(state.dX_src[:, 0]).median()
        st[4] = torch.abs(state.dX_src[:, 1]).median()
        st[5] = torch.abs(state.dX_src[:, 2]).median()
        st[6] = torch.sqrt(state.dX_src[:, 0]**2 + state.dX_src[:, 1]**2 + state.dX_src[:, 2]**2).max()
        st[7] = torch.quantile(torch.sqrt(state.dX_src[:, 0]**2 + state.dX_src[:, 1]**2 + state.dX_src[:, 2]**2), 0.90).item()
        
        stats_cpu = st.detach().cpu().numpy()

        # Time component (ΔT / origin-time correction) stats are not stored in stats_tensor
        # to preserve checkpoint/backward compatibility. Compute separately for logging.
        #
        # IMPORTANT interpretability note:
        # In a pure differential-time likelihood, adding a constant to *all* origin times does not
        # change dt_pred (only differences matter). That can make raw dt_* metrics drift slowly.
        # To diagnose meaningful relative-time mixing, also compute "centered" stats where we
        # subtract the per-component (cluster) mean if clusters exist, else the global mean.
        try:
            dT = state.dX_src[:, 3].to(torch.float32)
            dt_mean = float(dT.mean().detach().cpu().item())
            dt_med_abs = float(torch.abs(dT).median().detach().cpu().item())
            dt_max_abs = float(torch.abs(dT).max().detach().cpu().item())
            dt_p90_abs = float(torch.quantile(torch.abs(dT), 0.90).detach().cpu().item())
            # Centered ΔT (remove per-connected-component mean when available)
            try:
                dT0 = None
                cid = getattr(state, "cluster_ids", None)
                cc = getattr(state, "cluster_counts", None)
                if isinstance(cid, torch.Tensor) and isinstance(cc, torch.Tensor) and cid.numel() == dT.numel():
                    K = int(cc.numel())
                    if K > 0:
                        sums = torch.zeros((K,), device=dT.device, dtype=dT.dtype)
                        sums.index_add_(0, cid.to(torch.int64), dT)
                        denom = cc.to(device=dT.device, dtype=dT.dtype).view(-1).clamp_min(1.0)
                        means = sums / denom
                        dT0 = dT - means.index_select(0, cid.to(torch.int64))
                if dT0 is None:
                    dT0 = dT - dT.mean()
                dt_centered_std = float(dT0.std(unbiased=False).detach().cpu().item())
                dt_centered_med_abs = float(torch.abs(dT0).median().detach().cpu().item())
                dt_centered_p90_abs = float(torch.quantile(torch.abs(dT0), 0.90).detach().cpu().item())
            except Exception:
                dt_centered_std = float("nan")
                dt_centered_med_abs = float("nan")
                dt_centered_p90_abs = float("nan")
        except Exception:
            dt_mean = float("nan")
            dt_med_abs = float("nan")
            dt_max_abs = float("nan")
            dt_p90_abs = float("nan")
            dt_centered_std = float("nan")
            dt_centered_med_abs = float("nan")
            dt_centered_p90_abs = float("nan")
        
        metrics = {
            "loss": total_loss_mean,
            "epoch_time": epoch_time,
            "permute_time": float(permute_time_s),
            "dx_mean": stats_cpu[0],
            "dy_mean": stats_cpu[1],
            "dz_mean": stats_cpu[2],
            "dx_med_abs": stats_cpu[3],
            "dy_med_abs": stats_cpu[4],
            "dz_med_abs": stats_cpu[5],
            "dr_max": stats_cpu[6],
            "dr_90": stats_cpu[7],
            # ΔT (seconds)
            "dt_mean": dt_mean,
            "dt_med_abs": dt_med_abs,
            "dt_max_abs": dt_max_abs,
            "dt_p90_abs": dt_p90_abs,
            "dt_centered_std": dt_centered_std,
            "dt_centered_med_abs": dt_centered_med_abs,
            "dt_centered_p90_abs": dt_centered_p90_abs,
        }
        # dd_graph_re RMS metrics (per epoch)
        try:
            if bool(state.params.get("_dd_graph_re_enabled", False)):
                for k in (
                    "_dd_graph_re_pred_rms",
                    "_dd_graph_re_resid_rms",
                    "_dd_graph_re_pred_rms_p",
                    "_dd_graph_re_pred_rms_s",
                    "_dd_graph_re_resid_rms_p",
                    "_dd_graph_re_resid_rms_s",
                    "_dd_graph_re_loss_delta",
                ):
                    if k in state.params:
                        metrics[f"dd_graph_re/{k[1:]}"] = float(state.params.get(k))
        except Exception:
            pass
        # sigma_inflation removed (start fresh).
        # Optional: timing summary for shared_event_latent nuisance reconstruction (per epoch).
        try:
            diag = _get_diagnostics_cfg(state.params)
            if isinstance(diag, dict) and bool(diag.get("profile_shared_event_latent", False)):
                c = int(state.params.get("_se_lat_time_ms_count", 0) or 0)
                s = float(state.params.get("_se_lat_time_ms_sum", 0.0) or 0.0)
                if c > 0:
                    metrics["shared_event_latent/time_ms_mean"] = float(s / float(c))
                    metrics["shared_event_latent/time_ms_sum"] = float(s)
                    metrics["shared_event_latent/time_batches"] = float(c)
        except Exception:
            pass
        # Optional: timing + workload summary for collapsed shared_event_re likelihood (per epoch).
        try:
            diag = _get_diagnostics_cfg(state.params)
            if isinstance(diag, dict) and bool(diag.get("profile_shared_event_re", False)) and bool(state.params.get("_shared_event_re_enabled", False)):
                c = int(state.params.get("_se_re_time_ms_count", 0) or 0)
                s = float(state.params.get("_se_re_time_ms_sum", 0.0) or 0.0)
                if c > 0:
                    metrics["shared_event_re/time_ms_mean"] = float(s / float(c))
                    metrics["shared_event_re/time_ms_sum"] = float(s)
                    metrics["shared_event_re/time_batches"] = float(c)
                    # Workload (means over batches + max over epoch)
                    metrics["shared_event_re/groups_mean"] = float(int(state.params.get("_se_re_groups_sum", 0) or 0) / float(c))
                    metrics["shared_event_re/groups_pcg_mean"] = float(int(state.params.get("_se_re_groups_pcg_sum", 0) or 0) / float(c))
                    metrics["shared_event_re/groups_fallback_diag_mean"] = float(int(state.params.get("_se_re_groups_fallback_sum", 0) or 0) / float(c))
                    metrics["shared_event_re/max_rows_max"] = float(int(state.params.get("_se_re_max_rows_max", 0) or 0))
                    metrics["shared_event_re/max_nodes_max"] = float(int(state.params.get("_se_re_max_nodes_max", 0) or 0))
                if bool(state.params.get("_shared_event_re_station_phase_enabled", False)):
                    metrics["shared_event_re/station_phase_groups"] = float(int(state.params.get("_shared_event_re_station_phase_last_groups", 0) or 0))
                    metrics["shared_event_re/station_phase_quad"] = float(state.params.get("_shared_event_re_station_phase_last_quad", 0.0) or 0.0)
        except Exception:
            pass
        # Optional: timing + breakdown summary for collapsed slowness_re likelihood (per epoch).
        try:
            diag = _get_diagnostics_cfg(state.params)
            if isinstance(diag, dict) and bool(diag.get("profile_slowness_re", False)) and bool(state.params.get("_slowness_re_enabled", False)):
                c = int(state.params.get("_sl_re_time_ms_count", 0) or 0)
                s = float(state.params.get("_sl_re_time_ms_sum", 0.0) or 0.0)
                if c > 0:
                    metrics["slowness_re/time_ms_mean"] = float(s / float(c))
                    metrics["slowness_re/time_ms_sum"] = float(s)
                    metrics["slowness_re/time_batches"] = float(c)
                # Grouping is measured once per batch; average over batches.
                if c > 0:
                    metrics["slowness_re/grouping_ms_mean"] = float(float(state.params.get("_sl_re_grouping_ms_sum", 0.0) or 0.0) / float(c))
                # Kernel/assemble/solve are measured per *profiled group*; average over profiled groups.
                gc = int(state.params.get("_sl_re_profiled_groups_sum", 0) or 0)
                if gc > 0:
                    metrics["slowness_re/kernel_ms_mean"] = float(float(state.params.get("_sl_re_kernel_ms_sum", 0.0) or 0.0) / float(gc))
                    metrics["slowness_re/assemble_ms_mean"] = float(float(state.params.get("_sl_re_assemble_ms_sum", 0.0) or 0.0) / float(gc))
                    metrics["slowness_re/solve_ms_mean"] = float(float(state.params.get("_sl_re_solve_ms_sum", 0.0) or 0.0) / float(gc))
                    metrics["slowness_re/profiled_groups"] = float(gc)
                    metrics["slowness_re/pcg_iters_mean"] = float(int(state.params.get("_sl_re_pcg_iters_sum", 0) or 0) / float(gc))
                # Workload (means over batches + max over epoch)
                if c > 0:
                    metrics["slowness_re/groups_mean"] = float(int(state.params.get("_sl_re_groups_sum", 0) or 0) / float(c))
                    metrics["slowness_re/groups_woodbury_mean"] = float(int(state.params.get("_sl_re_groups_woodbury_sum", 0) or 0) / float(c))
                    metrics["slowness_re/groups_fallback_diag_mean"] = float(int(state.params.get("_sl_re_groups_fallback_sum", 0) or 0) / float(c))
                    metrics["slowness_re/max_rows_max"] = float(int(state.params.get("_sl_re_max_rows_max", 0) or 0))
                    metrics["slowness_re/max_nodes_max"] = float(int(state.params.get("_sl_re_max_nodes_max", 0) or 0))
                    metrics["slowness_re/max_M_max"] = float(int(state.params.get("_sl_re_max_M_max", 0) or 0))
        except Exception:
            pass
        # Optional uncollapsed shared-event latent diagnostics.
        # Note: some of these are expensive (e.g., inducing prior energy), so we allow amortization.
        try:
            if bool(state.params.get("_shared_event_latent_enabled", False)) and _want_wandb_group(state.params, "shared_event_latent"):
                # Allow amortizing heavy shared_event_latent diagnostics:
                # inference.diagnostics.shared_event_latent_every_epochs (int, default=1)
                se_every = 1
                try:
                    diag = _get_diagnostics_cfg(state.params)
                    if isinstance(diag, dict) and "shared_event_latent_every_epochs" in diag:
                        se_every = int(diag.get("shared_event_latent_every_epochs", 1))
                except Exception:
                    se_every = 1
                if se_every < 1:
                    se_every = 1
                if (int(epoch_index) % int(se_every)) != 0:
                    raise RuntimeError("skip shared_event_latent diagnostics this epoch")

                b = getattr(state, "shared_event_latent_b", None)
                # Coefficient diagnostics:
                # - full / graph_gmrf / inducing_gp: b is scalar per phase (..,*,2)
                # - slowness_inducing_gp: b is vector per phase (..,*,2,3)
                if isinstance(b, torch.Tensor) and b.ndim == 3 and int(b.shape[2]) == 2:
                    bP = b[:, :, 0]
                    bS = b[:, :, 1]
                    metrics["shared_event_latent/bP_rms"] = float(torch.sqrt((bP * bP).mean()).item())
                    metrics["shared_event_latent/bS_rms"] = float(torch.sqrt((bS * bS).mean()).item())
                    # Global mean drift (should be ~0 under Q with q_diag>0)
                    metrics["shared_event_latent/bP_mean"] = float(bP.mean().item())
                    metrics["shared_event_latent/bS_mean"] = float(bS.mean().item())
                elif isinstance(b, torch.Tensor) and b.ndim == 4 and int(b.shape[2]) == 2 and int(b.shape[3]) == 3:
                    bP = b[:, :, 0, :]  # (S_or_R, M, 3)
                    bS = b[:, :, 1, :]
                    metrics["shared_event_latent/bP_rms"] = float(torch.sqrt((bP * bP).mean()).item())
                    metrics["shared_event_latent/bS_rms"] = float(torch.sqrt((bS * bS).mean()).item())
                    metrics["shared_event_latent/bP_mean"] = float(bP.mean().item())
                    metrics["shared_event_latent/bS_mean"] = float(bS.mean().item())
                    # Direct check of the *station-common mode* constraint:
                    # We care about mean over stations *per event* (K,2), not just global mean.
                    try:
                        n_stations = int(getattr(state, "n_stations", 0) or 0)
                        if n_stations > 0:
                            if int(b.shape[0]) == int(n_stations):
                                mu = b.to(torch.float32).mean(dim=0)  # scalar: (K,2) ; vector: (M,2,3)
                            else:
                                W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                                if (
                                    isinstance(W_sta, torch.Tensor)
                                    and W_sta.ndim == 2
                                    and int(W_sta.shape[0]) == int(n_stations)
                                    and int(b.shape[0]) == int(W_sta.shape[1])
                                ):
                                    W = W_sta.to(device=b.device, dtype=torch.float32)
                                    ones = torch.ones((n_stations,), device=b.device, dtype=torch.float32)
                                    r = (W.transpose(0, 1).matmul(ones)) / float(max(1, n_stations))  # (R,)
                                    mu = torch.tensordot(r, b.to(torch.float32), dims=([0], [0]))  # scalar: (K,2) ; vector: (M,2,3)
                                else:
                                    mu = None
                            if isinstance(mu, torch.Tensor) and mu.numel() > 0:
                                if mu.ndim == 2 and int(mu.shape[1]) == 2:
                                    metrics["shared_event_latent/station_common_P_rms"] = float(torch.sqrt((mu[:, 0] * mu[:, 0]).mean()).item())
                                    metrics["shared_event_latent/station_common_S_rms"] = float(torch.sqrt((mu[:, 1] * mu[:, 1]).mean()).item())
                                elif mu.ndim == 3 and int(mu.shape[1]) == 2 and int(mu.shape[2]) == 3:
                                    metrics["shared_event_latent/station_common_P_rms"] = float(torch.sqrt((mu[:, 0, :] * mu[:, 0, :]).mean()).item())
                                    metrics["shared_event_latent/station_common_S_rms"] = float(torch.sqrt((mu[:, 1, :] * mu[:, 1, :]).mean()).item())
                    except Exception:
                        pass
                    # If station_basis is enabled, b is in basis space (R,M,2). The physically relevant
                    # station-field inducing coefficients are C_sta = W_sta @ A, shape (n_stations,M,2).
                    # Log their scale as well so we can tell if the station field is actually “big” even
                    # when basis coefficients look small (or vice versa).
                    try:
                        mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                        use_sta_basis = bool(state.params.get("_shared_event_latent_station_basis_enabled", False))
                        if mode == "inducing_gp" and use_sta_basis:
                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                            # Only do this when shapes line up; otherwise skip quietly.
                            if (
                                isinstance(W_sta, torch.Tensor)
                                and W_sta.ndim == 2
                                and int(b.shape[0]) == int(W_sta.shape[1])
                                and int(W_sta.shape[0]) > 0
                                and int(b.shape[1]) > 0
                            ):
                                A = b.to(torch.float32)  # (R,M,2)
                                W = W_sta.to(device=A.device, dtype=torch.float32)  # (S,R)
                                C_P = torch.matmul(W, A[:, :, 0])  # (S,M)
                                C_S = torch.matmul(W, A[:, :, 1])  # (S,M)
                                metrics["shared_event_latent/station_coeff_P_rms"] = float(torch.sqrt((C_P * C_P).mean()).item())
                                metrics["shared_event_latent/station_coeff_S_rms"] = float(torch.sqrt((C_S * C_S).mean()).item())
                                metrics["shared_event_latent/station_coeff_P_mean"] = float(C_P.mean().item())
                                metrics["shared_event_latent/station_coeff_S_mean"] = float(C_S.mean().item())
                    except Exception:
                        pass
                    # Prior energy under the shared_event_latent prior.
                    # This is a useful amplitude diagnostic: too small -> b not used; too large -> b drifting / overpowering data fit.
                    mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                    if mode not in {"full", "inducing_gp", "slowness_inducing_gp", "graph_gmrf"}:
                        mode = "full"
                    if mode in {"inducing_gp", "slowness_inducing_gp"}:
                        # Inducing-point GP coefficients prior:
                        #   c ~ N(0, K_UU^{-1})  ⇔  log p(c) ∝ -0.5 * c^T K_UU c
                        offs = state.params.get("_shared_event_latent_inducing_offsets", None)
                        K_blocks = state.params.get("_shared_event_latent_inducing_K_blocks", None)
                        if isinstance(offs, torch.Tensor) and isinstance(K_blocks, list) and K_blocks:
                            tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                            rho = float(state.params.get("_shared_event_latent_rho_ps", 0.0))
                            if (tau_p > 0.0) and (tau_s > 0.0) and (abs(rho) < 1.0):
                                det = (tau_p * tau_p) * (tau_s * tau_s) * (1.0 - rho * rho)
                                inv00 = (tau_s * tau_s) / det
                                inv11 = (tau_p * tau_p) / det
                                inv01 = (-rho * tau_p * tau_s) / det

                                e00 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                                e11 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                                e01 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                                nb = int(max(0, int(offs.numel()) - 1))
                                for bi in range(nb):
                                    i0 = int(offs[bi].item())
                                    i1 = int(offs[bi + 1].item())
                                    if i1 <= i0:
                                        continue
                                    try:
                                        K = K_blocks[bi]
                                    except Exception:
                                        continue
                                    if not isinstance(K, torch.Tensor) or K.numel() == 0:
                                        continue
                                    Kt = K.to(device=b.device, dtype=b.dtype)
                                    if isinstance(b, torch.Tensor) and b.ndim == 3 and int(b.shape[2]) == 2:
                                        cP = bP[:, i0:i1]
                                        cS = bS[:, i0:i1]
                                        KP = torch.matmul(cP, Kt)
                                        KS = torch.matmul(cS, Kt)
                                        e00 = e00 + (cP * KP).sum()
                                        e11 = e11 + (cS * KS).sum()
                                        e01 = e01 + (cP * KS).sum()
                                    elif isinstance(b, torch.Tensor) and b.ndim == 4 and int(b.shape[2]) == 2 and int(b.shape[3]) == 3:
                                        # Vector slowness coefficients: sum energies across xyz components.
                                        for d in range(3):
                                            cP = b[:, i0:i1, 0, d]
                                            cS = b[:, i0:i1, 1, d]
                                            KP = torch.matmul(cP, Kt)
                                            KS = torch.matmul(cS, Kt)
                                            e00 = e00 + (cP * KP).sum()
                                            e11 = e11 + (cS * KS).sum()
                                            e01 = e01 + (cP * KS).sum()
                                energy = 0.5 * (float(inv00) * e00 + float(inv11) * e11 + 2.0 * float(inv01) * e01)
                                # Degrees of freedom for normalization (scalar or vector)
                                dof = float(max(1, int(b.numel())))
                                metrics["shared_event_latent/prior_energy"] = float(energy.item())
                                metrics["shared_event_latent/prior_energy_per_dof"] = float((energy / dof).item())
                    else:
                        # Full parameterization: Laplacian-GMRF prior over events via fixed graph Q (kNN Laplacian + q_diag I).
                        u = state.params.get("_shared_event_latent_u", None)
                        v = state.params.get("_shared_event_latent_v", None)
                        w = state.params.get("_shared_event_latent_w", None)
                        q_diag = float(state.params.get("_shared_event_latent_q_diag_runtime", state.params.get("_shared_event_latent_q_diag", 0.0)))
                        if isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor):
                            tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                            rho = float(state.params.get("_shared_event_latent_rho_ps", 0.0))
                            if (tau_p > 0.0) and (tau_s > 0.0) and (abs(rho) < 1.0):
                                det = (tau_p * tau_p) * (tau_s * tau_s) * (1.0 - rho * rho)
                                inv00 = (tau_s * tau_s) / det
                                inv11 = (tau_p * tau_p) / det
                                inv01 = (-rho * tau_p * tau_s) / det

                                u_i = u.to(torch.int64)
                                v_i = v.to(torch.int64)
                                w_f = w.to(dtype=b.dtype)

                                def _apply_Q(xSN: torch.Tensor) -> torch.Tensor:
                                    y = xSN * float(max(0.0, q_diag))
                                    if int(u_i.numel()) > 0:
                                        xu = xSN.index_select(1, u_i)
                                        xv = xSN.index_select(1, v_i)
                                        diff = xu - xv  # [S,E]
                                        dw = diff * w_f.unsqueeze(0)
                                        y.index_add_(1, u_i, dw)
                                        y.index_add_(1, v_i, -dw)
                                    return y

                                qP = _apply_Q(bP)
                                qS = _apply_Q(bS)
                                e00 = (bP * qP).sum()
                                e11 = (bS * qS).sum()
                                e01 = (bP * qS).sum()
                                energy = 0.5 * (float(inv00) * e00 + float(inv11) * e11 + 2.0 * float(inv01) * e01)
                                dof = float(max(1, int(bP.numel() + bS.numel())))
                                metrics["shared_event_latent/prior_energy"] = float(energy.item())
                                metrics["shared_event_latent/prior_energy_per_dof"] = float((energy / dof).item())
        except Exception:
            pass

        # Decompose total loss into avg likelihood term + (1/N_total) prior term (prior is constant across minibatches).
        try:
            if bool(state.params.get("_shared_event_latent_enabled", False)) and _want_wandb_group(state.params, "shared_event_latent"):
                σp_now, σs_now = _current_noise_scales(state)
                l_prior = compute_prior_loss(
                    state.dX_src,
                    state.prior_event,
                    σp_now,
                    σs_now,
                    int(state.N),
                    state.params,
                    cluster_ids=state.cluster_ids,
                    cluster_counts=state.cluster_counts,
                    event_precision_matrix=state.event_precision_matrix,
                    corr_error_b=getattr(state, "corr_error_b", None),
                )
                l_prior_f = float(l_prior.detach().cpu().item())
                metrics["shared_event_latent/loss_total"] = float(total_loss_mean)
                metrics["shared_event_latent/loss_prior"] = float(l_prior_f)
                metrics["shared_event_latent/loss_like_avg"] = float(total_loss_mean - l_prior_f)
        except Exception:
            pass

        # Per-epoch stats of delta_b nuisance term (accumulated without per-batch sync).
        try:
            if want_lat_diag and (b_delta_count is not None):
                cnt = float(b_delta_count.detach().cpu().item())
                if cnt > 0.0:
                    mean = (b_delta_sum / b_delta_count)
                    rms = torch.sqrt(b_delta_sumsq / b_delta_count)
                    metrics["shared_event_latent/delta_b_mean"] = float(mean.detach().cpu().item())
                    metrics["shared_event_latent/delta_b_rms"] = float(rms.detach().cpu().item())
                    metrics["shared_event_latent/delta_b_maxabs"] = float(b_delta_maxabs.detach().cpu().item())
                # Endpoint RMS (event-level b at e1/e2), split by phase
                if (b_end_count_p is not None) and (b_end_sumsq_p is not None):
                    cntp = float(b_end_count_p.detach().cpu().item())
                    if cntp > 0.0:
                        brms_p = torch.sqrt(b_end_sumsq_p / b_end_count_p)
                        metrics["shared_event_latent/bP_endpoint_rms"] = float(brms_p.detach().cpu().item())
                if (b_end_count_s is not None) and (b_end_sumsq_s is not None):
                    cnts = float(b_end_count_s.detach().cpu().item())
                    if cnts > 0.0:
                        brms_s = torch.sqrt(b_end_sumsq_s / b_end_count_s)
                        metrics["shared_event_latent/bS_endpoint_rms"] = float(brms_s.detach().cpu().item())
                # slowness_inducing_gp: endpoint u-field RMS (per-phase)
                if (u_end_count_p is not None) and (u_end_sumsq_p is not None):
                    cntp = float(u_end_count_p.detach().cpu().item())
                    if cntp > 0.0:
                        urms_p = torch.sqrt(u_end_sumsq_p / u_end_count_p)
                        metrics["shared_event_latent/uP_endpoint_rms"] = float(urms_p.detach().cpu().item())
                if (u_end_count_s is not None) and (u_end_sumsq_s is not None):
                    cnts = float(u_end_count_s.detach().cpu().item())
                    if cnts > 0.0:
                        urms_s = torch.sqrt(u_end_sumsq_s / u_end_count_s)
                        metrics["shared_event_latent/uS_endpoint_rms"] = float(urms_s.detach().cpu().item())
                if u_end_maxnorm_p is not None:
                    metrics["shared_event_latent/uP_endpoint_maxnorm"] = float(u_end_maxnorm_p.detach().cpu().item())
                if u_end_maxnorm_s is not None:
                    metrics["shared_event_latent/uS_endpoint_maxnorm"] = float(u_end_maxnorm_s.detach().cpu().item())
        except Exception:
            pass

        # For inducing-point shared_event_latent modes, also log coefficient magnitudes.
        # This is the highest-signal debug aid when users report "latents blowing up".
        try:
            if want_lat_diag and bool(state.params.get("_shared_event_latent_enabled", False)) and _want_wandb_group(state.params, "shared_event_latent"):
                mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                tau_units = str(state.params.get("_shared_event_latent_slowness_tau_units", "abs")).strip().lower()
                tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                try:
                    tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                except Exception:
                    tau_p = float(tau_ps) if tau_ps is not None else 0.0
                    tau_s = float(tau_ps) if tau_ps is not None else 0.0
                metrics["shared_event_latent/tau_units"] = float(1.0 if tau_units == "vel_frac" else 0.0)
                metrics["shared_event_latent/tau_p_config"] = float(tau_p)
                metrics["shared_event_latent/tau_s_config"] = float(tau_s)
                # Flag whether EikoNet v(z) arrays are available (vel_frac scaling depends on this).
                try:
                    vp_arr = np.asarray(state.params.get("_eikonet_v1d_vp_km_s", []), dtype=np.float64).reshape(-1)
                    vs_arr = np.asarray(state.params.get("_eikonet_v1d_vs_km_s", []), dtype=np.float64).reshape(-1)
                    ok = bool(vp_arr.size >= 2 and vs_arr.size == vp_arr.size and np.isfinite(np.nanmedian(vp_arr)) and np.isfinite(np.nanmedian(vs_arr)))
                    metrics["shared_event_latent/eikonet_v1d_available"] = float(1.0 if ok else 0.0)
                except Exception:
                    metrics["shared_event_latent/eikonet_v1d_available"] = float(0.0)

                b = getattr(state, "shared_event_latent_b", None)
                if mode in {"inducing_gp", "slowness_inducing_gp"} and isinstance(b, torch.Tensor):
                    bb = b.detach()
                    # Scalar inducing_gp: (S_or_R, M_total, 2)
                    if bb.ndim == 3 and int(bb.shape[2]) == 2:
                        rms_p = torch.sqrt((bb[:, :, 0] * bb[:, :, 0]).mean().to(torch.float32))
                        rms_s = torch.sqrt((bb[:, :, 1] * bb[:, :, 1]).mean().to(torch.float32))
                        maxabs = bb.abs().max().to(torch.float32)
                        metrics["shared_event_latent/b_coeff_rms_p"] = float(rms_p.cpu().item())
                        metrics["shared_event_latent/b_coeff_rms_s"] = float(rms_s.cpu().item())
                        metrics["shared_event_latent/b_coeff_maxabs"] = float(maxabs.cpu().item())
                    # Vector slowness_inducing_gp: (S_or_R, M_total, 2, 3)
                    elif bb.ndim == 4 and int(bb.shape[2]) == 2 and int(bb.shape[3]) == 3:
                        # RMS across stations/ranks, inducing points, and xyz components
                        rms_p = torch.sqrt((bb[:, :, 0, :] * bb[:, :, 0, :]).mean().to(torch.float32))
                        rms_s = torch.sqrt((bb[:, :, 1, :] * bb[:, :, 1, :]).mean().to(torch.float32))
                        maxabs = bb.abs().max().to(torch.float32)
                        metrics["shared_event_latent/b_coeff_rms_p"] = float(rms_p.cpu().item())
                        metrics["shared_event_latent/b_coeff_rms_s"] = float(rms_s.cpu().item())
                        metrics["shared_event_latent/b_coeff_maxabs"] = float(maxabs.cpu().item())

                    # If we're in vel_frac mode, also log the implied slowness amplitude tau/v (s/km).
                    if tau_units == "vel_frac":
                        try:
                            vp = np.asarray(state.params.get("_eikonet_v1d_vp_km_s", []), dtype=np.float64).reshape(-1)
                            vs = np.asarray(state.params.get("_eikonet_v1d_vs_km_s", []), dtype=np.float64).reshape(-1)
                            vp_med = float(np.nanmedian(vp)) if vp.size > 0 else float("nan")
                            vs_med = float(np.nanmedian(vs)) if vs.size > 0 else float("nan")
                            if np.isfinite(vp_med) and vp_med > 0:
                                metrics["shared_event_latent/vp_v1d_med_km_s"] = float(vp_med)
                                metrics["shared_event_latent/tau_p_over_v_med_s_per_km"] = float(tau_p / vp_med)
                            if np.isfinite(vs_med) and vs_med > 0:
                                metrics["shared_event_latent/vs_v1d_med_km_s"] = float(vs_med)
                                metrics["shared_event_latent/tau_s_over_v_med_s_per_km"] = float(tau_s / vs_med)
                        except Exception:
                            pass
        except Exception:
            pass
        # Online ESS: re-log last-known values to avoid W&B gaps (only if we're actually logging ess_online).
        if bool(state.params.get("ess_online_enabled", False)) and _want_wandb_group(state.params, "ess_online"):
            last = getattr(state, "_ess_online_last_metrics", None)
            if isinstance(last, dict) and len(last) > 0:
                metrics.update(last)
            # Debug/health indicators (safe scalars)
            try:
                metrics["ess_online/updated"] = float(1.0 if ess_online_updated_this_epoch else 0.0)
                metrics["ess_online/error_count"] = float(int(getattr(state, "_ess_online_error_count", 0)))
            except Exception:
                pass

        # Optional diagnostic: evaluate loss on a FIXED subset of rows (same indices every time).
        # This helps distinguish "true drift" from minibatch sampling noise and from changing data permutations.
        try:
            if not _want_wandb_group(state.params, "fixed_eval"):
                raise RuntimeError("skip fixed_eval")
            diag = _get_diagnostics_cfg(state.params)
            fe = diag.get("fixed_eval", {}) if isinstance(diag, dict) else {}
            fe_enabled = bool(fe.get("enabled", False))
            if fe_enabled:
                every = int(fe.get("every_epochs", 25))
                if every < 1:
                    every = 25
                if (epoch_index % every) == 0:
                    n_rows = int(fe.get("n_rows", 200_000))
                    n_rows = max(1, min(n_rows, int(state.N)))
                    seed = int(fe.get("seed", 0))
                    bs_eval = int(fe.get("batch_size", 50_000))
                    bs_eval = max(1, bs_eval)

                    # Cache chosen subset so it stays constant across epochs.
                    if "_fixed_eval_rows" not in state.params:
                        rng = np.random.default_rng(seed)
                        idx_np = rng.choice(int(state.N), size=n_rows, replace=False).astype(np.int64)
                        state.params["_fixed_eval_rows"] = idx_np.tolist()
                    idx_np = np.asarray(state.params.get("_fixed_eval_rows", []), dtype=np.int64)
                    if idx_np.size > 0:
                        # Compute weighted mean loss over the subset in chunks
                        total = 0.0
                        denom = 0
                        σp_eval, σs_eval = _current_noise_scales(state)
                        for i0 in range(0, int(idx_np.size), bs_eval):
                            i1 = min(i0 + bs_eval, int(idx_np.size))
                            rows_t = torch.as_tensor(idx_np[i0:i1], device=state.device, dtype=torch.int64)
                            II_b = state.II.index_select(0, rows_t)
                            YY_b = state.YY.index_select(0, rows_t)
                            nuisance_delta = None
                            l = posterior_loss(
                                idx=II_b,
                                y=YY_b,
                                X_src=state.X_src,
                                ΔX_src=state.dX_src,
                                model=state.model,
                                prior_event=state.prior_event,
                                σ_p=σp_eval,
                                σ_s=σs_eval,
                                N_total=state.N,
                                params=state.params,
                                nuisance_delta=nuisance_delta,
                                cluster_ids=state.cluster_ids,
                                cluster_counts=state.cluster_counts,
                                event_precision_matrix=state.event_precision_matrix,
                                corr_error_b=(getattr(state, "corr_error_b", None) if bool(state.params.get("_corr_error_enabled", False)) else None),
                            )
                            # l is mean over this chunk
                            w = int(i1 - i0)
                            total += float(l.item()) * float(w)
                            denom += w
                        if denom > 0:
                            metrics["loss_fixed_eval"] = total / float(denom)
        except Exception:
            pass

        # Optional diagnostic: RMS of residuals (base and/or corrected) on a fixed subset of rows.
        # This is the most direct sanity check for posterior calibration in synthetic/noise-free tests.
        try:
            if not _want_wandb_group(state.params, "resid_rms"):
                raise RuntimeError("skip resid_rms")
            diag = _get_diagnostics_cfg(state.params)
            rr_cfg = diag.get("resid_rms", {}) if isinstance(diag, dict) else {}
            rr_enabled = bool(rr_cfg.get("enabled", False))
            if rr_enabled:
                every = int(rr_cfg.get("every_epochs", 10))
                if every < 1:
                    every = 10
                if (epoch_index % every) == 0:
                    n_rows = int(rr_cfg.get("n_rows", 200_000))
                    n_rows = max(1, min(n_rows, int(state.N)))
                    seed = int(rr_cfg.get("seed", 0))
                    bs_eval = int(rr_cfg.get("batch_size", 50_000))
                    bs_eval = max(1, bs_eval)
                    log_base = bool(rr_cfg.get("log_base", True))
                    log_corr = bool(rr_cfg.get("log_corrected", True))

                    # Cache chosen subset so it stays constant across epochs.
                    key_name = "_resid_rms_rows"
                    if key_name not in state.params:
                        rng = np.random.default_rng(seed)
                        idx_np = rng.choice(int(state.N), size=n_rows, replace=False).astype(np.int64)
                        state.params[key_name] = idx_np.tolist()
                    idx_np = np.asarray(state.params.get(key_name, []), dtype=np.int64)
                    if idx_np.size > 0:
                        sumsq_base_p = 0.0
                        sumsq_base_s = 0.0
                        sumsq_base_all = 0.0
                        sumsq_corr_p = 0.0
                        sumsq_corr_s = 0.0
                        sumsq_corr_all = 0.0
                        n_p = 0
                        n_s = 0
                        n_all = 0

                        # Mean correction: corr_error (if enabled and available).
                        def _corr_error_delta_for_rows(II_b: torch.Tensor, YY_b: torch.Tensor, sta_b: torch.Tensor) -> torch.Tensor | None:
                            try:
                                if not bool(state.params.get("_corr_error_enabled", False)):
                                    return None
                                W = getattr(state, "corr_error_station_basis_W", None)
                                b = getattr(state, "corr_error_b", None)
                                if not (isinstance(b, torch.Tensor) and isinstance(sta_b, torch.Tensor)):
                                    return None
                                if b.ndim != 3 or int(b.shape[2]) != 2:
                                    return None
                                e1 = II_b[:, 0].to(torch.int64)
                                e2 = II_b[:, 1].to(torch.int64)
                                bi = b.index_select(0, e1)
                                bj = b.index_select(0, e2)
                                db = (bj - bi).to(torch.float32)
                                ph = YY_b[:, 4]
                                is_s = (ph >= 0.5)
                                db_phase = torch.where(is_s.view(-1, 1), db[:, :, 1], db[:, :, 0])
                                if isinstance(W, torch.Tensor):
                                    Wr = W.index_select(0, sta_b.to(torch.int64)).to(torch.float32)
                                    return (Wr * db_phase).sum(dim=1)
                                sidx = sta_b.to(torch.int64).view(-1, 1)
                                return db_phase.gather(1, sidx).squeeze(1)
                            except Exception:
                                return None

                        σp_eval, σs_eval = _current_noise_scales(state)
                        sigma_p_base = float(σp_eval.item())
                        sigma_s_base = float(σs_eval.item())

                        for i0 in range(0, int(idx_np.size), bs_eval):
                            i1 = min(i0 + bs_eval, int(idx_np.size))
                            rows_t = torch.as_tensor(idx_np[i0:i1], device=state.device, dtype=torch.int64)
                            II_b = state.II.index_select(0, rows_t)
                            YY_b = state.YY.index_select(0, rows_t)

                            # Base residual: dt_obs - dt_pred (NO nuisance corrections)
                            rb = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().to(torch.float32)
                            ph = YY_b[:, 4].detach()
                            is_p = (ph < 0.5)

                            if log_base:
                                r2 = (rb * rb)
                                sumsq_base_all += float(r2.sum().item())
                                n_all += int(r2.numel())
                                if bool(is_p.any().item()):
                                    sumsq_base_p += float(r2[is_p].sum().item())
                                    n_p += int(is_p.sum().item())
                                if bool((~is_p).any().item()):
                                    sumsq_base_s += float(r2[~is_p].sum().item())
                                    n_s += int((~is_p).sum().item())

                            if log_corr:
                                sta = getattr(state, "row_station_index", None)
                                if isinstance(sta, torch.Tensor):
                                    sta_b = sta.index_select(0, rows_t).to(torch.int64)
                                    dc = _corr_error_delta_for_rows(II_b, YY_b, sta_b)
                                else:
                                    dc = None
                                rc = rb if dc is None else (rb - dc)
                                r2c = (rc * rc)
                                sumsq_corr_all += float(r2c.sum().item())
                                if bool(is_p.any().item()):
                                    sumsq_corr_p += float(r2c[is_p].sum().item())
                                if bool((~is_p).any().item()):
                                    sumsq_corr_s += float(r2c[~is_p].sum().item())

                        # Final RMS
                        metrics["resid_rms/n_rows"] = float(int(idx_np.size))
                        metrics["resid_rms/sigma_base_p"] = float(sigma_p_base)
                        metrics["resid_rms/sigma_base_s"] = float(sigma_s_base)
                        if log_base:
                            metrics["resid_rms/base_all"] = float(np.sqrt(sumsq_base_all / max(1, n_all)))
                            metrics["resid_rms/base_p"] = float(np.sqrt(sumsq_base_p / max(1, n_p))) if n_p > 0 else float("nan")
                            metrics["resid_rms/base_s"] = float(np.sqrt(sumsq_base_s / max(1, n_s))) if n_s > 0 else float("nan")
                        if log_corr:
                            metrics["resid_rms/corr_all"] = float(np.sqrt(sumsq_corr_all / max(1, n_all)))
                            metrics["resid_rms/corr_p"] = float(np.sqrt(sumsq_corr_p / max(1, n_p))) if n_p > 0 else float("nan")
                            metrics["resid_rms/corr_s"] = float(np.sqrt(sumsq_corr_s / max(1, n_s))) if n_s > 0 else float("nan")
        except Exception:
            pass

        # Report hierarchical event prior implied stds if enabled (for W&B parity across phases).
        # Keys match existing W&B dashboards: hier_std_x/y/z/t and hier_corr_zt.
        # Also report spatial correlations for quick health checks of the learned covariance.
        try:
            if not _want_wandb_group(state.params, "priors"):
                raise RuntimeError("skip priors metrics")
            if bool(getattr(state, "hierarchical_prior_enable", False)) and state.event_precision_matrix is not None:
                P0_all = state.event_precision_matrix
                P0_mean = P0_all.mean(dim=0) if isinstance(P0_all, torch.Tensor) and P0_all.ndim == 3 else P0_all
                Cov = torch.linalg.inv(P0_mean)
                stds = torch.sqrt(torch.diag(Cov)).clamp_min(torch.tensor(0.0, device=Cov.device, dtype=Cov.dtype))
                s0 = float(stds[0].item())
                s1 = float(stds[1].item())
                s2 = float(stds[2].item())
                s3 = float(stds[3].item())
                metrics.update({
                    "hier_std_x": s0,
                    "hier_std_y": s1,
                    "hier_std_z": s2,
                    "hier_std_t": s3,
                    "hier_corr_xy": float(Cov[0, 1].item()) / (s0 * s1 + 1e-12),
                    "hier_corr_xz": float(Cov[0, 2].item()) / (s0 * s2 + 1e-12),
                    "hier_corr_yz": float(Cov[1, 2].item()) / (s1 * s2 + 1e-12),
                    "hier_corr_zt": float(Cov[2, 3].item()) / (s2 * s3 + 1e-12),
                })
        except Exception:
            pass

        # --- corr_error diagnostics (W&B; cheap subsampled proxies) ---
        # These answer: is b blowing up? are neighbor diffs reasonable? is the graph present?
        try:
            if not (_want_wandb_group(state.params, "corr_error") or _want_wandb_group(state.params, "priors")):
                raise RuntimeError("skip corr_error metrics")
            if ddp_enabled and (not ddp_is_main):
                raise RuntimeError("skip corr_error metrics on non-main rank")
            if bool(state.params.get("_corr_error_enabled", False)) and isinstance(getattr(state, "corr_error_b", None), torch.Tensor):
                b = getattr(state, "corr_error_b").detach()
                if b.ndim == 3 and int(b.shape[2]) == 2:
                    n_ev = int(b.shape[0])
                    R = int(b.shape[1])
                    metrics["corr_error/R"] = float(R)
                    metrics["corr_error/n_events"] = float(n_ev)
                    metrics["corr_error/radius_km"] = float(state.params.get("_corr_error_event_graph_radius_km", float("nan")))
                    metrics["corr_error/k"] = float(state.params.get("_corr_error_event_graph_k", float("nan")))
                    metrics["corr_error/q_diag"] = float(state.params.get("_corr_error_q_diag", state.params.get("_corr_error_event_graph_q_diag", float("nan"))))
                    metrics["corr_error/graph_dt_s"] = float(state.params.get("_corr_error_event_graph_dt_s", float("nan")))
                    metrics["corr_error/graph_backend_id"] = float(state.params.get("_corr_error_event_graph_backend_id", float("nan")))

                    # Hierarchical tau metrics
                    tau_ps = state.params.get("_corr_error_tau_s", [0.0, 0.0])
                    rho = float(state.params.get("_corr_error_rho_ps", 0.0))
                    metrics["corr_error/tau_p"] = float(tau_ps[0])
                    metrics["corr_error/tau_s"] = float(tau_ps[1])
                    metrics["corr_error/rho_ps"] = float(rho)

                    # Subsample events for amplitude stats.
                    m_ev = int(min(max(1, 4096), n_ev))
                    seed0 = int(state.params.get("runtime_seed", 0) or 0)
                    gen = torch.Generator(device=b.device)
                    try:
                        gen.manual_seed(int(seed0 + 1000003 * int(epoch_index)))
                    except Exception:
                        pass
                    ev_idx = torch.randint(0, n_ev, (m_ev,), device=b.device, dtype=torch.int64, generator=gen)
                    b_s = b.index_select(0, ev_idx)  # [m,R,2]
                    bP = b_s[:, :, 0]
                    bS = b_s[:, :, 1]
                    metrics["corr_error/b_rms_p"] = float(torch.sqrt(torch.mean(bP * bP)).item())
                    metrics["corr_error/b_rms_s"] = float(torch.sqrt(torch.mean(bS * bS)).item())
                    metrics["corr_error/b_maxabs_p"] = float(torch.max(torch.abs(bP)).item())
                    metrics["corr_error/b_maxabs_s"] = float(torch.max(torch.abs(bS)).item())

                    # Edge-difference stats (subsample edges; proxy for Laplacian energy).
                    u = getattr(state, "corr_error_u", state.params.get("_corr_error_u", None))
                    v = getattr(state, "corr_error_v", state.params.get("_corr_error_v", None))
                    w = getattr(state, "corr_error_w", state.params.get("_corr_error_w", None))
                    if isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor):
                        E = int(u.numel())
                        metrics["corr_error/graph_edges"] = float(E)
                        if (E > 0) and (n_ev > 0):
                            metrics["corr_error/graph_deg_mean_approx"] = float(2.0 * float(E) / float(n_ev))
                        if E > 0:
                            m_e = int(min(max(1, 20000), E))
                            e_idx = torch.randint(0, E, (m_e,), device=u.device, dtype=torch.int64, generator=gen)
                            uu = u.index_select(0, e_idx).to(torch.int64)
                            vv = v.index_select(0, e_idx).to(torch.int64)
                            ww = w.index_select(0, e_idx).to(dtype=b.dtype, device=b.device)

                            b_full_P = b[:, :, 0]
                            b_full_S = b[:, :, 1]
                            dP = b_full_P.index_select(0, uu) - b_full_P.index_select(0, vv)  # [m_e,R]
                            dS = b_full_S.index_select(0, uu) - b_full_S.index_select(0, vv)
                            metrics["corr_error/edge_diff_rms_p"] = float(torch.sqrt(torch.mean(dP * dP)).item())
                            metrics["corr_error/edge_diff_rms_s"] = float(torch.sqrt(torch.mean(dS * dS)).item())
                            metrics["corr_error/edge_wdiff_rms_p"] = float(torch.sqrt(torch.mean((dP * dP) * ww.unsqueeze(1))).item())
                            metrics["corr_error/edge_wdiff_rms_s"] = float(torch.sqrt(torch.mean((dS * dS) * ww.unsqueeze(1))).item())
                # Also report RMS/max|.| of the *actual per-row* correction delta_corr (accumulated over the epoch).
                try:
                    if isinstance(corr_dc_sumsq_p, torch.Tensor) and isinstance(corr_dc_count_p, torch.Tensor):
                        s2p = corr_dc_sumsq_p.detach().clone()
                        cP = corr_dc_count_p.detach().clone()
                        mP = corr_dc_maxabs_p.detach().clone() if isinstance(corr_dc_maxabs_p, torch.Tensor) else None
                        s2s = corr_dc_sumsq_s.detach().clone()
                        cS = corr_dc_count_s.detach().clone()
                        mS = corr_dc_maxabs_s.detach().clone() if isinstance(corr_dc_maxabs_s, torch.Tensor) else None
                        if ddp_enabled:
                            dist.all_reduce(s2p, op=dist.ReduceOp.SUM)
                            dist.all_reduce(cP, op=dist.ReduceOp.SUM)
                            dist.all_reduce(s2s, op=dist.ReduceOp.SUM)
                            dist.all_reduce(cS, op=dist.ReduceOp.SUM)
                            if isinstance(mP, torch.Tensor):
                                dist.all_reduce(mP, op=dist.ReduceOp.MAX)
                            if isinstance(mS, torch.Tensor):
                                dist.all_reduce(mS, op=dist.ReduceOp.MAX)
                        if float(cP.item()) > 0.0:
                            metrics["corr_error/delta_corr_rms_p"] = float(torch.sqrt(s2p / cP.clamp_min(1.0)).item())
                            if isinstance(mP, torch.Tensor):
                                metrics["corr_error/delta_corr_maxabs_p"] = float(mP.item())
                        if float(cS.item()) > 0.0:
                            metrics["corr_error/delta_corr_rms_s"] = float(torch.sqrt(s2s / cS.clamp_min(1.0)).item())
                            if isinstance(mS, torch.Tensor):
                                metrics["corr_error/delta_corr_maxabs_s"] = float(mS.item())
                except Exception:
                    pass
        except Exception:
            pass

        # (Laplacian prior diagnostics removed.)
        # Optional: display preconditioner G stats periodically
        try:
            disp_every = int(state.params.get("display_precond_every", 0))
        except Exception:
            disp_every = 0
        if disp_every > 0 and (epoch_index % disp_every == 0) and _want_wandb_group(state.params, "precond"):
            try:
                if hasattr(optimizer, "preconditioner_stats"):
                    stats = optimizer.preconditioner_stats()  # type: ignore[attr-defined]
                    if isinstance(stats, dict):
                        g_min = float(stats.get("min", float("nan")))
                        g_p25 = float(stats.get("p25", float("nan")))
                        g_med = float(stats.get("median", float("nan")))
                        g_p75 = float(stats.get("p75", float("nan")))
                        g_max = float(stats.get("max", float("nan")))
                        if all([x == x for x in (g_p25, g_med, g_p75)]):  # not NaN
                            if (not ddp_enabled) or ddp_is_main:
                                print(f"Preconditioner G stats: p25={g_p25:.3e}, median={g_med:.3e}, p75={g_p75:.3e}")
                        if g_p25 == g_p25:
                            metrics["precond_g_p25"] = g_p25
                        if g_med == g_med:
                            metrics["precond_g_med"] = g_med
                        if g_p75 == g_p75:
                            metrics["precond_g_p75"] = g_p75
                        if g_min == g_min:
                            metrics["precond_g_min"] = g_min
                        if g_max == g_max:
                            metrics["precond_g_max"] = g_max
                    else:
                        # Backwards compat: (min, median, max)
                        try:
                            g_min, g_med, g_max = stats  # type: ignore[misc]
                            if all([x == x for x in (g_min, g_med, g_max)]):  # not NaN
                                if (not ddp_enabled) or ddp_is_main:
                                    print(f"Preconditioner G stats: min={g_min:.3e}, median={g_med:.3e}, max={g_max:.3e}")
                                metrics["precond_g_min"] = float(g_min)
                                metrics["precond_g_med"] = float(g_med)
                                metrics["precond_g_max"] = float(g_max)
                        except Exception:
                            pass
                else:
                    pg0 = optimizer.param_groups[0] if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0 else {}
                    if (not ddp_enabled) or ddp_is_main:
                        print(
                            "Preconditioner G stats: unavailable (preconditioning disabled or not initialized). "
                            f"preconditioning={pg0.get('preconditioning', None)} preconditioner={pg0.get('preconditioner', None)}"
                        )
            except Exception:
                pass

        # Optional: exact ESS update for corr_error_b (blocked latent update) at end of Phase 4 epochs.
        try:
            st_stats = _student_t_scale_update_full(state=state, epoch_index=int(epoch_index))
            if isinstance(st_stats, dict) and st_stats:
                for k, v in st_stats.items():
                    try:
                        metrics[str(k)] = float(v)
                    except Exception:
                        pass
                if (not ddp_enabled) or ddp_is_main:
                    try:
                        info(
                            "student_t_scale lambda: "
                            f"mean={st_stats.get('student_t_scale/lambda_mean', float('nan')):.3g} "
                            f"p50={st_stats.get('student_t_scale/lambda_p50', float('nan')):.3g} "
                            f"p90={st_stats.get('student_t_scale/lambda_p90', float('nan')):.3g} "
                            f"p99={st_stats.get('student_t_scale/lambda_p99', float('nan')):.3g}",
                            section="LIKELIHOOD",
                        )
                    except Exception:
                        pass
        except Exception as e:
            try:
                warn(f"student_t_scale update failed: {e}", section="LIKELIHOOD")
            except Exception:
                pass

        # Optional: exact ESS update for corr_error_b (blocked latent update) at end of Phase 4 epochs.
        if bool(is_sampling):
            try:
                m_ess = _corr_error_ess_update(state=state, epoch_index=int(epoch_index))
                if isinstance(m_ess, dict) and m_ess:
                    for k, v in m_ess.items():
                        try:
                            metrics[str(k)] = float(v)
                        except Exception:
                            pass
            except Exception as e:
                warn(f"corr_error ESS update failed: {e}", section="ESS")
            try:
                m_sl_ess = _slowness_re_ess_update(state=state, epoch_index=int(epoch_index))
                if isinstance(m_sl_ess, dict) and m_sl_ess:
                    for k, v in m_sl_ess.items():
                        try:
                            metrics[str(k)] = float(v)
                        except Exception:
                            pass
            except Exception as e:
                warn(f"slowness_re ESS update failed: {e}", section="ESS")
            try:
                m_dd_ess = _dd_graph_re_ess_update(state=state, epoch_index=int(epoch_index))
                if isinstance(m_dd_ess, dict) and m_dd_ess:
                    for k, v in m_dd_ess.items():
                        try:
                            metrics[str(k)] = float(v)
                        except Exception:
                            pass
            except Exception as e:
                warn(f"dd_graph_re ESS update failed: {e}", section="ESS")

        # Optional: whitening PCG metrics (if available).
        try:
            if bool(state.params.get("_shared_event_re_whitening_enabled", False)):
                metrics["shared_event_re_whitening/pcg_groups"] = float(state.params.get("_shared_event_re_whitening_last_groups_pcg", 0))
                metrics["shared_event_re_whitening/pcg_iters_sum"] = float(state.params.get("_shared_event_re_whitening_last_pcg_iters_sum", 0))
                metrics["shared_event_re_whitening/pcg_iters_max"] = float(state.params.get("_shared_event_re_whitening_last_pcg_iters_max", 0))
                metrics["shared_event_re_whitening/pcg_fail"] = float(state.params.get("_shared_event_re_whitening_last_pcg_fail", 0))
        except Exception:
            pass

        return metrics

