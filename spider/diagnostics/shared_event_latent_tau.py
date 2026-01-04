from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from spider.core.modeling import compute_residuals, med_abs_dev_torch
from spider.core.state import _current_noise_scales


@dataclass
class SharedEventLatentTauEstimate:
    tau_p_s: float
    tau_s_s: float
    resid_std_p_s: float
    resid_std_s_s: float
    sigma_p_s: float
    sigma_s_s: float
    n_rows_used_p: int
    n_rows_used_s: int
    n_rows_used: int


def _robust_std_from_residuals(r: torch.Tensor) -> float:
    """
    Robust scale estimate in seconds from residual samples.
    Uses MAD with Gaussian consistency factor.
    """
    if not isinstance(r, torch.Tensor) or r.numel() == 0:
        return float("nan")
    mad = med_abs_dev_torch(r).detach().float().cpu().item()
    if not np.isfinite(mad) or mad <= 0.0:
        # Fallback: RMS
        rr = r.detach().float().cpu()
        return float(torch.sqrt(torch.mean(rr * rr)).item())
    return float(1.4826 * mad)


@torch.no_grad()
def estimate_shared_event_latent_tau_s(
    *,
    state,
    n_rows: int = 200_000,
    seed: int = 0,
    batch_size: int = 50_000,
) -> Optional[SharedEventLatentTauEstimate]:
    """
    Estimate an amplitude scale tau_s (and tau_p) for shared_event_latent from MAP residuals.

    Heuristic:
      residual = (b_e2 - b_e1) + noise
      Var(residual_phi) ≈ Var(Δb_phi) + σ_phi^2
      Var(Δb_phi) ≈ 2 * tau_phi^2   (very rough; assumes weak correlation across event pairs)

    So:
      tau_phi ≈ sqrt(max(Var(residual_phi) - σ_phi^2, 0) / 2).

    This is intended as a post-Phase1 *initialization/tuning* diagnostic, not a strict estimator.
    """
    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None

    # Sample row indices (CPU) then batch-evaluate residuals at MAP.
    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    rP = []
    rS = []
    for i0 in range(0, int(rows_np.size), batch_size):
        ii = rows_np[i0 : i0 + batch_size]
        idx = state.II.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        y = state.YY.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        r = compute_residuals(idx, y, state.X_src, state.dX_src, state.model).detach()
        # phase col is y[:,4]; <0.5 P, >0.5 S
        ph = y[:, 4].detach()
        mS = (ph > 0.5)
        if bool((~mS).any()):
            rP.append(r[~mS].detach().to("cpu", dtype=torch.float32))
        if bool(mS.any()):
            rS.append(r[mS].detach().to("cpu", dtype=torch.float32))

    if len(rP) == 0 and len(rS) == 0:
        return None
    rP_all = torch.cat(rP, dim=0) if len(rP) else torch.empty((0,), dtype=torch.float32)
    rS_all = torch.cat(rS, dim=0) if len(rS) else torch.empty((0,), dtype=torch.float32)

    # Robust residual std (seconds)
    resid_std_p = _robust_std_from_residuals(rP_all)
    resid_std_s = _robust_std_from_residuals(rS_all)

    # Current noise scales σ_p/σ_s (seconds)
    try:
        σp, σs = _current_noise_scales(state)
        sigma_p = float(σp.detach().cpu().item())
        sigma_s = float(σs.detach().cpu().item())
    except Exception:
        sigma_p, sigma_s = float("nan"), float("nan")

    def _tau_from(resid_std: float, sigma: float) -> float:
        if not (np.isfinite(resid_std) and resid_std > 0.0):
            return float("nan")
        if not (np.isfinite(sigma) and sigma >= 0.0):
            sigma = 0.0
        v = max(0.0, resid_std * resid_std - sigma * sigma)
        return float(np.sqrt(v / 2.0))

    tau_p = _tau_from(resid_std_p, sigma_p)
    tau_s = _tau_from(resid_std_s, sigma_s)

    return SharedEventLatentTauEstimate(
        tau_p_s=float(tau_p),
        tau_s_s=float(tau_s),
        resid_std_p_s=float(resid_std_p),
        resid_std_s_s=float(resid_std_s),
        sigma_p_s=float(sigma_p),
        sigma_s_s=float(sigma_s),
        n_rows_used_p=int(rP_all.numel()),
        n_rows_used_s=int(rS_all.numel()),
        n_rows_used=int(rP_all.numel() + rS_all.numel()),
    )


def maybe_estimate_shared_event_latent_tau_after_phase1(*, state) -> None:
    """
    Convenience wrapper to run after Phase 1.
    Reads optional knobs from:
      inference.diagnostics.shared_event_latent_tau_estimate {enabled,n_rows,seed,batch_size}

    Defaults to enabled when shared_event_latent is enabled.
    """
    try:
        if not bool(state.params.get("_shared_event_latent_enabled", False)):
            return
    except Exception:
        return

    dg = None
    try:
        inf = state.params.get("inference", None)
        dg = (inf.get("diagnostics", None) if isinstance(inf, dict) else None)
    except Exception:
        dg = None
    cfg = {}
    try:
        cfg = dg.get("shared_event_latent_tau_estimate", {}) if isinstance(dg, dict) else {}
    except Exception:
        cfg = {}

    enabled = True
    try:
        if isinstance(cfg, dict) and "enabled" in cfg and cfg.get("enabled", None) is not None:
            enabled = bool(cfg.get("enabled"))
    except Exception:
        enabled = True
    if not enabled:
        return

    est = estimate_shared_event_latent_tau_s(
        state=state,
        n_rows=int(cfg.get("n_rows", 200_000)) if isinstance(cfg, dict) else 200_000,
        seed=int(cfg.get("seed", 0)) if isinstance(cfg, dict) else 0,
        batch_size=int(cfg.get("batch_size", 50_000)) if isinstance(cfg, dict) else 50_000,
    )
    if est is None:
        return

    state.params["_shared_event_latent_tau_est_p_s"] = float(est.tau_p_s)
    state.params["_shared_event_latent_tau_est_s_s"] = float(est.tau_s_s)
    try:
        print(
            "Shared-event-latent tau estimate (post Phase 1): "
            f"tau_p≈{est.tau_p_s:.3g}s, tau_s≈{est.tau_s_s:.3g}s "
            f"(resid_std_p≈{est.resid_std_p_s:.3g}s, resid_std_s≈{est.resid_std_s_s:.3g}s, "
            f"sigma_p≈{est.sigma_p_s:.3g}s, sigma_s≈{est.sigma_s_s:.3g}s, rows={est.n_rows_used})",
            flush=True,
        )
    except Exception:
        pass


