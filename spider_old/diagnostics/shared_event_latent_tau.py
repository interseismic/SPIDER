from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

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
    # Optional CV diagnostics (may be nan/empty if not computed)
    method: str = "moment"
    tau_grid_s: Optional[np.ndarray] = None
    cv_mse_p: Optional[np.ndarray] = None
    cv_mse_s: Optional[np.ndarray] = None


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

def _logspace_centered(center: float, *, decades: float, n: int) -> np.ndarray:
    """
    Return a positive log-spaced grid centered on `center`:
        center * logspace(-decades, +decades, n)
    """
    c = float(center)
    if not (np.isfinite(c) and c > 0.0):
        c = 1e-3
    decades = float(decades)
    if not (np.isfinite(decades) and decades > 0.0):
        decades = 1.0
    n = int(max(3, n))
    xs = np.logspace(-decades, +decades, num=n, base=10.0).astype(np.float64, copy=False)
    return (c * xs).astype(np.float64, copy=False)

def _solve_ridge_event_potentials(
    *,
    n_events: int,
    e1: np.ndarray,
    e2: np.ndarray,
    r: np.ndarray,
    ridge: float,
    rtol: float,
    maxiter: int,
) -> np.ndarray:
    """
    Solve b minimizing:
        sum_k (r_k - (b[e2_k] - b[e1_k]))^2 + ridge * ||b||^2

    This is a fast stand-in for the full shared_event_latent prior when estimating tau:
    - ridge ≈ (sigma/tau)^2 (up to constants)
    - we intentionally keep it simple and scalable (no global Q construction here)
    """
    from scipy.sparse import coo_matrix  # type: ignore
    from scipy.sparse.linalg import cg  # type: ignore

    n = int(n_events)
    if n <= 1 or e1.size == 0:
        return np.zeros((n,), dtype=np.float32)
    e1 = e1.astype(np.int64, copy=False)
    e2 = e2.astype(np.int64, copy=False)
    r = r.astype(np.float64, copy=False)
    ridge = float(ridge)
    ridge = ridge if np.isfinite(ridge) and ridge > 0.0 else 1e-6
    rtol = float(rtol) if np.isfinite(float(rtol)) and float(rtol) > 0 else 1e-6
    maxiter = int(max(50, maxiter))

    data = np.ones((e1.size,), dtype=np.float64)
    L = coo_matrix((data, (e1, e1)), shape=(n, n)) + coo_matrix((data, (e2, e2)), shape=(n, n))
    L = L - coo_matrix((data, (e1, e2)), shape=(n, n)) - coo_matrix((data, (e2, e1)), shape=(n, n))
    # Ridge stabilizes the gauge and encodes tau shrinkage.
    L = L + coo_matrix((np.full((n,), ridge, dtype=np.float64), (np.arange(n), np.arange(n))), shape=(n, n))
    L = L.tocsr()

    b = np.zeros((n,), dtype=np.float64)
    np.add.at(b, e1, -r)
    np.add.at(b, e2, +r)

    x, _info = cg(L, b, atol=0.0, rtol=rtol, maxiter=maxiter)
    x = x.astype(np.float32, copy=False)
    # No need to fix gauge when ridge>0 (system is SPD), but subtracting a reference improves numeric stability.
    if x.size:
        x -= float(x[0])
    return x


@torch.no_grad()
def estimate_shared_event_latent_tau_s(
    *,
    state,
    n_rows: int = 200_000,
    seed: int = 0,
    batch_size: int = 50_000,
    residuals_at: str = "map",  # "map" | "initial"
) -> Optional[SharedEventLatentTauEstimate]:
    """
    Estimate an amplitude scale tau_s (and tau_p) for shared_event_latent from residuals.

    Heuristic:
      residual = (b_e2 - b_e1) + noise
      Var(residual_phi) ≈ Var(Δb_phi) + σ_phi^2
      Var(Δb_phi) ≈ 2 * tau_phi^2   (very rough; assumes weak correlation across event pairs)

    So:
      tau_phi ≈ sqrt(max(Var(residual_phi) - σ_phi^2, 0) / 2).

    This is intended as a post-Phase1 *initialization/tuning* diagnostic, not a strict estimator.

    Args:
      residuals_at:
        - "map": use residuals at the current MAP (state.dX_src)
        - "initial": use residuals at ΔX=0 (input catalog locations). This is useful for synthetic tests
          where the "true" DD residuals are defined relative to the input catalog.
    """
    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None

    # Sample row indices (CPU) then batch-evaluate residuals.
    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    # Choose which ΔX to use for residual evaluation.
    try:
        mode = str(residuals_at).strip().lower()
    except Exception:
        mode = "map"
    if mode not in {"map", "initial"}:
        mode = "map"
    dX_use = state.dX_src
    if mode == "initial":
        dX_use = torch.zeros_like(state.dX_src, device=state.dX_src.device)

    rP = []
    rS = []
    for i0 in range(0, int(rows_np.size), batch_size):
        ii = rows_np[i0 : i0 + batch_size]
        idx = state.II.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        y = state.YY.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        r = compute_residuals(idx, y, state.X_src, dX_use, state.model).detach()
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
        method="moment",
    )

@torch.no_grad()
def estimate_shared_event_latent_tau_cv(
    *,
    state,
    n_rows: int = 200_000,
    seed: int = 0,
    batch_size: int = 50_000,
    holdout_frac: float = 0.2,
    min_edges_per_group: int = 200,
    max_edges_per_group: int = 5000,
    max_groups_per_phase: int = 32,
    grid_decades: float = 1.0,
    grid_size: int = 9,
    cg_rtol: float = 1e-6,
    cg_maxiter: int = 2000,
    residuals_at: str = "map",  # "map" | "initial"
) -> Optional[SharedEventLatentTauEstimate]:
    """
    Cross-validated tau estimate to reduce overfitting of shared_event_latent b.

    Strategy (per phase P/S, station-by-station):
      - sample DD rows, compute residuals r (with b disabled)
      - for each station-phase group, split edges into train/val
      - for each candidate tau, fit b on train by ridge-regularized least squares:
            r ≈ b[e2] - b[e1]
        with ridge ≈ (sigma/tau)^2
      - pick tau minimizing held-out MSE (equivalently held-out Gaussian NLL up to constants)

    This is an empirical-Bayes *predictive* heuristic intended to select a tau that generalizes
    instead of memorizing noise (which can otherwise drive sigma too small).
    """
    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        if getattr(state, "row_station_index", None) is None:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None

    # Need current sigma for mapping tau <-> ridge
    try:
        σp, σs = _current_noise_scales(state)
        sigma_p = float(σp.detach().cpu().item())
        sigma_s = float(σs.detach().cpu().item())
        sigma_p = sigma_p if np.isfinite(sigma_p) and sigma_p > 0 else 1.0
        sigma_s = sigma_s if np.isfinite(sigma_s) and sigma_s > 0 else 1.0
    except Exception:
        sigma_p, sigma_s = 1.0, 1.0

    # Choose which ΔX to use for residual evaluation.
    try:
        mode = str(residuals_at).strip().lower()
    except Exception:
        mode = "map"
    if mode not in {"map", "initial"}:
        mode = "map"
    dX_use = state.dX_src
    if mode == "initial":
        dX_use = torch.zeros_like(state.dX_src, device=state.dX_src.device)

    # Start from the moment estimate as a reasonable center for the tau grid.
    mom = estimate_shared_event_latent_tau_s(state=state, n_rows=n_rows, seed=seed, batch_size=batch_size, residuals_at=str(mode))
    if mom is None:
        return None
    # If moment estimate is ~0 (can happen if residual_std ~ sigma), keep a small positive center.
    tau0_p = float(mom.tau_p_s) if np.isfinite(float(mom.tau_p_s)) and float(mom.tau_p_s) > 0 else float(max(1e-4, 0.5 * sigma_p))
    tau0_s = float(mom.tau_s_s) if np.isfinite(float(mom.tau_s_s)) and float(mom.tau_s_s) > 0 else float(max(1e-4, 0.5 * sigma_s))

    # Use a single shared grid scale so P/S curves are comparable; center on the larger of the two.
    tau_center = float(max(tau0_p, tau0_s, 1e-4))
    tau_grid = _logspace_centered(tau_center, decades=float(grid_decades), n=int(grid_size)).astype(np.float64, copy=False)
    tau_grid = np.clip(tau_grid, 1e-6, 1.0)  # safety: [1 microsecond, 1 second]
    tau_grid = np.unique(tau_grid)
    if tau_grid.size < 3:
        return None

    # Sample row indices (CPU) and compute residuals in batches.
    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    sta_all: list[np.ndarray] = []
    e1_all: list[np.ndarray] = []
    e2_all: list[np.ndarray] = []
    ph_all: list[np.ndarray] = []
    r_all: list[np.ndarray] = []

    for i0 in range(0, int(rows_np.size), batch_size):
        ii = rows_np[i0 : i0 + batch_size]
        rows_t = torch.from_numpy(ii).to(device=state.device, dtype=torch.int64)
        II_b = state.II.index_select(0, rows_t)
        YY_b = state.YY.index_select(0, rows_t)
        r = compute_residuals(II_b, YY_b, state.X_src, dX_use, state.model).detach()
        ph_is_s = (YY_b[:, 4].detach() > 0.5)
        sta = state.row_station_index.index_select(0, rows_t).detach()

        sta_all.append(sta.detach().to("cpu", dtype=torch.int64).numpy())
        e1_all.append(II_b[:, 0].detach().to("cpu", dtype=torch.int64).numpy())
        e2_all.append(II_b[:, 1].detach().to("cpu", dtype=torch.int64).numpy())
        ph_all.append(ph_is_s.detach().to("cpu", dtype=torch.bool).numpy())
        r_all.append(r.detach().to("cpu", dtype=torch.float32).numpy())

    sta = np.concatenate(sta_all, axis=0).astype(np.int64, copy=False)
    e1 = np.concatenate(e1_all, axis=0).astype(np.int64, copy=False)
    e2 = np.concatenate(e2_all, axis=0).astype(np.int64, copy=False)
    ph_is_s = np.concatenate(ph_all, axis=0).astype(np.bool_, copy=False)
    rr = np.concatenate(r_all, axis=0).astype(np.float32, copy=False)

    # Build groups by (station,phase). key = station*2 + phase_bit.
    key = (sta.astype(np.int64, copy=False) << 1) | (ph_is_s.astype(np.int64, copy=False))
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    sta = sta[order]
    e1 = e1[order]
    e2 = e2[order]
    ph_is_s = ph_is_s[order]
    rr = rr[order]

    # Extract groups with enough edges, cap edges per group.
    uniq_keys, start_idx, counts = np.unique(key, return_index=True, return_counts=True)
    groups: list[dict] = []
    for k, s0, cnt in zip(uniq_keys.tolist(), start_idx.tolist(), counts.tolist()):
        cnt_i = int(cnt)
        if cnt_i < int(min_edges_per_group):
            continue
        s1 = int(s0 + cnt_i)
        # Downsample edges per group to control compute
        if cnt_i > int(max_edges_per_group):
            sel = rng.choice(cnt_i, size=int(max_edges_per_group), replace=False)
            sel.sort()
            idx = (int(s0) + sel).astype(np.int64, copy=False)
        else:
            idx = np.arange(int(s0), int(s1), dtype=np.int64)
        sta_k = int(sta[idx[0]])
        is_s = bool(ph_is_s[idx[0]])
        groups.append(
            {
                "sta": sta_k,
                "is_s": is_s,
                "e1": e1[idx].copy(),
                "e2": e2[idx].copy(),
                "r": rr[idx].copy(),
            }
        )
    if not groups:
        return mom

    # Select a subset of groups per phase for speed and representativeness.
    gP = [g for g in groups if not bool(g["is_s"])]
    gS = [g for g in groups if bool(g["is_s"])]
    rng.shuffle(gP)
    rng.shuffle(gS)
    gP = gP[: int(max(1, max_groups_per_phase))] if gP else []
    gS = gS[: int(max(1, max_groups_per_phase))] if gS else []

    def _cv_for_phase(gs: list[dict], sigma: float) -> Tuple[np.ndarray, int]:
        if not gs:
            return np.full((tau_grid.size,), np.nan, dtype=np.float64), 0
        sigma = float(sigma) if np.isfinite(float(sigma)) and float(sigma) > 0 else 1.0
        mse = np.zeros((tau_grid.size,), dtype=np.float64)
        n_used = 0
        for g in gs:
            e1g = g["e1"].astype(np.int64, copy=False)
            e2g = g["e2"].astype(np.int64, copy=False)
            rg = g["r"].astype(np.float32, copy=False)
            M = int(rg.shape[0])
            if M < int(min_edges_per_group):
                continue

            # Build local node set on all edges (train+val) so val edges always index into b.
            nodes = np.unique(np.concatenate([e1g, e2g], axis=0))
            if nodes.size < 2:
                continue
            nodes.sort()
            inv1 = np.searchsorted(nodes, e1g).astype(np.int64, copy=False)
            inv2 = np.searchsorted(nodes, e2g).astype(np.int64, copy=False)
            n = int(nodes.size)

            # Fixed split per group (reused across taus)
            perm = rng.permutation(M)
            n_val = int(max(1, int(np.floor(float(holdout_frac) * M))))
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            if tr_idx.size < 2 or val_idx.size < 1:
                continue

            e1_tr = inv1[tr_idx]
            e2_tr = inv2[tr_idx]
            r_tr = rg[tr_idx].astype(np.float32, copy=False)
            e1_va = inv1[val_idx]
            e2_va = inv2[val_idx]
            r_va = rg[val_idx].astype(np.float32, copy=False)

            for ti, tau in enumerate(tau_grid):
                t = float(tau)
                # ridge ≈ (sigma/tau)^2
                ridge = (sigma / max(t, 1e-12)) ** 2
                b = _solve_ridge_event_potentials(
                    n_events=n,
                    e1=e1_tr,
                    e2=e2_tr,
                    r=r_tr,
                    ridge=float(ridge),
                    rtol=float(cg_rtol),
                    maxiter=int(cg_maxiter),
                )
                pred = (b[e2_va] - b[e1_va]).astype(np.float32, copy=False)
                err = (r_va - pred).astype(np.float64, copy=False)
                mse[ti] += float(np.mean(err * err))
            n_used += 1

        if n_used > 0:
            mse = mse / float(n_used)
        else:
            mse[:] = np.nan
        return mse, int(n_used)

    mse_p, n_g_p = _cv_for_phase(gP, float(sigma_p))
    mse_s, n_g_s = _cv_for_phase(gS, float(sigma_s))

    def _pick_tau(mse: np.ndarray, sigma: float) -> float:
        if mse is None or mse.size == 0 or not np.isfinite(mse).any():
            return float("nan")
        i = int(np.nanargmin(mse))
        t = float(tau_grid[i])
        # Basic sanity clamp relative to sigma
        sigma = float(sigma)
        if np.isfinite(sigma) and sigma > 0:
            t = float(np.clip(t, 0.05 * sigma, 20.0 * sigma))
        return t

    tau_p_cv = _pick_tau(mse_p, float(sigma_p))
    tau_s_cv = _pick_tau(mse_s, float(sigma_s))

    # Use CV taus when available; fall back to moment otherwise.
    tau_p = float(tau_p_cv) if np.isfinite(tau_p_cv) and tau_p_cv > 0 else float(mom.tau_p_s)
    tau_s = float(tau_s_cv) if np.isfinite(tau_s_cv) and tau_s_cv > 0 else float(mom.tau_s_s)

    return SharedEventLatentTauEstimate(
        tau_p_s=float(tau_p),
        tau_s_s=float(tau_s),
        resid_std_p_s=float(mom.resid_std_p_s),
        resid_std_s_s=float(mom.resid_std_s_s),
        sigma_p_s=float(mom.sigma_p_s),
        sigma_s_s=float(mom.sigma_s_s),
        n_rows_used_p=int(mom.n_rows_used_p),
        n_rows_used_s=int(mom.n_rows_used_s),
        n_rows_used=int(mom.n_rows_used),
        method="cv",
        tau_grid_s=tau_grid.astype(np.float64, copy=False),
        cv_mse_p=mse_p.astype(np.float64, copy=False),
        cv_mse_s=mse_s.astype(np.float64, copy=False),
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

    method = "moment"
    try:
        if isinstance(cfg, dict) and "method" in cfg and cfg.get("method", None) is not None:
            method = str(cfg.get("method")).strip().lower()
    except Exception:
        method = "moment"
    if method not in {"moment", "cv"}:
        method = "moment"

    n_rows_i = int(cfg.get("n_rows", 200_000)) if isinstance(cfg, dict) else 200_000
    seed_i = int(cfg.get("seed", 0)) if isinstance(cfg, dict) else 0
    bs_i = int(cfg.get("batch_size", 50_000)) if isinstance(cfg, dict) else 50_000
    # Optional: choose residual reference for tau estimation.
    #   inference.diagnostics.shared_event_latent_tau_estimate.residuals_at = "map" | "initial"
    try:
        residuals_at = str(cfg.get("residuals_at", "map")).strip().lower() if isinstance(cfg, dict) else "map"
    except Exception:
        residuals_at = "map"
    if residuals_at not in {"map", "initial"}:
        residuals_at = "map"

    if method == "cv":
        # CV knobs (all optional)
        holdout_frac = float(cfg.get("holdout_frac", 0.2)) if isinstance(cfg, dict) else 0.2
        min_edges = int(cfg.get("min_edges_per_group", 200)) if isinstance(cfg, dict) else 200
        max_edges = int(cfg.get("max_edges_per_group", 5000)) if isinstance(cfg, dict) else 5000
        max_groups = int(cfg.get("max_groups_per_phase", 32)) if isinstance(cfg, dict) else 32
        grid_decades = float(cfg.get("grid_decades", 1.0)) if isinstance(cfg, dict) else 1.0
        grid_size = int(cfg.get("grid_size", 9)) if isinstance(cfg, dict) else 9
        cg_rtol = float(cfg.get("cg_rtol", 1e-6)) if isinstance(cfg, dict) else 1e-6
        cg_maxiter = int(cfg.get("cg_maxiter", 2000)) if isinstance(cfg, dict) else 2000

        try:
            est = estimate_shared_event_latent_tau_cv(
                state=state,
                n_rows=n_rows_i,
                seed=seed_i,
                batch_size=bs_i,
                holdout_frac=holdout_frac,
                min_edges_per_group=min_edges,
                max_edges_per_group=max_edges,
                max_groups_per_phase=max_groups,
                grid_decades=grid_decades,
                grid_size=grid_size,
                cg_rtol=cg_rtol,
                cg_maxiter=cg_maxiter,
                residuals_at=str(residuals_at),
            )
        except Exception:
            # If SciPy isn't available or CV fails, fall back to moment.
            est = estimate_shared_event_latent_tau_s(state=state, n_rows=n_rows_i, seed=seed_i, batch_size=bs_i, residuals_at=str(residuals_at))
    else:
        est = estimate_shared_event_latent_tau_s(state=state, n_rows=n_rows_i, seed=seed_i, batch_size=bs_i, residuals_at=str(residuals_at))

    if est is None:
        return

    state.params["_shared_event_latent_tau_est_p_s"] = float(est.tau_p_s)
    state.params["_shared_event_latent_tau_est_s_s"] = float(est.tau_s_s)

    # Optional: apply estimated tau immediately for Phase 2–4 (before b is initialized).
    apply = False
    apply_scale = 1.0
    try:
        if isinstance(cfg, dict) and "apply" in cfg and cfg.get("apply", None) is not None:
            apply = bool(cfg.get("apply"))
        if isinstance(cfg, dict) and "apply_scale" in cfg and cfg.get("apply_scale", None) is not None:
            apply_scale = float(cfg.get("apply_scale"))
    except Exception:
        apply = False
        apply_scale = 1.0
    if not (np.isfinite(apply_scale) and apply_scale > 0.0):
        apply_scale = 1.0

    if apply:
        tau_p_use = float(est.tau_p_s) * float(apply_scale)
        tau_s_use = float(est.tau_s_s) * float(apply_scale)
        # Internal key used by the prior code path
        state.params["_shared_event_latent_tau_s"] = [float(tau_p_use), float(tau_s_use)]
        # Best-effort update nested config (for logging/debugging)
        try:
            m = state.params.get("model", None)
            lk = m.get("likelihood", None) if isinstance(m, dict) else None
            se = lk.get("shared_event_latent", None) if isinstance(lk, dict) else None
            if isinstance(se, dict):
                se["tau_s"] = [float(tau_p_use), float(tau_s_use)]
        except Exception:
            pass
    try:
        print(
            "Shared-event-latent tau estimate (post Phase 1): "
            f"[method={getattr(est, 'method', 'moment')}] "
            f"tau_p≈{est.tau_p_s:.3g}s, tau_s≈{est.tau_s_s:.3g}s "
            f"(resid_std_p≈{est.resid_std_p_s:.3g}s, resid_std_s≈{est.resid_std_s_s:.3g}s, "
            f"sigma_p≈{est.sigma_p_s:.3g}s, sigma_s≈{est.sigma_s_s:.3g}s, rows={est.n_rows_used})",
            flush=True,
        )
        if apply:
            print(
                f"Shared-event-latent tau applied for Phase 2–4: "
                f"tau_p={float(state.params['_shared_event_latent_tau_s'][0]):.3g}s "
                f"tau_s={float(state.params['_shared_event_latent_tau_s'][1]):.3g}s (apply_scale={apply_scale:g})",
                flush=True,
            )
    except Exception:
        pass


