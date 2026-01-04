from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from spider.core.modeling import compute_residuals
from spider.core.state import _current_noise_scales


@dataclass
class SharedEventLatentEventEllEstimate:
    ell_p_km: float
    ell_s_km: float
    ell_km: float
    plateau_p: float
    plateau_s: float
    n_rows_used: int
    n_events_used_p: int
    n_events_used_s: int
    n_pairs_used: int


def _variogram_half_plateau_ell_pairs(
    *,
    coords_km: torch.Tensor,  # (M,2) float32 CPU
    vals: torch.Tensor,       # (M,)  float32 CPU
    n_pairs: int,
    n_bins: int,
    frac: float,
    seed: int,
    curve_out: Optional[dict] = None,
    max_dist_km: Optional[float] = None,
    log_bins: bool = False,
) -> Tuple[float, float, int]:
    """
    Approximate semivariogram by sampling random pairs (avoids O(M^2)).
    Returns (ell_km, plateau, n_pairs_used).
    """
    M = int(coords_km.shape[0])
    if M < 2:
        return float("nan"), float("nan"), 0
    n_bins = max(5, int(n_bins))
    frac = float(frac)
    if not (0.0 < frac < 1.0):
        frac = 0.5
    n_pairs = int(max(1, n_pairs))
    # cap for safety
    n_pairs = int(min(n_pairs, 2_000_000))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    i = torch.randint(0, M, (n_pairs,), generator=gen, dtype=torch.int64)
    # Ensure j != i by construction
    j_off = torch.randint(1, M, (n_pairs,), generator=gen, dtype=torch.int64)
    j = (i + j_off) % M

    di = coords_km.index_select(0, i) - coords_km.index_select(0, j)
    d = torch.sqrt(torch.clamp_min((di * di).sum(dim=1), 0.0)).to(torch.float32)
    dv = vals.index_select(0, i) - vals.index_select(0, j)
    g = (0.5 * dv * dv).to(torch.float32)

    # Optional: restrict to short-range pairs only (useful for visually stable variograms
    # over a specific range, e.g. 0–4 km). This also makes `n_bins` meaningful within that range.
    try:
        if max_dist_km is not None and float(max_dist_km) > 0:
            md = float(max_dist_km)
            m = d <= md
            if bool(m.any().item()):
                d = d[m]
                g = g[m]
    except Exception:
        pass

    dmax = float(d.max().item()) if d.numel() else float("nan")
    if not np.isfinite(dmax) or dmax <= 0:
        return float("nan"), float("nan"), int(d.numel())

    if bool(log_bins) and float(dmax) > 0:
        # Log-spaced edges on (0, dmax], with an explicit 0 edge.
        # This improves resolution near 0 km where correlation-length cues live.
        eps = float(dmax) * 1e-3  # 0.1% of range (e.g., 4 km -> 4 m)
        eps = eps if np.isfinite(eps) and eps > 0 else 1e-3
        e1 = torch.logspace(
            float(np.log10(eps)),
            float(np.log10(float(dmax))),
            steps=int(n_bins),
            dtype=torch.float32,
        )
        edges = torch.cat([torch.zeros((1,), dtype=torch.float32), e1], dim=0)
    else:
        edges = torch.linspace(0.0, dmax, steps=n_bins + 1, dtype=torch.float32)
    bi = torch.bucketize(d, edges, right=False) - 1
    bi = bi.clamp(0, n_bins - 1)

    g_sum = torch.zeros((n_bins,), dtype=torch.float64)
    g_cnt = torch.zeros((n_bins,), dtype=torch.float64)
    g_sum.scatter_add_(0, bi.to(torch.int64), g.to(torch.float64))
    g_cnt.scatter_add_(0, bi.to(torch.int64), torch.ones_like(g, dtype=torch.float64))
    g_mean = (g_sum / torch.clamp_min(g_cnt, 1.0)).to(torch.float32)
    if bool(log_bins):
        # Geometric mean for log-spaced bins; first bin (0, e1] gets a simple midpoint.
        c0 = 0.5 * edges[1]
        cg = torch.sqrt(torch.clamp_min(edges[1:-1] * edges[2:], 0.0))
        centers = torch.cat([c0.view(1), cg], dim=0)
    else:
        centers = 0.5 * (edges[:-1] + edges[1:])

    tail0 = int(np.floor(0.8 * n_bins))
    tail = g_mean[tail0:]
    tail_cnt = g_cnt[tail0:]
    tail = tail[tail_cnt > 0]
    if tail.numel() == 0:
        plateau = float("nan")
    else:
        plateau = float(torch.median(tail).item())
    if not np.isfinite(plateau) or plateau <= 0:
        if isinstance(curve_out, dict):
            try:
                curve_out.clear()
                curve_out["centers_km"] = centers.detach().cpu().numpy().astype(np.float32, copy=False)
                curve_out["gamma"] = g_mean.detach().cpu().numpy().astype(np.float32, copy=False)
                curve_out["count"] = g_cnt.detach().cpu().numpy().astype(np.float32, copy=False)
                curve_out["plateau"] = np.asarray([plateau], dtype=np.float32)
            except Exception:
                pass
        return float("nan"), float(plateau), int(d.numel())

    # Nugget correction (see station_basis_ell): avoid ell collapsing to ~0 when short-range
    # noise dominates (common for S picks).
    head1 = max(1, int(np.floor(0.1 * n_bins)))
    head = g_mean[:head1]
    head_cnt = g_cnt[:head1]
    head = head[head_cnt > 0]
    nugget = float(torch.median(head).item()) if head.numel() > 0 else 0.0
    if (not np.isfinite(nugget)) or nugget < 0.0:
        nugget = 0.0
    sill = float(plateau) - float(nugget)
    if np.isfinite(sill) and sill > 0:
        target = float(nugget) + float(frac) * float(sill)
    else:
        target = float(frac) * plateau
    ok = (g_cnt > 0) & (g_mean >= target)
    if not bool(ok.any().item()):
        if isinstance(curve_out, dict):
            try:
                curve_out.clear()
                curve_out["centers_km"] = centers.detach().cpu().numpy().astype(np.float32, copy=False)
                curve_out["gamma"] = g_mean.detach().cpu().numpy().astype(np.float32, copy=False)
                curve_out["count"] = g_cnt.detach().cpu().numpy().astype(np.float32, copy=False)
                curve_out["plateau"] = np.asarray([plateau], dtype=np.float32)
                curve_out["nugget"] = np.asarray([nugget], dtype=np.float32)
                curve_out["target"] = np.asarray([target], dtype=np.float32)
            except Exception:
                pass
        return float("nan"), float(plateau), int(d.numel())
    k = int(torch.nonzero(ok, as_tuple=False)[0].item())
    ell = float(centers[k].item())
    if isinstance(curve_out, dict):
        try:
            curve_out.clear()
            curve_out["centers_km"] = centers.detach().cpu().numpy().astype(np.float32, copy=False)
            curve_out["gamma"] = g_mean.detach().cpu().numpy().astype(np.float32, copy=False)
            curve_out["count"] = g_cnt.detach().cpu().numpy().astype(np.float32, copy=False)
            curve_out["plateau"] = np.asarray([plateau], dtype=np.float32)
            curve_out["nugget"] = np.asarray([nugget], dtype=np.float32)
            curve_out["target"] = np.asarray([target], dtype=np.float32)
        except Exception:
            pass
    return float(ell), float(plateau), int(d.numel())


def _solve_event_potentials_subgraph(
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
    Solve g minimizing sum (r - (g[e2]-g[e1]))^2 + ridge*||g||^2 over a subgraph.
    Returns g (n_events,) float32 with gauge fixed (g[0]=0).
    """
    from scipy.sparse import coo_matrix  # type: ignore
    from scipy.sparse.linalg import cg  # type: ignore

    n = int(n_events)
    if n <= 1 or e1.size == 0:
        return np.zeros((n,), dtype=np.float32)
    e1 = e1.astype(np.int64, copy=False)
    e2 = e2.astype(np.int64, copy=False)
    r = r.astype(np.float64, copy=False)
    ridge = float(max(ridge, 0.0))
    rtol = float(rtol) if np.isfinite(float(rtol)) and float(rtol) > 0 else 1e-6
    maxiter = int(max(10, maxiter))

    data = np.ones((e1.size,), dtype=np.float64)
    L = coo_matrix((data, (e1, e1)), shape=(n, n)) + coo_matrix((data, (e2, e2)), shape=(n, n))
    L = L - coo_matrix((data, (e1, e2)), shape=(n, n)) - coo_matrix((data, (e2, e1)), shape=(n, n))
    if ridge > 0:
        L = L + coo_matrix((np.full((n,), ridge, dtype=np.float64), (np.arange(n), np.arange(n))), shape=(n, n))
    L = L.tocsr()

    b = np.zeros((n,), dtype=np.float64)
    np.add.at(b, e1, -r)
    np.add.at(b, e2, +r)

    g, _info = cg(L, b, atol=0.0, rtol=rtol, maxiter=maxiter)
    g = g.astype(np.float32, copy=False)
    if g.size:
        g -= float(g[0])
    return g


@torch.no_grad()
def estimate_shared_event_latent_event_ell_km(
    *,
    state,
    n_rows: int = 100_000,
    seed: int = 0,
    batch_size: int = 50_000,
    n_bins: int = 20,
    frac_of_plateau: float = 0.5,
    ridge: float = 1e-3,
    rtol: float = 1e-6,
    maxiter: int = 2000,
    variogram_pairs: int = 200_000,
    curves_out: Optional[dict] = None,
    max_dist_km: Optional[float] = None,
    log_bins: bool = False,
) -> Optional[SharedEventLatentEventEllEstimate]:
    """
    Estimate an event-space length scale ell_km for shared_event_latent from MAP residual structure.

    Approach (per phase):
      - sample DD rows
      - compute residuals at MAP
      - fit a scalar per-event potential g by least squares on DD edges: r ≈ g[e2]-g[e1]
      - compute approximate variogram of g vs event XY using random pairs, return half-plateau distance

    Notes:
      - This avoids O(n_events^2) by (a) solving only on the subgraph induced by sampled rows and
        (b) using pair-subsampled variograms.
    """
    try:
        if int(getattr(state, "N", 0)) <= 0:
            return None
        N = int(state.N)
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, batch_size))
        Ne = int(state.X_src.shape[0])
        if Ne <= 1:
            return None
    except Exception:
        return None

    # Sample row indices (CPU)
    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    e1_all = []
    e2_all = []
    ph_all = []
    r_all = []
    # Standardize residuals by phase noise scale so P/S ell are not driven apart by nugget magnitude.
    try:
        σp, σs = _current_noise_scales(state)
        σp_f = float(σp.detach().cpu().item())
        σs_f = float(σs.detach().cpu().item())
        σp_f = σp_f if np.isfinite(σp_f) and σp_f > 0 else 1.0
        σs_f = σs_f if np.isfinite(σs_f) and σs_f > 0 else 1.0
    except Exception:
        σp_f, σs_f = 1.0, 1.0

    for i0 in range(0, int(rows_np.size), batch_size):
        i1 = min(i0 + batch_size, int(rows_np.size))
        rows_t = torch.as_tensor(rows_np[i0:i1], device=state.device, dtype=torch.int64)
        II_b = state.II.index_select(0, rows_t)
        YY_b = state.YY.index_select(0, rows_t)
        ph = (YY_b[:, 4].detach().cpu() >= 0.5).to(torch.bool)  # True = S
        r = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().to(torch.float32).cpu()
        r = torch.where(ph, r / float(σs_f), r / float(σp_f))
        e1_all.append(II_b[:, 0].detach().cpu().to(torch.int64))
        e2_all.append(II_b[:, 1].detach().cpu().to(torch.int64))
        ph_all.append(ph)
        r_all.append(r)

    e1 = torch.cat(e1_all, dim=0).numpy()
    e2 = torch.cat(e2_all, dim=0).numpy()
    ph_is_s = torch.cat(ph_all, dim=0).numpy()
    r = torch.cat(r_all, dim=0).numpy().astype(np.float32, copy=False)

    # Event XY at MAP
    Xtot = (state.X_src + state.dX_src).detach().cpu()
    XY = Xtot[:, 0:2].to(torch.float32)  # (Ne,2)

    def _per_phase(mask: np.ndarray, seed_off: int, out_key: str) -> Tuple[float, float, int, int]:
        if mask.sum() < 10:
            return float("nan"), float("nan"), 0, 0
        e1m = e1[mask]
        e2m = e2[mask]
        rm = r[mask]
        # Subgraph relabel to shrink solve
        nodes = np.unique(np.concatenate([e1m, e2m], axis=0))
        M = int(nodes.size)
        if M < 2:
            return float("nan"), float("nan"), M, 0
        nodes.sort()
        inv1 = np.searchsorted(nodes, e1m).astype(np.int64, copy=False)
        inv2 = np.searchsorted(nodes, e2m).astype(np.int64, copy=False)
        g = _solve_event_potentials_subgraph(
            n_events=M, e1=inv1, e2=inv2, r=rm, ridge=ridge, rtol=rtol, maxiter=maxiter
        )
        coords = XY.index_select(0, torch.from_numpy(nodes).to(torch.int64)).cpu()
        vals = torch.from_numpy(g).to(torch.float32).cpu()
        # Expose (coords, vals) used for variogram/correlation plots when requested.
        if isinstance(curves_out, dict):
            try:
                cd = curves_out.get("_corr_data", None)
                if not isinstance(cd, dict):
                    cd = {}
                    curves_out["_corr_data"] = cd
                cd[out_key] = {
                    "coords_km": coords.detach().cpu().numpy().astype(np.float32, copy=False),
                    "vals": vals.detach().cpu().numpy().astype(np.float32, copy=False),
                }
            except Exception:
                pass
        curve = {}
        ell, plat, n_pairs_used = _variogram_half_plateau_ell_pairs(
            coords_km=coords,
            vals=vals,
            n_pairs=int(variogram_pairs),
            n_bins=int(n_bins),
            frac=float(frac_of_plateau),
            seed=int(seed) + int(seed_off),
            curve_out=curve if isinstance(curves_out, dict) else None,
            max_dist_km=max_dist_km,
            log_bins=bool(log_bins),
        )
        if isinstance(curves_out, dict):
            try:
                curves_out[out_key] = dict(curve)
            except Exception:
                pass
        return float(ell), float(plat), int(M), int(n_pairs_used)

    def _joint(seed_off: int, out_key: str) -> Tuple[float, float, int, int]:
        # Use all (P+S) edges together. This matches the model: ell_km is a shared hyperparameter,
        # while P/S are handled via tau_p/tau_s and rho_ps.
        mask = np.ones((r.shape[0],), dtype=bool)
        e1m = e1[mask]
        e2m = e2[mask]
        rm = r[mask]
        nodes = np.unique(np.concatenate([e1m, e2m], axis=0))
        M = int(nodes.size)
        if M < 2:
            return float("nan"), float("nan"), M, 0
        nodes.sort()
        inv1 = np.searchsorted(nodes, e1m).astype(np.int64, copy=False)
        inv2 = np.searchsorted(nodes, e2m).astype(np.int64, copy=False)
        g = _solve_event_potentials_subgraph(
            n_events=M, e1=inv1, e2=inv2, r=rm, ridge=ridge, rtol=rtol, maxiter=maxiter
        )
        coords = XY.index_select(0, torch.from_numpy(nodes).to(torch.int64)).cpu()
        vals = torch.from_numpy(g).to(torch.float32).cpu()
        # Expose (coords, vals) used for variogram/correlation plots when requested.
        if isinstance(curves_out, dict):
            try:
                cd = curves_out.get("_corr_data", None)
                if not isinstance(cd, dict):
                    cd = {}
                    curves_out["_corr_data"] = cd
                cd[out_key] = {
                    "coords_km": coords.detach().cpu().numpy().astype(np.float32, copy=False),
                    "vals": vals.detach().cpu().numpy().astype(np.float32, copy=False),
                }
            except Exception:
                pass
        curve = {}
        ell, plat, n_pairs_used = _variogram_half_plateau_ell_pairs(
            coords_km=coords,
            vals=vals,
            n_pairs=int(variogram_pairs),
            n_bins=int(n_bins),
            frac=float(frac_of_plateau),
            seed=int(seed) + int(seed_off),
            curve_out=curve if isinstance(curves_out, dict) else None,
            max_dist_km=max_dist_km,
            log_bins=bool(log_bins),
        )
        if isinstance(curves_out, dict):
            try:
                curves_out[out_key] = dict(curve)
            except Exception:
                pass
        return float(ell), float(plat), int(M), int(n_pairs_used)

    if isinstance(curves_out, dict):
        try:
            curves_out.clear()
            curves_out["meta"] = {
                "n_rows_used": int(rows_np.size),
                "n_bins": int(n_bins),
                "frac_of_plateau": float(frac_of_plateau),
                "variogram_pairs": int(variogram_pairs),
                "ridge": float(ridge),
                "rtol": float(rtol),
                "maxiter": int(maxiter),
            }
        except Exception:
            pass

    ell_p, plat_p, n_ev_p, n_pairs_p = _per_phase(~ph_is_s, 17, "p")
    ell_s, plat_s, n_ev_s, n_pairs_s = _per_phase(ph_is_s, 29, "s")
    ell_joint, plat_joint, n_ev_joint, n_pairs_joint = _joint(41, "joint_ps")
    # Recommendation: prefer joint estimate (matches shared ell_km hyperparameter).
    ell = float(ell_joint) if np.isfinite(float(ell_joint)) else float(np.nanmax([ell_p, ell_s]))
    n_pairs_used = int(max(n_pairs_p, n_pairs_s, n_pairs_joint))
    return SharedEventLatentEventEllEstimate(
        ell_p_km=float(ell_p),
        ell_s_km=float(ell_s),
        ell_km=float(ell),
        plateau_p=float(plat_p),
        plateau_s=float(plat_s),
        n_rows_used=int(rows_np.size),
        n_events_used_p=int(n_ev_p if n_ev_p > 0 else n_ev_joint),
        n_events_used_s=int(n_ev_s if n_ev_s > 0 else n_ev_joint),
        n_pairs_used=int(n_pairs_used),
    )


def maybe_estimate_shared_event_latent_event_ell_after_phase1(*, state) -> None:
    """
    Convenience wrapper to run after Phase 1.
    Reads optional knobs from:
      inference.diagnostics.shared_event_latent_event_ell_estimate
        {enabled,n_rows,seed,batch_size,n_bins,frac_of_plateau,ridge,rtol,maxiter,variogram_pairs}

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
        cfg = dg.get("shared_event_latent_event_ell_estimate", {}) if isinstance(dg, dict) else {}
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

    est = estimate_shared_event_latent_event_ell_km(
        state=state,
        n_rows=int(cfg.get("n_rows", 100_000)) if isinstance(cfg, dict) else 100_000,
        seed=int(cfg.get("seed", 0)) if isinstance(cfg, dict) else 0,
        batch_size=int(cfg.get("batch_size", 50_000)) if isinstance(cfg, dict) else 50_000,
        n_bins=int(cfg.get("n_bins", 20)) if isinstance(cfg, dict) else 20,
        frac_of_plateau=float(cfg.get("frac_of_plateau", 0.5)) if isinstance(cfg, dict) else 0.5,
        ridge=float(cfg.get("ridge", 1e-3)) if isinstance(cfg, dict) else 1e-3,
        rtol=float(cfg.get("rtol", 1e-6)) if isinstance(cfg, dict) else 1e-6,
        maxiter=int(cfg.get("maxiter", 2000)) if isinstance(cfg, dict) else 2000,
        variogram_pairs=int(cfg.get("variogram_pairs", 200_000)) if isinstance(cfg, dict) else 200_000,
    )
    if est is None:
        return

    state.params["_shared_event_latent_event_ell_est_p_km"] = float(est.ell_p_km)
    state.params["_shared_event_latent_event_ell_est_s_km"] = float(est.ell_s_km)
    state.params["_shared_event_latent_event_ell_est_km"] = float(est.ell_km)
    try:
        print(
            "Shared-event-latent event ell_km estimate (post Phase 1): "
            f"ell_p≈{est.ell_p_km:.3g} km, ell_s≈{est.ell_s_km:.3g} km, recommend≈{est.ell_km:.3g} km "
            f"(rows={est.n_rows_used}, events_p={est.n_events_used_p}, events_s={est.n_events_used_s}, pairs≈{est.n_pairs_used})",
            flush=True,
        )
    except Exception:
        pass


