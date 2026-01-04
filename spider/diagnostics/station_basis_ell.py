from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import polars as pl
import torch

from spider.core.modeling import compute_residuals
from spider.core.state import _current_noise_scales


@dataclass
class StationBasisEllEstimate:
    ell_p_km: float
    ell_s_km: float
    ell_km: float
    plateau_p: float
    plateau_s: float
    n_rows_used: int
    n_stations_used_p: int
    n_stations_used_s: int


def recommend_station_basis_rank(
    *,
    coords_km: torch.Tensor,   # (S,2) CPU float32/float64
    ell_km: float,
    jitter: float = 0.0,
    frac_var: float = 0.95,
    r_max: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """
    Recommend a station-basis rank R for an RBF station kernel at length scale ell_km.

    Returns a dict with:
      - r_rec: recommended smallest R such that cum_eig/total >= frac_var
      - eff_rank: effective rank (sum λ)^2 / sum λ^2
      - frac_var: requested fraction
      - ell_km: used ell
      - n_stations: station count
    """
    try:
        S = int(coords_km.shape[0])
        if S <= 1:
            return None
        ell = float(ell_km)
        if not np.isfinite(ell) or ell <= 0:
            return None
        frac = float(frac_var)
        if (not np.isfinite(frac)) or frac <= 0.0 or frac >= 1.0:
            frac = 0.95
        jit = float(jitter)
        if (not np.isfinite(jit)) or jit < 0.0:
            jit = 0.0
        r_max_i = S if r_max is None else int(max(1, min(int(r_max), S)))

        XY = coords_km.to(torch.float64)
        D = torch.cdist(XY, XY).to(torch.float64)
        K = torch.exp(-0.5 * (D / ell) ** 2)
        if jit > 0:
            K = K + (jit * torch.eye(S, dtype=torch.float64))
        w = torch.linalg.eigvalsh(K).to(torch.float64)  # ascending
        w = torch.clamp(w, min=0.0)
        w_desc = torch.flip(w, dims=[0])
        total = float(w_desc.sum().item())
        if not np.isfinite(total) or total <= 0:
            return None
        c = torch.cumsum(w_desc, dim=0) / total
        idx = torch.nonzero(c >= frac, as_tuple=False)
        r_rec = int(idx[0].item()) + 1 if idx.numel() > 0 else r_max_i
        r_rec = int(min(r_rec, r_max_i))

        s1 = float(w_desc.sum().item())
        s2 = float((w_desc * w_desc).sum().item())
        eff = (s1 * s1 / max(s2, 1e-30)) if np.isfinite(s1) and np.isfinite(s2) else float("nan")
        return {
            "r_rec": float(r_rec),
            "eff_rank": float(eff),
            "frac_var": float(frac),
            "ell_km": float(ell),
            "n_stations": float(S),
        }
    except Exception:
        return None


def _variogram_half_plateau_ell(
    *,
    coords_km: torch.Tensor,  # (S,2) float32 CPU
    mean_r: torch.Tensor,     # (S,) float32 CPU with nan for missing
    n_bins: int,
    frac: float,
) -> Tuple[float, float]:
    """
    Compute empirical semivariogram and return (ell_km, plateau).
    We define ell_km as the smallest distance where semivariance >= frac * plateau.
    """
    S = int(coords_km.shape[0])
    if S < 2:
        return float("nan"), float("nan")
    # Pairwise distances (km)
    D = torch.cdist(coords_km, coords_km).to(torch.float32)  # (S,S)
    # Upper triangle pairs
    iu = torch.triu_indices(S, S, offset=1)
    d = D[iu[0], iu[1]]
    r1 = mean_r[iu[0]]
    r2 = mean_r[iu[1]]
    m = torch.isfinite(r1) & torch.isfinite(r2) & torch.isfinite(d)
    if not bool(m.any().item()):
        return float("nan"), float("nan")
    d = d[m]
    g = 0.5 * (r1[m] - r2[m]).pow(2)  # semivariance
    dmax = float(d.max().item())
    if not np.isfinite(dmax) or dmax <= 0:
        return float("nan"), float("nan")
    n_bins = max(5, int(n_bins))
    # Bin edges [0,dmax]
    edges = torch.linspace(0.0, dmax, steps=n_bins + 1, dtype=torch.float32)
    # Bin index for each pair
    # (Avoid torch.bucketize corner cases by clamping)
    bi = torch.bucketize(d, edges, right=False) - 1
    bi = bi.clamp(0, n_bins - 1)
    # Mean semivariance per bin
    g_sum = torch.zeros((n_bins,), dtype=torch.float64)
    g_cnt = torch.zeros((n_bins,), dtype=torch.float64)
    g_sum.scatter_add_(0, bi.to(torch.int64), g.to(torch.float64))
    g_cnt.scatter_add_(0, bi.to(torch.int64), torch.ones_like(g, dtype=torch.float64))
    g_mean = (g_sum / torch.clamp_min(g_cnt, 1.0)).to(torch.float32)

    # Plateau estimate: median of last 20% bins with at least 1 count
    tail0 = int(np.floor(0.8 * n_bins))
    tail = g_mean[tail0:]
    tail_cnt = g_cnt[tail0:]
    tail = tail[tail_cnt > 0]
    if tail.numel() == 0:
        plateau = float("nan")
    else:
        plateau = float(torch.median(tail).item())
    if not np.isfinite(plateau) or plateau <= 0:
        return float("nan"), float(plateau)

    # Nugget correction: if there is a big short-range jump (often stronger for S due to pick noise),
    # a naive frac*plateau threshold can be hit immediately, yielding unrealistically tiny ell.
    # Use target = nugget + frac*(plateau - nugget) when possible.
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
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Find first bin reaching target
    ok = (g_cnt > 0) & (g_mean >= target)
    if not bool(ok.any().item()):
        return float("nan"), float(plateau)
    i = int(torch.nonzero(ok, as_tuple=False)[0].item())
    ell = float(centers[i].item())
    return ell, float(plateau)


def _variogram_curve(
    *,
    coords_km: torch.Tensor,  # (S,2) float32 CPU
    mean_r: torch.Tensor,     # (S,) float32 CPU with nan for missing
    n_bins: int,
    frac: float,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute the empirical semivariogram curve (station means).

    Returns a dict of numpy arrays:
      - centers_km: (B,)
      - gamma:      (B,) semivariance
      - count:      (B,) pair counts
      - plateau:    (1,)
      - nugget:     (1,)
      - target:     (1,) nugget-corrected frac-of-plateau target
    """
    S = int(coords_km.shape[0])
    if S < 2:
        return None

    D = torch.cdist(coords_km, coords_km).to(torch.float32)  # (S,S)
    iu = torch.triu_indices(S, S, offset=1)
    d = D[iu[0], iu[1]]
    r1 = mean_r[iu[0]]
    r2 = mean_r[iu[1]]
    m = torch.isfinite(r1) & torch.isfinite(r2) & torch.isfinite(d)
    if not bool(m.any().item()):
        return None
    d = d[m]
    g = 0.5 * (r1[m] - r2[m]).pow(2)

    dmax = float(d.max().item())
    if not np.isfinite(dmax) or dmax <= 0:
        return None

    n_bins = max(5, int(n_bins))
    edges = torch.linspace(0.0, dmax, steps=n_bins + 1, dtype=torch.float32)
    bi = torch.bucketize(d, edges, right=False) - 1
    bi = bi.clamp(0, n_bins - 1)

    g_sum = torch.zeros((n_bins,), dtype=torch.float64)
    g_cnt = torch.zeros((n_bins,), dtype=torch.float64)
    g_sum.scatter_add_(0, bi.to(torch.int64), g.to(torch.float64))
    g_cnt.scatter_add_(0, bi.to(torch.int64), torch.ones_like(g, dtype=torch.float64))
    g_mean = (g_sum / torch.clamp_min(g_cnt, 1.0)).to(torch.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    tail0 = int(np.floor(0.8 * n_bins))
    tail = g_mean[tail0:]
    tail_cnt = g_cnt[tail0:]
    tail = tail[tail_cnt > 0]
    plateau = float(torch.median(tail).item()) if tail.numel() else float("nan")

    head1 = max(1, int(np.floor(0.1 * n_bins)))
    head = g_mean[:head1]
    head_cnt = g_cnt[:head1]
    head = head[head_cnt > 0]
    nugget = float(torch.median(head).item()) if head.numel() else 0.0
    if (not np.isfinite(nugget)) or nugget < 0.0:
        nugget = 0.0

    if np.isfinite(plateau) and plateau > 0:
        sill = float(plateau) - float(nugget)
        if np.isfinite(sill) and sill > 0:
            target = float(nugget) + float(frac) * float(sill)
        else:
            target = float(frac) * float(plateau)
    else:
        target = float("nan")

    return {
        "centers_km": centers.detach().cpu().numpy().astype(np.float32, copy=False),
        "gamma": g_mean.detach().cpu().numpy().astype(np.float32, copy=False),
        "count": g_cnt.detach().cpu().numpy().astype(np.float32, copy=False),
        "plateau": np.asarray([plateau], dtype=np.float32),
        "nugget": np.asarray([nugget], dtype=np.float32),
        "target": np.asarray([target], dtype=np.float32),
    }


def _variogram_curve_joint_ps(
    *,
    coords_km: torch.Tensor,  # (S,2) float32 CPU
    mean_p: torch.Tensor,     # (S,) float32 CPU with nan for missing
    mean_s: torch.Tensor,     # (S,) float32 CPU with nan for missing
    n_bins: int,
    frac: float,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Joint P+S variogram curve for vector value v=[mean_p,mean_s]:
        γ = 0.5 * ((Δp)^2 + (Δs)^2)
    """
    S = int(coords_km.shape[0])
    if S < 2:
        return None
    m = torch.isfinite(mean_p) & torch.isfinite(mean_s)
    if not bool(m.any().item()):
        return None
    coords = coords_km[m]
    p = mean_p[m]
    s = mean_s[m]
    S2 = int(coords.shape[0])
    if S2 < 2:
        return None

    D = torch.cdist(coords, coords).to(torch.float32)
    iu = torch.triu_indices(S2, S2, offset=1)
    d = D[iu[0], iu[1]]
    dp = (p[iu[0]] - p[iu[1]]).to(torch.float32)
    ds = (s[iu[0]] - s[iu[1]]).to(torch.float32)
    g = 0.5 * (dp * dp + ds * ds)
    mm = torch.isfinite(d) & torch.isfinite(g)
    if not bool(mm.any().item()):
        return None
    d = d[mm]
    g = g[mm]

    dmax = float(d.max().item())
    if not np.isfinite(dmax) or dmax <= 0:
        return None

    n_bins = max(5, int(n_bins))
    edges = torch.linspace(0.0, dmax, steps=n_bins + 1, dtype=torch.float32)
    bi = torch.bucketize(d, edges, right=False) - 1
    bi = bi.clamp(0, n_bins - 1)

    g_sum = torch.zeros((n_bins,), dtype=torch.float64)
    g_cnt = torch.zeros((n_bins,), dtype=torch.float64)
    g_sum.scatter_add_(0, bi.to(torch.int64), g.to(torch.float64))
    g_cnt.scatter_add_(0, bi.to(torch.int64), torch.ones_like(g, dtype=torch.float64))
    g_mean = (g_sum / torch.clamp_min(g_cnt, 1.0)).to(torch.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    tail0 = int(np.floor(0.8 * n_bins))
    tail = g_mean[tail0:]
    tail_cnt = g_cnt[tail0:]
    tail = tail[tail_cnt > 0]
    plateau = float(torch.median(tail).item()) if tail.numel() else float("nan")

    head1 = max(1, int(np.floor(0.1 * n_bins)))
    head = g_mean[:head1]
    head_cnt = g_cnt[:head1]
    head = head[head_cnt > 0]
    nugget = float(torch.median(head).item()) if head.numel() else 0.0
    if (not np.isfinite(nugget)) or nugget < 0.0:
        nugget = 0.0

    if np.isfinite(plateau) and plateau > 0:
        sill = float(plateau) - float(nugget)
        if np.isfinite(sill) and sill > 0:
            target = float(nugget) + float(frac) * float(sill)
        else:
            target = float(frac) * float(plateau)
    else:
        target = float("nan")

    return {
        "centers_km": centers.detach().cpu().numpy().astype(np.float32, copy=False),
        "gamma": g_mean.detach().cpu().numpy().astype(np.float32, copy=False),
        "count": g_cnt.detach().cpu().numpy().astype(np.float32, copy=False),
        "plateau": np.asarray([plateau], dtype=np.float32),
        "nugget": np.asarray([nugget], dtype=np.float32),
        "target": np.asarray([target], dtype=np.float32),
    }


def _variogram_half_plateau_ell_joint_ps(
    *,
    coords_km: torch.Tensor,  # (S,2) float32 CPU
    mean_p: torch.Tensor,     # (S,) float32 CPU with nan for missing
    mean_s: torch.Tensor,     # (S,) float32 CPU with nan for missing
    n_bins: int,
    frac: float,
) -> float:
    """
    Joint P+S variogram ell estimate.

    Treat each station as a 2D vector value v=[mean_p, mean_s] and define semivariance:
        γ = 0.5 * ||v_i - v_j||^2 = 0.5 * ((Δp)^2 + (Δs)^2)

    Returns ell_km (plateau is not returned because we only use ell for recommendation).
    """
    S = int(coords_km.shape[0])
    if S < 2:
        return float("nan")
    m = torch.isfinite(mean_p) & torch.isfinite(mean_s)
    if not bool(m.any().item()):
        return float("nan")
    coords = coords_km[m]
    p = mean_p[m]
    s = mean_s[m]
    S2 = int(coords.shape[0])
    if S2 < 2:
        return float("nan")

    D = torch.cdist(coords, coords).to(torch.float32)
    iu = torch.triu_indices(S2, S2, offset=1)
    d = D[iu[0], iu[1]]
    dp = (p[iu[0]] - p[iu[1]]).to(torch.float32)
    ds = (s[iu[0]] - s[iu[1]]).to(torch.float32)
    g = 0.5 * (dp * dp + ds * ds)
    mm = torch.isfinite(d) & torch.isfinite(g)
    if not bool(mm.any().item()):
        return float("nan")
    d = d[mm]
    g = g[mm]

    dmax = float(d.max().item())
    if not np.isfinite(dmax) or dmax <= 0:
        return float("nan")
    n_bins = max(5, int(n_bins))
    edges = torch.linspace(0.0, dmax, steps=n_bins + 1, dtype=torch.float32)
    bi = torch.bucketize(d, edges, right=False) - 1
    bi = bi.clamp(0, n_bins - 1)
    g_sum = torch.zeros((n_bins,), dtype=torch.float64)
    g_cnt = torch.zeros((n_bins,), dtype=torch.float64)
    g_sum.scatter_add_(0, bi.to(torch.int64), g.to(torch.float64))
    g_cnt.scatter_add_(0, bi.to(torch.int64), torch.ones_like(g, dtype=torch.float64))
    g_mean = (g_sum / torch.clamp_min(g_cnt, 1.0)).to(torch.float32)

    tail0 = int(np.floor(0.8 * n_bins))
    tail = g_mean[tail0:]
    tail_cnt = g_cnt[tail0:]
    tail = tail[tail_cnt > 0]
    if tail.numel() == 0:
        return float("nan")
    plateau = float(torch.median(tail).item())
    if not np.isfinite(plateau) or plateau <= 0:
        return float("nan")

    # Nugget-corrected target
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
        target = float(frac) * float(plateau)

    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = (g_cnt > 0) & (g_mean >= target)
    if not bool(ok.any().item()):
        return float("nan")
    i = int(torch.nonzero(ok, as_tuple=False)[0].item())
    return float(centers[i].item())

@torch.no_grad()
def estimate_station_basis_ell_km(
    *,
    state,
    n_rows: int = 50_000,
    seed: int = 0,
    batch_size: int = 50_000,
    n_bins: int = 20,
    frac_of_plateau: float = 0.5,
    curves_out: Optional[Dict[str, Any]] = None,
) -> Optional[StationBasisEllEstimate]:
    """
    Estimate a station-basis length scale ell_km from Phase-1 MAP residual structure.

    Steps:
      - sample rows
      - compute base residuals at current MAP ΔX (no shared_event_latent correction)
      - per-station mean residuals for P and S
      - variogram vs station distance, pick half-plateau distance as ell
    """
    if getattr(state, "row_station_index", None) is None or int(getattr(state, "n_stations", 0)) <= 1:
        return None
    if not isinstance(state.dtimes, pl.DataFrame) or "sta_idx" not in state.dtimes.columns:
        return None

    S = int(state.n_stations)
    N = int(state.N)
    if N <= 0:
        return None
    n_rows = int(max(1, min(int(n_rows), N)))
    batch_size = int(max(1, batch_size))

    # Station coordinates from dtimes (one per sta_idx)
    try:
        sta_xy = (
            state.dtimes.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y")])
            .unique(subset=["sta_idx"], maintain_order=True)
            .sort("sta_idx")
        )
        if int(sta_xy.shape[0]) != S:
            # If mismatch, station coords may be incomplete; bail rather than produce nonsense.
            return None
        coords = torch.from_numpy(
            sta_xy.select([pl.col("X"), pl.col("Y")]).to_numpy().astype(np.float32, copy=False)
        ).cpu()
    except Exception:
        return None

    # Sample row indices
    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    # Accumulate per-station residual sums/counts (CPU float64 for stability)
    sum_p = torch.zeros((S,), dtype=torch.float64)
    cnt_p = torch.zeros((S,), dtype=torch.float64)
    sum_s = torch.zeros((S,), dtype=torch.float64)
    cnt_s = torch.zeros((S,), dtype=torch.float64)

    for i0 in range(0, int(rows_np.size), batch_size):
        i1 = min(i0 + batch_size, int(rows_np.size))
        rows_t = torch.as_tensor(rows_np[i0:i1], device=state.device, dtype=torch.int64)
        II_b = state.II.index_select(0, rows_t)
        YY_b = state.YY.index_select(0, rows_t)
        r = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().to(torch.float32)
        sta_b = state.row_station_index.index_select(0, rows_t).to(torch.int64).detach().cpu()
        ph = YY_b[:, 4].detach().cpu()
        is_p = (ph < 0.5)
        is_s = ~is_p
        # Standardize by phase noise scale to reduce P/S differences driven purely by nugget/pick scatter.
        try:
            σp, σs = _current_noise_scales(state)
            σp_f = float(σp.detach().cpu().item())
            σs_f = float(σs.detach().cpu().item())
            σp_f = σp_f if np.isfinite(σp_f) and σp_f > 0 else 1.0
            σs_f = σs_f if np.isfinite(σs_f) and σs_f > 0 else 1.0
        except Exception:
            σp_f, σs_f = 1.0, 1.0
        r_cpu = r.detach().cpu()
        r_cpu = torch.where(is_p, r_cpu / float(σp_f), r_cpu / float(σs_f))

        if is_p.any():
            sp = sta_b[is_p]
            rp = r_cpu[is_p].to(torch.float64)
            sum_p.scatter_add_(0, sp, rp)
            cnt_p.scatter_add_(0, sp, torch.ones_like(rp, dtype=torch.float64))
        if is_s.any():
            ss = sta_b[is_s]
            rs = r_cpu[is_s].to(torch.float64)
            sum_s.scatter_add_(0, ss, rs)
            cnt_s.scatter_add_(0, ss, torch.ones_like(rs, dtype=torch.float64))

    mean_p = (sum_p / torch.clamp_min(cnt_p, 1.0)).to(torch.float32)
    mean_s = (sum_s / torch.clamp_min(cnt_s, 1.0)).to(torch.float32)
    # Mask stations with no data as NaN
    mean_p = torch.where(cnt_p > 0, mean_p, torch.full_like(mean_p, float("nan")))
    mean_s = torch.where(cnt_s > 0, mean_s, torch.full_like(mean_s, float("nan")))

    ell_p, plat_p = _variogram_half_plateau_ell(
        coords_km=coords, mean_r=mean_p, n_bins=n_bins, frac=float(frac_of_plateau)
    )
    ell_s, plat_s = _variogram_half_plateau_ell(
        coords_km=coords, mean_r=mean_s, n_bins=n_bins, frac=float(frac_of_plateau)
    )
    # Recommendation: ell_km is a *shared* hyperparameter for station_basis across phases.
    # Prefer a joint P+S estimate; fall back to max(P,S) if joint is not available.
    ell_joint = _variogram_half_plateau_ell_joint_ps(
        coords_km=coords, mean_p=mean_p, mean_s=mean_s, n_bins=n_bins, frac=float(frac_of_plateau)
    )
    ell = float(ell_joint) if np.isfinite(float(ell_joint)) else float(np.nanmax([ell_p, ell_s]))

    if isinstance(curves_out, dict):
        try:
            curves_out.clear()
            curves_out["coords_km"] = coords.detach().cpu().numpy().astype(np.float32, copy=False)
            curves_out["mean_p"] = mean_p.detach().cpu().numpy().astype(np.float32, copy=False)
            curves_out["mean_s"] = mean_s.detach().cpu().numpy().astype(np.float32, copy=False)
            curves_out["p"] = _variogram_curve(coords_km=coords, mean_r=mean_p, n_bins=n_bins, frac=float(frac_of_plateau))
            curves_out["s"] = _variogram_curve(coords_km=coords, mean_r=mean_s, n_bins=n_bins, frac=float(frac_of_plateau))
            curves_out["joint_ps"] = _variogram_curve_joint_ps(coords_km=coords, mean_p=mean_p, mean_s=mean_s, n_bins=n_bins, frac=float(frac_of_plateau))
            curves_out["meta"] = {
                "n_rows_used": int(rows_np.size),
                "n_bins": int(n_bins),
                "frac_of_plateau": float(frac_of_plateau),
                "ell_p_km": float(ell_p),
                "ell_s_km": float(ell_s),
                "ell_km": float(ell),
            }
        except Exception:
            pass

    return StationBasisEllEstimate(
        ell_p_km=float(ell_p),
        ell_s_km=float(ell_s),
        ell_km=float(ell),
        plateau_p=float(plat_p),
        plateau_s=float(plat_s),
        n_rows_used=int(rows_np.size),
        n_stations_used_p=int((cnt_p > 0).sum().item()),
        n_stations_used_s=int((cnt_s > 0).sum().item()),
    )


def maybe_estimate_station_basis_ell_after_phase1(*, state) -> None:
    """
    Convenience wrapper to run after Phase 1. Reads optional knobs from params:
      inference.diagnostics.station_basis_ell_estimate.{enabled,n_rows,seed,batch_size,n_bins,frac_of_plateau}
    Defaults to enabled when station_basis is enabled.
    """
    try:
        if not bool(state.params.get("_shared_event_latent_enabled", False)):
            return
        if not bool(state.params.get("_shared_event_latent_station_basis_enabled", False)):
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
        cfg = dg.get("station_basis_ell_estimate", {}) if isinstance(dg, dict) else {}
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

    est = estimate_station_basis_ell_km(
        state=state,
        n_rows=int(cfg.get("n_rows", 50_000)) if isinstance(cfg, dict) else 50_000,
        seed=int(cfg.get("seed", 0)) if isinstance(cfg, dict) else 0,
        batch_size=int(cfg.get("batch_size", 50_000)) if isinstance(cfg, dict) else 50_000,
        n_bins=int(cfg.get("n_bins", 20)) if isinstance(cfg, dict) else 20,
        frac_of_plateau=float(cfg.get("frac_of_plateau", 0.5)) if isinstance(cfg, dict) else 0.5,
    )
    if est is None:
        return
    # Store for later logging / config tuning
    state.params["_station_basis_ell_est_p_km"] = float(est.ell_p_km)
    state.params["_station_basis_ell_est_s_km"] = float(est.ell_s_km)
    state.params["_station_basis_ell_est_km"] = float(est.ell_km)

    # Optional: recommend rank R from the station-kernel eigenspectrum
    coords = None
    try:
        sta_xy = (
            state.dtimes.select([pl.col("sta_idx"), pl.col("X"), pl.col("Y")])
            .unique(subset=["sta_idx"], maintain_order=True)
            .sort("sta_idx")
        )
        if int(sta_xy.shape[0]) == int(state.n_stations):
            coords = torch.from_numpy(
                sta_xy.select([pl.col("X"), pl.col("Y")]).to_numpy().astype(np.float32, copy=False)
            ).cpu()
    except Exception:
        coords = None

    if coords is not None:
        frac_var = float(cfg.get("rank_frac_var", 0.95)) if isinstance(cfg, dict) else 0.95
        r_max = cfg.get("rank_r_max", None) if isinstance(cfg, dict) else None
        r_max_i = int(r_max) if r_max is not None else None
        jit_cfg = float(state.params.get("_shared_event_latent_station_basis_jitter", 0.0))
        ell_cfg = float(state.params.get("_shared_event_latent_station_basis_ell_km", float("nan")))

        rec_cfg = recommend_station_basis_rank(
            coords_km=coords, ell_km=ell_cfg, jitter=jit_cfg, frac_var=frac_var, r_max=r_max_i
        )
        rec_est = recommend_station_basis_rank(
            coords_km=coords, ell_km=float(est.ell_km), jitter=jit_cfg, frac_var=frac_var, r_max=r_max_i
        )
        if rec_cfg is not None:
            state.params["_station_basis_rank_rec_cfg"] = float(rec_cfg["r_rec"])
            state.params["_station_basis_eff_rank_cfg"] = float(rec_cfg["eff_rank"])
        if rec_est is not None:
            state.params["_station_basis_rank_rec_est"] = float(rec_est["r_rec"])
            state.params["_station_basis_eff_rank_est"] = float(rec_est["eff_rank"])

        if rec_cfg is not None and np.isfinite(ell_cfg):
            print(
                f"Station-basis rank estimate (kernel eigenspectrum @ configured ell={ell_cfg:.3g}km): "
                f"R≈{int(rec_cfg['r_rec'])} for {float(rec_cfg['frac_var'])*100:.1f}% var (eff_rank≈{rec_cfg['eff_rank']:.2f})",
                flush=True,
            )
        if rec_est is not None:
            print(
                f"Station-basis rank estimate (kernel eigenspectrum @ estimated ell={float(est.ell_km):.3g}km): "
                f"R≈{int(rec_est['r_rec'])} for {float(rec_est['frac_var'])*100:.1f}% var (eff_rank≈{rec_est['eff_rank']:.2f})",
                flush=True,
            )
    try:
        print(
            "Station-basis ell_km estimate (post Phase 1): "
            f"ell_p≈{est.ell_p_km:.3g} km, ell_s≈{est.ell_s_km:.3g} km, recommend≈{est.ell_km:.3g} km "
            f"(rows={est.n_rows_used}, stations_p={est.n_stations_used_p}/{int(state.n_stations)}, stations_s={est.n_stations_used_s}/{int(state.n_stations)})",
            flush=True,
        )
    except Exception:
        pass


