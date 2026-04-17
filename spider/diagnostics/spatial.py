import math
from typing import Tuple
from spider.utils.console import info, warn

import numpy as np
import polars as pl
import torch

from ..core.modeling import compute_residuals_full



# Standardized stdout helper
def _log(*parts, section: str = "DIAG", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

@torch.no_grad()
def _standardized_residuals(state) -> torch.Tensor:
    """
    Compute standardized residuals r/σ for all rows at the current (X_src + dX_src),
    """
    if state.N <= 0:
        return torch.zeros(0, dtype=torch.float32, device=state.device)

    bs = max(int(state.batch_size_warmup or state.batch_size_sgld or 1024), 1)
    resid = compute_residuals_full(
        state.II, state.YY, state.X_src, state.dX_src, state.model, bs, state.N, params=state.params
    )  # observed - predicted

    # Build per-row σ from fixed phase_unc (noise learning removed).
    try:
        σ_pair = state.scale_theta.to(resid.device)  # type: ignore[union-attr]
        σ_p = σ_pair[0]
        σ_s = σ_pair[1]
    except Exception:
        vals = state.params.get("phase_unc", [0.05, 0.08])
        σ_p = torch.tensor(float(vals[0]), device=resid.device)
        σ_s = torch.tensor(float(vals[1]), device=resid.device)
    phase_mask = state.YY[:, 4] < 0.5
    σ_row = torch.where(phase_mask, σ_p, σ_s).clamp_min(1e-12)
    return resid / σ_row


def _event_median_from_residuals_std(state, resid_std: torch.Tensor) -> np.ndarray:
    """
    Compute a robust per-event scalar:
      r̄_e = median of signed standardized residual contributions touching event e,
      where contributions are {-r_std for e as first (e1), +r_std for e as second (e2)}.
    Returns np.ndarray shape (Ne,).
    """
    # Prepare contributions on CPU for grouping
    r = resid_std.detach().cpu().numpy().astype(np.float32, copy=False)
    e1 = state.dtimes["evid1_idx"].to_numpy()
    e2 = state.dtimes["evid2_idx"].to_numpy()
    a1 = pl.DataFrame({"event": e1, "contrib": -r})
    a2 = pl.DataFrame({"event": e2, "contrib": +r})
    c = pl.concat([a1, a2], how="vertical")
    # Optional minimum observations per event to stabilize estimates
    min_obs = int(state.params.get("spatial_diag_min_obs", 5))
    grp = (
        c.group_by("event")
        .agg([pl.len().alias("cnt"), pl.col("contrib").median().alias("rbar")])
        .sort("event")
    )
    # Build full vector, filling missing events with 0.0 so diagnostic does not crash.
    Ne = int(state.X_src.shape[0])
    rbar = np.zeros((Ne,), dtype=np.float32)
    if grp.height > 0:
        # Filter small-count events, leave zeros otherwise
        grp = grp.with_columns(pl.when(pl.col("cnt") >= min_obs).then(pl.col("rbar")).otherwise(None).alias("rbar"))
        idx = grp["event"].to_numpy()
        vals = grp["rbar"].to_numpy()
        # Replace NaNs with 0 for events failing min_obs
        vals = np.where(np.isfinite(vals), vals, 0.0).astype(np.float32, copy=False)
        rbar[idx] = vals
    return rbar


def _unique_edges_with_distance(XY: np.ndarray, II_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduplicate event edges from observation pairs (ignoring direction),
    and compute inter-event distances in km for those edges.
    Returns (i_idx, j_idx, d_ij).
    """
    a = II_np[:, 0].astype(np.int64, copy=False)
    b = II_np[:, 1].astype(np.int64, copy=False)
    u = np.minimum(a, b)
    v = np.maximum(a, b)
    edges = np.stack([u, v], axis=1)
    # Unique undirected edges; keep order stable using np.unique with return_index
    uniq, idx = np.unique(edges, axis=0, return_index=True)
    i = uniq[:, 0]
    j = uniq[:, 1]
    # Compute Euclidean distance in projected (km) coordinates
    dxy = XY[i, :] - XY[j, :]
    d = np.sqrt((dxy * dxy).sum(axis=1)).astype(np.float32, copy=False)
    return i, j, d


def _morans_i_from_graph(values: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray, w: np.ndarray) -> float:
    """
    Compute Moran's I on an undirected weighted graph given by edge lists (i_idx, j_idx) with weights w.
    """
    v = values.astype(np.float64, copy=False)
    dv = v - v.mean()
    denom = float((dv * dv).sum())
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    # Symmetric sum over edges
    num = float((w * (dv[i_idx] * dv[j_idx])).sum()) * 2.0
    S0 = float(w.sum()) * 2.0
    N = float(values.shape[0])
    return (N / S0) * (num / denom)


def _perm_pvalue(values: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray, w: np.ndarray, I_obs: float, *, n_perm: int, seed: int = 0) -> Tuple[float, float, float]:
    """
    Permutation test for Moran's I (two-sided). Returns (p_value, mean_perm, std_perm).
    """
    if n_perm <= 0 or not np.isfinite(I_obs):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    vals = values.copy()
    I_vals = np.zeros(n_perm, dtype=np.float64)
    for t in range(n_perm):
        rng.shuffle(vals)
        I_vals[t] = _morans_i_from_graph(vals, i_idx, j_idx, w)
    mu = float(np.nanmean(I_vals))
    sd = float(np.nanstd(I_vals, ddof=1)) if n_perm > 1 else float("nan")
    # Two-sided p-value using permutation distribution
    ge = float(np.nanmean(I_vals >= I_obs))
    le = float(np.nanmean(I_vals <= I_obs))
    p = 2.0 * min(ge, le)
    p = min(max(p, 0.0), 1.0)
    return p, mu, sd


@torch.no_grad()
def run_spatial_diag_end_phase1(state) -> None:
    """
    Run a one-shot spatial diagnostic at the end of Phase 1:
      - Compute standardized residuals (with SSST applied if available)
      - Build a per-event robust scalar (median of signed standardized residuals)
      - Build a distance-weighted event graph from unique observed pairs
      - Compute Moran's I and a permutation p-value
    Prints a concise summary line to the console.
    """
    if not bool(state.params.get("spatial_diag_enable", False)):
        return
    try:
        resid_std = _standardized_residuals(state)
        rbar = _event_median_from_residuals_std(state, resid_std)
        # Event coordinates in projected km
        Xtot = (state.X_src + state.dX_src).detach().cpu().numpy()
        XY = Xtot[:, 0:2].astype(np.float32, copy=False)
        # Unique edges from observations and their distances
        II_np = state.II.detach().cpu().numpy()
        ei, ej, dij = _unique_edges_with_distance(XY, II_np)
        if ei.size == 0:
            _log("Spatial diag (end Phase 1): no edges available; skipping.")
            return
        # Distance-based weights (Gaussian kernel with automatic bandwidth from median distance)
        ell = float(np.median(dij))
        if not math.isfinite(ell) or ell <= 0.0:
            ell = 100.0  # km fallback
        w = np.exp(-0.5 * (dij / ell) ** 2).astype(np.float64, copy=False)
        I_obs = _morans_i_from_graph(rbar, ei, ej, w)
        n_perm = int(state.params.get("spatial_diag_permutations", 199))
        seed = int(state.params.get("spatial_diag_seed", 0))
        p, mu, sd = _perm_pvalue(rbar, ei, ej, w, I_obs, n_perm=n_perm, seed=seed)
        z = (I_obs - mu) / sd if (np.isfinite(mu) and np.isfinite(sd) and sd > 0.0) else float("nan")
        _log(
            f"Spatial diag (end Phase 1): Moran's I={I_obs:.4f} "
            f"(z={z:.2f}, p={p:.3f}) | events={rbar.size} edges={ei.size} "
            f"kernel_ell≈{ell:.1f} km perms={n_perm}"
        )
    except Exception as e:
        _log(f"Spatial diag (end Phase 1): failed with error: {e}")


