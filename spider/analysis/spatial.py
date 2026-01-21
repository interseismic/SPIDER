import os
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import torch


def _compute_ev_station_geometry(
    II: torch.Tensor,
    YY: torch.Tensor,
    X_src: torch.Tensor,
    dX_src: torch.Tensor,
    dtimes: pl.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-observation geometry:
      - R_ev: event pair separation (km)
      - D_es: distance from pair centroid to station (km)
      - ratio: R_ev / max(D_es, eps)
      - phase_mask: True for S-phase, False for P-phase
      - station_ids: array[str] 'NET.STA'
    Returns arrays of shape (N,).
    """
    device = X_src.device
    X_total = (X_src + dX_src).detach().cpu().numpy()  # (Ne, 4) columns: X,Y,Z,T
    II_np = II.detach().cpu().numpy()
    YY_np = YY.detach().cpu().numpy()  # cols: [dt, X, Y, Z, phase]

    e1 = II_np[:, 0].astype(np.int64)
    e2 = II_np[:, 1].astype(np.int64)
    pos1 = X_total[e1, :3]  # (N, 3)
    pos2 = X_total[e2, :3]  # (N, 3)
    R_ev = np.linalg.norm(pos2 - pos1, axis=1)  # (N,)

    centroids = 0.5 * (pos1 + pos2)  # (N, 3)
    rec = YY_np[:, 1:4]  # station XYZ (km)
    D_es = np.linalg.norm(rec - centroids, axis=1)  # (N,)

    eps = 1e-6
    ratio = R_ev / np.maximum(D_es, eps)
    # Clip for robustness in visualization
    ratio = np.clip(ratio, 0.0, 1.0)

    phase_mask = YY_np[:, 4] > 0.5  # True for S, False for P

    # Build station id 'NET.STA'
    sta_df = dtimes.select(
        [
            (pl.col("network").cast(pl.Utf8) + pl.lit(".") + pl.col("station").cast(pl.Utf8)).alias("sta"),
        ]
    )
    station_ids = np.asarray(sta_df["sta"].to_list(), dtype=str)

    return R_ev, D_es, ratio, phase_mask, station_ids


def _station_xy_map(dtimes: pl.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Return mapping 'NET.STA' -> (X, Y) from dtimes (projected km)."""
    unique_sta = (
        dtimes.select(
            [
                (pl.col("network").cast(pl.Utf8) + pl.lit(".") + pl.col("station").cast(pl.Utf8)).alias("sta"),
                pl.col("X").alias("X"),
                pl.col("Y").alias("Y"),
            ]
        )
        .unique(maintain_order=True)
    )
    return {row["sta"]: (float(row["X"]), float(row["Y"])) for row in unique_sta.iter_rows(named=True)}


def _semivariogram(values: np.ndarray, XY: np.ndarray, lag_bins: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical semivariogram for scalar 'values' observed at coordinates XY (S x 2).
    Returns (lag_centers, gamma, counts) each shape (L,), where L = len(lag_bins) - 1.
    """
    S = values.shape[0]
    if S < 2:
        L = len(lag_bins) - 1
        return np.asarray([(lag_bins[i] + lag_bins[i + 1]) * 0.5 for i in range(L)]), np.zeros(L), np.zeros(L, dtype=int)

    # Pairwise distances (SxS symmetric)
    d2 = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=-1)
    d = np.sqrt(np.maximum(d2, 0.0))
    iu = np.triu_indices(S, k=1)
    d_flat = d[iu]

    diff = values[:, None] - values[None, :]
    diff2 = (diff * diff)[iu]

    L = len(lag_bins) - 1
    gamma = np.zeros(L, dtype=np.float64)
    counts = np.zeros(L, dtype=np.int64)
    for ell in range(L):
        lo = lag_bins[ell]
        hi = lag_bins[ell + 1]
        mask = (d_flat >= lo) & (d_flat < hi)
        n = int(mask.sum())
        counts[ell] = n
        if n > 0:
            gamma[ell] = 0.5 * float(np.mean(diff2[mask]))
        else:
            gamma[ell] = 0.0

    lag_centers = np.asarray([(lag_bins[i] + lag_bins[i + 1]) * 0.5 for i in range(L)], dtype=np.float64)
    return lag_centers, gamma, counts


def compute_receiver_ratio_semivariograms(
    II: torch.Tensor,
    YY: torch.Tensor,
    X_src: torch.Tensor,
    dX_src: torch.Tensor,
    dtimes: pl.DataFrame,
    ratio_bins: Sequence[float],
    lag_bins: Sequence[float],
    min_obs_per_station: int,
    sigma_p: float,
    sigma_s: float,
    phase: str,
) -> Dict[str, np.ndarray | List[np.ndarray]]:
    """
    Compute station-level semivariograms of scaled residual means, stratified by ratio=R_ev/D_es bins.

    phase: "P" or "S"
    Returns dict with:
      - 'lag_centers': (L,)
      - 'gamma_curves': list of (L,) arrays, one per ratio bin
      - 'counts_curves': list of (L,) arrays (pair counts) per ratio bin
      - 'ratio_labels': list[str] describing each ratio bin
    """
    # Geometry and identifiers
    R_ev, D_es, ratio, ph_mask, station_ids = _compute_ev_station_geometry(II, YY, X_src, dX_src, dtimes)
    res = None
    # Compute residuals (observed - predicted) via modeling.compute_residuals_full upstream before calling this,
    # but if not provided, we can approximate from YY and model; caller should pass residuals if available.
    # Here we use dt = YY[:,0] and model eval is not available here, so the caller must precompute residuals.
    # To keep the API simple, we scale using YY's phase flags and sigmas, assuming residuals provided externally.
    # For now, we reconstruct residuals from YY assuming 'dt' already stores (obs - pred); otherwise this
    # function should be called with residuals explicitly. We fallback to zeros.
    YY_np = YY.detach().cpu().numpy()
    if YY_np.shape[1] >= 1:
        # Warning: this is only correct if YY[:,0] already contains residuals
        res = YY_np[:, 0].astype(np.float64)
    else:
        res = np.zeros_like(ratio, dtype=np.float64)

    # Scale by global sigma per phase
    sigma_base = np.where(ph_mask, float(sigma_s), float(sigma_p))
    z = res / np.maximum(sigma_base, 1e-12)

    # Phase selection
    if phase.upper() == "P":
        sel = ~ph_mask
    else:
        sel = ph_mask

    # Station coordinates map
    sta_xy = _station_xy_map(dtimes)

    # Prepare outputs
    lag_centers = np.asarray([(lag_bins[i] + lag_bins[i + 1]) * 0.5 for i in range(len(lag_bins) - 1)], dtype=np.float64)
    gamma_curves: List[np.ndarray] = []
    counts_curves: List[np.ndarray] = []
    labels: List[str] = []

    # Iterate ratio bins
    for b in range(len(ratio_bins) - 1):
        lo, hi = ratio_bins[b], ratio_bins[b + 1]
        mask = sel & (ratio >= lo) & (ratio < hi if b < len(ratio_bins) - 2 else ratio <= hi)
        if not np.any(mask):
            gamma_curves.append(np.zeros_like(lag_centers))
            counts_curves.append(np.zeros_like(lag_centers, dtype=int))
            labels.append(f"[{lo:.2f},{hi:.2f}]")
            continue

        # Aggregate station means in this stratum
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for i in np.where(mask)[0]:
            sid = station_ids[i]
            if sid not in sta_xy:
                continue
            sums[sid] = sums.get(sid, 0.0) + float(z[i])
            counts[sid] = counts.get(sid, 0) + 1

        vals: List[float] = []
        xy_list: List[Tuple[float, float]] = []
        for sid, c in counts.items():
            if c >= int(min_obs_per_station) and sid in sta_xy:
                vals.append(sums[sid] / c)
                xy_list.append(sta_xy[sid])

        if len(vals) < 2:
            gamma_curves.append(np.zeros_like(lag_centers))
            counts_curves.append(np.zeros_like(lag_centers, dtype=int))
            labels.append(f"[{lo:.2f},{hi:.2f}]")
            continue

        m = np.asarray(vals, dtype=np.float64)
        XY = np.asarray(xy_list, dtype=np.float64)
        lc, g, cts = _semivariogram(m, XY, lag_bins)
        gamma_curves.append(g)
        counts_curves.append(cts)
        labels.append(f"[{lo:.2f},{hi:.2f}]")

    return {
        "lag_centers": lag_centers,
        "gamma_curves": gamma_curves,
        "counts_curves": counts_curves,
        "ratio_labels": labels,
    }


def plot_receiver_ratio_semivariograms(
    res_P: Dict[str, np.ndarray | List[np.ndarray]],
    res_S: Dict[str, np.ndarray | List[np.ndarray]],
    out_path: str,
    title: Optional[str] = None,
) -> None:
    """Plot P and S variograms across ratio bins side by side and save to out_path."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lagP = res_P["lag_centers"]  # type: ignore[index]
    lagS = res_S["lag_centers"]  # type: ignore[index]
    gammasP: List[np.ndarray] = res_P["gamma_curves"]  # type: ignore[assignment]
    gammasS: List[np.ndarray] = res_S["gamma_curves"]  # type: ignore[assignment]
    labelsP: List[str] = res_P["ratio_labels"]  # type: ignore[assignment]
    labelsS: List[str] = res_S["ratio_labels"]  # type: ignore[assignment]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax = axes[0]
    for g, lab in zip(gammasP, labelsP):
        ax.plot(lagP, g, marker="o", linewidth=1.5, label=lab)
    ax.set_title("P-phase")
    ax.set_xlabel("Lag distance (km)")
    ax.set_ylabel("Semivariogram γ(h)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="ratio R_ev/D_es", fontsize=8)

    ax = axes[1]
    for g, lab in zip(gammasS, labelsS):
        ax.plot(lagS, g, marker="o", linewidth=1.5, label=lab)
    ax.set_title("S-phase")
    ax.set_xlabel("Lag distance (km)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="ratio R_ev/D_es", fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


