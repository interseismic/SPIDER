import math
from typing import Dict, List, Tuple
from spider.utils.console import info, warn

import numpy as np
import polars as pl
import torch

from .spatial import _standardized_residuals



# Standardized stdout helper
def _log(*parts, section: str = "DIAG", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

def _wrap_angle(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    y = (x + np.pi) % (2.0 * np.pi) - np.pi
    # Map exact -pi to +pi for symmetry
    y[(y <= -np.pi) | (y > np.pi)] = np.pi
    return y


def _robust_scale(values: np.ndarray) -> float:
    """
    Robust scale ~ 90th percentile of absolute deviation from median (or circular mean for angles).
    Avoids O(N^2) pairwise computations.
    """
    if values.size == 0:
        return 1.0
    if values.ndim != 1:
        values = values.ravel()
    med = np.median(values)
    dev = np.abs(values - med)
    s = float(np.quantile(dev, 0.90))
    return max(s, 1e-6)


def _robust_scale_angle(theta: np.ndarray) -> float:
    """Robust scale for angular data using circular mean as center, then 90th percentile of |wrapped diff|."""
    if theta.size == 0:
        return 1.0
    c = complex(np.mean(np.cos(theta)), np.mean(np.sin(theta)))
    theta0 = math.atan2(c.imag, c.real)
    d = _wrap_angle(theta - theta0)
    d = np.abs(d)
    s = float(np.quantile(d, 0.90))
    return max(s, 1e-3)  # prevent division by very small sigma


def _bin_stat(x: np.ndarray, y: np.ndarray, *, bin_width: float, max_x: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean(y) within bins of x with width bin_width over [0, max_x).
    Returns (x_centers, means) where unpopulated bins are NaN.
    """
    if x.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    nb = max(1, int(math.ceil(max_x / bin_width)))
    edges = np.linspace(0.0, nb * bin_width, nb + 1, dtype=np.float64)
    idx = np.floor(np.clip(x, 0.0, edges[-1] - 1e-12) / bin_width).astype(np.int64)
    sums = np.bincount(idx, weights=y, minlength=nb).astype(np.float64)
    cnts = np.bincount(idx, minlength=nb).astype(np.int64)
    means = np.full(nb, np.nan, dtype=np.float64)
    mask = cnts > 0
    means[mask] = sums[mask] / cnts[mask]
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers.astype(np.float32), means.astype(np.float32)


@torch.no_grad()
def run_pathcorr_diag_end_phase1(state) -> None:
    """
    Per station–phase (network,station,phase), compute:
      - r_i = median of standardized residual contributions for event i at that station–phase
      - features per event: epicentral distance d_i (km), azimuth θ_i (rad), depth z_i (km)
      - sample event pairs, compute Δπ_ij from (Δd, Δθ, Δz) with robust per-feature scales
      - estimate correlation vs path similarity: ρ̂(Δπ) = E[r_i r_j]/Var(r)
    Write per-(s,p) summary CSV and print a short console summary.
    Controlled by params:
      - pathcorr_enable: bool
      - pathcorr_max_pairs: int (default 50000)
      - pathcorr_bin_width: float (default 0.5)
      - pathcorr_max_pi: float (default computed from data)
      - pathcorr_min_obs_per_event: int (default 3)
      - pathcorr_top_k_print: int (default 8)
      - pathcorr_seed: int (default 0)
    """
    if not bool(state.params.get("pathcorr_enable", False)):
        return
    try:
        # 1) Standardized residuals per row (with SSST applied if available)
        r_std = _standardized_residuals(state)  # shape (N,)
        r_np = r_std.detach().cpu().numpy().astype(np.float32, copy=False)

        # 2) Build contributions for (network,station,phase,event) using Polars
        df = state.dtimes.with_columns(pl.Series("r_std", r_np))
        a = df.select(
            [
                pl.col("network"),
                pl.col("station"),
                pl.col("phase"),
                pl.col("X").alias("X_sta"),
                pl.col("Y").alias("Y_sta"),
                pl.col("evid1_idx").alias("e1"),
                pl.col("evid2_idx").alias("e2"),
                (-pl.col("r_std")).alias("c1"),
                pl.col("r_std").alias("c2"),
            ]
        )
        a1 = a.select(
            [
                "network",
                "station",
                "phase",
                "X_sta",
                "Y_sta",
                pl.col("e1").alias("event"),
                pl.col("c1").alias("contrib"),
            ]
        )
        a2 = a.select(
            [
                "network",
                "station",
                "phase",
                "X_sta",
                "Y_sta",
                pl.col("e2").alias("event"),
                pl.col("c2").alias("contrib"),
            ]
        )
        contrib = pl.concat([a1, a2], how="vertical")
        min_obs = int(state.params.get("pathcorr_min_obs_per_event", 3))
        # Aggregate to per-(s,p,event)
        ge = (
            contrib.group_by(["network", "station", "phase", "X_sta", "Y_sta", "event"])
            .agg([pl.len().alias("cnt"), pl.col("contrib").median().alias("r_event")])
            .filter(pl.col("cnt") >= min_obs)
        )

        if ge.height == 0:
            _log("PathCorr: no per-event residuals available after filtering; skipping.")
            return

        # 3) Event coordinates (X,Y,depth) for all events
        Xtot = (state.X_src + state.dX_src).detach().cpu().numpy().astype(np.float32, copy=False)
        Xe = Xtot[:, 0]
        Ye = Xtot[:, 1]
        Ze = Xtot[:, 2]  # depth (km)

        # 4) For each (s,p), compute features, sample pairs, bin correlation vs Δπ
        rng = np.random.default_rng(int(state.params.get("pathcorr_seed", 0)))
        max_pairs = int(state.params.get("pathcorr_max_pairs", 50000))
        bin_width = float(state.params.get("pathcorr_bin_width", 0.5))
        top_k_print = int(state.params.get("pathcorr_top_k_print", 8))

        summaries: List[Dict[str, object]] = []
        group_keys = ["network", "station", "phase", "X_sta", "Y_sta"]
        # Partition into per-(network,station,phase) groups
        try:
            partitions = ge.partition_by(group_keys, maintain_order=True)
        except Exception:
            # Fallback: unique keys + filtering (slower)
            uniq = ge.select(group_keys).unique()
            partitions = []
            for row in uniq.iter_rows(named=True):
                filt = (
                    (pl.col("network") == row["network"])
                    & (pl.col("station") == row["station"])
                    & (pl.col("phase") == row["phase"])
                    & (pl.col("X_sta") == row["X_sta"])
                    & (pl.col("Y_sta") == row["Y_sta"])
                )
                partitions.append(ge.filter(filt))

        for sub in partitions:
            network = sub["network"][0]
            station = sub["station"][0]
            phase = int(sub["phase"][0])
            Xs = float(sub["X_sta"][0])
            Ys = float(sub["Y_sta"][0])
            ev_idx = sub["event"].to_numpy()
            r_event = sub["r_event"].to_numpy().astype(np.float32, copy=False)
            n_e = int(ev_idx.size)
            if n_e < 4:
                continue

            # Features
            dx = Xe[ev_idx] - Xs
            dy = Ye[ev_idx] - Ys
            d_i = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)
            theta_i = np.arctan2(dy, dx).astype(np.float32, copy=False)
            z_i = Ze[ev_idx].astype(np.float32, copy=False)

            # Robust scales
            σd = _robust_scale(d_i)
            σθ = _robust_scale_angle(theta_i)
            σz = _robust_scale(z_i)

            # Center r and variance
            r_c = r_event - float(np.mean(r_event))
            σr2 = float(np.var(r_c, ddof=1))
            if not np.isfinite(σr2) or σr2 <= 0.0:
                continue

            # Pair sampling
            # number of possible pairs
            tot_pairs = n_e * (n_e - 1) // 2
            use_pairs = min(max_pairs, tot_pairs)
            if use_pairs <= 0:
                continue
            # sample indices for i<j
            if use_pairs == tot_pairs and tot_pairs <= 200000:
                # generate all pairs when small enough
                idx_i, idx_j = np.triu_indices(n_e, k=1)
                # optionally downsample
                if tot_pairs > max_pairs:
                    sel = rng.choice(tot_pairs, size=max_pairs, replace=False)
                    idx_i = idx_i[sel]
                    idx_j = idx_j[sel]
            else:
                # random pairs without enforced uniqueness
                idx_i = rng.integers(0, n_e, size=use_pairs, endpoint=False)
                idx_j = rng.integers(0, n_e, size=use_pairs, endpoint=False)
                mask = idx_i != idx_j
                idx_i = idx_i[mask]
                idx_j = idx_j[mask]
                # keep the two arrays aligned and truncate if needed
                if idx_i.size > max_pairs:
                    idx_i = idx_i[:max_pairs]
                    idx_j = idx_j[:max_pairs]

            Δd = np.abs(d_i[idx_i] - d_i[idx_j])
            Δθ = np.abs(_wrap_angle(theta_i[idx_i] - theta_i[idx_j]))
            Δz = np.abs(z_i[idx_i] - z_i[idx_j])
            Δπ = np.sqrt((Δd / σd) ** 2 + (Δθ / σθ) ** 2 + (Δz / σz) ** 2).astype(np.float32, copy=False)
            prod = (r_c[idx_i] * r_c[idx_j]).astype(np.float32, copy=False)

            max_pi = float(state.params.get("pathcorr_max_pi", float(np.nanquantile(Δπ, 0.99))))
            if not np.isfinite(max_pi) or max_pi <= 0.0:
                max_pi = float(np.nanmax(Δπ))
                if not np.isfinite(max_pi) or max_pi <= 0.0:
                    max_pi = 3.0

            centers, mean_prod = _bin_stat(Δπ, prod, bin_width=bin_width, max_x=max_pi)
            rho = mean_prod / σr2
            # summarize
            rho0 = float(rho[0]) if rho.size > 0 else float("nan")
            # find first center where rho<=0.1 (approx decay scale)
            pi_at_01 = float("nan")
            if rho.size > 0:
                for c, val in zip(centers, rho):
                    if np.isfinite(val) and val <= 0.1:
                        pi_at_01 = float(c)
                        break

            summaries.append(
                {
                    "network": network,
                    "station": station,
                    "phase": phase,
                    "n_events": n_e,
                    "pairs_used": int(Δπ.size),
                    "sigma_d": float(σd),
                    "sigma_theta": float(σθ),
                    "sigma_z": float(σz),
                    "rho0": rho0,
                    "pi_at_0.1": pi_at_01,
                }
            )

        if len(summaries) == 0:
            _log("PathCorr: no station–phase groups yielded usable statistics.")
            return

        # Write CSV
        out_path = state.params.get("pathcorr_outfile", None)
        if not out_path:
            base = state.params.get("catalog_outfile", "catalog")
            out_path = f"{base}_pathcorr_phase1.csv"
        try:
            pl.DataFrame(summaries).write_csv(out_path)
            _log(f"PathCorr: wrote per-station–phase summary to {out_path}")
        except Exception as e:
            _log(f"PathCorr: failed to write CSV: {e}")

        # Console summary
        df_sum = pl.DataFrame(summaries)
        try:
            # Overall medians
            med_rho0 = float(df_sum.select(pl.col("rho0").median()).item())
            n_groups = int(df_sum.height)
            _log(f"PathCorr: groups={n_groups} median rho0={med_rho0:.3f}")
            # Top-k by rho0
            top_k = df_sum.sort("rho0", descending=True).head(top_k_print)
            try:
                pdf = top_k.select(["network", "station", "phase", "n_events", "rho0", "pi_at_0.1"]).to_pandas()
                import pandas as _pd  # type: ignore

                _pd.set_option("display.max_columns", None)
                _pd.set_option("display.width", 0)
                _log("PathCorr (top by rho0):")
                _log(pdf.to_string(index=False))
            except Exception:
                _log("PathCorr (top by rho0):")
                _log(top_k)
        except Exception:
            pass
    except Exception as e:
        _log(f"PathCorr: failed with error: {e}")


