from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from pyproj import Proj
except Exception:  # pragma: no cover
    Proj = None  # type: ignore


@dataclass(frozen=True)
class CalibrationResult:
    """
    Outputs for posterior-vs-truth calibration checks.

    - merged: per-event DataFrame (requires pandas) with posterior, truth, and calibration columns
    - summary: lightweight dict of aggregate calibration metrics (coverage etc.)
    """

    merged: Any
    summary: Dict[str, float]


def _require_pandas() -> None:
    if pd is None:
        raise ImportError("pandas is required for calibration utilities (pip install pandas).")


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for calibration utilities.")


def _require_pyproj() -> None:
    if Proj is None:
        raise ImportError("pyproj is required for lon/lat -> XY km projection (pip install pyproj).")


def project_lonlat_depth_to_xyz_km(
    lon: np.ndarray,
    lat: np.ndarray,
    depth_km: np.ndarray,
    *,
    lat0: float,
    lon0: float,
) -> np.ndarray:
    """
    Project WGS84 lon/lat to local LAEA X/Y (km) and append depth (km).
    Returns (N,3) array [X_km, Y_km, Z_km].
    """
    _require_pyproj()
    projector = Proj(proj="laea", lat_0=float(lat0), lon_0=float(lon0), datum="WGS84", units="km")
    xx, yy = projector(np.asarray(lon), np.asarray(lat))
    zz = np.asarray(depth_km)
    return np.column_stack([np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64), zz.astype(np.float64)])


def _chi2_ppf_df3(p: float) -> float:
    """
    Chi-square(df=3) quantiles for common p without requiring SciPy.
    Values from standard tables.
    """
    p = float(p)
    # Commonly used for coverage checks
    table = {
        0.50: 2.3659738843753377,
        0.80: 4.641627676087454,
        0.90: 6.251388631170325,
        0.95: 7.814727903251179,
        0.99: 11.344866730144373,
    }
    if p in table:
        return float(table[p])
    # Fallback: try SciPy if available
    try:  # pragma: no cover
        from scipy.stats import chi2  # type: ignore

        return float(chi2.ppf(p, df=3))
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Unsupported p={p} without SciPy installed. Supported: {sorted(table.keys())}") from e


def _norm_ppf_two_sided_coverage(p: float) -> float:
    """
    Return zcrit such that P(|Z| <= zcrit) = p for Z~N(0,1).
    """
    p = float(p)
    table = {
        0.50: 0.6744897501960817,
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    if p in table:
        return float(table[p])
    try:  # pragma: no cover
        from scipy.stats import norm  # type: ignore

        return float(norm.ppf((1.0 + p) * 0.5))
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Unsupported p={p} without SciPy installed. Supported: {sorted(table.keys())}") from e


def calibrate_event_posteriors_against_truth(
    summary: Any,
    truth_catalog: "pd.DataFrame",
    *,
    lat0: float,
    lon0: float,
    post_cat_dd: Optional["pd.DataFrame"] = None,
    evid_col: str = "evid",
    truth_lon_col: str = "longitude",
    truth_lat_col: str = "latitude",
    truth_depth_col: str = "depth",
    use_map_if_available: bool = False,
    use_full_cov_from_samples: bool = True,
    dedupe_evid: bool = True,
    coverages: Sequence[float] = (0.50, 0.90, 0.95),
) -> CalibrationResult:
    """
    Compare per-event posterior locations to a truth catalog and compute calibration diagnostics.

    This answers: "do my posteriors reflect the true errors?"

    What it computes:
    - per-event error in projected km: err_x/err_y/err_z and dr (km)
    - per-event normalized errors: z_x/z_y/z_z using posterior std (or full cov)
    - multivariate calibration: Mahalanobis distance squared (maha2), and coverage of chi-square ellipsoids

    Inputs:
    - summary: typically `EventSamplesSummary` returned by `compute_cat_dd_and_xyz(...)`.
              If `use_full_cov_from_samples=True`, requires summary.X/Y/Z arrays to exist.
    - truth_catalog: pandas DataFrame with columns [evid, longitude, latitude, depth] (depth in km).
    - lat0/lon0: projection center, should match the run (for SPIDER this is params["lat_min"], params["lon_min"]).
    - post_cat_dd: if provided, use this instead of `summary.cat_dd`.
    """
    _require_pandas()
    if post_cat_dd is None:
        if getattr(summary, "cat_dd", None) is None:
            raise ValueError("summary.cat_dd is required (or pass post_cat_dd=...).")
        cat_dd = summary.cat_dd
    else:
        cat_dd = post_cat_dd

    if evid_col not in cat_dd.columns:
        raise ValueError(f"Posterior cat_dd is missing '{evid_col}' column.")
    for c in (truth_lon_col, truth_lat_col, truth_depth_col, evid_col):
        if c not in truth_catalog.columns:
            raise ValueError(f"Truth catalog is missing '{c}' column.")

    # Select posterior delta columns (used for covariance / optional fallback only).
    # Note: in SPIDER, cat_dd.X/Y/Z are typically *deltas* (km) relative to the input catalog, while
    # cat_dd.longitude/latitude/depth are absolute posterior locations.
    use_map = bool(use_map_if_available) and all(c in cat_dd.columns for c in ("X_map", "Y_map", "Z_map"))
    post_dX = "X_map" if use_map else "X"
    post_dY = "Y_map" if use_map else "Y"
    post_dZ = "Z_map" if use_map else "Z"
    for c in (post_dX, post_dY, post_dZ):
        if c not in cat_dd.columns:
            raise ValueError(f"Posterior cat_dd is missing '{c}' column needed for calibration.")

    # Merge by evid, but keep a stable event-row index so we can align merged rows
    # to summary.X/Y/Z even if merges later duplicate rows.
    dd = cat_dd.copy()
    # Prefer the stable per-event row if present (added by EventSamplesSummary.compute).
    if "_event_row" in dd.columns:
        dd["_sample_row"] = np.asarray(dd["_event_row"].to_numpy(), dtype=np.int64)
    else:
        dd["_sample_row"] = np.arange(dd.shape[0], dtype=np.int64)
    tt = truth_catalog[[evid_col, truth_lon_col, truth_lat_col, truth_depth_col]].copy()
    dd[evid_col] = dd[evid_col].astype(np.int64)
    tt[evid_col] = tt[evid_col].astype(np.int64)
    merged = dd.merge(tt, on=evid_col, how="inner", suffixes=("_post", "_true"), copy=False)
    if merged.shape[0] == 0:
        raise ValueError("No overlapping events between posterior cat_dd and truth_catalog (after merge on evid).")

    # Optional: enforce one row per evid (take the smallest sample-row index deterministically).
    if bool(dedupe_evid):
        try:
            merged = (
                merged.sort_values("_sample_row", kind="stable")
                .drop_duplicates(subset=[evid_col], keep="first")
                .reset_index(drop=True)
            )
        except Exception:
            # Fallback without stable sort
            try:
                merged = merged.drop_duplicates(subset=[evid_col], keep="first").reset_index(drop=True)
            except Exception:
                pass

    # Resolve truth column names after merge (they may be suffixed if cat_dd already had lon/lat/depth)
    def _resolve_truth_col(base: str) -> str:
        if base in merged.columns:
            return base
        suff = f"{base}_true"
        if suff in merged.columns:
            return suff
        raise KeyError(
            f"Truth column '{base}' not found after merge. "
            f"Available columns include: {list(merged.columns)[:20]}{'...' if len(merged.columns) > 20 else ''}"
        )

    def _resolve_post_col(base: str) -> Optional[str]:
        # Prefer the suffixed post column when merge detected overlap
        suff = f"{base}_post"
        if suff in merged.columns:
            return suff
        if base in merged.columns:
            return base
        return None

    lon_col_m = _resolve_truth_col(truth_lon_col)
    lat_col_m = _resolve_truth_col(truth_lat_col)
    dep_col_m = _resolve_truth_col(truth_depth_col)

    # Preferred posterior location for comparison: absolute lon/lat/dep (projected to km).
    # Fall back to X/Y/Z if lon/lat/dep are not available.
    post_lon_col = _resolve_post_col("longitude")
    post_lat_col = _resolve_post_col("latitude")
    post_dep_col = _resolve_post_col("depth")

    # Project truth lon/lat to XY km in the same frame as posterior X/Y
    truth_xyz = project_lonlat_depth_to_xyz_km(
        merged[lon_col_m].to_numpy(),
        merged[lat_col_m].to_numpy(),
        merged[dep_col_m].to_numpy(),
        lat0=float(lat0),
        lon0=float(lon0),
    )
    merged["X_true_km"] = truth_xyz[:, 0]
    merged["Y_true_km"] = truth_xyz[:, 1]
    merged["Z_true_km"] = truth_xyz[:, 2]

    if post_lon_col is not None and post_lat_col is not None and post_dep_col is not None:
        post_xyz = project_lonlat_depth_to_xyz_km(
            merged[post_lon_col].to_numpy(),
            merged[post_lat_col].to_numpy(),
            merged[post_dep_col].to_numpy(),
            lat0=float(lat0),
            lon0=float(lon0),
        )
        merged["X_post_km"] = post_xyz[:, 0]
        merged["Y_post_km"] = post_xyz[:, 1]
        merged["Z_post_km"] = post_xyz[:, 2]
    else:
        # Fallback: assume X/Y/Z in cat_dd are already absolute km (rare; usually they are deltas)
        merged["X_post_km"] = merged[post_dX].to_numpy(dtype=np.float64)
        merged["Y_post_km"] = merged[post_dY].to_numpy(dtype=np.float64)
        merged["Z_post_km"] = merged[post_dZ].to_numpy(dtype=np.float64)

    ex = merged["X_post_km"].to_numpy(dtype=np.float64) - merged["X_true_km"].to_numpy(dtype=np.float64)
    ey = merged["Y_post_km"].to_numpy(dtype=np.float64) - merged["Y_true_km"].to_numpy(dtype=np.float64)
    ez = merged["Z_post_km"].to_numpy(dtype=np.float64) - merged["Z_true_km"].to_numpy(dtype=np.float64)
    merged["err_x_km"] = ex
    merged["err_y_km"] = ey
    merged["err_z_km"] = ez
    merged["dr_km"] = np.sqrt(ex * ex + ey * ey + ez * ez)

    # Compute covariance either from samples (full 3x3 per event) or from std_* columns (diag)
    if use_full_cov_from_samples:
        _require_torch()
        if any(getattr(summary, k, None) is None for k in ("X", "Y", "Z")):
            raise ValueError("use_full_cov_from_samples=True requires summary.X, summary.Y, summary.Z to be present.")

        # summary.{X,Y,Z} are centered by mean; cov = E[xx], E[xy], ...
        Xc = torch.as_tensor(np.asarray(summary.X), dtype=torch.float64, device="cpu")
        Yc = torch.as_tensor(np.asarray(summary.Y), dtype=torch.float64, device="cpu")
        Zc = torch.as_tensor(np.asarray(summary.Z), dtype=torch.float64, device="cpu")

        # Align to merged event ordering via the original sample row index.
        n_events_samples = int(Xc.shape[0])
        if "_sample_row" not in merged.columns:
            raise RuntimeError("Internal error: missing _sample_row column for sample alignment.")
        rows = merged["_sample_row"].to_numpy(dtype=np.int64)
        bad = (rows < 0) | (rows >= n_events_samples)
        if np.any(bad):
            n_bad = int(np.sum(bad))
            raise ValueError(
                f"Sample alignment failed: {n_bad} merged rows have _sample_row outside [0, {n_events_samples}). "
                f"This suggests inconsistent cat_dd vs sample arrays."
            )

        Xc = Xc[rows]
        Yc = Yc[rows]
        Zc = Zc[rows]

        # Per-event covariances (population; unbiased=False)
        # shape: (N,)
        cxx = torch.mean(Xc * Xc, dim=1)
        cyy = torch.mean(Yc * Yc, dim=1)
        czz = torch.mean(Zc * Zc, dim=1)
        cxy = torch.mean(Xc * Yc, dim=1)
        cxz = torch.mean(Xc * Zc, dim=1)
        cyz = torch.mean(Yc * Zc, dim=1)

        # Assemble (N,3,3)
        cov = torch.stack(
            [
                torch.stack([cxx, cxy, cxz], dim=1),
                torch.stack([cxy, cyy, cyz], dim=1),
                torch.stack([cxz, cyz, czz], dim=1),
            ],
            dim=1,
        )

        # Small jitter for numerical stability (in km^2)
        jitter = 1e-12
        cov = cov + jitter * torch.eye(3, dtype=torch.float64)[None, :, :]
        cov_inv = torch.linalg.inv(cov)

        e = torch.as_tensor(np.column_stack([ex, ey, ez]), dtype=torch.float64)
        maha2 = torch.einsum("ni,nij,nj->n", e, cov_inv, e).detach().cpu().numpy()

        merged["std_x"] = np.sqrt(cxx.detach().cpu().numpy())
        merged["std_y"] = np.sqrt(cyy.detach().cpu().numpy())
        merged["std_z"] = np.sqrt(czz.detach().cpu().numpy())
    else:
        # Diagonal-only using whatever std columns exist
        for c in ("std_x", "std_y", "std_z"):
            if c not in merged.columns:
                raise ValueError(
                    f"Missing '{c}' in cat_dd. Either compute it via uncertainty_metrics=['std'] "
                    f"or set use_full_cov_from_samples=True."
                )
        sx = merged["std_x"].to_numpy(dtype=np.float64)
        sy = merged["std_y"].to_numpy(dtype=np.float64)
        sz = merged["std_z"].to_numpy(dtype=np.float64)
        maha2 = (ex / np.maximum(sx, 1e-12)) ** 2 + (ey / np.maximum(sy, 1e-12)) ** 2 + (ez / np.maximum(sz, 1e-12)) ** 2

    merged["z_x"] = merged["err_x_km"].to_numpy(dtype=np.float64) / np.maximum(merged["std_x"].to_numpy(dtype=np.float64), 1e-12)
    merged["z_y"] = merged["err_y_km"].to_numpy(dtype=np.float64) / np.maximum(merged["std_y"].to_numpy(dtype=np.float64), 1e-12)
    merged["z_z"] = merged["err_z_km"].to_numpy(dtype=np.float64) / np.maximum(merged["std_z"].to_numpy(dtype=np.float64), 1e-12)
    merged["maha2_xyz"] = maha2

    # Aggregate calibration summary
    out: Dict[str, float] = {}
    out["n_events"] = float(merged.shape[0])
    out["rmse_x_km"] = float(np.sqrt(np.mean(ex * ex)))
    out["rmse_y_km"] = float(np.sqrt(np.mean(ey * ey)))
    out["rmse_z_km"] = float(np.sqrt(np.mean(ez * ez)))
    out["rmse_r_km"] = float(np.sqrt(np.mean(merged["dr_km"].to_numpy(dtype=np.float64) ** 2)))

    # 1D calibration: fraction within +/- zcrit
    for p in coverages:
        zc = _norm_ppf_two_sided_coverage(p)
        out[f"cover_{p:.2f}_x"] = float(np.mean(np.abs(merged["z_x"].to_numpy(dtype=np.float64)) <= zc))
        out[f"cover_{p:.2f}_y"] = float(np.mean(np.abs(merged["z_y"].to_numpy(dtype=np.float64)) <= zc))
        out[f"cover_{p:.2f}_z"] = float(np.mean(np.abs(merged["z_z"].to_numpy(dtype=np.float64)) <= zc))
        thr = _chi2_ppf_df3(p)
        out[f"cover_{p:.2f}_xyz"] = float(np.mean(merged["maha2_xyz"].to_numpy(dtype=np.float64) <= thr))

    return CalibrationResult(merged=merged, summary=out)


