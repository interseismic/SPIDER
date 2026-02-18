"""
One-off coverage sanity check (no CLI, no pandas).

This script is meant to replace notebook state for debugging posterior calibration.
It loads:
  - SPIDER samples (H5) referenced by a params JSON
  - a truth catalog (lon/lat/depth in km) with evid mapping

Then it prints:
  - 1D coverage using standardized errors (|err| / std <= zcrit)
  - 3D ellipsoid coverage using per-event sample covariance (maha2 <= chi2_ppf(df=3))
  - sample-quantile credible-interval coverage (truth inside per-event central interval)

Edit the paths in the CONFIG block below and run:
  python /home/zross/git/SPIDER/yifan_redo/check_coverage_once.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from pyproj import Proj

from spider.io import read_all_samples


# -----------------
# CONFIG (edit me)
# -----------------
PARAM_FILE = Path("/home/zross/git/SPIDER/yifan_redo/SPIDER_yifan.json")
TRUE_CAT = Path("/home/zross/git/SPIDER/yifan_redo/true_cat_from_yifan.csv")
# Used to attach evid to TRUE_CAT (TRUE_CAT is typically in the same row order but lacks evid).
PRE_CAT = Path("/home/zross/git/SPIDER/synthetic_yifan/pre_cat.csv")

DEVICE = "cpu"
THIN = 1
BURN_IN = 0

# Optional: apply a simple local bias correction in projected XYZ space before computing errors.
# This matches the *intent* of the notebook but avoids the unit mismatch (km vs degrees).
#
# IMPORTANT: This uses truth-space neighbors to estimate a local translation field.
# That is appropriate for diagnosing "centroid pinned" behavior in synthetic tests,
# but it is an *optimistic* evaluation choice (not applicable to real data).
APPLY_LOCAL_BIAS_CORRECTION = True
LOCAL_BIAS_K_NEIGHBORS = 5

# Projection center: match your notebook (lon0=lon_min, lat0=lat_min).
# (This is not necessarily the best projection center, but keeps comparisons consistent.)


def _as_numpy(a):
    try:
        import torch

        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)


def _load_params() -> Dict:
    with PARAM_FILE.open("r") as f:
        return json.load(f)


def _read_csv_cols(path: Path, cols: Iterable[str]) -> Dict[str, np.ndarray]:
    import csv

    want = list(cols)
    out: Dict[str, List[float]] = {c: [] for c in want}
    out_str: Dict[str, List[str]] = {c: [] for c in want}

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for c in want:
                v = row.get(c, "")
                if v is None or v == "":
                    out_str[c].append("")
                    continue
                out_str[c].append(v)

    for c in want:
        # Try int first, then float
        vals = out_str[c]
        try:
            out[c] = np.asarray([int(v) for v in vals], dtype=np.int64)
            continue
        except Exception:
            pass
        out[c] = np.asarray([float(v) for v in vals], dtype=np.float64)
    return out


def _load_truth_catalog() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (evid, lon, lat, depth_km), all 1D arrays.
    """
    true_cols = _read_csv_cols(TRUE_CAT, ["evid", "longitude", "latitude", "depth"])
    if "evid" in true_cols and true_cols["evid"].dtype == np.int64 and true_cols["evid"].size > 0:
        evid = true_cols["evid"].astype(np.int64, copy=False)
        lon = true_cols["longitude"].astype(np.float64, copy=False)
        lat = true_cols["latitude"].astype(np.float64, copy=False)
        dep = true_cols["depth"].astype(np.float64, copy=False)
        return evid, lon, lat, dep

    # If TRUE_CAT lacks evid, attach from PRE_CAT by row order (matches your notebook)
    pre = _read_csv_cols(PRE_CAT, ["evid"])
    evid = pre["evid"].astype(np.int64, copy=False)
    lon = true_cols["longitude"].astype(np.float64, copy=False)
    lat = true_cols["latitude"].astype(np.float64, copy=False)
    dep = true_cols["depth"].astype(np.float64, copy=False)
    if evid.size != lon.size:
        raise ValueError(f"TRUE_CAT rows ({lon.size}) != PRE_CAT rows ({evid.size}); cannot map evid by row order.")
    return evid, lon, lat, dep


def _projector(lat0: float, lon0: float) -> Proj:
    return Proj(proj="laea", lat_0=float(lat0), lon_0=float(lon0), datum="WGS84", units="km")


def _project_lonlat_to_xy_km(proj: Proj, lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xx, yy = proj(np.asarray(lon_deg), np.asarray(lat_deg))
    return np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64)


def _norm_ppf_two_sided(p: float) -> float:
    table = {
        0.50: 0.6744897501960817,
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    p = float(p)
    if p in table:
        return float(table[p])
    try:
        from scipy.stats import norm  # type: ignore

        return float(norm.ppf((1.0 + p) * 0.5))
    except Exception as e:
        raise ValueError(f"Unsupported p={p} without SciPy installed. Supported: {sorted(table.keys())}") from e


def _chi2_ppf_df3(p: float) -> float:
    table = {
        0.50: 2.3659738843753377,
        0.80: 4.641627676087454,
        0.90: 6.251388631170325,
        0.95: 7.814727903251179,
        0.99: 11.344866730144373,
    }
    p = float(p)
    if p in table:
        return float(table[p])
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.ppf(p, df=3))
    except Exception as e:
        raise ValueError(f"Unsupported p={p} without SciPy installed. Supported: {sorted(table.keys())}") from e


def _local_bias_correction_xyz(
    *,
    x_true: np.ndarray,
    y_true: np.ndarray,
    z_true: np.ndarray,
    x_post: np.ndarray,
    y_post: np.ndarray,
    z_post: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple local bias correction using kNN in truth space:
      bias_x[i] = mean( x_true[nbr] - x_post[nbr] )
    and likewise for y/z.
    """
    pts = np.column_stack([x_true, y_true, z_true]).astype(np.float64, copy=False)
    # Pure-numpy kNN (O(N^2)) for small-ish synthetic runs; avoids sklearn binary-wheel issues.
    # If this becomes too slow, disable APPLY_LOCAL_BIAS_CORRECTION or re-enable sklearn in a clean env.
    k = max(2, int(k))
    # squared distances (N,N)
    # d2[i,j] = ||pts[i]-pts[j]||^2
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum("nij,nij->ni", diff, diff)
    idx = np.argsort(d2, axis=1)[:, :k]  # includes self at col 0
    bias_x = np.zeros_like(x_true, dtype=np.float64)
    bias_y = np.zeros_like(y_true, dtype=np.float64)
    bias_z = np.zeros_like(z_true, dtype=np.float64)
    for i in range(pts.shape[0]):
        nbr = idx[i]
        nbr = nbr[nbr != i]
        if nbr.size >= 1:
            bias_x[i] = float(np.mean(x_true[nbr] - x_post[nbr]))
            bias_y[i] = float(np.mean(y_true[nbr] - y_post[nbr]))
            bias_z[i] = float(np.mean(z_true[nbr] - z_post[nbr]))
    return bias_x, bias_y, bias_z


def _coverage_xyz_from_samples(
    *,
    proj: Proj,
    event_ids: np.ndarray,
    lon_s: np.ndarray,
    lat_s: np.ndarray,
    dep_s: np.ndarray,
    truth_evid: np.ndarray,
    truth_lon: np.ndarray,
    truth_lat: np.ndarray,
    truth_dep: np.ndarray,
    coverages=(0.50, 0.90, 0.95),
) -> Dict[str, float]:
    """
    Compute central-interval coverage per component using posterior samples directly.

    For each event and component:
      truth in [q_lo, q_hi] where q_lo/q_hi are per-event sample quantiles.
    """
    if lon_s.ndim != 2 or lat_s.ndim != 2 or dep_s.ndim != 2:
        raise ValueError(f"Expected samples lon/lat/dep to be 2D arrays (N,S). Got lon={lon_s.shape} lat={lat_s.shape} dep={dep_s.shape}")
    N, S = lon_s.shape
    if lat_s.shape != (N, S) or dep_s.shape != (N, S):
        raise ValueError("Inconsistent sample shapes between longitude/latitude/depth.")

    # Build truth lookup by evid, then align to sample ordering
    truth_map = {int(e): i for i, e in enumerate(truth_evid.tolist())}
    keep = np.asarray([int(e) in truth_map for e in event_ids.tolist()], dtype=bool)
    if not np.all(keep):
        n_drop = int(np.sum(~keep))
        print(f"[warn] dropping {n_drop} events missing from truth catalog (by evid)")
    event_ids = event_ids[keep]
    lon_s = lon_s[keep]
    lat_s = lat_s[keep]
    dep_s = dep_s[keep]
    N, S = lon_s.shape

    idx = np.asarray([truth_map[int(e)] for e in event_ids.tolist()], dtype=np.int64)
    lon_true = truth_lon[idx]
    lat_true = truth_lat[idx]
    dep_true = truth_dep[idx]

    x_true, y_true = _project_lonlat_to_xy_km(proj, lon_true, lat_true)
    z_true = dep_true.astype(np.float64, copy=False)

    # Project all posterior lon/lat samples to XY km.
    # Flatten for one pyproj call (much faster).
    x_flat, y_flat = _project_lonlat_to_xy_km(proj, lon_s.reshape(-1), lat_s.reshape(-1))
    X = x_flat.reshape(N, S)
    Y = y_flat.reshape(N, S)
    Z = dep_s.astype(np.float64, copy=False)

    # Posterior mean (in km)
    x_mean = np.mean(X, axis=1)
    y_mean = np.mean(Y, axis=1)
    z_mean = np.mean(Z, axis=1)

    # Also compute MAP point estimate if present (helps compare to notebook)
    x_map = y_map = z_map = None
    try:
        map_lon = _as_numpy(samples.get("map_longitude"))  # type: ignore[name-defined]
        map_lat = _as_numpy(samples.get("map_latitude"))  # type: ignore[name-defined]
        map_dep = _as_numpy(samples.get("map_depth"))  # type: ignore[name-defined]
        if map_lon is not None and map_lat is not None and map_dep is not None:
            x_map0, y_map0 = _project_lonlat_to_xy_km(proj, np.asarray(map_lon, dtype=np.float64), np.asarray(map_lat, dtype=np.float64))
            x_map = np.asarray(x_map0, dtype=np.float64)
            y_map = np.asarray(y_map0, dtype=np.float64)
            z_map = np.asarray(map_dep, dtype=np.float64)
            # align to truth-kept events
            x_map = x_map[keep]
            y_map = y_map[keep]
            z_map = z_map[keep]
    except Exception:
        x_map = y_map = z_map = None

    # Optional local bias correction (in km)
    if APPLY_LOCAL_BIAS_CORRECTION:
        bx, by, bz = _local_bias_correction_xyz(
            x_true=x_true, y_true=y_true, z_true=z_true, x_post=x_mean, y_post=y_mean, z_post=z_mean, k=int(LOCAL_BIAS_K_NEIGHBORS)
        )
        x_mean = x_mean + bx
        y_mean = y_mean + by
        z_mean = z_mean + bz
        # Shift samples by the same local bias (approximate; strictly, the bias is a function of event position).
        X = X + bx[:, None]
        Y = Y + by[:, None]
        Z = Z + bz[:, None]
        if x_map is not None and y_map is not None and z_map is not None:
            x_map = x_map + bx
            y_map = y_map + by
            z_map = z_map + bz

    # Errors
    ex = x_mean - x_true
    ey = y_mean - y_true
    ez = z_mean - z_true

    # Sample covariance per event (population)
    Xc = X - x_mean[:, None]
    Yc = Y - y_mean[:, None]
    Zc = Z - z_mean[:, None]

    cxx = np.mean(Xc * Xc, axis=1)
    cyy = np.mean(Yc * Yc, axis=1)
    czz = np.mean(Zc * Zc, axis=1)
    cxy = np.mean(Xc * Yc, axis=1)
    cxz = np.mean(Xc * Zc, axis=1)
    cyz = np.mean(Yc * Zc, axis=1)

    cov = np.zeros((N, 3, 3), dtype=np.float64)
    cov[:, 0, 0] = cxx
    cov[:, 1, 1] = cyy
    cov[:, 2, 2] = czz
    cov[:, 0, 1] = cov[:, 1, 0] = cxy
    cov[:, 0, 2] = cov[:, 2, 0] = cxz
    cov[:, 1, 2] = cov[:, 2, 1] = cyz

    cov = cov + (1e-12 * np.eye(3, dtype=np.float64)[None, :, :])
    cov_inv = np.linalg.inv(cov)

    e = np.column_stack([ex, ey, ez]).astype(np.float64, copy=False)
    maha2 = np.einsum("ni,nij,nj->n", e, cov_inv, e)

    sx = np.sqrt(np.maximum(cxx, 1e-24))
    sy = np.sqrt(np.maximum(cyy, 1e-24))
    sz = np.sqrt(np.maximum(czz, 1e-24))
    zx = ex / sx
    zy = ey / sy
    zz = ez / sz

    out: Dict[str, float] = {}
    out["n_events"] = float(N)
    out["rmse_x_km"] = float(np.sqrt(np.mean(ex * ex)))
    out["rmse_y_km"] = float(np.sqrt(np.mean(ey * ey)))
    out["rmse_z_km"] = float(np.sqrt(np.mean(ez * ez)))

    if x_map is not None and y_map is not None and z_map is not None:
        out["rmse_x_km_map"] = float(np.sqrt(np.mean((x_map - x_true) ** 2)))
        out["rmse_y_km_map"] = float(np.sqrt(np.mean((y_map - y_true) ** 2)))
        out["rmse_z_km_map"] = float(np.sqrt(np.mean((z_map - z_true) ** 2)))
    out["median_abs_err_over_std_x"] = float(np.median(np.abs(ex)) / max(np.median(sx), 1e-12))
    out["median_abs_err_over_std_y"] = float(np.median(np.abs(ey)) / max(np.median(sy), 1e-12))
    out["median_abs_err_over_std_z"] = float(np.median(np.abs(ez)) / max(np.median(sz), 1e-12))

    # 1D z-based coverage + 3D chi2 ellipsoid coverage
    for p in coverages:
        zc = _norm_ppf_two_sided(p)
        out[f"cover_{p:.2f}_x"] = float(np.mean(np.abs(zx) <= zc))
        out[f"cover_{p:.2f}_y"] = float(np.mean(np.abs(zy) <= zc))
        out[f"cover_{p:.2f}_z"] = float(np.mean(np.abs(zz) <= zc))
        thr = _chi2_ppf_df3(p)
        out[f"cover_{p:.2f}_xyz"] = float(np.mean(maha2 <= thr))

    # Sample-quantile coverage (central interval)
    for p in coverages:
        lo = 0.5 * (1.0 - float(p))
        hi = 1.0 - lo
        out[f"qcover_{p:.2f}_x"] = float(np.mean((x_true >= np.quantile(X, lo, axis=1)) & (x_true <= np.quantile(X, hi, axis=1))))
        out[f"qcover_{p:.2f}_y"] = float(np.mean((y_true >= np.quantile(Y, lo, axis=1)) & (y_true <= np.quantile(Y, hi, axis=1))))
        out[f"qcover_{p:.2f}_z"] = float(np.mean((z_true >= np.quantile(Z, lo, axis=1)) & (z_true <= np.quantile(Z, hi, axis=1))))

    return out


def main() -> None:
    params = _load_params()
    truth_evid, truth_lon, truth_lat, truth_dep = _load_truth_catalog()

    # Match notebook behavior: use domain lon/lat minima as projection center.
    dom = params.get("domain", {})
    lat0 = float(dom.get("lat_min"))
    lon0 = float(dom.get("lon_min"))
    proj = _projector(lat0, lon0)

    print(f"[load] params={PARAM_FILE}")
    samples_out = params.get("samples_outfile")
    if samples_out is None:
        samples_out = params.get("io", {}).get("samples_outfile", "(missing)")
    print(f"[load] samples_outfile={samples_out}")
    print(f"[load] truth={TRUE_CAT} (rows={int(truth_evid.size)})")

    # Load samples
    samples = read_all_samples(params, backend="torch", device=DEVICE, thin=int(THIN))
    if not samples:
        raise RuntimeError("read_all_samples(...) returned empty samples dict.")

    event_ids = np.asarray(samples["event_ids"], dtype=np.int64)
    lon_s = _as_numpy(samples["longitude"]).astype(np.float64, copy=False)
    lat_s = _as_numpy(samples["latitude"]).astype(np.float64, copy=False)
    dep_s = _as_numpy(samples["depth"]).astype(np.float64, copy=False)

    out = _coverage_xyz_from_samples(
        proj=proj,
        event_ids=event_ids,
        lon_s=lon_s,
        lat_s=lat_s,
        dep_s=dep_s,
        truth_evid=truth_evid,
        truth_lon=truth_lon,
        truth_lat=truth_lat,
        truth_dep=truth_dep,
        coverages=(0.50, 0.90, 0.95),
    )

    print("\n[coverage summary] (computed without pandas)")
    # Keep output ordering stable
    for k in [
        "n_events",
        "rmse_x_km",
        "rmse_y_km",
        "rmse_z_km",
        "median_abs_err_over_std_x",
        "median_abs_err_over_std_y",
        "median_abs_err_over_std_z",
        "cover_0.50_x",
        "cover_0.50_y",
        "cover_0.50_z",
        "cover_0.50_xyz",
        "cover_0.90_x",
        "cover_0.90_y",
        "cover_0.90_z",
        "cover_0.90_xyz",
        "cover_0.95_x",
        "cover_0.95_y",
        "cover_0.95_z",
        "cover_0.95_xyz",
        "qcover_0.50_x",
        "qcover_0.50_y",
        "qcover_0.50_z",
        "qcover_0.90_x",
        "qcover_0.90_y",
        "qcover_0.90_z",
        "qcover_0.95_x",
        "qcover_0.95_y",
        "qcover_0.95_z",
    ]:
        if k in out:
            print(f"{k}: {out[k]}")

    if out["rmse_x_km"] < 1e-12 and out["rmse_y_km"] < 1e-12 and out["rmse_z_km"] < 1e-12:
        raise RuntimeError(
            "RMSE is ~0 in all components. This strongly suggests the truth catalog is identical to the posterior "
            "(or was mapped incorrectly). Double-check TRUE_CAT/PRE_CAT mapping."
        )


if __name__ == "__main__":
    main()


