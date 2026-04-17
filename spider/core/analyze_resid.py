from __future__ import annotations

from typing import Optional, Any

import os
import math
import torch
import torch.nn as nn
import numpy as np
import polars as pl

from spider.io.phase_bundle import load_phase2_bundle
from spider.core.init_state import _build_initial_state
from spider.utils.console import info, warn

from spider.core.modeling import compute_residuals
from spider.core.state import _current_noise_scales
from spider.core.shared_event_re_whitening import build_whitening_cache_entry



# Standardized stdout helper
def _log(*parts, section: str = "ANALYZE", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

@torch.no_grad()
def _corr_error_delta_for_rows(*, state, II_b: torch.Tensor, YY_b: torch.Tensor, sta_b: torch.Tensor) -> Optional[torch.Tensor]:
    """corr_error was removed; keep a no-op stub for legacy diagnostics."""
    return None


@torch.no_grad()
def _compute_residuals_numpy(
    *,
    state,
    II_b: torch.Tensor,
    YY_b: torch.Tensor,
    dX_use: torch.Tensor,
    sta_b: Optional[torch.Tensor] = None,
    variant: str = "base",
) -> np.ndarray:
    """
    Residuals for diagnostics, optionally including corr_error as a nuisance correction.

    Args:
      variant:
        - "base": residual = obs - pred
        - "corr_error": residual = obs - (pred + delta_corr)  [if available]
    """
    from spider.core.modeling import compute_residuals

    rb = compute_residuals(II_b, YY_b, state.X_src, dX_use, state.model)
    v = str(variant).strip().lower()
    if v in {"corr_error", "with_corr_error", "corr"}:
        if sta_b is not None:
            dc = _corr_error_delta_for_rows(state=state, II_b=II_b, YY_b=YY_b, sta_b=sta_b)
            if isinstance(dc, torch.Tensor):
                rb = rb - dc.to(dtype=rb.dtype, device=rb.device)
    return rb.detach().to("cpu").numpy().astype(np.float64, copy=False)


def _get_diag_cfg(state, key: str) -> dict:
    try:
        # Removed diagnostics: keep hard-disabled at runtime.
        if key in {"resid_distribution", "shared_event_legcorr2d", "resid_scalar_metrics"}:
            return {}
        inf = state.params.get("inference", None)
        dg = (inf.get("diagnostics", None) if isinstance(inf, dict) else None)
        cfg = dg.get(key, {}) if isinstance(dg, dict) else {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _diag_enabled(cfg: dict, *, default: bool = True) -> bool:
    try:
        if "enabled" in cfg and cfg.get("enabled", None) is not None:
            return bool(cfg.get("enabled"))
    except Exception:
        pass
    return bool(default)


def _get_truth_catalog_cfg(state) -> dict:
    cfg = _get_diag_cfg(state, "truth_catalog")
    if isinstance(cfg, dict) and cfg:
        return cfg
    cfg = _get_diag_cfg(state, "truth_locations")
    return cfg if isinstance(cfg, dict) else {}


def _truth_dX_from_catalog(state) -> Optional[torch.Tensor]:
    """
    Build a ΔX_src tensor from a truth catalog (CSV).
    Falls back to MAP dX_src for missing events or missing time.
    """
    try:
        cfg = _get_truth_catalog_cfg(state)
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict) or not cfg:
        warn("residuals_at='truth' requested but no diagnostics.truth_catalog config found.", section="DIAG")
        return None

    path = cfg.get("path", None)
    if path is None:
        path = cfg.get("catalog_path", None)
    if path is None:
        path = cfg.get("truth_catalog", None)
    if path is None:
        warn("truth_catalog config missing 'path' (or 'catalog_path')", section="DIAG")
        return None
    path = str(path)

    require_all = bool(cfg.get("require_all", False))
    time_source = str(cfg.get("time_source", "map")).strip().lower()
    if time_source not in {"map", "initial", "truth"}:
        time_source = "map"
    time_ref = str(cfg.get("time_ref", "min")).strip().lower()
    if time_ref not in {"min", "median", "first"}:
        time_ref = "min"

    try:
        truth = pl.read_csv(path)
    except Exception as e:
        warn(f"Failed to read truth_catalog '{path}': {e}", section="DIAG")
        return None

    missing_cols = [c for c in ("evid", "longitude", "latitude", "depth") if c not in truth.columns]
    if missing_cols:
        warn(f"truth_catalog missing required columns: {missing_cols}", section="DIAG")
        return None

    truth = truth.with_columns(pl.col("evid").cast(pl.Utf8).alias("evid"))
    evid_truth = truth["evid"].to_list()
    truth_idx = {str(ev): i for i, ev in enumerate(evid_truth)}

    try:
        lon = truth["longitude"].to_numpy()
        lat = truth["latitude"].to_numpy()
        dep = truth["depth"].to_numpy()
    except Exception:
        warn("truth_catalog could not be converted to numpy arrays", section="DIAG")
        return None

    try:
        xx, yy = state.projector(lon, lat)
        xyz = np.column_stack([np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64), np.asarray(dep, dtype=np.float64)])
    except Exception as e:
        warn(f"Failed to project truth_catalog lon/lat: {e}", section="DIAG")
        return None

    base = state.X_src.detach().cpu().numpy().astype(np.float64, copy=False)
    dX_map = state.dX_src.detach().cpu().numpy().astype(np.float64, copy=False)
    dX = dX_map.copy()

    # Optional truth time (seconds relative to reference)
    t_truth = None
    if time_source == "truth":
        if "time" not in truth.columns:
            warn("truth_catalog missing 'time' column; falling back to MAP times.", section="DIAG")
            time_source = "map"
        else:
            try:
                tcol = truth["time"].cast(pl.Utf8).str.strptime(pl.Datetime("ns"), strict=False)
                if tcol.null_count() > 0:
                    raise ValueError("truth_catalog.time parse failed")
                t_ns = tcol.cast(pl.Int64).to_numpy()
                if time_ref == "first":
                    ref_ns = int(t_ns[0]) if t_ns.size > 0 else 0
                elif time_ref == "median":
                    ref_ns = int(np.median(t_ns)) if t_ns.size > 0 else 0
                else:
                    ref_ns = int(np.min(t_ns)) if t_ns.size > 0 else 0
                t_truth = (t_ns - ref_ns).astype(np.float64) / 1e9
            except Exception as e:
                warn(f"truth_catalog time parse failed; falling back to MAP times. ({e})", section="DIAG")
                time_source = "map"

    n_events = int(base.shape[0])
    missing = 0
    for i, row in enumerate(state.origins0.iter_rows(named=True)):
        ev = str(row.get("evid"))
        j = truth_idx.get(ev, None)
        if j is None:
            missing += 1
            continue
        dX[i, 0] = float(xyz[j, 0] - base[i, 0])
        dX[i, 1] = float(xyz[j, 1] - base[i, 1])
        dX[i, 2] = float(xyz[j, 2] - base[i, 2])
        if time_source == "truth" and t_truth is not None and int(j) < int(t_truth.size):
            dX[i, 3] = float(t_truth[int(j)] - base[i, 3])
        elif time_source == "initial":
            dX[i, 3] = 0.0
        # else keep MAP time from dX_map

    if missing > 0:
        msg = f"truth_catalog missing {missing}/{n_events} events; falling back to MAP for those."
        if require_all:
            raise ValueError(msg)
        warn(msg, section="DIAG")

    return torch.tensor(dX, dtype=torch.float32, device=state.device)


def _resolve_dX_use(state, residuals_at: str) -> torch.Tensor:
    mode = str(residuals_at).strip().lower()
    if mode in {"init", "initial"}:
        return torch.zeros_like(state.dX_src, device=state.dX_src.device)
    if mode in {"truth", "true"}:
        dX_truth = getattr(state, "_truth_dX_cache", None)
        if not isinstance(dX_truth, torch.Tensor):
            dX_truth = _truth_dX_from_catalog(state)
            setattr(state, "_truth_dX_cache", dX_truth)
        if isinstance(dX_truth, torch.Tensor):
            return dX_truth
        warn("truth residuals requested but truth catalog unavailable; using MAP.", section="DIAG")
    return state.dX_src


def _mpl_pyplot():
    """
    Import matplotlib in headless mode. Returns (plt, ok).
    """
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt  # type: ignore
        return plt, True
    except Exception:
        return None, False


def _safe_excess_kurtosis(x: np.ndarray) -> float:
    """
    Excess kurtosis (Fisher) using a simple moment estimator.
    Outlier-sensitive but a useful rough tail diagnostic.
    """
    try:
        z = np.asarray(x, dtype=np.float64)
        z = z[np.isfinite(z)]
        if z.size < 4:
            return float("nan")
        m = float(np.mean(z))
        c = z - m
        v = float(np.mean(c * c))
        if (not np.isfinite(v)) or v <= 0.0:
            return float("nan")
        m4 = float(np.mean((c * c) * (c * c)))
        return float(m4 / (v * v) - 3.0)
    except Exception:
        return float("nan")


def _safe_skew(x: np.ndarray) -> float:
    """Skewness using a simple moment estimator."""
    try:
        z = np.asarray(x, dtype=np.float64)
        z = z[np.isfinite(z)]
        if z.size < 3:
            return float("nan")
        m = float(np.mean(z))
        c = z - m
        v = float(np.mean(c * c))
        if (not np.isfinite(v)) or v <= 0.0:
            return float("nan")
        s = float(np.mean(c * c * c))
        return float(s / (v ** 1.5))
    except Exception:
        return float("nan")


def _tail_ratio(abs_z: np.ndarray, *, q_hi: float = 0.95, q_lo: float = 0.75) -> float:
    """
    Tail-heaviness proxy: q_hi(|z|) / q_lo(|z|).

    Rough reference:
      - Normal ≈ 2.44 (0.95/0.75)
      - Laplace (unit var) ≈ 3.32
    """
    try:
        a = np.asarray(abs_z, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size < 16:
            return float("nan")
        hi = float(np.quantile(a, float(q_hi)))
        lo = float(np.quantile(a, float(q_lo)))
        if (not np.isfinite(hi)) or (not np.isfinite(lo)) or lo <= 0.0:
            return float("nan")
        return float(hi / lo)
    except Exception:
        return float("nan")


def _robust_std_from_residuals(r: Any) -> float:
    """
    Robust scale estimate using MAD (with std fallback).
    Accepts torch.Tensor or numpy arrays.
    """
    try:
        if isinstance(r, np.ndarray):
            x = np.asarray(r, dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size < 4:
                return float("nan")
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            s = 1.4826 * mad
            if (not np.isfinite(s)) or s <= 0.0:
                s = float(np.std(x, ddof=0))
            return float(s)
        if not isinstance(r, torch.Tensor):
            return float("nan")
        x = r.detach().to(dtype=torch.float32).flatten()
        if x.numel() < 4:
            return float("nan")
        finite = torch.isfinite(x)
        if not bool(finite.any()):
            return float("nan")
        x = x[finite]
        med = torch.median(x)
        mad = torch.median(torch.abs(x - med))
        s = 1.4826 * mad
        if (not torch.isfinite(s)) or float(s.item()) <= 0.0:
            s = torch.std(x, unbiased=False)
        return float(s.detach().cpu().item())
    except Exception:
        return float("nan")


@torch.no_grad()
def estimate_shared_event_re_tau_s(
    *,
    state,
    n_rows: int = 200_000,
    seed: int = 0,
    batch_size: int = 50_000,
    residuals_at: str = "map",  # "map" | "initial"
    sigma_quantile: float = 0.2,
    sigma_source: str = "quantile",  # "quantile" | "params"
) -> Optional[dict]:
    """
    Estimate shared_event_re tau from residual variance:
      Var(r) ≈ sigma^2 + 2*tau^2  =>  tau ≈ sqrt(max(0, Var - sigma^2) / 2)
    """
    try:
        if not bool(state.params.get("_shared_event_re_enabled", False)):
            return None
    except Exception:
        return None

    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None

    try:
        mode = str(residuals_at).strip().lower()
    except Exception:
        mode = "map"
    if mode not in {"map", "initial", "truth"}:
        mode = "map"
    dX_use = _resolve_dX_use(state, mode)

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    rP_all: list[torch.Tensor] = []
    rS_all: list[torch.Tensor] = []
    for i0 in range(0, int(rows_np.size), batch_size):
        ii = rows_np[i0 : i0 + batch_size]
        idx = state.II.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        y = state.YY.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        r = compute_residuals(idx, y, state.X_src, dX_use, state.model).detach()
        ph = y[:, 4].detach()
        is_s = (ph > 0.5)
        finite = torch.isfinite(r)
        if not bool(finite.any()):
            continue
        if not bool(finite.all()):
            r = r[finite]
            is_s = is_s[finite]
        if bool((~is_s).any()):
            rP_all.append(r[~is_s].detach().to("cpu", dtype=torch.float32))
        if bool(is_s.any()):
            rS_all.append(r[is_s].detach().to("cpu", dtype=torch.float32))

    rP = torch.cat(rP_all, dim=0) if rP_all else torch.empty((0,), dtype=torch.float32)
    rS = torch.cat(rS_all, dim=0) if rS_all else torch.empty((0,), dtype=torch.float32)
    if rP.numel() + rS.numel() == 0:
        return None

    def _sigma_from_quantile(r: torch.Tensor, q: float) -> float:
        try:
            if r.numel() == 0:
                return float("nan")
            q = float(q)
            if not (0.01 <= q <= 0.99):
                q = 0.5
            a = r.detach().float()
            a = a[torch.isfinite(a)]
            if a.numel() == 0:
                return float("nan")
            a_np = a.detach().cpu().numpy()
            aq = float(np.quantile(np.abs(a_np), q))
            if not (np.isfinite(aq) and aq >= 0.0):
                return float("nan")
            zf = float("nan")
            try:
                from scipy.stats import norm  # type: ignore
                zf = float(norm.ppf((q + 1.0) / 2.0))
            except Exception:
                try:
                    zt = torch.distributions.Normal(0.0, 1.0).icdf(
                        torch.tensor((q + 1.0) / 2.0, dtype=torch.float32)
                    )
                    zf = float(zt.detach().cpu().item())
                except Exception:
                    zf = float("nan")
            if not (np.isfinite(zf) and zf > 1e-6):
                return float("nan")
            return float(aq / zf)
        except Exception:
            return float("nan")

    stdP = _robust_std_from_residuals(rP)
    stdS = _robust_std_from_residuals(rS)

    sigma_p_est = _sigma_from_quantile(rP, float(sigma_quantile))
    sigma_s_est = _sigma_from_quantile(rS, float(sigma_quantile))
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _robust_std_from_residuals(rP)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _robust_std_from_residuals(rS)

    try:
        vv = state.params.get("phase_unc", [float("nan"), float("nan")])
        sigma_p_param = float(vv[0])
        sigma_s_param = float(vv[1])
    except Exception:
        sigma_p_param = float("nan")
        sigma_s_param = float("nan")
    if not (np.isfinite(sigma_p_param) and sigma_p_param > 0.0):
        sigma_p_param = float("nan")
    if not (np.isfinite(sigma_s_param) and sigma_s_param > 0.0):
        sigma_s_param = float("nan")

    sigma_source = str(sigma_source).strip().lower()
    if sigma_source not in {"params", "quantile"}:
        sigma_source = "quantile"

    sigma_p = sigma_p_est
    sigma_s = sigma_s_est
    sigma_source_used = "quantile"
    if sigma_source == "params" and np.isfinite(sigma_p_param) and np.isfinite(sigma_s_param):
        sigma_p = sigma_p_param
        sigma_s = sigma_s_param
        sigma_source_used = "params"

    def _tau_from_std(std: float, sigma: float) -> float:
        if not (np.isfinite(std) and std > 0.0):
            return float("nan")
        if not (np.isfinite(sigma) and sigma >= 0.0):
            sigma = 0.0
        v = max(0.0, std * std - sigma * sigma)
        return float(np.sqrt(v / 2.0))

    tau_p = _tau_from_std(stdP, sigma_p)
    tau_s = _tau_from_std(stdS, sigma_s)

    return {
        "tau_p": float(tau_p),
        "tau_s": float(tau_s),
        "sigma_p_used": float(sigma_p),
        "sigma_s_used": float(sigma_s),
        "sigma_source": str(sigma_source_used),
        "sigma_p_est": float(sigma_p_est),
        "sigma_s_est": float(sigma_s_est),
        "sigma_p_param": float(sigma_p_param),
        "sigma_s_param": float(sigma_s_param),
        "resid_std_p": float(stdP),
        "resid_std_s": float(stdS),
        "n_rows_used_p": int(rP.numel()),
        "n_rows_used_s": int(rS.numel()),
    }


@torch.no_grad()
def estimate_shared_event_re_tau_logdet(
    *,
    state,
    n_rows: int,
    seed: int,
    batch_size: int,
    residuals_at: str,
    sigma_source: str,
    sigma_quantile: float,
    tau_grid: list[float],
    max_groups: int,
    max_nodes: int,
    max_edges: int,
) -> Optional[dict]:
    """
    Estimate tau by minimizing (0.5 r^T Σ^{-1} r + 0.5 log|Σ|) over a tau grid,
    using a small subset of groups and dense Cholesky (node-space).
    """
    try:
        if not bool(state.params.get("_shared_event_re_enabled", False)):
            return None
    except Exception:
        return None

    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None

    try:
        mode = str(residuals_at).strip().lower()
    except Exception:
        mode = "map"
    if mode not in {"map", "initial", "truth"}:
        mode = "map"
    dX_use = _resolve_dX_use(state, mode)

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()
    rows_t = torch.tensor(rows_np, device=state.device, dtype=torch.int64)

    idx = state.II.index_select(0, rows_t)
    y = state.YY.index_select(0, rows_t)
    r = compute_residuals(idx, y, state.X_src, dX_use, state.model).detach()
    ph = y[:, 4].detach()
    is_s = (ph > 0.5)
    finite = torch.isfinite(r)
    if not bool(finite.any()):
        return None
    if not bool(finite.all()):
        r = r[finite]
        is_s = is_s[finite]
        idx = idx[finite]

    # sigma selection
    try:
        vv = state.params.get("phase_unc", [float("nan"), float("nan")])
        sigma_p_param = float(vv[0])
        sigma_s_param = float(vv[1])
    except Exception:
        sigma_p_param = float("nan")
        sigma_s_param = float("nan")
    if not (np.isfinite(sigma_p_param) and sigma_p_param > 0.0):
        sigma_p_param = float("nan")
    if not (np.isfinite(sigma_s_param) and sigma_s_param > 0.0):
        sigma_s_param = float("nan")

    def _sigma_from_quantile(r0: torch.Tensor, q: float) -> float:
        try:
            if r0.numel() == 0:
                return float("nan")
            q = float(q)
            if not (0.01 <= q <= 0.99):
                q = 0.5
            a = r0.detach().float()
            a = a[torch.isfinite(a)]
            if a.numel() == 0:
                return float("nan")
            a_np = a.detach().cpu().numpy()
            aq = float(np.quantile(np.abs(a_np), q))
            if not (np.isfinite(aq) and aq >= 0.0):
                return float("nan")
            zf = float("nan")
            try:
                from scipy.stats import norm  # type: ignore
                zf = float(norm.ppf((q + 1.0) / 2.0))
            except Exception:
                try:
                    zt = torch.distributions.Normal(0.0, 1.0).icdf(
                        torch.tensor((q + 1.0) / 2.0, dtype=torch.float32)
                    )
                    zf = float(zt.detach().cpu().item())
                except Exception:
                    zf = float("nan")
            if not (np.isfinite(zf) and zf > 1e-6):
                return float("nan")
            return float(aq / zf)
        except Exception:
            return float("nan")

    rP = r[~is_s]
    rS = r[is_s]
    sigma_p_est = _sigma_from_quantile(rP, float(sigma_quantile))
    sigma_s_est = _sigma_from_quantile(rS, float(sigma_quantile))
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _robust_std_from_residuals(rP)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _robust_std_from_residuals(rS)

    sigma_source = str(sigma_source).strip().lower()
    if sigma_source not in {"params", "quantile"}:
        sigma_source = "quantile"
    sigma_p = sigma_p_est
    sigma_s = sigma_s_est
    sigma_source_used = "quantile"
    if sigma_source == "params" and np.isfinite(sigma_p_param) and np.isfinite(sigma_s_param):
        sigma_p = sigma_p_param
        sigma_s = sigma_s_param
        sigma_source_used = "params"

    # Build grouping keys
    grouping = str(state.params.get("_shared_event_re_grouping", "station_phase")).strip().lower()
    if grouping in {"stationphase", "station-phase"}:
        grouping = "station_phase"
    if grouping != "station_phase":
        warn("shared_event_re tau logdet: only station_phase grouping is supported.", section="DIAG")
        return None
    ph_id = torch.where(is_s, torch.ones_like(r, dtype=torch.int64), torch.zeros_like(r, dtype=torch.int64))
    sta_idx = getattr(state, "row_station_index", None)
    if not isinstance(sta_idx, torch.Tensor):
        warn("shared_event_re tau logdet: missing row_station_index.", section="DIAG")
        return None
    sta = sta_idx.index_select(0, rows_t)[finite]
    keys = (sta.to(dtype=torch.int64) * 2) + ph_id

    # Whitening weights configuration
    edge_weighting = str(state.params.get("_shared_event_re_edge_weight_mode", "uniform")).strip().lower()
    edge_weight_ell_km = float(state.params.get("_shared_event_re_edge_weight_ell_km", 1.0))
    edge_weight_eps_km = float(state.params.get("_shared_event_re_edge_weight_eps_km", 1e-3))
    edge_weight_power = float(state.params.get("_shared_event_re_edge_weight_power", 1.0))
    edge_weight_scale_km = float(state.params.get("_shared_event_re_edge_weight_scale_km", 1.0))
    edge_weight_global_scale = float(state.params.get("_shared_event_re_edge_weight_global_scale", 1.0))
    edge_weight_normalize = bool(state.params.get("_shared_event_re_edge_weight_normalize", False))

    X_event = (state.X_src + dX_use)[:, :3].detach()

    cache = build_whitening_cache_entry(
        idx=idx,
        keys=keys,
        ph_id=ph_id,
        sigma_p=torch.tensor(float(sigma_p), device=state.device),
        sigma_s=torch.tensor(float(sigma_s), device=state.device),
        tau_p=1.0,
        tau_s=1.0,
        jitter0=0.0,
        max_rows_per_group=int(max_edges),
        max_nodes_per_group=int(max_nodes),
        solver="pcg",
        edge_weighting=edge_weighting,
        edge_weight_ell_km=float(edge_weight_ell_km),
        edge_weight_eps_km=float(edge_weight_eps_km),
        edge_weight_power=float(edge_weight_power),
        edge_weight_scale_km=float(edge_weight_scale_km),
        edge_weight_global_scale=float(edge_weight_global_scale),
        edge_weight_normalize=bool(edge_weight_normalize),
        X_event=X_event,
        grouping_cache=None,
    )

    perm = cache.get("perm", None)
    groups = cache.get("groups", None)
    group_ph = cache.get("group_ph", None)
    if not isinstance(perm, torch.Tensor) or not isinstance(groups, list):
        return None
    r_perm = r.index_select(0, perm)

    # Gather group data and filter by size
    gdata_p = []
    gdata_s = []
    for gi, gd in enumerate(groups):
        if not isinstance(gd, dict):
            continue
        n_nodes = int(gd.get("n_nodes", 0) or 0)
        s0 = int(gd.get("start", 0))
        e0 = int(gd.get("end", 0))
        m = int(max(e0 - s0, 0))
        if n_nodes <= 1 or m <= 1:
            continue
        if n_nodes > int(max_nodes) or m > int(max_edges):
            continue
        local_u = gd.get("local_u", None)
        local_v = gd.get("local_v", None)
        w = gd.get("w", None)
        if not (isinstance(local_u, torch.Tensor) and isinstance(local_v, torch.Tensor) and isinstance(w, torch.Tensor)):
            continue
        ph_bit = None
        try:
            if isinstance(group_ph, torch.Tensor) and int(group_ph.numel()) > gi:
                ph_bit = float(group_ph[gi].item())
        except Exception:
            ph_bit = None
        sigma_g = float(gd.get("sigma", float("nan")))
        r_g = r_perm[s0:e0].detach().to("cpu", dtype=torch.float64)
        if not torch.isfinite(r_g).any():
            continue
        # Precompute Laplacian + weights (CPU, dense)
        u = local_u.detach().to("cpu")
        v = local_v.detach().to("cpu")
        w_cpu = w.detach().to("cpu", dtype=torch.float64)
        w_sqrt = torch.sqrt(w_cpu.clamp_min(0.0))
        L = torch.zeros((n_nodes, n_nodes), dtype=torch.float64)
        deg = torch.zeros((n_nodes,), dtype=torch.float64)
        deg.index_add_(0, u, w_cpu)
        deg.index_add_(0, v, w_cpu)
        L[u, v] -= w_cpu
        L[v, u] -= w_cpu
        L.diagonal().add_(deg)
        entry = {
            "r": r_g,
            "u": u,
            "v": v,
            "w": w_cpu,
            "w_sqrt": w_sqrt,
            "L": L,
            "n_nodes": int(n_nodes),
            "m": int(m),
            "sigma": float(sigma_g),
        }
        if ph_bit is not None and float(ph_bit) < 0.5:
            gdata_p.append(entry)
        else:
            gdata_s.append(entry)

    def _take_top(groups_in: list[dict]) -> list[dict]:
        if not groups_in:
            return []
        groups_in = sorted(groups_in, key=lambda d: int(d.get("m", 0)), reverse=True)
        return groups_in[: int(max_groups)]

    gdata_p = _take_top(gdata_p)
    gdata_s = _take_top(gdata_s)

    def _logdet_fit(groups_in: list[dict], tau_grid_v: list[float]) -> tuple[float, float]:
        best_tau = float("nan")
        best_obj = float("inf")
        for tau in tau_grid_v:
            t = float(tau)
            if not (np.isfinite(t) and t > 0.0):
                continue
            obj = 0.0
            n_used = 0
            for gd in groups_in:
                r_g = gd["r"]
                u = gd["u"]
                v = gd["v"]
                w = gd["w"]
                w_sqrt = gd["w_sqrt"]
                L = gd["L"]
                n_nodes = int(gd["n_nodes"])
                m = int(gd["m"])
                sig = float(gd["sigma"])
                if not (np.isfinite(sig) and sig > 0.0):
                    continue
                alpha = 1.0 / (t * t)
                beta = 1.0 / (sig * sig)
                M = L.mul(beta)
                M.diagonal().add_(alpha)
                try:
                    chol = torch.linalg.cholesky(M)
                except Exception:
                    continue
                # logdet(M)
                logdet_M = 2.0 * torch.log(torch.diagonal(chol)).sum().item()
                # b = A^T (beta * w_sqrt * r)
                edge_vals = (beta * w_sqrt * r_g)
                b = torch.zeros((n_nodes,), dtype=torch.float64)
                b.index_add_(0, u, -edge_vals)
                b.index_add_(0, v, edge_vals)
                x = torch.cholesky_solve(b.unsqueeze(1), chol).squeeze(1)
                ax = x.index_select(0, v) - x.index_select(0, u)
                u_edge = beta * (r_g - (w_sqrt * ax))
                quad = 0.5 * (r_g * u_edge).sum().item()
                logdet_sigma = (float(m) * math.log(sig * sig)) + logdet_M - (float(n_nodes) * math.log(alpha))
                obj += float(quad + 0.5 * logdet_sigma)
                n_used += 1
            if n_used <= 0:
                continue
            if obj < best_obj:
                best_obj = obj
                best_tau = t
        return best_tau, float(best_obj)

    tau_p_best, obj_p = _logdet_fit(gdata_p, tau_grid)
    tau_s_best, obj_s = _logdet_fit(gdata_s, tau_grid)

    return {
        "tau_p": float(tau_p_best),
        "tau_s": float(tau_s_best),
        "obj_p": float(obj_p),
        "obj_s": float(obj_s),
        "sigma_p_used": float(sigma_p),
        "sigma_s_used": float(sigma_s),
        "sigma_source": str(sigma_source_used),
        "n_groups_p": int(len(gdata_p)),
        "n_groups_s": int(len(gdata_s)),
        "n_rows_used": int(r.numel()),
    }


@torch.no_grad()
def estimate_shared_event_re_hier_tau_s(
    *,
    state,
    n_rows: int = 200_000,
    seed: int = 0,
    batch_size: int = 50_000,
    residuals_at: str = "map",  # "map" | "initial"
    sigma_quantile: float = 0.2,
    min_events_per_cluster: int = 1,
) -> Optional[dict]:
    """
    Estimate hierarchical shared_event_re scales from MAP residuals:
      Var(within)  ≈ sigma^2 + 2*tau_event^2
      Var(cross)   ≈ sigma^2 + 2*tau_event^2 + 2*tau_cluster^2
    """
    try:
        if not bool(state.params.get("_shared_event_re_hierarchical", False)):
            return None
    except Exception:
        return None
    try:
        cm = str(state.params.get("_shared_event_re_cluster_mode", "none")).strip().lower()
        if cm not in {"dd_khop", "component"}:
            return None
    except Exception:
        return None
    cluster_ids = state.params.get("_shared_event_re_cluster_ids", None)
    if not isinstance(cluster_ids, torch.Tensor):
        warn("shared_event_re hier tau estimate: missing cluster ids", section="DIAG")
        return None
    try:
        min_events_per_cluster = int(min_events_per_cluster)
        if min_events_per_cluster < 1:
            min_events_per_cluster = 1
    except Exception:
        min_events_per_cluster = 1

    # Build cluster-size mask (CPU) so tiny clusters are excluded.
    try:
        c_cpu = cluster_ids.detach().to("cpu")
        counts = torch.bincount(c_cpu, minlength=int(c_cpu.max().item() + 1) if c_cpu.numel() > 0 else 0)
        keep_c = counts >= int(min_events_per_cluster)
        keep_c = keep_c.to(device=state.device)
    except Exception:
        keep_c = None

    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None

    try:
        mode = str(residuals_at).strip().lower()
    except Exception:
        mode = "map"
    if mode not in {"map", "initial", "truth"}:
        mode = "map"
    dX_use = _resolve_dX_use(state, mode)

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    rP_within = []
    rP_cross = []
    rS_within = []
    rS_cross = []
    rP_all = []
    rS_all = []
    for i0 in range(0, int(rows_np.size), batch_size):
        ii = rows_np[i0 : i0 + batch_size]
        idx = state.II.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        y = state.YY.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        r = compute_residuals(idx, y, state.X_src, dX_use, state.model).detach()
        ph = y[:, 4].detach()
        is_s = (ph > 0.5)
        e1 = idx[:, 0].to(torch.int64)
        e2 = idx[:, 1].to(torch.int64)
        c1 = cluster_ids.index_select(0, e1)
        c2 = cluster_ids.index_select(0, e2)
        finite = torch.isfinite(r)
        if not bool(finite.any()):
            continue
        if not bool(finite.all()):
            r = r[finite]
            ph = ph[finite]
            is_s = is_s[finite]
            c1 = c1[finite]
            c2 = c2[finite]
        if keep_c is not None:
            ok = keep_c.index_select(0, c1) & keep_c.index_select(0, c2)
            if not bool(ok.any()):
                continue
            c1 = c1[ok]
            c2 = c2[ok]
            r = r[ok]
            ph = ph[ok]
            is_s = is_s[ok]
        same = (c1 == c2)
        if bool((~is_s).any()):
            rP = r[~is_s]
            rP_all.append(rP.detach().to("cpu", dtype=torch.float32))
            sameP = same[~is_s]
            if bool(sameP.any()):
                rP_within.append(rP[sameP].detach().to("cpu", dtype=torch.float32))
            if bool((~sameP).any()):
                rP_cross.append(rP[~sameP].detach().to("cpu", dtype=torch.float32))
        if bool(is_s.any()):
            rS = r[is_s]
            rS_all.append(rS.detach().to("cpu", dtype=torch.float32))
            sameS = same[is_s]
            if bool(sameS.any()):
                rS_within.append(rS[sameS].detach().to("cpu", dtype=torch.float32))
            if bool((~sameS).any()):
                rS_cross.append(rS[~sameS].detach().to("cpu", dtype=torch.float32))

    def _finite_only(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            return x
        m = torch.isfinite(x)
        if not bool(m.any()):
            return x[:0]
        if not bool(m.all()):
            return x[m]
        return x

    rP_w = torch.cat(rP_within, dim=0) if len(rP_within) else torch.empty((0,), dtype=torch.float32)
    rP_c = torch.cat(rP_cross, dim=0) if len(rP_cross) else torch.empty((0,), dtype=torch.float32)
    rS_w = torch.cat(rS_within, dim=0) if len(rS_within) else torch.empty((0,), dtype=torch.float32)
    rS_c = torch.cat(rS_cross, dim=0) if len(rS_cross) else torch.empty((0,), dtype=torch.float32)
    rP_a = torch.cat(rP_all, dim=0) if len(rP_all) else torch.empty((0,), dtype=torch.float32)
    rS_a = torch.cat(rS_all, dim=0) if len(rS_all) else torch.empty((0,), dtype=torch.float32)

    rP_w = _finite_only(rP_w)
    rP_c = _finite_only(rP_c)
    rS_w = _finite_only(rS_w)
    rS_c = _finite_only(rS_c)
    rP_a = _finite_only(rP_a)
    rS_a = _finite_only(rS_a)

    if rP_w.numel() + rS_w.numel() == 0:
        return None

    stdP_w = _robust_std_from_residuals(rP_w)
    stdP_c = _robust_std_from_residuals(rP_c) if rP_c.numel() > 0 else float("nan")
    stdS_w = _robust_std_from_residuals(rS_w)
    stdS_c = _robust_std_from_residuals(rS_c) if rS_c.numel() > 0 else float("nan")
    if not np.isfinite(stdP_c):
        stdP_c = 0.0
    if not np.isfinite(stdS_c):
        stdS_c = 0.0

    try:
        σp, σs = _current_noise_scales(state)
        sigma_p = float(σp.detach().cpu().item())
        sigma_s = float(σs.detach().cpu().item())
    except Exception:
        sigma_p, sigma_s = float("nan"), float("nan")

    def _tau_event(std_w: float, sigma: float) -> float:
        if not (np.isfinite(std_w) and std_w > 0.0):
            return float("nan")
        if not (np.isfinite(sigma) and sigma >= 0.0):
            sigma = 0.0
        v = max(0.0, std_w * std_w - sigma * sigma)
        return float(np.sqrt(v / 2.0))

    def _tau_cluster(std_c: float, sigma: float, tau_e: float) -> float:
        if not (np.isfinite(std_c) and std_c > 0.0):
            return float("nan")
        if not (np.isfinite(sigma) and sigma >= 0.0):
            sigma = 0.0
        if not (np.isfinite(tau_e) and tau_e >= 0.0):
            tau_e = 0.0
        v = max(0.0, std_c * std_c - sigma * sigma - 2.0 * tau_e * tau_e)
        return float(np.sqrt(v / 2.0))

    tau_e_p = _tau_event(stdP_w, sigma_p)
    tau_e_s = _tau_event(stdS_w, sigma_s)
    tau_c_p = _tau_cluster(stdP_c, sigma_p, tau_e_p)
    tau_c_s = _tau_cluster(stdS_c, sigma_s, tau_e_s)
    # Estimate sigma from a low-quantile of |residuals| (robust, avoids circular dependence on tau).
    def _sigma_from_quantile(r: torch.Tensor, q: float) -> float:
        try:
            if not isinstance(r, torch.Tensor) or r.numel() == 0:
                return float("nan")
            q = float(q)
            if not (0.01 <= q <= 0.99):
                q = 0.5
            a = r.detach().float()
            a = a[torch.isfinite(a)]
            if a.numel() == 0:
                return float("nan")
            a_np = a.detach().cpu().numpy()
            aq = float(np.quantile(np.abs(a_np), q))
            if not (np.isfinite(aq) and aq >= 0.0):
                return float("nan")
            zf = float("nan")
            try:
                from scipy.stats import norm  # type: ignore

                zf = float(norm.ppf((q + 1.0) / 2.0))
            except Exception:
                try:
                    zt = torch.distributions.Normal(0.0, 1.0).icdf(
                        torch.tensor((q + 1.0) / 2.0, dtype=torch.float32)
                    )
                    zf = float(zt.detach().cpu().item())
                except Exception:
                    zf = float("nan")
            if not (np.isfinite(zf) and zf > 1e-6):
                return float("nan")
            return float(aq / zf)
        except Exception:
            return float("nan")
    # Use within-cluster residuals by default (less contaminated by cluster shifts).
    sigma_q = float(sigma_quantile)
    sigma_p_est = _sigma_from_quantile(rP_w, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _sigma_from_quantile(rP_c, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _sigma_from_quantile(rP_a, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _robust_std_from_residuals(rP_a if rP_a.numel() > 0 else rP_w)
    sigma_s_est = _sigma_from_quantile(rS_w, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _sigma_from_quantile(rS_c, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _sigma_from_quantile(rS_a, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _robust_std_from_residuals(rS_a if rS_a.numel() > 0 else rS_w)

    return {
        "tau_event_p": float(tau_e_p),
        "tau_event_s": float(tau_e_s),
        "tau_cluster_p": float(tau_c_p),
        "tau_cluster_s": float(tau_c_s),
        "sigma_p_est": float(sigma_p_est),
        "sigma_s_est": float(sigma_s_est),
        "resid_std_within_p": float(stdP_w),
        "resid_std_within_s": float(stdS_w),
        "resid_std_cross_p": float(stdP_c),
        "resid_std_cross_s": float(stdS_c),
        "sigma_p": float(sigma_p),
        "sigma_s": float(sigma_s),
        "n_rows_used_p_within": int(rP_w.numel()),
        "n_rows_used_s_within": int(rS_w.numel()),
        "n_rows_used_p_cross": int(rP_c.numel()),
        "n_rows_used_s_cross": int(rS_c.numel()),
    }


@torch.no_grad()
def estimate_shared_event_re_station_phase_tau_s(
    *,
    state,
    n_rows: int,
    seed: int,
    batch_size: int,
    residuals_at: str,
    sigma_quantile: float,
    sigma_source: str,
    min_rows_per_group: int,
) -> Optional[dict]:
    try:
        sta_idx_all = state.row_station_index
        if not isinstance(sta_idx_all, torch.Tensor):
            warn("shared_event_re station_phase tau estimate: missing station index", section="DIAG")
            return None
    except Exception:
        return None

    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None

    try:
        min_rows_per_group = int(min_rows_per_group)
        if min_rows_per_group < 1:
            min_rows_per_group = 1
    except Exception:
        min_rows_per_group = 1

    try:
        mode = str(residuals_at).strip().lower()
    except Exception:
        mode = "map"
    if mode not in {"map", "initial", "truth"}:
        mode = "map"
    dX_use = _resolve_dX_use(state, mode)

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)

    # Accumulate per station-phase stats using bincount.
    sta_cpu = sta_idx_all.detach().to("cpu").numpy().astype(np.int64, copy=False)
    n_sta = int(sta_cpu.max() + 1) if sta_cpu.size > 0 else 0
    n_keys = int(max(1, n_sta * 2))
    counts = np.zeros((n_keys,), dtype=np.int64)
    sum_r = np.zeros((n_keys,), dtype=np.float64)
    sum_r2 = np.zeros((n_keys,), dtype=np.float64)
    rP_all, rS_all = [], []

    for i0 in range(0, int(rows_np.size), batch_size):
        ii = rows_np[i0 : i0 + batch_size]
        idx = state.II.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        y = state.YY.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        r = compute_residuals(idx, y, state.X_src, dX_use, state.model).detach()
        ph = y[:, 4].detach()
        is_s = (ph > 0.5)
        sta = sta_idx_all.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        finite = torch.isfinite(r)
        if not bool(finite.any()):
            continue
        if not bool(finite.all()):
            r = r[finite]
            is_s = is_s[finite]
            sta = sta[finite]
        r_cpu = r.to("cpu").numpy().astype(np.float64, copy=False)
        sta_cpu_b = sta.to("cpu").numpy().astype(np.int64, copy=False)
        ph_cpu = is_s.to("cpu").numpy().astype(np.int64, copy=False)
        keys = sta_cpu_b * 2 + ph_cpu
        counts += np.bincount(keys, minlength=n_keys)
        sum_r += np.bincount(keys, weights=r_cpu, minlength=n_keys)
        sum_r2 += np.bincount(keys, weights=(r_cpu * r_cpu), minlength=n_keys)
        rP_all.append(r_cpu[ph_cpu == 0])
        rS_all.append(r_cpu[ph_cpu == 1])

    rP_all = np.concatenate(rP_all, axis=0) if len(rP_all) > 0 else np.zeros((0,), dtype=np.float64)
    rS_all = np.concatenate(rS_all, axis=0) if len(rS_all) > 0 else np.zeros((0,), dtype=np.float64)

    def _sigma_from_quantile(r: np.ndarray, q: float) -> float:
        if r.size == 0:
            return float("nan")
        rq = np.quantile(np.abs(r), q)
        return float(rq / 0.6744897501960817)

    sigma_q = float(sigma_quantile)
    sigma_p_est = _sigma_from_quantile(rP_all, sigma_q)
    sigma_s_est = _sigma_from_quantile(rS_all, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = float(_robust_std_from_residuals(torch.from_numpy(rP_all)) if rP_all.size > 0 else float("nan"))
    if not np.isfinite(sigma_s_est):
        sigma_s_est = float(_robust_std_from_residuals(torch.from_numpy(rS_all)) if rS_all.size > 0 else float("nan"))

    try:
        vv = state.params.get("phase_unc", [float("nan"), float("nan")])
        sigma_p_param = float(vv[0])
        sigma_s_param = float(vv[1])
    except Exception:
        sigma_p_param = float("nan")
        sigma_s_param = float("nan")
    if not (np.isfinite(sigma_p_param) and sigma_p_param > 0.0):
        sigma_p_param = float("nan")
    if not (np.isfinite(sigma_s_param) and sigma_s_param > 0.0):
        sigma_s_param = float("nan")

    sigma_source = str(sigma_source).strip().lower()
    if sigma_source not in {"params", "quantile"}:
        sigma_source = "quantile"
    sigma_p_use = sigma_p_est
    sigma_s_use = sigma_s_est
    sigma_source_used = "quantile"
    if sigma_source == "params" and np.isfinite(sigma_p_param) and np.isfinite(sigma_s_param):
        sigma_p_use = sigma_p_param
        sigma_s_use = sigma_s_param
        sigma_source_used = "params"

    valid = counts >= int(min_rows_per_group)
    var = np.zeros_like(sum_r2)
    mask = valid & (counts > 0)
    var[mask] = (sum_r2[mask] / counts[mask]) - (sum_r[mask] / counts[mask]) ** 2

    var_p = var[0::2]
    var_s = var[1::2]
    n_groups_p = int(np.sum(mask[0::2]))
    n_groups_s = int(np.sum(mask[1::2]))
    var_p_med = float(np.median(var_p[mask[0::2]])) if n_groups_p > 0 else float("nan")
    var_s_med = float(np.median(var_s[mask[1::2]])) if n_groups_s > 0 else float("nan")

    tau_p = float(np.sqrt(max(0.0, var_p_med - sigma_p_use * sigma_p_use))) if np.isfinite(var_p_med) else float("nan")
    tau_s = float(np.sqrt(max(0.0, var_s_med - sigma_s_use * sigma_s_use))) if np.isfinite(var_s_med) else float("nan")

    return {
        "tau_p": float(tau_p),
        "tau_s": float(tau_s),
        "sigma_p_used": float(sigma_p_use),
        "sigma_s_used": float(sigma_s_use),
        "sigma_source": str(sigma_source_used),
        "sigma_p_est": float(sigma_p_est),
        "sigma_s_est": float(sigma_s_est),
        "sigma_p_param": float(sigma_p_param),
        "sigma_s_param": float(sigma_s_param),
        "var_p_med": float(var_p_med),
        "var_s_med": float(var_s_med),
        "n_groups_p": int(n_groups_p),
        "n_groups_s": int(n_groups_s),
    }


@torch.no_grad()
def _shared_event_re_station_phase_joint_holdout(
    *,
    state,
    n_rows: int,
    seed: int,
    batch_size: int,
    residuals_at: str,
    train_frac: float,
    sigma_quantile: float,
    min_events_per_cluster: int,
    min_rows_per_group: int,
) -> Optional[dict]:
    try:
        if not bool(state.params.get("_shared_event_re_hierarchical", False)):
            return None
    except Exception:
        return None
    try:
        cm = str(state.params.get("_shared_event_re_cluster_mode", "none")).strip().lower()
        if cm not in {"dd_khop", "component"}:
            return None
    except Exception:
        return None
    cluster_ids = state.params.get("_shared_event_re_cluster_ids", None)
    if not isinstance(cluster_ids, torch.Tensor):
        return None
    try:
        sta_idx_all = state.row_station_index
        if not isinstance(sta_idx_all, torch.Tensor):
            warn("shared_event_re joint holdout: missing station index", section="DIAG")
            return None
    except Exception:
        return None

    try:
        min_events_per_cluster = int(min_events_per_cluster)
        if min_events_per_cluster < 1:
            min_events_per_cluster = 1
    except Exception:
        min_events_per_cluster = 1
    try:
        min_rows_per_group = int(min_rows_per_group)
        if min_rows_per_group < 1:
            min_rows_per_group = 1
    except Exception:
        min_rows_per_group = 1

    try:
        c_cpu = cluster_ids.detach().to("cpu")
        counts = torch.bincount(c_cpu, minlength=int(c_cpu.max().item() + 1) if c_cpu.numel() > 0 else 0)
        keep_c = counts >= int(min_events_per_cluster)
        keep_c = keep_c.to(device=state.device)
    except Exception:
        keep_c = None

    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None
    try:
        train_frac = float(train_frac)
    except Exception:
        train_frac = 0.8
    train_frac = min(max(train_frac, 0.1), 0.95)

    try:
        mode = str(residuals_at).strip().lower()
    except Exception:
        mode = "map"
    if mode not in {"map", "initial", "truth"}:
        mode = "map"
    dX_use = _resolve_dX_use(state, mode)

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rng.shuffle(rows_np)
    n_train = int(max(1, int(round(train_frac * rows_np.size))))
    rows_train = np.sort(rows_np[:n_train])
    rows_test = np.sort(rows_np[n_train:])

    # Pass 1: collect train residuals and station-phase stats.
    rP_w_tr, rP_c_tr, rS_w_tr, rS_c_tr, rP_a_tr, rS_a_tr = [], [], [], [], [], []
    sta_cpu = sta_idx_all.detach().to("cpu").numpy().astype(np.int64, copy=False)
    n_sta = int(sta_cpu.max() + 1) if sta_cpu.size > 0 else 0
    n_keys = int(max(1, n_sta * 2))
    counts_sp = np.zeros((n_keys,), dtype=np.int64)
    sum_r_sp = np.zeros((n_keys,), dtype=np.float64)
    sum_r2_sp = np.zeros((n_keys,), dtype=np.float64)

    for i0 in range(0, int(rows_train.size), batch_size):
        ii = rows_train[i0 : i0 + batch_size]
        idx = state.II.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        y = state.YY.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        r = compute_residuals(idx, y, state.X_src, dX_use, state.model).detach()
        ph = y[:, 4].detach()
        is_s = (ph > 0.5)
        e1 = idx[:, 0].to(torch.int64)
        e2 = idx[:, 1].to(torch.int64)
        c1 = cluster_ids.index_select(0, e1)
        c2 = cluster_ids.index_select(0, e2)
        sta = sta_idx_all.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
        finite = torch.isfinite(r)
        if not bool(finite.any()):
            continue
        if not bool(finite.all()):
            r = r[finite]
            ph = ph[finite]
            is_s = is_s[finite]
            c1 = c1[finite]
            c2 = c2[finite]
            sta = sta[finite]
        if keep_c is not None:
            ok = keep_c.index_select(0, c1) & keep_c.index_select(0, c2)
            if not bool(ok.any()):
                continue
            c1 = c1[ok]
            c2 = c2[ok]
            r = r[ok]
            ph = ph[ok]
            is_s = is_s[ok]
            sta = sta[ok]
        same = (c1 == c2)
        if bool((~is_s).any()):
            rP = r[~is_s]
            rP_a_tr.append(rP.detach().to("cpu", dtype=torch.float32))
            sameP = same[~is_s]
            if bool(sameP.any()):
                rP_w_tr.append(rP[sameP].detach().to("cpu", dtype=torch.float32))
            if bool((~sameP).any()):
                rP_c_tr.append(rP[~sameP].detach().to("cpu", dtype=torch.float32))
        if bool(is_s.any()):
            rS = r[is_s]
            rS_a_tr.append(rS.detach().to("cpu", dtype=torch.float32))
            sameS = same[is_s]
            if bool(sameS.any()):
                rS_w_tr.append(rS[sameS].detach().to("cpu", dtype=torch.float32))
            if bool((~sameS).any()):
                rS_c_tr.append(rS[~sameS].detach().to("cpu", dtype=torch.float32))

        r_cpu = r.detach().to("cpu").numpy().astype(np.float64, copy=False)
        sta_cpu_b = sta.detach().to("cpu").numpy().astype(np.int64, copy=False)
        ph_cpu = is_s.detach().to("cpu").numpy().astype(np.int64, copy=False)
        keys = sta_cpu_b * 2 + ph_cpu
        counts_sp += np.bincount(keys, minlength=n_keys)
        sum_r_sp += np.bincount(keys, weights=r_cpu, minlength=n_keys)
        sum_r2_sp += np.bincount(keys, weights=(r_cpu * r_cpu), minlength=n_keys)

    rP_w_tr = torch.cat(rP_w_tr, dim=0) if rP_w_tr else torch.empty((0,), dtype=torch.float32)
    rP_c_tr = torch.cat(rP_c_tr, dim=0) if rP_c_tr else torch.empty((0,), dtype=torch.float32)
    rS_w_tr = torch.cat(rS_w_tr, dim=0) if rS_w_tr else torch.empty((0,), dtype=torch.float32)
    rS_c_tr = torch.cat(rS_c_tr, dim=0) if rS_c_tr else torch.empty((0,), dtype=torch.float32)
    rP_a_tr = torch.cat(rP_a_tr, dim=0) if rP_a_tr else torch.empty((0,), dtype=torch.float32)
    rS_a_tr = torch.cat(rS_a_tr, dim=0) if rS_a_tr else torch.empty((0,), dtype=torch.float32)

    def _sigma_from_quantile(a: torch.Tensor, q: float) -> float:
        try:
            if not isinstance(a, torch.Tensor) or a.numel() == 0:
                return float("nan")
            a_np = a.detach().cpu().numpy().astype(np.float64, copy=False)
            aq = float(np.quantile(np.abs(a_np), q))
            if not (np.isfinite(aq) and aq >= 0.0):
                return float("nan")
            zf = float("nan")
            try:
                from scipy.stats import norm  # type: ignore

                zf = float(norm.ppf((q + 1.0) / 2.0))
            except Exception:
                try:
                    zt = torch.distributions.Normal(0.0, 1.0).icdf(
                        torch.tensor((q + 1.0) / 2.0, dtype=torch.float32)
                    )
                    zf = float(zt.detach().cpu().item())
                except Exception:
                    zf = float("nan")
            if not (np.isfinite(zf) and zf > 1e-6):
                return float("nan")
            return float(aq / zf)
        except Exception:
            return float("nan")

    sigma_q = float(sigma_quantile)
    sigma_p_est = _sigma_from_quantile(rP_w_tr, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _sigma_from_quantile(rP_c_tr, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _sigma_from_quantile(rP_a_tr, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _robust_std_from_residuals(rP_a_tr if rP_a_tr.numel() > 0 else rP_w_tr)
    sigma_s_est = _sigma_from_quantile(rS_w_tr, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _sigma_from_quantile(rS_c_tr, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _sigma_from_quantile(rS_a_tr, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _robust_std_from_residuals(rS_a_tr if rS_a_tr.numel() > 0 else rS_w_tr)

    valid = counts_sp >= int(min_rows_per_group)
    var_sp = np.zeros_like(sum_r2_sp)
    mask = valid & (counts_sp > 0)
    var_sp[mask] = (sum_r2_sp[mask] / counts_sp[mask]) - (sum_r_sp[mask] / counts_sp[mask]) ** 2
    var_p = var_sp[0::2]
    var_s = var_sp[1::2]
    var_p_med = float(np.median(var_p[mask[0::2]])) if int(np.sum(mask[0::2])) > 0 else float("nan")
    var_s_med = float(np.median(var_s[mask[1::2]])) if int(np.sum(mask[1::2])) > 0 else float("nan")
    tau_sp_p = float(np.sqrt(max(0.0, var_p_med - sigma_p_est * sigma_p_est))) if np.isfinite(var_p_med) else float("nan")
    tau_sp_s = float(np.sqrt(max(0.0, var_s_med - sigma_s_est * sigma_s_est))) if np.isfinite(var_s_med) else float("nan")

    # Compute station-phase posterior mean (train) for adjustment.
    sum_w = np.zeros_like(sum_r_sp)
    sum_wr = np.zeros_like(sum_r_sp)
    inv_sig2_p = 0.0 if not (np.isfinite(sigma_p_est) and sigma_p_est > 0.0) else 1.0 / (sigma_p_est * sigma_p_est)
    inv_sig2_s = 0.0 if not (np.isfinite(sigma_s_est) and sigma_s_est > 0.0) else 1.0 / (sigma_s_est * sigma_s_est)
    sum_w[0::2] = counts_sp[0::2] * inv_sig2_p
    sum_w[1::2] = counts_sp[1::2] * inv_sig2_s
    sum_wr[0::2] = sum_r_sp[0::2] * inv_sig2_p
    sum_wr[1::2] = sum_r_sp[1::2] * inv_sig2_s
    denom_p = 1.0 + (tau_sp_p * tau_sp_p) * sum_w[0::2]
    denom_s = 1.0 + (tau_sp_s * tau_sp_s) * sum_w[1::2]
    a_hat = np.zeros_like(sum_r_sp)
    if np.isfinite(tau_sp_p) and tau_sp_p > 0.0:
        a_hat[0::2] = (tau_sp_p * tau_sp_p) * sum_wr[0::2] / denom_p
    if np.isfinite(tau_sp_s) and tau_sp_s > 0.0:
        a_hat[1::2] = (tau_sp_s * tau_sp_s) * sum_wr[1::2] / denom_s

    def _collect_adjusted(rows: np.ndarray):
        rP_w, rP_c, rS_w, rS_c, rP_a, rS_a = [], [], [], [], [], []
        for i0 in range(0, int(rows.size), batch_size):
            ii = rows[i0 : i0 + batch_size]
            idx = state.II.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
            y = state.YY.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
            r = compute_residuals(idx, y, state.X_src, dX_use, state.model).detach()
            ph = y[:, 4].detach()
            is_s = (ph > 0.5)
            e1 = idx[:, 0].to(torch.int64)
            e2 = idx[:, 1].to(torch.int64)
            c1 = cluster_ids.index_select(0, e1)
            c2 = cluster_ids.index_select(0, e2)
            sta = sta_idx_all.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
            finite = torch.isfinite(r)
            if not bool(finite.any()):
                continue
            if not bool(finite.all()):
                r = r[finite]
                ph = ph[finite]
                is_s = is_s[finite]
                c1 = c1[finite]
                c2 = c2[finite]
                sta = sta[finite]
            if keep_c is not None:
                ok = keep_c.index_select(0, c1) & keep_c.index_select(0, c2)
                if not bool(ok.any()):
                    continue
                c1 = c1[ok]
                c2 = c2[ok]
                r = r[ok]
                ph = ph[ok]
                is_s = is_s[ok]
                sta = sta[ok]
            sta_cpu_b = sta.detach().to("cpu").numpy().astype(np.int64, copy=False)
            ph_cpu = is_s.detach().to("cpu").numpy().astype(np.int64, copy=False)
            keys = sta_cpu_b * 2 + ph_cpu
            a_adj = torch.from_numpy(a_hat[keys]).to(device=r.device, dtype=r.dtype)
            r = r - a_adj
            same = (c1 == c2)
            if bool((~is_s).any()):
                rP = r[~is_s]
                rP_a.append(rP.detach().to("cpu", dtype=torch.float32))
                sameP = same[~is_s]
                if bool(sameP.any()):
                    rP_w.append(rP[sameP].detach().to("cpu", dtype=torch.float32))
                if bool((~sameP).any()):
                    rP_c.append(rP[~sameP].detach().to("cpu", dtype=torch.float32))
            if bool(is_s.any()):
                rS = r[is_s]
                rS_a.append(rS.detach().to("cpu", dtype=torch.float32))
                sameS = same[is_s]
                if bool(sameS.any()):
                    rS_w.append(rS[sameS].detach().to("cpu", dtype=torch.float32))
                if bool((~sameS).any()):
                    rS_c.append(rS[~sameS].detach().to("cpu", dtype=torch.float32))
        return (
            torch.cat(rP_w, dim=0) if rP_w else torch.empty((0,), dtype=torch.float32),
            torch.cat(rP_c, dim=0) if rP_c else torch.empty((0,), dtype=torch.float32),
            torch.cat(rS_w, dim=0) if rS_w else torch.empty((0,), dtype=torch.float32),
            torch.cat(rS_c, dim=0) if rS_c else torch.empty((0,), dtype=torch.float32),
            torch.cat(rP_a, dim=0) if rP_a else torch.empty((0,), dtype=torch.float32),
            torch.cat(rS_a, dim=0) if rS_a else torch.empty((0,), dtype=torch.float32),
        )

    rP_w_tr_adj, rP_c_tr_adj, rS_w_tr_adj, rS_c_tr_adj, rP_a_tr_adj, rS_a_tr_adj = _collect_adjusted(rows_train)
    rP_w_te, rP_c_te, rS_w_te, rS_c_te, _, _ = _collect_adjusted(rows_test)

    def _tau_event(std_w: float, sigma: float) -> float:
        if not (np.isfinite(std_w) and std_w > 0.0):
            return float("nan")
        if not (np.isfinite(sigma) and sigma >= 0.0):
            sigma = 0.0
        v = max(0.0, std_w * std_w - sigma * sigma)
        return float(np.sqrt(v / 2.0))

    def _tau_cluster(std_c: float, sigma: float, tau_e: float) -> float:
        if not (np.isfinite(std_c) and std_c > 0.0):
            return float("nan")
        if not (np.isfinite(sigma) and sigma >= 0.0):
            sigma = 0.0
        if not (np.isfinite(tau_e) and tau_e >= 0.0):
            tau_e = 0.0
        v = max(0.0, std_c * std_c - sigma * sigma - 2.0 * tau_e * tau_e)
        return float(np.sqrt(v / 2.0))

    stdP_w = _robust_std_from_residuals(rP_w_tr_adj)
    stdP_c = _robust_std_from_residuals(rP_c_tr_adj) if rP_c_tr_adj.numel() > 0 else float("nan")
    stdS_w = _robust_std_from_residuals(rS_w_tr_adj)
    stdS_c = _robust_std_from_residuals(rS_c_tr_adj) if rS_c_tr_adj.numel() > 0 else float("nan")
    tau_e_p = _tau_event(stdP_w, sigma_p_est)
    tau_e_s = _tau_event(stdS_w, sigma_s_est)
    tau_c_p = _tau_cluster(stdP_c, sigma_p_est, tau_e_p)
    tau_c_s = _tau_cluster(stdS_c, sigma_s_est, tau_e_s)

    def _z_stats(r: torch.Tensor, var: float) -> dict:
        if r.numel() == 0:
            return {"n": 0, "std": 0.0, "mean_abs": 0.0}
        if not (np.isfinite(var) and var > 0.0):
            return {"n": int(r.numel()), "std": float("nan"), "mean_abs": float("nan")}
        z = r.detach().float().cpu().numpy() / float(np.sqrt(var))
        return {"n": int(z.size), "std": float(np.std(z)), "mean_abs": float(np.mean(np.abs(z)))}

    var_p_within = sigma_p_est * sigma_p_est + 2.0 * tau_e_p * tau_e_p
    var_s_within = sigma_s_est * sigma_s_est + 2.0 * tau_e_s * tau_e_s
    var_p_cross = var_p_within + 2.0 * tau_c_p * tau_c_p
    var_s_cross = var_s_within + 2.0 * tau_c_s * tau_c_s
    zP_w = _z_stats(rP_w_te, var_p_within)
    zP_c = _z_stats(rP_c_te, var_p_cross)
    zS_w = _z_stats(rS_w_te, var_s_within)
    zS_c = _z_stats(rS_c_te, var_s_cross)

    def _tau_event_from_z(sigma: float, z_std: float) -> float:
        if not (np.isfinite(sigma) and sigma > 0.0 and np.isfinite(z_std) and z_std > 1.0):
            return float("nan")
        return float(sigma * np.sqrt((z_std * z_std - 1.0) / 2.0))

    def _tau_cluster_from_z(sigma: float, z_cross: float, tau_e: float) -> float:
        if not (np.isfinite(sigma) and sigma > 0.0 and np.isfinite(z_cross) and z_cross > 1.0):
            return 0.0
        if not (np.isfinite(tau_e) and tau_e >= 0.0):
            tau_e = 0.0
        v = (z_cross * z_cross - 1.0) * sigma * sigma - 2.0 * tau_e * tau_e
        if not (np.isfinite(v) and v > 0.0):
            return 0.0
        return float(np.sqrt(v / 2.0))

    tau_e_p_hold = _tau_event_from_z(sigma_p_est, float(zP_w.get("std", float("nan"))))
    tau_e_s_hold = _tau_event_from_z(sigma_s_est, float(zS_w.get("std", float("nan"))))
    tau_c_p_hold = _tau_cluster_from_z(sigma_p_est, float(zP_c.get("std", float("nan"))), tau_e_p_hold)
    tau_c_s_hold = _tau_cluster_from_z(sigma_s_est, float(zS_c.get("std", float("nan"))), tau_e_s_hold)

    return {
        "tau_sp_p": float(tau_sp_p),
        "tau_sp_s": float(tau_sp_s),
        "tau_event_p": float(tau_e_p),
        "tau_event_s": float(tau_e_s),
        "tau_cluster_p": float(tau_c_p),
        "tau_cluster_s": float(tau_c_s),
        "sigma_p_est": float(sigma_p_est),
        "sigma_s_est": float(sigma_s_est),
        "train_rows": int(rows_train.size),
        "test_rows": int(rows_test.size),
        "z_within_p": zP_w,
        "z_cross_p": zP_c,
        "z_within_s": zS_w,
        "z_cross_s": zS_c,
        "tau_event_p_holdout": float(tau_e_p_hold),
        "tau_event_s_holdout": float(tau_e_s_hold),
        "tau_cluster_p_holdout": float(tau_c_p_hold),
        "tau_cluster_s_holdout": float(tau_c_s_hold),
    }


@torch.no_grad()
def _shared_event_re_holdout_calibration(
    *,
    state,
    n_rows: int,
    seed: int,
    batch_size: int,
    residuals_at: str,
    train_frac: float,
    sigma_quantile: float,
    min_events_per_cluster: int,
) -> Optional[dict]:
    try:
        if not bool(state.params.get("_shared_event_re_hierarchical", False)):
            return None
    except Exception:
        return None
    try:
        cm = str(state.params.get("_shared_event_re_cluster_mode", "none")).strip().lower()
        if cm not in {"dd_khop", "component"}:
            return None
    except Exception:
        return None
    cluster_ids = state.params.get("_shared_event_re_cluster_ids", None)
    if not isinstance(cluster_ids, torch.Tensor):
        return None
    try:
        min_events_per_cluster = int(min_events_per_cluster)
        if min_events_per_cluster < 1:
            min_events_per_cluster = 1
    except Exception:
        min_events_per_cluster = 1
    try:
        c_cpu = cluster_ids.detach().to("cpu")
        counts = torch.bincount(c_cpu, minlength=int(c_cpu.max().item() + 1) if c_cpu.numel() > 0 else 0)
        keep_c = counts >= int(min_events_per_cluster)
        keep_c = keep_c.to(device=state.device)
    except Exception:
        keep_c = None

    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
    except Exception:
        return None
    try:
        train_frac = float(train_frac)
    except Exception:
        train_frac = 0.8
    train_frac = min(max(train_frac, 0.1), 0.95)

    try:
        mode = str(residuals_at).strip().lower()
    except Exception:
        mode = "map"
    if mode not in {"map", "initial", "truth"}:
        mode = "map"
    dX_use = _resolve_dX_use(state, mode)

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64)
    rng.shuffle(rows_np)
    n_train = int(max(1, int(round(train_frac * rows_np.size))))
    rows_train = np.sort(rows_np[:n_train])
    rows_test = np.sort(rows_np[n_train:])

    def _collect(rows: np.ndarray):
        rP_w, rP_c, rS_w, rS_c, rP_a, rS_a = [], [], [], [], [], []
        for i0 in range(0, int(rows.size), batch_size):
            ii = rows[i0 : i0 + batch_size]
            idx = state.II.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
            y = state.YY.index_select(0, torch.from_numpy(ii).to(device=state.device, dtype=torch.int64))
            r = compute_residuals(idx, y, state.X_src, dX_use, state.model).detach()
            ph = y[:, 4].detach()
            is_s = (ph > 0.5)
            e1 = idx[:, 0].to(torch.int64)
            e2 = idx[:, 1].to(torch.int64)
            c1 = cluster_ids.index_select(0, e1)
            c2 = cluster_ids.index_select(0, e2)
            finite = torch.isfinite(r)
            if not bool(finite.any()):
                continue
            if not bool(finite.all()):
                r = r[finite]
                ph = ph[finite]
                is_s = is_s[finite]
                c1 = c1[finite]
                c2 = c2[finite]
            if keep_c is not None:
                ok = keep_c.index_select(0, c1) & keep_c.index_select(0, c2)
                if not bool(ok.any()):
                    continue
                c1 = c1[ok]
                c2 = c2[ok]
                r = r[ok]
                ph = ph[ok]
                is_s = is_s[ok]
            same = (c1 == c2)
            if bool((~is_s).any()):
                rP = r[~is_s]
                rP_a.append(rP.detach().to("cpu", dtype=torch.float32))
                sameP = same[~is_s]
                if bool(sameP.any()):
                    rP_w.append(rP[sameP].detach().to("cpu", dtype=torch.float32))
                if bool((~sameP).any()):
                    rP_c.append(rP[~sameP].detach().to("cpu", dtype=torch.float32))
            if bool(is_s.any()):
                rS = r[is_s]
                rS_a.append(rS.detach().to("cpu", dtype=torch.float32))
                sameS = same[is_s]
                if bool(sameS.any()):
                    rS_w.append(rS[sameS].detach().to("cpu", dtype=torch.float32))
                if bool((~sameS).any()):
                    rS_c.append(rS[~sameS].detach().to("cpu", dtype=torch.float32))
        return (
            torch.cat(rP_w, dim=0) if rP_w else torch.empty((0,), dtype=torch.float32),
            torch.cat(rP_c, dim=0) if rP_c else torch.empty((0,), dtype=torch.float32),
            torch.cat(rS_w, dim=0) if rS_w else torch.empty((0,), dtype=torch.float32),
            torch.cat(rS_c, dim=0) if rS_c else torch.empty((0,), dtype=torch.float32),
            torch.cat(rP_a, dim=0) if rP_a else torch.empty((0,), dtype=torch.float32),
            torch.cat(rS_a, dim=0) if rS_a else torch.empty((0,), dtype=torch.float32),
        )

    def _finite_only(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            return x
        m = torch.isfinite(x)
        if not bool(m.any()):
            return x[:0]
        if not bool(m.all()):
            return x[m]
        return x

    rP_w_tr, rP_c_tr, rS_w_tr, rS_c_tr, rP_a_tr, rS_a_tr = _collect(rows_train)
    rP_w_te, rP_c_te, rS_w_te, rS_c_te, rP_a_te, rS_a_te = _collect(rows_test)

    rP_w_tr = _finite_only(rP_w_tr)
    rP_c_tr = _finite_only(rP_c_tr)
    rS_w_tr = _finite_only(rS_w_tr)
    rS_c_tr = _finite_only(rS_c_tr)
    rP_a_tr = _finite_only(rP_a_tr)
    rS_a_tr = _finite_only(rS_a_tr)
    rP_w_te = _finite_only(rP_w_te)
    rP_c_te = _finite_only(rP_c_te)
    rS_w_te = _finite_only(rS_w_te)
    rS_c_te = _finite_only(rS_c_te)
    rP_a_te = _finite_only(rP_a_te)
    rS_a_te = _finite_only(rS_a_te)

    def _sigma_from_quantile(r: torch.Tensor, q: float) -> float:
        try:
            if not isinstance(r, torch.Tensor) or r.numel() == 0:
                return float("nan")
            q = float(q)
            if not (0.01 <= q <= 0.99):
                q = 0.5
            a = r.detach().float()
            a = a[torch.isfinite(a)]
            if a.numel() == 0:
                return float("nan")
            a_np = a.detach().cpu().numpy()
            aq = float(np.quantile(np.abs(a_np), q))
            if not (np.isfinite(aq) and aq >= 0.0):
                return float("nan")
            zf = float("nan")
            try:
                from scipy.stats import norm  # type: ignore

                zf = float(norm.ppf((q + 1.0) / 2.0))
            except Exception:
                try:
                    zt = torch.distributions.Normal(0.0, 1.0).icdf(
                        torch.tensor((q + 1.0) / 2.0, dtype=torch.float32)
                    )
                    zf = float(zt.detach().cpu().item())
                except Exception:
                    zf = float("nan")
            if not (np.isfinite(zf) and zf > 1e-6):
                return float("nan")
            return float(aq / zf)
        except Exception:
            return float("nan")

    stdP_w = _robust_std_from_residuals(rP_w_tr)
    stdP_c = _robust_std_from_residuals(rP_c_tr) if rP_c_tr.numel() > 0 else float("nan")
    stdS_w = _robust_std_from_residuals(rS_w_tr)
    stdS_c = _robust_std_from_residuals(rS_c_tr) if rS_c_tr.numel() > 0 else float("nan")
    sigma_q = float(sigma_quantile)
    sigma_p_est = _sigma_from_quantile(rP_w_tr, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _sigma_from_quantile(rP_c_tr, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _sigma_from_quantile(rP_a_tr, sigma_q)
    if not np.isfinite(sigma_p_est):
        sigma_p_est = _robust_std_from_residuals(rP_a_tr if rP_a_tr.numel() > 0 else rP_w_tr)
    sigma_s_est = _sigma_from_quantile(rS_w_tr, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _sigma_from_quantile(rS_c_tr, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _sigma_from_quantile(rS_a_tr, sigma_q)
    if not np.isfinite(sigma_s_est):
        sigma_s_est = _robust_std_from_residuals(rS_a_tr if rS_a_tr.numel() > 0 else rS_w_tr)

    def _tau_event(std_w: float, sigma: float) -> float:
        if not (np.isfinite(std_w) and std_w > 0.0):
            return float("nan")
        if not (np.isfinite(sigma) and sigma >= 0.0):
            sigma = 0.0
        v = max(0.0, std_w * std_w - sigma * sigma)
        return float(np.sqrt(v / 2.0))

    def _tau_cluster(std_c: float, sigma: float, tau_e: float) -> float:
        if not (np.isfinite(std_c) and std_c > 0.0):
            return float("nan")
        if not (np.isfinite(sigma) and sigma >= 0.0):
            sigma = 0.0
        if not (np.isfinite(tau_e) and tau_e >= 0.0):
            tau_e = 0.0
        v = max(0.0, std_c * std_c - sigma * sigma - 2.0 * tau_e * tau_e)
        return float(np.sqrt(v / 2.0))

    tau_e_p = _tau_event(stdP_w, sigma_p_est)
    tau_e_s = _tau_event(stdS_w, sigma_s_est)
    tau_c_p = _tau_cluster(stdP_c, sigma_p_est, tau_e_p)
    tau_c_s = _tau_cluster(stdS_c, sigma_s_est, tau_e_s)

    def _z_stats(r: torch.Tensor, var: float) -> dict:
        if r.numel() == 0:
            return {"n": 0, "std": 0.0, "mean_abs": 0.0}
        if not (np.isfinite(var) and var > 0.0):
            return {"n": int(r.numel()), "std": float("nan"), "mean_abs": float("nan")}
        z = r.detach().float().cpu().numpy() / float(np.sqrt(var))
        return {
            "n": int(z.size),
            "std": float(np.std(z)),
            "mean_abs": float(np.mean(np.abs(z))),
        }

    var_p_within = sigma_p_est * sigma_p_est + 2.0 * tau_e_p * tau_e_p
    var_s_within = sigma_s_est * sigma_s_est + 2.0 * tau_e_s * tau_e_s
    var_p_cross = var_p_within + 2.0 * tau_c_p * tau_c_p
    var_s_cross = var_s_within + 2.0 * tau_c_s * tau_c_s

    def _tau_event_from_z(sigma: float, z_std: float) -> float:
        if not (np.isfinite(sigma) and sigma > 0.0 and np.isfinite(z_std) and z_std > 1.0):
            return float("nan")
        return float(sigma * np.sqrt((z_std * z_std - 1.0) / 2.0))

    def _tau_cluster_from_z(sigma: float, z_cross: float, tau_e: float) -> float:
        if not (np.isfinite(sigma) and sigma > 0.0 and np.isfinite(z_cross) and z_cross > 1.0):
            return 0.0
        if not (np.isfinite(tau_e) and tau_e >= 0.0):
            tau_e = 0.0
        v = (z_cross * z_cross - 1.0) * sigma * sigma - 2.0 * tau_e * tau_e
        if not (np.isfinite(v) and v > 0.0):
            return 0.0
        return float(np.sqrt(v / 2.0))

    zP_w = _z_stats(rP_w_te, var_p_within)
    zP_c = _z_stats(rP_c_te, var_p_cross)
    zS_w = _z_stats(rS_w_te, var_s_within)
    zS_c = _z_stats(rS_c_te, var_s_cross)

    tau_e_p_hold = _tau_event_from_z(sigma_p_est, float(zP_w.get("std", float("nan"))))
    tau_e_s_hold = _tau_event_from_z(sigma_s_est, float(zS_w.get("std", float("nan"))))
    tau_c_p_hold = _tau_cluster_from_z(sigma_p_est, float(zP_c.get("std", float("nan"))), tau_e_p_hold)
    tau_c_s_hold = _tau_cluster_from_z(sigma_s_est, float(zS_c.get("std", float("nan"))), tau_e_s_hold)

    return {
        "tau_event_p": float(tau_e_p),
        "tau_event_s": float(tau_e_s),
        "tau_cluster_p": float(tau_c_p),
        "tau_cluster_s": float(tau_c_s),
        "sigma_p_est": float(sigma_p_est),
        "sigma_s_est": float(sigma_s_est),
        "train_rows": int(rows_train.size),
        "test_rows": int(rows_test.size),
        "z_within_p": zP_w,
        "z_cross_p": zP_c,
        "z_within_s": zS_w,
        "z_cross_s": zS_c,
        "tau_event_p_holdout": float(tau_e_p_hold),
        "tau_event_s_holdout": float(tau_e_s_hold),
        "tau_cluster_p_holdout": float(tau_c_p_hold),
        "tau_cluster_s_holdout": float(tau_c_s_hold),
    }


@torch.no_grad()
def _exceedance_rates(abs_z: np.ndarray, thresholds: list[float]) -> dict[float, float]:
    """Return exceedance rates: P(|z| > t) for each threshold t."""
    out: dict[float, float] = {}
    try:
        a = np.asarray(abs_z, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            for t in thresholds:
                out[float(t)] = float("nan")
            return out
        for t in thresholds:
            tt = float(t)
            out[tt] = float(np.mean(a > tt))
        return out
    except Exception:
        for t in thresholds:
            out[float(t)] = float("nan")
        return out


def _fit_aic_normal_vs_laplace(z: np.ndarray) -> dict:
    """
    Fit Normal and Laplace by MLE and report log-likelihood and AIC for each.

    Normal MLE:  mu = mean, sigma = std
    Laplace MLE: loc = median, b = mean(|x-loc|)
    """
    out = {
        "n": 0,
        "ll_norm": float("nan"),
        "aic_norm": float("nan"),
        "ll_laplace": float("nan"),
        "aic_laplace": float("nan"),
        "mu_norm": float("nan"),
        "sigma_norm": float("nan"),
        "loc_laplace": float("nan"),
        "b_laplace": float("nan"),
        "aic_laplace_minus_norm": float("nan"),
    }
    try:
        x = np.asarray(z, dtype=np.float64)
        x = x[np.isfinite(x)]
        n = int(x.size)
        out["n"] = n
        if n < 8:
            return out

        mu = float(np.mean(x))
        sig = float(np.std(x, ddof=0))
        if (not np.isfinite(sig)) or sig <= 0.0:
            sig = float("nan")
        out["mu_norm"] = mu
        out["sigma_norm"] = sig

        loc = float(np.median(x))
        b = float(np.mean(np.abs(x - loc)))
        if (not np.isfinite(b)) or b <= 0.0:
            b = float("nan")
        out["loc_laplace"] = loc
        out["b_laplace"] = b

        if np.isfinite(sig) and sig > 0:
            s2 = float(sig * sig)
            ss = float(np.sum((x - mu) * (x - mu)))
            ll_norm = -0.5 * n * np.log(2.0 * np.pi) - n * np.log(sig) - 0.5 * (ss / s2)
            out["ll_norm"] = float(ll_norm)
            out["aic_norm"] = float(2 * 2 - 2 * ll_norm)

        if np.isfinite(b) and b > 0:
            sad = float(np.sum(np.abs(x - loc)))
            ll_lap = -n * np.log(2.0 * b) - (sad / b)
            out["ll_laplace"] = float(ll_lap)
            out["aic_laplace"] = float(2 * 2 - 2 * ll_lap)

        if np.isfinite(out["aic_laplace"]) and np.isfinite(out["aic_norm"]):
            out["aic_laplace_minus_norm"] = float(out["aic_laplace"] - out["aic_norm"])
        return out
    except Exception:
        return out


def _huber_logZ(k: float) -> float:
    """
    Normalizing constant for the Huber pseudo-density:
      p(z|k) ∝ exp(-rho_k(z))
      rho_k(z) = 0.5 z^2                 if |z|<=k
               = k(|z| - 0.5 k)          if |z|>k

    Z(k) = ∫ exp(-rho_k(z)) dz
         = sqrt(2π)*erf(k/sqrt(2)) + 2*exp(-k^2/2)/k
    """
    try:
        kk = float(k)
        if (not np.isfinite(kk)) or kk <= 0.0:
            return float("nan")
        import math

        A = math.sqrt(2.0 * math.pi) * math.erf(kk / math.sqrt(2.0))
        B = 2.0 * math.exp(-0.5 * kk * kk) / kk
        Z = A + B
        return float(math.log(Z))
    except Exception:
        return float("nan")


def _huber_rho(z: np.ndarray, k: float) -> np.ndarray:
    kk = float(k)
    a = np.abs(np.asarray(z, dtype=np.float64))
    core = a <= kk
    out = np.empty_like(a)
    out[core] = 0.5 * (a[core] ** 2)
    out[~core] = kk * (a[~core] - 0.5 * kk)
    return out


def _fit_huber_k_mle(z: np.ndarray, *, k_min: float = 0.2, k_max: float = 8.0, n_grid: int = 240) -> dict:
    """
    Fit k by maximizing log pseudo-likelihood for p(z|k) ∝ exp(-rho_k(z)).
    Returns best k and the grid score curve summary.
    """
    out = {"n": 0, "k_hat": float("nan"), "ll_hat": float("nan")}
    x = np.asarray(z, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out["n"] = n
    if n < 64:
        return out
    k_min = float(k_min); k_max = float(k_max)
    if (not np.isfinite(k_min)) or (not np.isfinite(k_max)) or k_min <= 0.0 or k_max <= k_min:
        k_min, k_max = 0.2, 8.0
    n_grid = int(max(32, n_grid))

    # Use a mildly log-spaced grid to cover small k well.
    ks = np.exp(np.linspace(np.log(k_min), np.log(k_max), n_grid)).astype(np.float64, copy=False)
    best_ll = -float("inf")
    best_k = float("nan")
    for kk in ks.tolist():
        logZ = _huber_logZ(float(kk))
        if not np.isfinite(logZ):
            continue
        ll = -float(np.sum(_huber_rho(x, float(kk)))) - float(n) * float(logZ)
        if ll > best_ll:
            best_ll = ll
            best_k = float(kk)
    out["k_hat"] = best_k
    out["ll_hat"] = float(best_ll) if np.isfinite(best_ll) else float("nan")
    return out


def _huber_ppf(p: np.ndarray, k: float) -> np.ndarray:
    """
    Quantile function for the symmetric Huber pseudo-density.
    Uses analytic inversion with erfinv for the core and log for the tails.
    """
    import math

    kk = float(k)
    if (not np.isfinite(kk)) or kk <= 0.0:
        return np.asarray(p, dtype=np.float64)

    # A = ∫_0^k exp(-t^2/2) dt = sqrt(pi/2) * erf(k/sqrt(2))
    A = math.sqrt(math.pi / 2.0) * math.erf(kk / math.sqrt(2.0))
    B = math.exp(-0.5 * kk * kk) / kk
    Z = 2.0 * (A + B)

    pp = np.asarray(p, dtype=np.float64)
    q = np.empty_like(pp)

    # Use torch.special.erfinv if available, else fall back to a numeric bisection in core.
    def _erfinv_np(xv: np.ndarray) -> np.ndarray:
        try:
            xt = torch.as_tensor(xv, dtype=torch.float64)
            yt = torch.special.erfinv(xt)
            return yt.detach().cpu().numpy().astype(np.float64, copy=False)
        except Exception:
            # crude fallback: bisection on erf for each element
            outv = np.empty_like(xv)
            for i, y in enumerate(xv.tolist()):
                lo, hi = -5.0, 5.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if math.erf(mid) < y:
                        lo = mid
                    else:
                        hi = mid
                outv[i] = 0.5 * (lo + hi)
            return outv

    # symmetry: q(p) = -q(1-p) for p<0.5
    m_lo = pp < 0.5
    pp_hi = np.where(m_lo, 1.0 - pp, pp)

    # Map to positive side mass: u = (p-0.5)*Z in [0, A+B]
    u = (pp_hi - 0.5) * Z
    core = u <= A
    # core: u = sqrt(pi/2)*erf(z/sqrt(2)) -> z = sqrt(2)*erfinv(u/sqrt(pi/2))
    if np.any(core):
        y = u[core] / math.sqrt(math.pi / 2.0)
        y = np.clip(y, -0.999999999999, 0.999999999999)
        qpos = math.sqrt(2.0) * _erfinv_np(y)
        # cap to k (numerical)
        qpos = np.clip(qpos, 0.0, kk)
        q[core] = qpos
    # tail: u = A + B*(1-exp(-k(z-k))) -> z = k - (1/k)*log(1-(u-A)/B)
    if np.any(~core):
        u2 = u[~core] - A
        frac = np.clip(u2 / B, 0.0, 1.0 - 1e-15)
        qpos = kk - (1.0 / kk) * np.log(1.0 - frac)
        q[~core] = qpos

    # apply symmetry
    q = np.where(m_lo, -q, q)
    return q.astype(np.float64, copy=False)


def _qqplot_huber(*, z: np.ndarray, k: float, title: str, out_png: str, n_points: int = 5000) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")
    x = np.asarray(z, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 32:
        raise RuntimeError("Not enough residuals for QQ plot")
    if int(x.size) > int(n_points):
        stride = max(1, int(x.size // int(n_points)))
        x = x[::stride]
    x = np.sort(x)
    n = int(x.size)
    p = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    q = _huber_ppf(p, float(k))

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    ax.scatter(q, x, s=6, alpha=0.35)
    lo = float(np.nanmin([np.nanmin(q), np.nanmin(x)]))
    hi = float(np.nanmax([np.nanmax(q), np.nanmax(x)]))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.plot([lo, hi], [lo, hi], color="k", linewidth=1.0, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_title(title + f" (QQ vs Huber k={float(k):.3g})")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.grid(True, alpha=0.25)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _qqplot(
    *,
    z: np.ndarray,
    dist: str,
    title: str,
    out_png: str,
    n_points: int = 5000,
) -> None:
    """
    QQ plot against a fitted Normal or Laplace distribution.
    """
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")
    x = np.asarray(z, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 32:
        raise RuntimeError("Not enough residuals for QQ plot")

    if int(x.size) > int(n_points):
        stride = max(1, int(x.size // int(n_points)))
        x = x[::stride]
    x = np.sort(x)
    n = int(x.size)
    p = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    dist = str(dist).strip().lower()

    if dist in {"normal", "norm", "gaussian"}:
        mu = float(np.mean(x))
        sig = float(np.std(x, ddof=0))
        sig = sig if np.isfinite(sig) and sig > 0 else 1.0
        try:
            from scipy.stats import norm  # type: ignore
            q = norm.ppf(p, loc=mu, scale=sig)
        except Exception:
            q = x.copy()
        ref = "Normal"
    elif dist in {"laplace", "double_exponential"}:
        loc = float(np.median(x))
        b = float(np.mean(np.abs(x - loc)))
        b = b if np.isfinite(b) and b > 0 else 1.0
        try:
            from scipy.stats import laplace  # type: ignore
            q = laplace.ppf(p, loc=loc, scale=b)
        except Exception:
            q = np.empty_like(p)
            m = p < 0.5
            q[m] = loc + b * np.log(2.0 * p[m])
            q[~m] = loc - b * np.log(2.0 * (1.0 - p[~m]))
        ref = "Laplace"
    else:
        raise ValueError(f"Unsupported dist for QQ: {dist}")

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    ax.scatter(q, x, s=6, alpha=0.35)
    lo = float(np.nanmin([np.nanmin(q), np.nanmin(x)]))
    hi = float(np.nanmax([np.nanmax(q), np.nanmax(x)]))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.plot([lo, hi], [lo, hi], color="k", linewidth=1.0, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_title(title + f" (QQ vs {ref})")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.grid(True, alpha=0.25)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _resid_distribution_summary(*, r: np.ndarray, sigma: Optional[float], label: str) -> dict:
    """
    Summarize residual distribution for a 1D array.
    Computes raw-residual stats and standardized stats:
      - z_sigma: r / sigma (if sigma provided)
      - z_mad:   (r - median) / (1.4826*MAD)
    """
    out: dict[str, Any] = {"label": str(label)}
    x = np.asarray(r, dtype=np.float64)
    x = x[np.isfinite(x)]
    out["n"] = int(x.size)
    if x.size < 8:
        return out

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    out["mean"] = float(np.mean(x))
    out["std"] = float(np.std(x, ddof=0))
    out["med"] = med
    out["mad"] = mad
    out["q75_abs"] = float(np.quantile(np.abs(x), 0.75))
    out["q95_abs"] = float(np.quantile(np.abs(x), 0.95))
    out["q99_abs"] = float(np.quantile(np.abs(x), 0.99))
    out["skew"] = _safe_skew(x)
    out["excess_kurtosis"] = _safe_excess_kurtosis(x)

    # Sigma-standardized
    if sigma is not None and np.isfinite(float(sigma)) and float(sigma) > 0:
        z = x / float(sigma)
        out["z_sigma_tail_ratio_95_75"] = _tail_ratio(np.abs(z), q_hi=0.95, q_lo=0.75)
        out["z_sigma_fit"] = _fit_aic_normal_vs_laplace(z)

    # Robust MAD-standardized
    s = 1.4826 * mad
    if np.isfinite(s) and s > 0:
        z0 = (x - med) / float(s)
        out["z_mad_tail_ratio_95_75"] = _tail_ratio(np.abs(z0), q_hi=0.95, q_lo=0.75)
        out["z_mad_fit"] = _fit_aic_normal_vs_laplace(z0)

    return out

def _leg_edges_km(*, max_leg_km: float, n_bins: int, log_bins: bool) -> np.ndarray:
    """
    Construct bin edges in km for leg-length heatmaps, always starting at 0.
    """
    max_leg = float(max_leg_km)
    n_bins = int(max(3, n_bins))
    if (not np.isfinite(max_leg)) or max_leg <= 0:
        max_leg = 4.0
    if bool(log_bins):
        eps = max_leg * 1e-3  # 0.1% of range (e.g., 4 km -> 4 m)
        eps = eps if np.isfinite(eps) and eps > 0 else 1e-3
        e1 = np.logspace(np.log10(eps), np.log10(max_leg), n_bins, dtype=np.float64)
        edges = np.concatenate([np.array([0.0], dtype=np.float64), e1])
    else:
        edges = np.linspace(0.0, max_leg, n_bins + 1, dtype=np.float64)
    edges = edges[np.isfinite(edges)]
    edges = np.unique(np.sort(edges))
    if edges.size < 2 or float(edges[0]) != 0.0:
        edges = np.unique(np.concatenate([np.array([0.0], dtype=np.float64), edges]))
    return edges.astype(np.float64, copy=False)


def _plot_legcorr2d(
    *,
    corr: np.ndarray,
    count: np.ndarray,
    edges_km: np.ndarray,
    title: str,
    out_png: str,
    corr_vmax: Optional[float] = None,
    count_vmax: Optional[float] = None,
) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")
    edges = np.asarray(edges_km, dtype=np.float64)
    C = np.asarray(corr, dtype=np.float64)
    N = np.asarray(count, dtype=np.float64)

    if corr_vmax is None:
        vals = C[np.isfinite(C)]
        if vals.size:
            v = float(np.nanquantile(np.abs(vals), 0.98))
            v = v if np.isfinite(v) and v > 0 else float(np.nanmax(np.abs(vals)))
        else:
            v = 1.0
        v = float(max(v, 0.05))
    else:
        v = float(corr_vmax)
        if (not np.isfinite(v)) or v <= 0.0:
            v = 1.0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 11), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    X, Y = np.meshgrid(edges, edges, indexing="xy")
    im = ax0.pcolormesh(X, Y, C, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    ax0.set_title(title)
    ax0.set_ylabel("leg2 length (km)")
    ax0.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax0, shrink=0.85, label="corr")

    logN = np.log1p(N)
    if count_vmax is None:
        im2 = ax1.pcolormesh(X, Y, logN, cmap="viridis", shading="auto")
    else:
        vmax2 = float(count_vmax)
        if (not np.isfinite(vmax2)) or vmax2 <= 0.0:
            vmax2 = float(np.nanmax(logN)) if np.isfinite(np.nanmax(logN)) else 1.0
        im2 = ax1.pcolormesh(X, Y, logN, cmap="viridis", vmin=0.0, vmax=vmax2, shading="auto")
    ax1.set_xlabel("leg1 length (km)")
    ax1.set_ylabel("leg2 length (km)")
    ax1.set_aspect("equal", adjustable="box")
    fig.colorbar(im2, ax=ax1, shrink=0.85, label="log(1+n_pairs)")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_legcorr2d_pair(
    *,
    corr_p: np.ndarray,
    corr_s: np.ndarray,
    count_p: np.ndarray,
    count_s: np.ndarray,
    edges_km: np.ndarray,
    title: str,
    out_png: str,
    corr_vmax: Optional[float] = None,
    count_vmax: Optional[float] = None,
) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")
    edges = np.asarray(edges_km, dtype=np.float64)
    C_p = np.asarray(corr_p, dtype=np.float64)
    C_s = np.asarray(corr_s, dtype=np.float64)
    N_p = np.asarray(count_p, dtype=np.float64)
    N_s = np.asarray(count_s, dtype=np.float64)

    if corr_vmax is None:
        vals = np.concatenate([C_p[np.isfinite(C_p)], C_s[np.isfinite(C_s)]])
        if vals.size:
            v = float(np.nanquantile(np.abs(vals), 0.98))
            v = v if np.isfinite(v) and v > 0 else float(np.nanmax(np.abs(vals)))
        else:
            v = 1.0
        v = float(max(v, 0.05))
    else:
        v = float(corr_vmax)
        if (not np.isfinite(v)) or v <= 0.0:
            v = 1.0

    logN_p = np.log1p(N_p)
    logN_s = np.log1p(N_s)
    if count_vmax is None:
        vmax2 = float(np.nanmax(np.concatenate([logN_p, logN_s])))
        if (not np.isfinite(vmax2)) or vmax2 <= 0.0:
            vmax2 = 1.0
    else:
        vmax2 = float(count_vmax)
        if (not np.isfinite(vmax2)) or vmax2 <= 0.0:
            vmax2 = float(np.nanmax(np.concatenate([logN_p, logN_s])))
            vmax2 = vmax2 if np.isfinite(vmax2) and vmax2 > 0 else 1.0

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.08, hspace=0.1)
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1], sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 0], sharey=axes[0, 0])
    cax_corr = fig.add_subplot(gs[0, 2])
    cax_cnt = fig.add_subplot(gs[1, 2])
    X, Y = np.meshgrid(edges, edges, indexing="xy")

    im0 = axes[0, 0].pcolormesh(X, Y, C_p, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    im1 = axes[0, 1].pcolormesh(X, Y, C_s, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    axes[0, 0].set_title("P")
    axes[0, 1].set_title("S")
    axes[0, 0].set_ylabel("leg2 length (km)")
    for ax in axes[0, :]:
        ax.set_aspect("equal", adjustable="box")

    im2 = axes[1, 0].pcolormesh(X, Y, logN_p, cmap="viridis", vmin=0.0, vmax=vmax2, shading="auto")
    im3 = axes[1, 1].pcolormesh(X, Y, logN_s, cmap="viridis", vmin=0.0, vmax=vmax2, shading="auto")
    axes[1, 0].set_xlabel("leg1 length (km)")
    axes[1, 1].set_xlabel("leg1 length (km)")
    axes[1, 0].set_ylabel("leg2 length (km)")
    for ax in axes[1, :]:
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title, y=0.97)
    fig.colorbar(im0, cax=cax_corr, label="corr")
    fig.colorbar(im2, cax=cax_cnt, label="log(1+n_pairs)")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.94, hspace=0.1, wspace=0.08)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def _plot_variogram_payload(
    *,
    curves: dict,
    keys: list[str],
    title: str,
    out_png: str,
    xmax_km: Optional[float] = None,
) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for key in keys:
        c = curves.get(key, None)
        if not isinstance(c, dict):
            continue
        x = c.get("centers_km", None)
        y = c.get("gamma", None)
        n = c.get("count", None)
        if x is None or y is None:
            continue
        # We plot bin centers, so the smallest x is typically > 0. Prepend an explicit
        # "distance=0" point for readability. For semivariograms, γ(0)=0 by definition,
        # but with nugget effects a more informative visible intercept is the estimated nugget.
        try:
            nug = c.get("nugget", None)
            nug0 = float(nug[0]) if nug is not None else 0.0
            if (x is not None) and (len(x) > 0) and float(x[0]) > 0.0:
                x = np.concatenate([np.array([0.0], dtype=x.dtype), x])
                y = np.concatenate([np.array([nug0], dtype=y.dtype), y])
                if n is not None:
                    n = np.concatenate([np.array([0.0], dtype=n.dtype), n])
        except Exception:
            pass
        try:
            if xmax_km is not None and float(xmax_km) > 0:
                m = x <= float(xmax_km)
                x = x[m]
                y = y[m]
                if n is not None:
                    n = n[m]
        except Exception:
            pass
        ax0.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=key)
        if n is not None:
            ax1.plot(x, n, marker="o", linewidth=1.0, markersize=2, label=key)
        try:
            tgt = float(c.get("target", [float("nan")])[0])
            if tgt == tgt:
                ax0.axhline(tgt, linestyle="--", linewidth=1.0, alpha=0.5)
        except Exception:
            pass

    ax0.set_title(title)
    ax0.set_ylabel("semivariance γ")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=9)

    ax1.set_xlabel("distance (km)")
    ax1.set_ylabel("pairs/bin")
    ax1.grid(True, alpha=0.3)
    try:
        if xmax_km is not None and float(xmax_km) > 0:
            ax0.set_xlim(0.0, float(xmax_km))
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _acf_from_variogram_curve(c: dict) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert a binned semivariogram curve to an approximate correlation/ACF curve:

        ρ(d) ≈ 1 - (γ(d) - nugget) / (plateau - nugget)

    where plateau is the semivariance plateau (≈ variance) and nugget is the short-range jump.
    Returns (x_km, rho, count).
    """
    try:
        x = c.get("centers_km", None)
        g = c.get("gamma", None)
        cnt = c.get("count", None)
        if x is None or g is None:
            return None
        x = np.asarray(x)
        g = np.asarray(g)
        cnt = np.asarray(cnt) if cnt is not None else np.zeros_like(x)
        nug = c.get("nugget", None)
        plat = c.get("plateau", None)
        nug0 = float(np.asarray(nug)[0]) if nug is not None else 0.0
        plat0 = float(np.asarray(plat)[0]) if plat is not None else float("nan")

        # Nugget-corrected semivariance for correlation conversion.
        g_adj = np.maximum(g - nug0, 0.0)

        # "Sill" (variance) estimation is noisy when the variogram hasn't clearly plateaued.
        # Use a robust maximum of a few candidates to avoid pathological tiny/negative sills that
        # can drive the derived ACF immediately negative.
        sill_cands = []
        try:
            if np.isfinite(plat0):
                sill_cands.append(float(plat0) - float(nug0))
                sill_cands.append(float(plat0))
        except Exception:
            pass
        try:
            if g_adj.size:
                sill_cands.append(float(np.nanquantile(g_adj, 0.9)))
                sill_cands.append(float(np.nanmax(g_adj)))
        except Exception:
            pass
        sill = float(np.nanmax(np.asarray(sill_cands, dtype=np.float64))) if sill_cands else float("nan")
        if (not np.isfinite(sill)) or sill <= 0.0:
            return None

        rho = 1.0 - (g_adj / sill)
        # For "length scale by eye" plots, negative lobes are usually just noise/nonstationarity.
        # Clip to [0,1] so the plot is readable as an ACF-like decay curve.
        rho = np.clip(rho, 0.0, 1.0)
        return x.astype(np.float32, copy=False), rho.astype(np.float32, copy=False), cnt.astype(np.float32, copy=False)
    except Exception:
        return None


def _empirical_corr_by_distance_sampled(
    *,
    coords_km: np.ndarray,   # (M,2)
    vals: np.ndarray,        # (M,)
    n_pairs: int,
    n_bins: int,
    xmax_km: float,
    seed: int = 0,
    binning: str = "linear",
    # Backward-compatibility alias (deprecated): when True, behaves like binning="log".
    log_bins: bool = False,
) -> Optional[dict]:
    """
    Direct empirical correlation-by-distance using random pair sampling (for large M).

      ρ_bin = E[(x_i-μ)(x_j-μ) | d_ij in bin] / Var(x)

    Binning modes:
      - linear: fixed-width bins on [0, xmax_km]
      - log: log-spaced bins on (0, xmax_km] with explicit 0 edge
      - equal_count: quantile bins so each bin has ~equal number of sampled pairs (stabilizes tails)

    For linear/log, bins are fixed over [0, xmax_km] so `n_bins` controls resolution in the plotted range.
    """
    try:
        xy = np.asarray(coords_km, dtype=np.float64)
        x = np.asarray(vals, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        if m.sum() < 10:
            return None
        xy = xy[m]
        x = x[m]
        M = int(x.shape[0])
        if M < 10:
            return None
        mu = float(np.nanmean(x))
        xc = x - mu
        var = float(np.nanmean(xc * xc))
        if (not np.isfinite(var)) or var <= 0.0:
            return None
        xmax = float(xmax_km)
        if (not np.isfinite(xmax)) or xmax <= 0.0:
            return None
        n_bins = int(max(5, int(n_bins)))
        n_pairs = int(max(1, int(n_pairs)))

        rng = np.random.default_rng(int(seed))
        i = rng.integers(0, M, size=n_pairs, dtype=np.int64)
        j_off = rng.integers(1, M, size=n_pairs, dtype=np.int64)
        j = (i + j_off) % M
        dx = xy[i] - xy[j]
        d = np.sqrt(np.maximum(0.0, np.sum(dx * dx, axis=1)))
        mm = np.isfinite(d) & (d <= xmax)
        if not np.any(mm):
            return None
        d = d[mm]
        prod = (xc[i[mm]] * xc[j[mm]])

        # Resolve binning mode (log_bins is legacy alias).
        try:
            if bool(log_bins):
                binning = "log"
        except Exception:
            pass
        bmode = str(binning).strip().lower()
        if bmode in {"log_bins", "logspace", "log-spaced"}:
            bmode = "log"
        if bmode in {"equal_count", "equal-count", "quantile", "quantiles"}:
            bmode = "equal_count"
        if bmode not in {"linear", "log", "equal_count"}:
            bmode = "linear"

        centers = None
        if bmode == "log":
            eps = float(xmax) * 1e-3
            eps = eps if np.isfinite(eps) and eps > 0 else 1e-3
            e1 = np.logspace(np.log10(eps), np.log10(xmax), n_bins, dtype=np.float64)
            edges = np.concatenate([np.array([0.0], dtype=np.float64), e1])
            centers = np.concatenate([np.array([0.5 * edges[1]], dtype=np.float64), np.sqrt(edges[1:-1] * edges[2:])])
        elif bmode == "equal_count":
            qs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
            edges = np.quantile(d.astype(np.float64, copy=False), qs)
            edges[0] = 0.0
            edges = np.unique(np.sort(edges))
            if edges.size < 2:
                edges = np.linspace(0.0, xmax, n_bins + 1, dtype=np.float64)
            # If edges collapsed, reduce bin count.
            n_bins_eff = int(max(1, int(edges.size) - 1))
            if n_bins_eff != int(n_bins):
                n_bins = int(n_bins_eff)
        else:
            edges = np.linspace(0.0, xmax, n_bins + 1, dtype=np.float64)
            centers = 0.5 * (edges[:-1] + edges[1:])

        bi = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)
        sum_prod = np.zeros((n_bins,), dtype=np.float64)
        cnt = np.zeros((n_bins,), dtype=np.float64)
        np.add.at(sum_prod, bi, prod)
        np.add.at(cnt, bi, 1.0)
        mean_prod = sum_prod / np.maximum(cnt, 1.0)
        rho = mean_prod / var
        rho = np.where(cnt > 0, rho, np.nan)
        if centers is None:
            # Mean distance per bin (stable for variable-width quantile bins).
            sum_d = np.zeros((n_bins,), dtype=np.float64)
            np.add.at(sum_d, bi, d.astype(np.float64, copy=False))
            centers = sum_d / np.maximum(cnt, 1.0)
        return {
            "centers_km": centers.astype(np.float32, copy=False),
            "rho": rho.astype(np.float32, copy=False),
            "count": cnt.astype(np.float32, copy=False),
            "mu": np.asarray([mu], dtype=np.float32),
            "var": np.asarray([var], dtype=np.float32),
            "binning": np.asarray([str(bmode)], dtype=object),
        }
    except Exception:
        return None


def _plot_empirical_corr_payload(
    *,
    corr: dict,
    title: str,
    out_png: str,
    xmax_km: Optional[float] = None,
) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")
    x = np.asarray(corr.get("centers_km", []), dtype=np.float32)
    rho = np.asarray(corr.get("rho", []), dtype=np.float32)
    cnt = np.asarray(corr.get("count", np.zeros_like(x)), dtype=np.float32)
    if x.size == 0 or rho.size == 0:
        raise RuntimeError("empty correlation curve")

    # Prepend rho(0)=1 for readability.
    if float(x[0]) > 0.0:
        x = np.concatenate([np.array([0.0], dtype=x.dtype), x])
        rho = np.concatenate([np.array([1.0], dtype=rho.dtype), rho])
        cnt = np.concatenate([np.array([0.0], dtype=cnt.dtype), cnt])
    if xmax_km is not None and float(xmax_km) > 0:
        m = x <= float(xmax_km)
        x = x[m]; rho = rho[m]; cnt = cnt[m]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax0.plot(x, rho, marker="o", linewidth=1.5, markersize=3)
    ax0.set_title(title)
    ax0.set_ylabel("empirical corr ρ")
    ax0.grid(True, alpha=0.3)
    ax0.axhline(0.0, linewidth=1.0, alpha=0.4)

    ax1.plot(x, cnt, marker="o", linewidth=1.0, markersize=2)
    ax1.set_xlabel("distance (km)")
    ax1.set_ylabel("pairs/bin")
    ax1.grid(True, alpha=0.3)
    if xmax_km is not None and float(xmax_km) > 0:
        ax0.set_xlim(0.0, float(xmax_km))

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_empirical_corr_payload_multi(
    *,
    curves: list[dict],
    title: str,
    out_png: str,
    xmax_km: Optional[float] = None,
    show_legend: bool = False,
) -> None:
    """
    Plot multiple correlation-by-distance curves on a single axis.
    Each curve dict should contain: centers_km, rho, and optional label.
    """
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for c in curves:
        try:
            x = np.asarray(c.get("centers_km", []), dtype=np.float32)
            rho = np.asarray(c.get("rho", []), dtype=np.float32)
            if x.size == 0 or rho.size == 0:
                continue
            if xmax_km is not None and float(xmax_km) > 0:
                m = x <= float(xmax_km)
                x = x[m]
                rho = rho[m]
            if x.size == 0 or rho.size == 0:
                continue
            lab = c.get("label", None)
            if isinstance(lab, str) and lab:
                ax.plot(x, rho, linewidth=1.2, alpha=0.55, label=lab)
            else:
                ax.plot(x, rho, linewidth=1.2, alpha=0.55)
        except Exception:
            continue

    ax.set_title(title)
    ax.set_xlabel("distance (km)")
    ax.set_ylabel("empirical corr ρ")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, linewidth=1.0, alpha=0.4)
    if xmax_km is not None and float(xmax_km) > 0:
        ax.set_xlim(0.0, float(xmax_km))
    if bool(show_legend):
        ax.legend(loc="best", fontsize=8, ncols=2)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _fit_rbf_ell_from_corr(
    *,
    x_km: np.ndarray,
    rho: np.ndarray,
    count: Optional[np.ndarray] = None,
    rho_min: float = 0.05,
    rho_max: float = 0.999,
) -> Optional[float]:
    """
    Fit an RBF length scale ell (km) from empirical correlation data assuming:
        rho(d) ≈ exp(-0.5 * (d/ell)^2)

    Use a weighted least squares fit of y = -2 log(rho) against d^2 through the origin:
        y ≈ (1/ell^2) * d^2
    """
    try:
        x = np.asarray(x_km, dtype=np.float64)
        r = np.asarray(rho, dtype=np.float64)
        w = np.ones_like(x, dtype=np.float64) if count is None else np.asarray(count, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(r) & np.isfinite(w) & (x > 0.0) & (w > 0.0) & (r > rho_min) & (r < rho_max)
        if m.sum() < 2:
            return None
        x = x[m]
        r = r[m]
        w = w[m]
        d2 = x * x
        y = -2.0 * np.log(np.clip(r, 1e-12, 1.0))
        # Weighted slope through origin
        num = np.sum(w * d2 * y)
        den = np.sum(w * d2 * d2)
        if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
            return None
        slope = num / den
        if not np.isfinite(slope) or slope <= 0.0:
            return None
        ell = 1.0 / np.sqrt(slope)
        return float(ell) if np.isfinite(ell) and ell > 0 else None
    except Exception:
        return None


def _fmt_km(v: Optional[float]) -> str:
    try:
        return f"{float(v):.3g}" if v is not None and np.isfinite(float(v)) else "nan"
    except Exception:
        return "nan"


def _plot_acf_payload(
    *,
    curves: dict,
    keys: list[str],
    title: str,
    out_png: str,
    xmax_km: Optional[float] = None,
) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for key in keys:
        c = curves.get(key, None)
        if not isinstance(c, dict):
            continue
        out = _acf_from_variogram_curve(c)
        if out is None:
            continue
        x, rho, cnt = out
        # Prepend explicit 0-distance point: rho(0)=1
        try:
            if x.size and float(x[0]) > 0.0:
                x = np.concatenate([np.array([0.0], dtype=x.dtype), x])
                rho = np.concatenate([np.array([1.0], dtype=rho.dtype), rho])
                cnt = np.concatenate([np.array([0.0], dtype=cnt.dtype), cnt])
        except Exception:
            pass
        if xmax_km is not None and float(xmax_km) > 0:
            m = x <= float(xmax_km)
            x = x[m]; rho = rho[m]; cnt = cnt[m]
        ax0.plot(x, rho, marker="o", linewidth=1.5, markersize=3, label=key)
        ax1.plot(x, cnt, marker="o", linewidth=1.0, markersize=2, label=key)

    ax0.set_title(title)
    ax0.set_ylabel("correlation ρ")
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=9)
    ax0.axhline(0.0, linewidth=1.0, alpha=0.4)

    ax1.set_xlabel("distance (km)")
    ax1.set_ylabel("pairs/bin")
    ax1.grid(True, alpha=0.3)
    try:
        if xmax_km is not None and float(xmax_km) > 0:
            ax0.set_xlim(0.0, float(xmax_km))
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _legcorr2d_mean_abs_corr(
    *,
    corr: np.ndarray,
    count: np.ndarray,
    min_pairs_cell: int,
) -> dict[str, float]:
    """
    Compute simple scalar scores for a legcorr2d heatmap.

    "Supported bins" means bins where corr is finite and count >= min_pairs_cell.
    Returns both unweighted and count-weighted mean(|corr|).
    """
    C = np.asarray(corr, dtype=np.float64)
    N = np.asarray(count, dtype=np.float64)
    mask = np.isfinite(C) & np.isfinite(N) & (N >= float(min_pairs_cell))
    n_bins = int(np.count_nonzero(mask))
    if n_bins <= 0:
        return {
            "n_supported_bins": 0.0,
            "mean_abs_corr": float("nan"),
            "mean_abs_corr_weighted": float("nan"),
            "sum_pairs_supported": float("nan"),
        }
    vals = np.abs(C[mask])
    w = N[mask]
    wsum = float(np.sum(w))
    mean_abs = float(np.mean(vals)) if vals.size else float("nan")
    mean_abs_w = float(np.sum(vals * w) / wsum) if (vals.size and wsum > 0.0) else float("nan")
    return {
        "n_supported_bins": float(n_bins),
        "mean_abs_corr": mean_abs,
        "mean_abs_corr_weighted": mean_abs_w,
        "sum_pairs_supported": float(wsum),
    }


def _legcorr2d_fit_distance(
    *,
    corr: np.ndarray,
    count: np.ndarray,
    edges_km: np.ndarray,
    min_pairs_cell: int,
) -> dict[str, float] | None:
    """
    Fit corr ~ k * d^p using per-cell average leg length d.
    Returns dict with power p, scale_km, and linear-fit slope/intercept.
    """
    C = np.asarray(corr, dtype=np.float64)
    N = np.asarray(count, dtype=np.float64)
    if C.size == 0 or N.size == 0:
        return None
    if edges_km.size < 2:
        return None
    centers = 0.5 * (edges_km[:-1] + edges_km[1:])
    h1, h2 = np.meshgrid(centers, centers, indexing="xy")
    d = 0.5 * (h1 + h2)
    mask = np.isfinite(C) & np.isfinite(N) & (N >= float(min_pairs_cell)) & (d > 0.0) & (C > 0.0)
    if np.count_nonzero(mask) < 10:
        return None
    d1 = d[mask]
    c1 = C[mask]
    w1 = N[mask]
    try:
        # Power-law fit on log-log scale: log(c) = log(k) + p log(d)
        logd = np.log(d1)
        logc = np.log(c1)
        p, logk = np.polyfit(logd, logc, deg=1, w=np.sqrt(w1))
        k = float(np.exp(logk))
        if abs(p) < 1e-6:
            scale = float("inf")
        else:
            scale = float((1.0 / max(k, 1e-12)) ** (1.0 / p))
    except Exception:
        p, k, scale = float("nan"), float("nan"), float("nan")
    try:
        # Linear fit: c = a*d + b
        a, b = np.polyfit(d1, c1, deg=1, w=np.sqrt(w1))
    except Exception:
        a, b = float("nan"), float("nan")
    return {
        "power": float(p),
        "scale_km": float(scale),
        "k": float(k),
        "linear_slope": float(a),
        "linear_intercept": float(b),
        "n_points": float(int(np.count_nonzero(mask))),
    }


@torch.no_grad()
def _maybe_shared_event_legcorr2d(*, state, plot_dir: str) -> None:
    """
    Shared-event cross-observation correlation binned by BOTH leg lengths.

    For dt rows sharing one event i at a fixed station-phase:
      r_ij = f(x_j) - f(x_i) + noise
      r_ik = f(x_k) - f(x_i) + noise

    We estimate Corr(r_ij, r_ik) on a grid of:
      h1 = ||x_j - x_i||,  h2 = ||x_k - x_i||

    Config (optional):
      inference.diagnostics.shared_event_legcorr2d:
        enabled: bool (default False)
        n_rows: int (default 500000)   # dt rows sampled from full dataset
        batch_size: int (default 50000)  # residual eval batch size
        seed: int (default 0)
        max_leg_km: float (default 4.0)
        n_bins: int (default 20)
        log_bins: bool (default True)
        max_pairs_per_event: int (default 200)
        max_pairs_total: int (default 400000)
        min_pairs_cell: int (default 300)
        aggregate_phases: bool (default True)  # if True, output one PS heatmap; else separate P/S
        standardize_by_sigma: bool (default True)
        locations_at: str (default "map")  # "map" or "initial"
    """
    cfg = _get_diag_cfg(state, "shared_event_legcorr2d")
    if not _diag_enabled(cfg, default=False):
        return

    try:
        N_total = int(getattr(state, "N", 0))
        if N_total <= 0:
            return
        n_rows = int(cfg.get("n_rows", 500_000))
        n_rows = int(max(1, min(n_rows, N_total)))
        batch_size = int(cfg.get("batch_size", 50_000))
        batch_size = max(1, batch_size)
        seed = int(cfg.get("seed", 0))
        max_leg_km = float(cfg.get("max_leg_km", 4.0))
        n_bins = int(cfg.get("n_bins", 20))
        log_bins = bool(cfg.get("log_bins", True))
        max_pairs_per_event = int(cfg.get("max_pairs_per_event", 200))
        max_pairs_total = int(cfg.get("max_pairs_total", 400_000))
        min_pairs_cell = int(cfg.get("min_pairs_cell", 300))
        aggregate_phases = bool(cfg.get("aggregate_phases", True))
        standardize_by_sigma = bool(cfg.get("standardize_by_sigma", True))
        locations_at = str(cfg.get("locations_at", "map")).strip().lower()
        residual_variant = str(cfg.get("residual_variant", "base")).strip().lower()
    except Exception:
        return

    # Which residuals to analyze:
    # - base: observed - predicted (no corr_error)
    # - corr_error: observed - (predicted + corr_error_delta)
    # - both: produce both plots for comparison
    if residual_variant in {"with_corr_error", "corr"}:
        residual_variant = "corr_error"
    if residual_variant not in {"base", "corr_error", "both"}:
        residual_variant = "base"
    variants = ["base", "corr_error"] if residual_variant == "both" else [residual_variant]

    edges_km = _leg_edges_km(max_leg_km=max_leg_km, n_bins=n_bins, log_bins=log_bins)
    n_leg = int(edges_km.size - 1)
    if n_leg <= 0:
        return

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N_total, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    # Event positions (km) at MAP or initial
    if locations_at in {"init", "initial"}:
        Xtot = state.X_src.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        Xtot = (state.X_src + state.dX_src).detach().cpu().numpy().astype(np.float64, copy=False)
    x_ev = Xtot[:, 0]
    y_ev = Xtot[:, 1]
    z_ev = Xtot[:, 2]

    rows_t_all = torch.as_tensor(rows_np, device=state.device, dtype=torch.int64)
    II_s = state.II.index_select(0, rows_t_all).detach().cpu().numpy().astype(np.int64, copy=False)
    e1 = II_s[:, 0]
    e2 = II_s[:, 1]
    YY_s = state.YY.index_select(0, rows_t_all).detach()
    ph = (YY_s[:, 4].detach().cpu().numpy() >= 0.5).astype(np.int8, copy=False)

    # Station index per row (required)
    sta = None
    try:
        if getattr(state, "row_station_index", None) is not None:
            sta = state.row_station_index.index_select(0, rows_t_all).detach().cpu().numpy().astype(np.int64, copy=False)
    except Exception:
        sta = None
    if sta is None:
        warn("shared_event_legcorr2d: missing row_station_index; skipping.", section="DIAG")
        return

    # Residuals for sampled rows (observed - predicted), optionally subtracting corr_error delta.
    from spider.core.modeling import compute_residuals
    need_corr = ("corr_error" in variants)
    sta_all_t: Optional[torch.Tensor] = None
    if need_corr:
        try:
            sta_all_t = state.row_station_index.index_select(0, rows_t_all)  # type: ignore[union-attr]
        except Exception:
            sta_all_t = None
    r_chunks_base: list[np.ndarray] = []
    r_chunks_corr: list[np.ndarray] = []
    for i0 in range(0, int(rows_t_all.numel()), batch_size):
        i1 = min(i0 + batch_size, int(rows_t_all.numel()))
        rt = rows_t_all[i0:i1]
        II_b = state.II.index_select(0, rt)
        YY_b = state.YY.index_select(0, rt)
        rb_t = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model)
        r_chunks_base.append(rb_t.detach().to("cpu").numpy().astype(np.float64, copy=False))
        if need_corr:
            sta_b = (sta_all_t[i0:i1] if isinstance(sta_all_t, torch.Tensor) else None)
            if isinstance(sta_b, torch.Tensor):
                dc = _corr_error_delta_for_rows(state=state, II_b=II_b, YY_b=YY_b, sta_b=sta_b)
            else:
                dc = None
            if isinstance(dc, torch.Tensor):
                rb_corr_t = rb_t - dc.to(dtype=rb_t.dtype, device=rb_t.device)
            else:
                rb_corr_t = rb_t
            r_chunks_corr.append(rb_corr_t.detach().to("cpu").numpy().astype(np.float64, copy=False))
    resid_base = np.concatenate(r_chunks_base, axis=0) if r_chunks_base else np.zeros((rows_np.size,), dtype=np.float64)
    resid_corr = np.concatenate(r_chunks_corr, axis=0) if r_chunks_corr else resid_base

    # Precompute sigma scales for standardization (same for base/corr_error).
    σp_f, σs_f = 1.0, 1.0
    if standardize_by_sigma:
        try:
            from spider.core.state import _current_noise_scales
            σp, σs = _current_noise_scales(state)
            σp_f = float(σp.detach().cpu().item()); σs_f = float(σs.detach().cpu().item())
            σp_f = σp_f if np.isfinite(σp_f) and σp_f > 0 else 1.0
            σs_f = σs_f if np.isfinite(σs_f) and σs_f > 0 else 1.0
        except Exception:
            σp_f, σs_f = 1.0, 1.0

    # Leg lengths
    dx = x_ev[e2] - x_ev[e1]
    dy = y_ev[e2] - y_ev[e1]
    dz = z_ev[e2] - z_ev[e1]
    h = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64, copy=False)

    def _run_one(*, resid_raw: np.ndarray, tag: str) -> dict[str, dict[str, np.ndarray]]:
        # Standardize after applying any nuisance correction.
        resid = resid_raw
        if standardize_by_sigma:
            resid = np.where(ph == 1, resid / float(σs_f), resid / float(σp_f))

        max_leg = float(edges_km[-1])
        keep = np.isfinite(resid) & np.isfinite(h) & (h <= max_leg)
        if np.count_nonzero(keep) < 1000:
            warn(f"shared_event_legcorr2d[{tag}]: too few sampled rows after filtering; skipping.", section="DIAG")
            return {}
        resid_k = resid[keep]
        h_k = h[keep]
        e1_k = e1[keep]
        e2_k = e2[keep]
        ph_k = ph[keep]
        sta_k = sta[keep]

        # Center residuals per group
        if aggregate_phases:
            gid = sta_k.astype(np.int64)
        else:
            gid = (sta_k.astype(np.int64) * 2 + ph_k.astype(np.int64))
        K = int(gid.max()) + 1
        sum_r = np.bincount(gid, weights=resid_k, minlength=K).astype(np.float64, copy=False)
        cnt_r = np.bincount(gid, minlength=K).astype(np.float64, copy=False)
        mean_r = sum_r / np.maximum(cnt_r, 1.0)
        r0 = (resid_k - mean_r[gid]).astype(np.float64, copy=False)

        # Build per-(gid, shared_event) entry lists
        entries: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for j in range(int(r0.size)):
            gk = int(gid[j])
            a = int(e1_k[j]); b0 = int(e2_k[j])
            hj = float(h_k[j])
            rj = float(r0[j])
            entries.setdefault((gk, a), []).append((rj, hj))
            entries.setdefault((gk, b0), []).append((-rj, hj))

        mats: dict[str, dict[str, np.ndarray]] = {}
        if aggregate_phases:
            mats["PS"] = {
                "n": np.zeros((n_leg, n_leg), dtype=np.int64),
                "sprod": np.zeros((n_leg, n_leg), dtype=np.float64),
                "s1": np.zeros((n_leg, n_leg), dtype=np.float64),
                "s2": np.zeros((n_leg, n_leg), dtype=np.float64),
            }
        else:
            for lab in ("P", "S"):
                mats[lab] = {
                    "n": np.zeros((n_leg, n_leg), dtype=np.int64),
                    "sprod": np.zeros((n_leg, n_leg), dtype=np.float64),
                    "s1": np.zeros((n_leg, n_leg), dtype=np.float64),
                    "s2": np.zeros((n_leg, n_leg), dtype=np.float64),
                }

        picked_total = 0
        idx_buf = None
        for (gk, _ev), lst in entries.items():
            if picked_total >= max_pairs_total:
                break
            m = len(lst)
            if m < 2:
                continue
            if not aggregate_phases:
                phv = int(gk % 2)
                lab = "S" if phv == 1 else "P"
            else:
                lab = "PS"
            mm = mats[lab]
            nmat = mm["n"]; sprod = mm["sprod"]; s1 = mm["s1"]; s2 = mm["s2"]
            tot_pairs = m * (m - 1) // 2
            k_pairs = int(min(int(max_pairs_per_event), int(tot_pairs)))
            if k_pairs <= 0:
                continue
            if idx_buf is None or idx_buf.size != m:
                idx_buf = np.arange(m, dtype=np.int64)
            for _ in range(k_pairs):
                if picked_total >= max_pairs_total:
                    break
                ia, ib = rng.choice(idx_buf, size=2, replace=False).tolist()
                ra, h1 = lst[int(ia)]
                rb, h2 = lst[int(ib)]
                if not (np.isfinite(ra) and np.isfinite(rb) and np.isfinite(h1) and np.isfinite(h2)):
                    continue
                b1 = int(np.searchsorted(edges_km, float(h1), side="right") - 1)
                b2 = int(np.searchsorted(edges_km, float(h2), side="right") - 1)
                if b1 < 0 or b1 >= n_leg or b2 < 0 or b2 >= n_leg:
                    continue
                # Symmetrize
                nmat[b1, b2] += 1; nmat[b2, b1] += 1
                sprod[b1, b2] += ra * rb; sprod[b2, b1] += ra * rb
                s1[b1, b2] += ra * ra; s1[b2, b1] += ra * ra
                s2[b1, b2] += rb * rb; s2[b2, b1] += rb * rb
                picked_total += 1

        if picked_total <= 0:
            warn(f"shared_event_legcorr2d[{tag}]: no pairs picked; skipping.", section="DIAG")
            return {}

        out: dict[str, dict[str, np.ndarray]] = {}
        for label, mm in mats.items():
            nmat = mm["n"]
            sprod = mm["sprod"]; s1 = mm["s1"]; s2 = mm["s2"]
            corr = np.full((n_leg, n_leg), np.nan, dtype=np.float64)
            for i in range(n_leg):
                for j in range(n_leg):
                    nij = int(nmat[i, j])
                    if nij < min_pairs_cell:
                        continue
                    cov = float(sprod[i, j] / max(nij, 1))
                    v1 = float(s1[i, j] / max(nij, 1))
                    v2 = float(s2[i, j] / max(nij, 1))
                    corr[i, j] = cov / float(np.sqrt(max(v1, 1e-20) * max(v2, 1e-20)))
            out[str(label)] = {"corr": corr, "count": nmat.astype(np.float64, copy=False)}
        return out

    results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    if "base" in variants:
        results["base"] = _run_one(resid_raw=resid_base, tag="base")
    if "corr_error" in variants:
        results["corr_error"] = _run_one(resid_raw=resid_corr, tag="corr_error")

    # Plot. If both variants are present, enforce shared color scales for fair comparison.
    tags = list(results.keys())
    if not tags:
        return
    labels = set()
    for tag in tags:
        labels |= set(results.get(tag, {}).keys())
    for label in sorted(labels):
        # Determine shared scales across available variants for this label
        corr_vmax = None
        count_vmax = None
        if len(tags) > 1:
            # corr: use max of per-variant 98th-quantile(abs) to be robust to outliers but comparable
            vv = []
            vv2 = []
            for tag in tags:
                dd = results.get(tag, {}).get(label, None)
                if not isinstance(dd, dict):
                    continue
                C = np.asarray(dd.get("corr", np.zeros((0, 0))), dtype=np.float64)
                N = np.asarray(dd.get("count", np.zeros((0, 0))), dtype=np.float64)
                vals = C[np.isfinite(C)]
                if vals.size:
                    v0 = float(np.nanquantile(np.abs(vals), 0.98))
                    if (not np.isfinite(v0)) or v0 <= 0.0:
                        v0 = float(np.nanmax(np.abs(vals)))
                    if np.isfinite(v0) and v0 > 0:
                        vv.append(v0)
                try:
                    m2 = float(np.nanmax(np.log1p(N)))
                    if np.isfinite(m2) and m2 > 0:
                        vv2.append(m2)
                except Exception:
                    pass
            if vv:
                corr_vmax = float(max(vv))
            if vv2:
                count_vmax = float(max(vv2))

        for tag in tags:
            dd = results.get(tag, {}).get(label, None)
            if not isinstance(dd, dict):
                continue
            # Scalar score for quick comparison across runs (and across base vs corr_error).
            try:
                score = _legcorr2d_mean_abs_corr(
                    corr=np.asarray(dd["corr"], dtype=np.float64),
                    count=np.asarray(dd["count"], dtype=np.float64),
                    min_pairs_cell=int(min_pairs_cell),
                )
                _log(
                    "Shared-event legcorr2d score "
                    f"(label={label} variant={tag} supported_bins={int(score['n_supported_bins'])}): "
                    f"mean|corr|={score['mean_abs_corr']:.4g} "
                    f"weighted={score['mean_abs_corr_weighted']:.4g} "
                    f"(sum_pairs={score['sum_pairs_supported']:.3g})",
                    flush=True,
                )
                fit = _legcorr2d_fit_distance(
                    corr=np.asarray(dd["corr"], dtype=np.float64),
                    count=np.asarray(dd["count"], dtype=np.float64),
                    edges_km=np.asarray(edges_km, dtype=np.float64),
                    min_pairs_cell=int(min_pairs_cell),
                )
                if isinstance(fit, dict):
                    _log(
                        "Shared-event legcorr2d distance fit "
                        f"(label={label} variant={tag} n_points={int(fit.get('n_points', 0))}): "
                        f"power≈{fit.get('power', float('nan')):.3g} "
                        f"scale_km≈{fit.get('scale_km', float('nan')):.3g} "
                        f"linear_slope≈{fit.get('linear_slope', float('nan')):.3g} "
                        f"linear_intercept≈{fit.get('linear_intercept', float('nan')):.3g}",
                        flush=True,
                    )
                # Persist scores as JSON next to plots for easy run-to-run diffing.
                try:
                    scores_path = os.path.join(str(plot_dir), "shared_event_legcorr2d_scores.json")
                    payload = {
                        "label": str(label),
                        "variant": str(tag),
                        "min_pairs_cell": int(min_pairs_cell),
                        **{k: float(v) for k, v in score.items()},
                    }
                    if isinstance(fit, dict):
                        for k, v in fit.items():
                            payload[f"fit_{k}"] = float(v)
                    # Append as JSONL for simplicity and robustness.
                    with open(scores_path, "a") as f:
                        import json as _json
                        f.write(_json.dumps(payload) + "\n")
                except Exception:
                    pass
            except Exception:
                pass
            out_png = os.path.join(str(plot_dir), f"shared_event_legcorr2d_{label}_{tag}.pdf")
            _plot_legcorr2d(
                corr=np.asarray(dd["corr"], dtype=np.float64),
                count=np.asarray(dd["count"], dtype=np.float64),
                edges_km=edges_km,
                title=f"Shared-event corr (leg1,leg2) aggregated={label} [{tag}]",
                out_png=str(out_png),
                corr_vmax=corr_vmax,
                count_vmax=count_vmax,
            )
            _log(f"Shared-event legcorr2d -> {out_png}", flush=True)

    # If phases are separated, also emit a side-by-side P/S figure per variant.
    if not aggregate_phases:
        for tag in tags:
            dd_p = results.get(tag, {}).get("P", None)
            dd_s = results.get(tag, {}).get("S", None)
            if not (isinstance(dd_p, dict) and isinstance(dd_s, dict)):
                continue
            # Shared scales across P/S for this tag.
            corr_vals = []
            for C in (dd_p.get("corr", None), dd_s.get("corr", None)):
                if C is None:
                    continue
                C = np.asarray(C, dtype=np.float64)
                vals = C[np.isfinite(C)]
                if vals.size:
                    corr_vals.append(vals)
            if corr_vals:
                vv = np.concatenate(corr_vals)
                corr_vmax_ps = float(np.nanquantile(np.abs(vv), 0.98))
                if (not np.isfinite(corr_vmax_ps)) or corr_vmax_ps <= 0.0:
                    corr_vmax_ps = float(np.nanmax(np.abs(vv)))
            else:
                corr_vmax_ps = None

            count_vals = []
            for N in (dd_p.get("count", None), dd_s.get("count", None)):
                if N is None:
                    continue
                N = np.asarray(N, dtype=np.float64)
                vals = np.log1p(N[np.isfinite(N)])
                if vals.size:
                    count_vals.append(vals)
            if count_vals:
                vv2 = np.concatenate(count_vals)
                count_vmax_ps = float(np.nanmax(vv2))
            else:
                count_vmax_ps = None

            out_png = os.path.join(str(plot_dir), f"shared_event_legcorr2d_PS_{tag}.pdf")
            _plot_legcorr2d_pair(
                corr_p=np.asarray(dd_p["corr"], dtype=np.float64),
                corr_s=np.asarray(dd_s["corr"], dtype=np.float64),
                count_p=np.asarray(dd_p["count"], dtype=np.float64),
                count_s=np.asarray(dd_s["count"], dtype=np.float64),
                edges_km=edges_km,
                title=f"Shared-event corr (leg1,leg2) P/S [{tag}]",
                out_png=str(out_png),
                corr_vmax=corr_vmax_ps,
                count_vmax=count_vmax_ps,
            )
            _log(f"Shared-event legcorr2d P/S -> {out_png}", flush=True)


def _mean_by_key_sorted(key: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given key sorted ascending, compute mean(val) per unique key.
    Returns (uniq_key, mean_val) sorted by uniq_key.
    """
    key = np.asarray(key)
    val = np.asarray(val, dtype=np.float64)
    if key.size == 0:
        return key.astype(np.int64, copy=False), val.astype(np.float32, copy=False)
    # boundaries
    change = np.ones((key.size,), dtype=bool)
    change[1:] = key[1:] != key[:-1]
    idx = np.nonzero(change)[0].astype(np.int64, copy=False)
    uniq = key[idx]
    # reduce sums/counts
    sums = np.add.reduceat(val, idx)
    # counts are distances between idx entries
    cnt = np.diff(np.append(idx, key.size)).astype(np.float64, copy=False)
    mean = (sums / np.maximum(cnt, 1.0)).astype(np.float32, copy=False)
    return uniq.astype(np.int64, copy=False), mean


def _plot_station_corr_matrix(
    *,
    corr: np.ndarray,
    count: np.ndarray,
    labels: list[str],
    title: str,
    out_png: str,
) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")
    C = np.asarray(corr, dtype=np.float64)
    N = np.asarray(count, dtype=np.float64)
    K = int(C.shape[0])
    if K <= 0:
        raise RuntimeError("empty correlation matrix")

    vals = C[np.isfinite(C)]
    v = float(np.nanquantile(np.abs(vals), 0.98)) if vals.size else 1.0
    v = float(max(0.1, min(v, 1.0)))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    im = ax0.imshow(C, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest", aspect="equal")
    ax0.set_title(title)
    fig.colorbar(im, ax=ax0, shrink=0.85, label="corr")

    im2 = ax1.imshow(np.log1p(N), cmap="viridis", interpolation="nearest", aspect="equal")
    fig.colorbar(im2, ax=ax1, shrink=0.85, label="log(1+n_common_pairs)")

    # Ticks/labels (only if manageable)
    try:
        if labels and len(labels) == K and K <= 40:
            ax1.set_xticks(np.arange(K))
            ax1.set_xticklabels(labels, rotation=90, fontsize=7)
            ax0.set_yticks(np.arange(K))
            ax0.set_yticklabels(labels, fontsize=7)
        else:
            ax1.set_xticks([])
            ax0.set_yticks([])
    except Exception:
        pass

    ax1.set_xlabel("station")
    ax1.set_ylabel("station")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_resid_histogram(
    *,
    rP: np.ndarray,
    rS: np.ndarray,
    bins: int,
    hist_range: Optional[tuple[float, float]],
    title: str,
    out_png: str,
) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")
    fig, (ax, ax_cdf, ax_sf) = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    rng = hist_range if hist_range is not None else None
    ax.hist(rP, bins=bins, range=rng, alpha=0.6, color="tab:blue", label="P", density=True)
    ax.hist(rS, bins=bins, range=rng, alpha=0.6, color="tab:orange", label="S", density=True)
    try:
        rp = np.asarray(rP, dtype=np.float64)
        rs = np.asarray(rS, dtype=np.float64)
        rp = rp[np.isfinite(rp)]
        rs = rs[np.isfinite(rs)]
        if rng is not None:
            x_min, x_max = float(rng[0]), float(rng[1])
        else:
            x_min = float(np.min(np.concatenate([rp, rs], axis=0))) if (rp.size + rs.size) > 0 else -1.0
            x_max = float(np.max(np.concatenate([rp, rs], axis=0))) if (rp.size + rs.size) > 0 else 1.0
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = -1.0, 1.0
        xs = np.linspace(x_min, x_max, 600)

        def _norm_pdf(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
            s = float(sig)
            if (not np.isfinite(s)) or s <= 0:
                s = 1.0
            z = (x - float(mu)) / s
            return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * s)

        def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            xx = np.asarray(x, dtype=np.float64)
            xx = xx[np.isfinite(xx)]
            if xx.size == 0:
                return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
            xx.sort()
            yy = (np.arange(xx.size, dtype=np.float64) + 1.0) / float(xx.size)
            return xx, yy

        def _norm_cdf(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
            s = float(sig)
            if (not np.isfinite(s)) or s <= 0:
                s = 1.0
            z = (x - float(mu)) / (s * np.sqrt(2.0))
            return 0.5 * (1.0 + np.vectorize(math.erf)(z))

        def _survival(y: np.ndarray) -> np.ndarray:
            yy = np.asarray(y, dtype=np.float64)
            yy = np.clip(yy, 0.0, 1.0)
            return np.clip(1.0 - yy, 1e-12, 1.0)

        if rp.size >= 8:
            mu_p = float(np.mean(rp))
            sig_p = float(np.std(rp, ddof=0))
            ax.plot(xs, _norm_pdf(xs, mu_p, sig_p), color="tab:blue", lw=2.0, alpha=0.9, label="P fit")
            xec, yec = _ecdf(rp)
            if xec.size > 0:
                ax_cdf.plot(xec, yec, color="tab:blue", lw=1.5, alpha=0.6, label="P ECDF")
                ax_sf.plot(xec, _survival(yec), color="tab:blue", lw=1.5, alpha=0.6, label="P survival")
            ax_cdf.plot(xs, _norm_cdf(xs, mu_p, sig_p), color="tab:blue", lw=2.0, alpha=0.9, label="P CDF fit")
            ax_sf.plot(xs, _survival(_norm_cdf(xs, mu_p, sig_p)), color="tab:blue", lw=2.0, alpha=0.9, label="P fit survival")
        if rs.size >= 8:
            mu_s = float(np.mean(rs))
            sig_s = float(np.std(rs, ddof=0))
            ax.plot(xs, _norm_pdf(xs, mu_s, sig_s), color="tab:orange", lw=2.0, alpha=0.9, label="S fit")
            xec, yec = _ecdf(rs)
            if xec.size > 0:
                ax_cdf.plot(xec, yec, color="tab:orange", lw=1.5, alpha=0.6, label="S ECDF")
                ax_sf.plot(xec, _survival(yec), color="tab:orange", lw=1.5, alpha=0.6, label="S survival")
            ax_cdf.plot(xs, _norm_cdf(xs, mu_s, sig_s), color="tab:orange", lw=2.0, alpha=0.9, label="S CDF fit")
            ax_sf.plot(xs, _survival(_norm_cdf(xs, mu_s, sig_s)), color="tab:orange", lw=2.0, alpha=0.9, label="S fit survival")
    except Exception:
        pass
    ax.set_title(title)
    ax.set_xlabel("residual (s)")
    ax.set_ylabel("density")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    ax_cdf.set_title("Residual CDF")
    ax_cdf.set_xlabel("residual (s)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.grid(True, alpha=0.2)
    ax_cdf.legend(loc="best")
    ax_sf.set_title("Residual survival (1 - CDF)")
    ax_sf.set_xlabel("residual (s)")
    ax_sf.set_ylabel("survival")
    ax_sf.set_yscale("log")
    ax_sf.set_ylim(1e-6, 1.0)
    ax_sf.grid(True, alpha=0.2)
    ax_sf.legend(loc="best")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


@torch.no_grad()
def _maybe_event_pair_station_corr(*, state, plot_dir: str) -> None:
    """
    Cross-station correlation for the *same event pair* (and phase) across many pairs.

    This answers: "Are residuals for a fixed (e1,e2,phase) correlated between station A and station B
    when we look across many event pairs that are observed at both stations?"

    We estimate:
      Corr( r_{(e1,e2), A, phase} , r_{(e1,e2), B, phase} )
    across event pairs, for a selected set of stations (top-K by sample count).

    Config (optional):
      inference.diagnostics.event_pair_station_corr:
        enabled: bool (default False)
        n_rows: int (default 300000)         # dt rows sampled from full dataset
        batch_size: int (default 50000)      # residual eval batch size
        seed: int (default 0)
        top_k_stations: int (default 30)
        min_common_pairs: int (default 50)   # min shared event-pairs needed to compute corr
        standardize_by_sigma: bool (default True)
      residuals_at: str (default "map")    # "map" | "initial" | "truth"
        outfile_prefix: str (default "event_pair_station_corr")
    """
    cfg = _get_diag_cfg(state, "event_pair_station_corr")
    if not _diag_enabled(cfg, default=False):
        return

    try:
        N_total = int(getattr(state, "N", 0))
        if N_total <= 0:
            return
        n_rows = int(cfg.get("n_rows", 300_000))
        n_rows = int(max(1, min(n_rows, N_total)))
        batch_size = int(cfg.get("batch_size", 50_000))
        batch_size = max(1, batch_size)
        seed = int(cfg.get("seed", 0))
        top_k = int(cfg.get("top_k_stations", 30))
        top_k = int(max(2, min(top_k, 200)))
        min_common = int(cfg.get("min_common_pairs", 50))
        min_common = int(max(10, min_common))
        standardize_by_sigma = bool(cfg.get("standardize_by_sigma", True))
        residuals_at = str(cfg.get("residuals_at", "map")).strip().lower()
        if residuals_at not in {"map", "initial", "truth"}:
            residuals_at = "map"
        out_prefix = str(cfg.get("outfile_prefix", "event_pair_station_corr") or "event_pair_station_corr")
        residual_variant = str(cfg.get("residual_variant", "base")).strip().lower()
    except Exception:
        return

    if residual_variant in {"with_corr_error", "corr"}:
        residual_variant = "corr_error"
    if residual_variant not in {"base", "corr_error"}:
        residual_variant = "base"

    # Require station indices
    sta_idx_t = getattr(state, "row_station_index", None)
    if not isinstance(sta_idx_t, torch.Tensor):
        warn("event_pair_station_corr: missing row_station_index; skipping.", section="DIAG")
        return

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N_total, size=n_rows, replace=False).astype(np.int64, copy=False)
    rows_np.sort()
    rows_t_all = torch.as_tensor(rows_np, device=state.device, dtype=torch.int64)

    II_s = state.II.index_select(0, rows_t_all).detach().cpu().numpy().astype(np.int64, copy=False)
    e1 = II_s[:, 0]
    e2 = II_s[:, 1]
    YY_s = state.YY.index_select(0, rows_t_all).detach()
    ph = (YY_s[:, 4].detach().cpu().numpy() >= 0.5).astype(np.int8, copy=False)  # 0=P, 1=S
    sta = sta_idx_t.index_select(0, rows_t_all).detach().cpu().numpy().astype(np.int64, copy=False)

    # Residuals for sampled rows
    from spider.core.modeling import compute_residuals
    dX_use = _resolve_dX_use(state, residuals_at)

    r_chunks = []
    for i0 in range(0, int(rows_t_all.numel()), batch_size):
        i1 = min(i0 + batch_size, int(rows_t_all.numel()))
        rt = rows_t_all[i0:i1]
        II_b = state.II.index_select(0, rt)
        YY_b = state.YY.index_select(0, rt)
        sta_b = sta_idx_t.index_select(0, rt)
        rb = _compute_residuals_numpy(state=state, II_b=II_b, YY_b=YY_b, dX_use=dX_use, sta_b=sta_b, variant=residual_variant)
        r_chunks.append(rb)
    resid = np.concatenate(r_chunks, axis=0) if r_chunks else np.zeros((rows_np.size,), dtype=np.float64)

    # Filter invalid stations
    keep = (sta >= 0) & np.isfinite(resid)
    if np.count_nonzero(keep) < 1000:
        warn("event_pair_station_corr: too few sampled rows after filtering; skipping.", section="DIAG")
        return
    e1 = e1[keep]; e2 = e2[keep]; ph = ph[keep]; sta = sta[keep]; resid = resid[keep]

    if standardize_by_sigma:
        try:
            from spider.core.state import _current_noise_scales
            σp, σs = _current_noise_scales(state)
            σp_f = float(σp.detach().cpu().item()); σs_f = float(σs.detach().cpu().item())
            σp_f = σp_f if np.isfinite(σp_f) and σp_f > 0 else 1.0
            σs_f = σs_f if np.isfinite(σs_f) and σs_f > 0 else 1.0
        except Exception:
            σp_f, σs_f = 1.0, 1.0
        resid = np.where(ph == 1, resid / float(σs_f), resid / float(σp_f))

    # Canonicalize event-pair orientation so (e1,e2) matches across stations:
    # key uses unordered (u=min, v=max); residual sign is flipped if original is reversed.
    u = np.minimum(e1, e2).astype(np.int64, copy=False)
    v = np.maximum(e1, e2).astype(np.int64, copy=False)
    sgn = np.where(e1 == u, 1.0, -1.0).astype(np.float64, copy=False)
    resid = resid * sgn

    # Make a compact key_id for unordered pair: (u<<32) | v
    key_id = ((u.astype(np.uint64) << np.uint64(32)) | v.astype(np.uint64)).astype(np.uint64, copy=False)

    # Best-effort station labels
    sta_label: dict[int, str] = {}
    try:
        import polars as pl  # type: ignore
        dt = state.dtimes
        if isinstance(dt, pl.DataFrame) and ("sta_idx" in dt.columns) and ("network" in dt.columns) and ("station" in dt.columns):
            st = (
                dt.select([pl.col("sta_idx"), pl.col("network"), pl.col("station")])
                .unique(subset=["sta_idx"], maintain_order=True)
                .sort("sta_idx")
            )
            for row in st.iter_rows(named=True):
                sta_label[int(row["sta_idx"])] = f"{row['network']}.{row['station']}"
    except Exception:
        sta_label = {}

    def _do_phase(phase_bit: int, phase_name: str) -> None:
        m = (ph == int(phase_bit))
        if np.count_nonzero(m) < 1000:
            return
        sta_p = sta[m]
        key_p = key_id[m]
        r_p = resid[m].astype(np.float64, copy=False)

        # pick top-K stations by count in this phase sample
        uniq_sta, cnt_sta = np.unique(sta_p, return_counts=True)
        ord0 = np.argsort(-cnt_sta)
        uniq_sta = uniq_sta[ord0][: int(min(top_k, uniq_sta.size))]
        if uniq_sta.size < 2:
            return

        # Build per-station mean residual per event-pair key
        keys_by_sta: list[np.ndarray] = []
        vals_by_sta: list[np.ndarray] = []
        labels: list[str] = []
        for s_id in uniq_sta.tolist():
            s_id = int(s_id)
            ms = (sta_p == s_id)
            if np.count_nonzero(ms) < 10:
                continue
            k = key_p[ms]
            x = r_p[ms]
            # sort by key and average duplicates
            ordk = np.argsort(k, kind="mergesort")
            k_s = k[ordk]
            x_s = x[ordk]
            ku, xm = _mean_by_key_sorted(k_s, x_s)
            if ku.size < int(min_common):
                continue
            keys_by_sta.append(ku.astype(np.uint64, copy=False))
            vals_by_sta.append(xm.astype(np.float32, copy=False))
            labels.append(sta_label.get(s_id, f"sta{s_id}"))

        K = int(len(keys_by_sta))
        if K < 2:
            return

        corr = np.full((K, K), np.nan, dtype=np.float32)
        cnt = np.zeros((K, K), dtype=np.int32)
        np.fill_diagonal(corr, 1.0)

        for i in range(K):
            ki = keys_by_sta[i]
            xi = vals_by_sta[i].astype(np.float64, copy=False)
            for j in range(i + 1, K):
                kj = keys_by_sta[j]
                xj = vals_by_sta[j].astype(np.float64, copy=False)
                common, ii, jj = np.intersect1d(ki, kj, assume_unique=True, return_indices=True)
                m0 = int(common.size)
                cnt[i, j] = cnt[j, i] = int(m0)
                if m0 < int(min_common):
                    continue
                a = xi[ii]
                b = xj[jj]
                ma = float(np.mean(a)); mb = float(np.mean(b))
                ac = a - ma; bc = b - mb
                va = float(np.mean(ac * ac)); vb = float(np.mean(bc * bc))
                if (not np.isfinite(va)) or (not np.isfinite(vb)) or va <= 0.0 or vb <= 0.0:
                    continue
                c = float(np.mean(ac * bc) / np.sqrt(va * vb))
                corr[i, j] = corr[j, i] = float(np.clip(c, -0.999, 0.999))

        out_png = os.path.join(str(plot_dir), f"{out_prefix}_{phase_name}.png")
        _plot_station_corr_matrix(
            corr=corr,
            count=cnt.astype(np.float64, copy=False),
            labels=labels,
            title=f"Event-pair residual corr across stations (phase={phase_name}, sampled_rows={int(n_rows):,})",
            out_png=str(out_png),
        )
        _log(f"Event-pair station corr -> {out_png} (phase={phase_name}, stations={K})", flush=True)

    _do_phase(0, "P")
    _do_phase(1, "S")

@torch.no_grad()
def analyze_resid_from_bundle(
    *,
    params: dict,
    bundle_path: str,
    model: nn.Module,
    device: torch.device,
    plot_variograms: bool = True,
    plot_dir: Optional[str] = None,
    use_latest_checkpoint: bool = False,
) -> None:
    """
    Run residual/variogram diagnostics from an existing Phase-2 bundle (output of `locate-map`),
    without rerunning Phase 1 (MAP).

    This is meant for fast iteration on residual diagnostics via CLI.
    """
    bun = load_phase2_bundle(path=str(bundle_path))

    # Prefer current runtime flags, but keep any materialized flat keys stored in the bundle.
    run_params = params
    try:
        if isinstance(bun.params, dict):
            run_params = {**bun.params, **params}
    except Exception:
        run_params = params

    # Build a normal state, but restore the MAP solution from the bundle.
    state = _build_initial_state(run_params, bun.origins0, bun.dtimes, model, device)
    # Optionally override ΔX (and corr_error_b) from the most recent sampling checkpoint.
    # This is useful to evaluate diagnostics "after spider sample" (phase4) rather than at MAP.
    ckpt = None
    if bool(use_latest_checkpoint):
        try:
            from spider.io.checkpoint import load_checkpoint
            ckpt = load_checkpoint(run_params, state.device)
        except Exception:
            ckpt = None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("ΔX_src", None), torch.Tensor):
        state.dX_src.data.copy_(ckpt["ΔX_src"].to(device=state.device, dtype=torch.float32))
        try:
            info(
                f"analyze-resid: using latest checkpoint phase={str(ckpt.get('phase','?'))} epoch={int(ckpt.get('epoch',0))} for ΔX_src",
                section="DIAG",
            )
        except Exception:
            pass
    else:
        try:
            state.dX_src.data.copy_(bun.dX_src.to(device=state.device, dtype=torch.float32))
        except Exception as e:
            raise RuntimeError(f"Could not restore MAP dX_src from bundle: {e}") from e
    # Note: Phase-2 bundles no longer persist noise scale / optimizer state by default.

    info(
        f"analyze-resid: loaded bundle={bundle_path} events={int(state.X_src.shape[0])} dtimes={int(state.N)}",
        section="DIAG",
    )

    # Initialize corr_error (if enabled) so diagnostics can optionally evaluate residuals
    # after applying the corr_error nuisance correction.
    try:
        # Prefer checkpoint corr_error_b if available; otherwise use the bundle's MAP corr_error_b.
        b_resume = None
        try:
            if isinstance(ckpt, dict) and isinstance(ckpt.get("corr_error_b", None), torch.Tensor):
                b_resume = ckpt.get("corr_error_b")
        except Exception:
            b_resume = None
        if b_resume is None:
            b_resume = bun.corr_error_b
        if isinstance(b_resume, torch.Tensor):
            setattr(state, "_resume_corr_error_b", b_resume.to(device=state.device, dtype=torch.float32))
        # Build graph/W and create state.corr_error_b (will use _resume_corr_error_b if present).
        from spider.core.locate import _maybe_init_corr_error
        _maybe_init_corr_error(state)
    except Exception:
        pass

    # For analyze-resid, default plot_dir should be next to the bundle; set it early so
    # all optional diagnostics (including residual distribution) write to the same place.
    if plot_dir is None:
        try:
            plot_dir = os.path.dirname(os.path.abspath(str(bundle_path))) or "."
        except Exception:
            plot_dir = "."

    # Residual distribution diagnostics (Normal vs Laplace; tail ratios; QQ plots).
    # Controlled by `inference.diagnostics.resid_distribution` (optional; default disabled).
    try:
        cfg = _get_diag_cfg(state, "resid_distribution")
        if _diag_enabled(cfg, default=False):
            residuals_at = str(cfg.get("residuals_at", "map")).strip().lower()
            if residuals_at not in {"map", "initial", "truth"}:
                residuals_at = "map"
            n_rows = int(cfg.get("n_rows", 200_000))
            seed = int(cfg.get("seed", 0))
            bs = int(cfg.get("batch_size", 50_000))
            max_groups = int(cfg.get("max_groups", 16))
            min_rows_per_group = int(cfg.get("min_rows_per_group", 500))
            make_plots = bool(cfg.get("make_plots", True))
            hist_enable = bool(cfg.get("histogram", False))
            hist_bins = int(cfg.get("hist_bins", 200))
            hist_range_s = cfg.get("hist_range_s", None)
            hist_initial_only = bool(cfg.get("histogram_initial_only", True))
            qq_sigma = bool(cfg.get("qq_sigma", True))
            qq_mad = bool(cfg.get("qq_mad", True))
            qq_huber = bool(cfg.get("qq_huber", True))
            by_station_phase = bool(cfg.get("by_station_phase", True))
            exceedance_thresholds = cfg.get("exceedance_thresholds", [3, 5, 10])
            huber_fit = bool(cfg.get("huber_fit", True))
            huber_k_min = float(cfg.get("huber_k_min", 0.2))
            huber_k_max = float(cfg.get("huber_k_max", 8.0))
            huber_k_n = int(cfg.get("huber_k_n", 240))
            residual_variant = str(cfg.get("residual_variant", "base")).strip().lower()

            bs = max(1024, int(bs))
            max_groups = max(0, int(max_groups))
            min_rows_per_group = max(0, int(min_rows_per_group))
            # sanitize exceedance thresholds
            thr: list[float] = []
            try:
                if isinstance(exceedance_thresholds, (list, tuple)):
                    for v in exceedance_thresholds:
                        try:
                            fv = float(v)
                            if np.isfinite(fv) and fv > 0:
                                thr.append(float(fv))
                        except Exception:
                            continue
                else:
                    fv = float(exceedance_thresholds)
                    if np.isfinite(fv) and fv > 0:
                        thr.append(float(fv))
            except Exception:
                thr = [3.0, 5.0, 10.0]
            if not thr:
                thr = [3.0, 5.0, 10.0]
            thr = sorted(set(thr))

            N = int(state.N)
            if n_rows <= 0 or n_rows >= N:
                rows_np = np.arange(N, dtype=np.int64)
            else:
                rng = np.random.default_rng(int(seed))
                rows_np = rng.choice(N, size=int(n_rows), replace=False).astype(np.int64, copy=False)
                rows_np.sort()

            rows_t = torch.tensor(rows_np, device=state.device, dtype=torch.int64)
            II_all = state.II.index_select(0, rows_t)
            YY_all = state.YY.index_select(0, rows_t)
            ph = YY_all[:, 4].detach().to("cpu").numpy().astype(np.int64, copy=False)
            sta = None
            sta_all_t: Optional[torch.Tensor] = None
            try:
                if getattr(state, "row_station_index", None) is not None:
                    sta_t = state.row_station_index.index_select(0, rows_t)  # type: ignore[union-attr]
                    sta = sta_t.detach().to("cpu").numpy().astype(np.int64, copy=False)
                    sta_all_t = sta_t
            except Exception:
                sta = None
                sta_all_t = None

            if residual_variant in {"with_corr_error", "corr"}:
                residual_variant = "corr_error"
            if residual_variant not in {"base", "corr_error"}:
                residual_variant = "base"

            # Sigma scales (for z=r/sigma standardization): fixed phase_unc only (noise learning removed)
            sigma_p: Optional[float]
            sigma_s: Optional[float]
            try:
                    vv = state.params.get("phase_unc", [float("nan"), float("nan")])
                    sigma_p = float(vv[0]); sigma_s = float(vv[1])
            except Exception:
                sigma_p = None; sigma_s = None
            if sigma_p is not None and (not np.isfinite(sigma_p) or sigma_p <= 0):
                sigma_p = None
            if sigma_s is not None and (not np.isfinite(sigma_s) or sigma_s <= 0):
                sigma_s = None

            # Residuals (seconds)
            r_chunks: list[np.ndarray] = []
            dX_use = _resolve_dX_use(state, residuals_at)
            for i0 in range(0, int(II_all.shape[0]), bs):
                i1 = min(i0 + bs, int(II_all.shape[0]))
                II_b = II_all[i0:i1]
                YY_b = YY_all[i0:i1]
                sta_b = (sta_all_t[i0:i1] if isinstance(sta_all_t, torch.Tensor) else None)
                rb = _compute_residuals_numpy(state=state, II_b=II_b, YY_b=YY_b, dX_use=dX_use, sta_b=sta_b, variant=residual_variant)
                r_chunks.append(rb)
            resid = np.concatenate(r_chunks, axis=0) if r_chunks else np.zeros((rows_np.size,), dtype=np.float64)

            keep = np.isfinite(resid) & np.isfinite(ph) & ((ph == 0) | (ph == 1))
            resid = resid[keep]
            ph = ph[keep]
            if isinstance(sta, np.ndarray) and sta.shape[0] == keep.shape[0]:
                sta = sta[keep]
            else:
                sta = None

            rP = resid[ph == 0]
            rS = resid[ph == 1]

            _log("\nResidual distribution diagnostics (analyze-resid)", flush=True)
            _log(f"- residuals_at={residuals_at} n_rows_used={int(resid.size):,}", flush=True)
            _log(f"- sigma_p={sigma_p} sigma_s={sigma_s}", flush=True)

            sumP = _resid_distribution_summary(r=rP, sigma=sigma_p, label="P")
            sumS = _resid_distribution_summary(r=rS, sigma=sigma_s, label="S")

            def _robust_z_mad_and_scale(x: np.ndarray) -> tuple[np.ndarray, float, float]:
                xx = np.asarray(x, dtype=np.float64)
                xx = xx[np.isfinite(xx)]
                if xx.size < 8:
                    return np.zeros((0,), dtype=np.float64), float("nan"), float("nan")
                med = float(np.median(xx))
                mad = float(np.median(np.abs(xx - med)))
                s = 1.4826 * mad
                if (not np.isfinite(s)) or s <= 0.0:
                    # fallback to std to avoid divide-by-zero
                    s = float(np.std(xx, ddof=0))
                    if (not np.isfinite(s)) or s <= 0.0:
                        s = 1.0
                return (xx - med) / float(s), med, float(s)

            zP_mad, medP_all, sP_all = _robust_z_mad_and_scale(rP)
            zS_mad, medS_all, sS_all = _robust_z_mad_and_scale(rS)
            exP = _exceedance_rates(np.abs(zP_mad), thr)
            exS = _exceedance_rates(np.abs(zS_mad), thr)

            # Huber k fit (on MAD-standardized residuals)
            kP = float("nan")
            kS = float("nan")
            if huber_fit:
                try:
                    fitP = _fit_huber_k_mle(zP_mad, k_min=huber_k_min, k_max=huber_k_max, n_grid=huber_k_n)
                    fitS = _fit_huber_k_mle(zS_mad, k_min=huber_k_min, k_max=huber_k_max, n_grid=huber_k_n)
                    kP = float(fitP.get("k_hat", float("nan")))
                    kS = float(fitS.get("k_hat", float("nan")))
                    if np.isfinite(kP) and np.isfinite(kS):
                        _log(f"  Huber fit on z_mad: k_hat(P)={kP:.3g}  k_hat(S)={kS:.3g}", flush=True)
                        if np.isfinite(sP_all) and np.isfinite(sS_all) and sP_all > 0 and sS_all > 0:
                            dP = float(kP) * float(sP_all)
                            dS = float(kS) * float(sS_all)
                            _log(
                                f"  z_mad scale (sec): P={float(sP_all):.3g} (med={float(medP_all):.3g})  "
                                f"S={float(sS_all):.3g} (med={float(medS_all):.3g})",
                                flush=True,
                            )
                            _log(
                                f"  implied Huber delta (sec): P={dP:.3g} ({1e3*dP:.3g} ms)  "
                                f"S={dS:.3g} ({1e3*dS:.3g} ms)",
                                flush=True,
                            )
                except Exception as e:
                    warn(f"Could not fit Huber k: {e}", section="DIAG")

            def _fmt_fit(d: dict) -> str:
                try:
                    dn = d.get("aic_laplace_minus_norm", float("nan"))
                    return f"ΔAIC(lap-norm)={float(dn):.3g}"
                except Exception:
                    return "ΔAIC(lap-norm)=nan"

            def _print_sum(s: dict) -> None:
                n = int(s.get("n", 0) or 0)
                lab = str(s.get("label", "?"))
                if n <= 0:
                    _log(f"  {lab}: n=0", flush=True)
                    return
                tr = s.get("z_mad_tail_ratio_95_75", float("nan"))
                fit = s.get("z_mad_fit", {})
                _log(
                    f"  {lab}: n={n:,} std={float(s.get('std', float('nan'))):.3g} "
                    f"q95|r|={float(s.get('q95_abs', float('nan'))):.3g} "
                    f"tail_ratio95/75(|z_mad|)={float(tr):.3g} {_fmt_fit(fit) if isinstance(fit, dict) else ''}",
                    flush=True,
                )

            _print_sum(sumP)
            _print_sum(sumS)

            def _fmt_exceed(ex: dict[float, float]) -> str:
                parts = []
                for t in thr:
                    v = ex.get(float(t), float("nan"))
                    parts.append(f"P(|z_mad|>{t:g})={float(v):.3g}")
                return "  ".join(parts)

            _log(f"  P exceedance: {_fmt_exceed(exP)}", flush=True)
            _log(f"  S exceedance: {_fmt_exceed(exS)}", flush=True)

            # Optional QQ plots (global per-phase)
            if make_plots:
                try:
                    wrote = []
                    if qq_sigma:
                        outP = os.path.join(str(plot_dir), "resid_qq_P_sigma_norm.png")
                        outS = os.path.join(str(plot_dir), "resid_qq_S_sigma_norm.png")
                        _qqplot(z=rP / float(sigma_p) if sigma_p is not None else rP, dist="normal", title="P residuals (sigma-standardized if available)", out_png=outP)
                        _qqplot(z=rS / float(sigma_s) if sigma_s is not None else rS, dist="normal", title="S residuals (sigma-standardized if available)", out_png=outS)
                        outP2 = os.path.join(str(plot_dir), "resid_qq_P_sigma_laplace.png")
                        outS2 = os.path.join(str(plot_dir), "resid_qq_S_sigma_laplace.png")
                        _qqplot(z=rP / float(sigma_p) if sigma_p is not None else rP, dist="laplace", title="P residuals (sigma-standardized if available)", out_png=outP2)
                        _qqplot(z=rS / float(sigma_s) if sigma_s is not None else rS, dist="laplace", title="S residuals (sigma-standardized if available)", out_png=outS2)
                        wrote += [outP, outS, outP2, outS2]
                    if qq_mad:
                        outPm = os.path.join(str(plot_dir), "resid_qq_P_mad_norm.png")
                        outSm = os.path.join(str(plot_dir), "resid_qq_S_mad_norm.png")
                        _qqplot(z=zP_mad, dist="normal", title="P residuals (MAD-standardized)", out_png=outPm)
                        _qqplot(z=zS_mad, dist="normal", title="S residuals (MAD-standardized)", out_png=outSm)
                        outPm2 = os.path.join(str(plot_dir), "resid_qq_P_mad_laplace.png")
                        outSm2 = os.path.join(str(plot_dir), "resid_qq_S_mad_laplace.png")
                        _qqplot(z=zP_mad, dist="laplace", title="P residuals (MAD-standardized)", out_png=outPm2)
                        _qqplot(z=zS_mad, dist="laplace", title="S residuals (MAD-standardized)", out_png=outSm2)
                        wrote += [outPm, outSm, outPm2, outSm2]
                    if qq_huber and huber_fit and np.isfinite(kP) and np.isfinite(kS):
                        outPh = os.path.join(str(plot_dir), "resid_qq_P_mad_huber.png")
                        outSh = os.path.join(str(plot_dir), "resid_qq_S_mad_huber.png")
                        _qqplot_huber(z=zP_mad, k=float(kP), title="P residuals (MAD-standardized)", out_png=outPh)
                        _qqplot_huber(z=zS_mad, k=float(kS), title="S residuals (MAD-standardized)", out_png=outSh)
                        wrote += [outPh, outSh]
                    if wrote:
                        _log(f"Residual QQ plots -> {str(plot_dir)}", flush=True)
                except Exception as e:
                    warn(f"Could not plot residual QQ: {e}", section="DIAG")

            # Optional histogram (default: only for initial model)
            if make_plots and hist_enable and (residuals_at == "initial" or not hist_initial_only):
                hist_range: Optional[tuple[float, float]] = None
                try:
                    if isinstance(hist_range_s, (list, tuple)) and len(hist_range_s) == 2:
                        lo = float(hist_range_s[0])
                        hi = float(hist_range_s[1])
                        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                            hist_range = (lo, hi)
                except Exception:
                    hist_range = None
                try:
                    out_png = os.path.join(str(plot_dir), f"resid_hist_{residuals_at}.png")
                    _plot_resid_histogram(
                        rP=rP,
                        rS=rS,
                        bins=max(16, int(hist_bins)),
                        hist_range=hist_range,
                        title=f"Residual histogram ({residuals_at})",
                        out_png=out_png,
                    )
                    _log(f"Residual histogram -> {str(plot_dir)}", flush=True)
                except Exception as e:
                    warn(f"Residual histogram failed: {e}", section="DIAG")

            # Optional per-station×phase tail summaries to detect mixture/heavy tails.
            if by_station_phase and isinstance(sta, np.ndarray):
                try:
                    k = (sta.astype(np.int64, copy=False) * 2) + ph.astype(np.int64, copy=False)
                    # Use MAD-standardization per phase (more robust to sigma misestimation).
                    medP = float(np.median(rP)) if rP.size else 0.0
                    madP = float(np.median(np.abs(rP - medP))) if rP.size else 0.0
                    sP = 1.4826 * madP
                    if (not np.isfinite(sP)) or sP <= 0.0:
                        sP = float(np.std(rP, ddof=0)) if rP.size else 1.0
                    medS = float(np.median(rS)) if rS.size else 0.0
                    madS = float(np.median(np.abs(rS - medS))) if rS.size else 0.0
                    sS = 1.4826 * madS
                    if (not np.isfinite(sS)) or sS <= 0.0:
                        sS = float(np.std(rS, ddof=0)) if rS.size else 1.0
                    sP = sP if np.isfinite(sP) and sP > 0 else 1.0
                    sS = sS if np.isfinite(sS) and sS > 0 else 1.0
                    z = np.where(ph == 1, (resid - medS) / float(sS), (resid - medP) / float(sP))
                    absz = np.abs(z)
                    # Pick top groups by count
                    Kmax = int(k.max()) + 1 if k.size else 0
                    cnt = np.bincount(k, minlength=max(Kmax, 0)).astype(np.int64, copy=False) if Kmax > 0 else np.zeros((0,), dtype=np.int64)
                    good = np.nonzero(cnt >= int(min_rows_per_group))[0]
                    if good.size > 0 and max_groups > 0:
                        order = good[np.argsort(-cnt[good], kind="mergesort")]
                        order = order[: int(max_groups)]
                        rows = []
                        for gid in order:
                            m = (k == int(gid))
                            if int(m.sum()) < int(min_rows_per_group):
                                continue
                            tr = _tail_ratio(absz[m], q_hi=0.95, q_lo=0.75)
                            ex = _exceedance_rates(absz[m], thr)
                            rows.append((int(gid), int(m.sum()), float(tr), ex))
                        if rows:
                            _log(f"\nTop station×phase groups by count (tail_ratio + exceedance on |z_mad|):", flush=True)
                            for gid, n0, tr, ex in rows:
                                sta_id = int(gid // 2)
                                ph_id = int(gid % 2)
                                ph_name = "S" if ph_id == 1 else "P"
                                exs = "  ".join([f">{t:g}:{float(ex.get(float(t), float('nan'))):.3g}" for t in thr])
                                _log(f"  sta={sta_id} phase={ph_name} n={n0:,} tail_ratio95/75={tr:.3g} exceed({exs})", flush=True)
                except Exception as e:
                    warn(f"Could not compute station×phase tail summaries: {e}", section="DIAG")

    except Exception as e:
        warn(f"Residual distribution diagnostics failed: {e}", section="DIAG")

    # Optional: scalar residual-correlation metrics (quick CLI diagnostics).
    try:
        cfg = _get_diag_cfg(state, "resid_scalar_metrics")
        if _diag_enabled(cfg, default=False):
            residuals_at = str(cfg.get("residuals_at", "map")).strip().lower()
            if residuals_at not in {"map", "initial", "truth"}:
                residuals_at = "map"
            n_rows = int(cfg.get("n_rows", 200_000))
            seed = int(cfg.get("seed", 0))
            bs = int(cfg.get("batch_size", 50_000))
            sigma_source = str(cfg.get("sigma_source", "quantile")).strip().lower()
            sigma_q = float(cfg.get("sigma_quantile", 0.2))
            min_rows_per_group = int(cfg.get("min_rows_per_group", 200))
            min_rows_per_event = int(cfg.get("min_rows_per_event", 50))
            dist_bins = int(cfg.get("dist_bins", 8))
            min_rows_per_bin = int(cfg.get("min_rows_per_bin", 50))
            residual_variant = str(cfg.get("residual_variant", "base")).strip().lower()

            N = int(state.N)
            if n_rows <= 0 or n_rows >= N:
                rows_np = np.arange(N, dtype=np.int64)
            else:
                rng = np.random.default_rng(int(seed))
                rows_np = rng.choice(N, size=int(n_rows), replace=False).astype(np.int64, copy=False)
                rows_np.sort()

            rows_t = torch.tensor(rows_np, device=state.device, dtype=torch.int64)
            II_all = state.II.index_select(0, rows_t)
            YY_all = state.YY.index_select(0, rows_t)
            ph = YY_all[:, 4].detach().to("cpu").numpy().astype(np.int64, copy=False)

            sta = None
            sta_all_t: Optional[torch.Tensor] = None
            try:
                if getattr(state, "row_station_index", None) is not None:
                    sta_t = state.row_station_index.index_select(0, rows_t)  # type: ignore[union-attr]
                    sta = sta_t.detach().to("cpu").numpy().astype(np.int64, copy=False)
                    sta_all_t = sta_t
            except Exception:
                sta = None
                sta_all_t = None

            if residual_variant in {"with_corr_error", "corr"}:
                residual_variant = "corr_error"
            if residual_variant not in {"base", "corr_error"}:
                residual_variant = "base"

            dX_use = _resolve_dX_use(state, residuals_at)
            r_chunks: list[np.ndarray] = []
            for i0 in range(0, int(II_all.shape[0]), bs):
                i1 = min(i0 + bs, int(II_all.shape[0]))
                II_b = II_all[i0:i1]
                YY_b = YY_all[i0:i1]
                sta_b = (sta_all_t[i0:i1] if isinstance(sta_all_t, torch.Tensor) else None)
                rb = _compute_residuals_numpy(state=state, II_b=II_b, YY_b=YY_b, dX_use=dX_use, sta_b=sta_b, variant=residual_variant)
                r_chunks.append(rb)
            resid = np.concatenate(r_chunks, axis=0) if r_chunks else np.zeros((rows_np.size,), dtype=np.float64)

            keep = np.isfinite(resid) & np.isfinite(ph) & ((ph == 0) | (ph == 1))
            resid = resid[keep]
            ph = ph[keep]
            II_np = II_all.detach().cpu().numpy().astype(np.int64, copy=False)[keep]
            if isinstance(sta, np.ndarray) and sta.shape[0] == keep.shape[0]:
                sta = sta[keep]
            else:
                sta = None

            rP = resid[ph == 0]
            rS = resid[ph == 1]

            def _sigma_from_quantile_np(r: np.ndarray, q: float) -> float:
                if r.size == 0:
                    return float("nan")
                rq = np.quantile(np.abs(r), q)
                return float(rq / 0.6744897501960817)

            try:
                vv = state.params.get("phase_unc", [float("nan"), float("nan")])
                sigma_p_param = float(vv[0])
                sigma_s_param = float(vv[1])
            except Exception:
                sigma_p_param = float("nan")
                sigma_s_param = float("nan")
            if not (np.isfinite(sigma_p_param) and sigma_p_param > 0.0):
                sigma_p_param = float("nan")
            if not (np.isfinite(sigma_s_param) and sigma_s_param > 0.0):
                sigma_s_param = float("nan")

            sigma_source = sigma_source if sigma_source in {"params", "quantile"} else "quantile"
            sigma_p = _sigma_from_quantile_np(rP, sigma_q)
            sigma_s = _sigma_from_quantile_np(rS, sigma_q)
            if not np.isfinite(sigma_p):
                sigma_p = float(_robust_std_from_residuals(torch.from_numpy(rP)) if rP.size > 0 else float("nan"))
            if not np.isfinite(sigma_s):
                sigma_s = float(_robust_std_from_residuals(torch.from_numpy(rS)) if rS.size > 0 else float("nan"))
            if sigma_source == "params" and np.isfinite(sigma_p_param) and np.isfinite(sigma_s_param):
                sigma_p = sigma_p_param
                sigma_s = sigma_s_param

            def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
                if x.size < 8 or y.size < 8:
                    return float("nan")
                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    return float("nan")
                sx = float(np.std(x))
                sy = float(np.std(y))
                if sx <= 0.0 or sy <= 0.0:
                    return float("nan")
                return float(np.corrcoef(x, y)[0, 1])

            def _lin_slope(x: np.ndarray, y: np.ndarray) -> float:
                if x.size < 8 or y.size < 8:
                    return float("nan")
                vx = float(np.var(x))
                if vx <= 0.0:
                    return float("nan")
                return float(np.cov(x, y, ddof=0)[0, 1] / vx)

            def _station_phase_bias_metrics(r: np.ndarray, sta_idx: np.ndarray, sigma: float) -> tuple[float, float]:
                if r.size < 8 or sta_idx is None:
                    return float("nan"), float("nan")
                k = sta_idx.astype(np.int64, copy=False)
                counts = np.bincount(k, minlength=int(k.max()) + 1 if k.size else 0)
                sum_r = np.bincount(k, weights=r, minlength=counts.size)
                m = (counts >= int(min_rows_per_group))
                if not np.any(m):
                    return float("nan"), float("nan")
                mean = np.zeros_like(sum_r, dtype=np.float64)
                mean[m] = sum_r[m] / counts[m]
                z = mean[m] / float(sigma if np.isfinite(sigma) and sigma > 0 else 1.0)
                return float(np.max(np.abs(z))), float(np.sqrt(np.mean(z * z)))

            def _event_bias_metrics(r: np.ndarray, e1: np.ndarray, e2: np.ndarray, sigma: float) -> tuple[float, float]:
                if r.size < 8:
                    return float("nan"), float("nan")
                n_ev = int(state.X_src.shape[0])
                sum_r = np.zeros((n_ev,), dtype=np.float64)
                cnt = np.zeros((n_ev,), dtype=np.int64)
                sum_r[e1] -= r
                sum_r[e2] += r
                cnt[e1] += 1
                cnt[e2] += 1
                m = cnt >= int(min_rows_per_event)
                if not np.any(m):
                    return float("nan"), float("nan")
                mean = np.zeros_like(sum_r, dtype=np.float64)
                mean[m] = sum_r[m] / cnt[m]
                z = mean[m] / float(sigma if np.isfinite(sigma) and sigma > 0 else 1.0)
                return float(np.max(np.abs(z))), float(np.std(z))

            # Compute per-phase metrics
            e1 = II_np[:, 0]
            e2 = II_np[:, 1]

            def _phase_metrics(phase_bit: int, r_phase: np.ndarray, sigma: float) -> dict:
                m = (ph == int(phase_bit))
                if np.count_nonzero(m) < 8:
                    return {}
                e1p = e1[m]; e2p = e2[m]
                rp = r_phase
                # station-phase bias (mean residual per station)
                sp_max, sp_rms = _station_phase_bias_metrics(rp, sta[m] if isinstance(sta, np.ndarray) else None, sigma)
                # event bias (mean signed residual per event)
                ev_max, ev_std = _event_bias_metrics(rp, e1p, e2p, sigma)
                # distance correlation (abs residual vs event separation)
                X = (state.X_src + dX_use)[:, :3].detach().cpu().numpy().astype(np.float64, copy=False)
                dist = np.linalg.norm(X[e2p] - X[e1p], axis=1)
                corr_abs_dist = _pearson_corr(np.abs(rp), dist)
                slope_abs_dist = _lin_slope(dist, np.abs(rp))
                # distance-binned |r| median trend (robust sigma(d) proxy)
                med_slope = float("nan")
                med_intercept = float("nan")
                med_r2 = float("nan")
                med_ratio = float("nan")
                try:
                    if dist_bins < 3:
                        nb = 3
                    else:
                        nb = int(dist_bins)
                    # quantile bins to balance counts
                    edges = np.quantile(dist, np.linspace(0.0, 1.0, nb + 1))
                    edges = np.unique(edges)
                    if edges.size >= 3:
                        meds = []
                        centers = []
                        for i in range(edges.size - 1):
                            lo = edges[i]
                            hi = edges[i + 1]
                            mask = (dist >= lo) & (dist <= hi) if i == edges.size - 2 else (dist >= lo) & (dist < hi)
                            if np.count_nonzero(mask) < int(min_rows_per_bin):
                                continue
                            meds.append(float(np.median(np.abs(rp[mask]))))
                            centers.append(float(0.5 * (lo + hi)))
                        if len(meds) >= 3:
                            x = np.asarray(centers, dtype=np.float64)
                            y = np.asarray(meds, dtype=np.float64)
                            med_slope = _lin_slope(x, y)
                            if np.isfinite(med_slope):
                                med_intercept = float(np.mean(y) - med_slope * np.mean(x))
                                y_hat = med_slope * x + med_intercept
                                ss_res = float(np.sum((y - y_hat) ** 2))
                                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                                if ss_tot > 0:
                                    med_r2 = 1.0 - (ss_res / ss_tot)
                            if len(meds) >= 2 and meds[0] > 0:
                                med_ratio = float(meds[-1] / meds[0])
                except Exception:
                    pass
                # time correlation (residual vs mean event time)
                t_corr = float("nan")
                try:
                    tcol = state.origins0.get_column("time")
                    t_ns = tcol.cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()
                    t0 = float(np.min(t_ns)) if t_ns.size else 0.0
                    tsec = (t_ns - t0).astype(np.float64) / 1e9
                    tpair = 0.5 * (tsec[e1p] + tsec[e2p])
                    t_corr = _pearson_corr(rp, tpair)
                except Exception:
                    t_corr = float("nan")
                return {
                    "sp_bias_max_z": sp_max,
                    "sp_bias_rms_z": sp_rms,
                    "event_bias_max_z": ev_max,
                    "event_bias_std_z": ev_std,
                    "corr_abs_r_dist": corr_abs_dist,
                    "slope_abs_r_dist": slope_abs_dist,
                    "med_slope_abs_r_dist": med_slope,
                    "med_r2_abs_r_dist": med_r2,
                    "med_ratio_abs_r_dist": med_ratio,
                    "corr_r_time": t_corr,
                }

            mP = _phase_metrics(0, rP, sigma_p)
            mS = _phase_metrics(1, rS, sigma_s)

            _log("\nResidual scalar metrics (analyze-resid)", flush=True)
            _log(
                f"- residuals_at={residuals_at} n_rows_used={int(resid.size):,} "
                f"sigma_source={sigma_source} sigma_q={sigma_q}",
                flush=True,
            )
            if mP:
                _log(
                    f"  P: sp_bias_max_z={mP.get('sp_bias_max_z', float('nan')):.3g} "
                    f"sp_bias_rms_z={mP.get('sp_bias_rms_z', float('nan')):.3g} "
                    f"event_bias_max_z={mP.get('event_bias_max_z', float('nan')):.3g} "
                    f"event_bias_std_z={mP.get('event_bias_std_z', float('nan')):.3g} "
                    f"corr(|r|,dist)={mP.get('corr_abs_r_dist', float('nan')):.3g} "
                    f"slope(|r|,dist)={mP.get('slope_abs_r_dist', float('nan')):.3g} "
                    f"med_slope(|r|,dist)={mP.get('med_slope_abs_r_dist', float('nan')):.3g} "
                    f"med_r2={mP.get('med_r2_abs_r_dist', float('nan')):.3g} "
                    f"med_ratio={mP.get('med_ratio_abs_r_dist', float('nan')):.3g} "
                    f"corr(r,time)={mP.get('corr_r_time', float('nan')):.3g}",
                    flush=True,
                )
            if mS:
                _log(
                    f"  S: sp_bias_max_z={mS.get('sp_bias_max_z', float('nan')):.3g} "
                    f"sp_bias_rms_z={mS.get('sp_bias_rms_z', float('nan')):.3g} "
                    f"event_bias_max_z={mS.get('event_bias_max_z', float('nan')):.3g} "
                    f"event_bias_std_z={mS.get('event_bias_std_z', float('nan')):.3g} "
                    f"corr(|r|,dist)={mS.get('corr_abs_r_dist', float('nan')):.3g} "
                    f"slope(|r|,dist)={mS.get('slope_abs_r_dist', float('nan')):.3g} "
                    f"med_slope(|r|,dist)={mS.get('med_slope_abs_r_dist', float('nan')):.3g} "
                    f"med_r2={mS.get('med_r2_abs_r_dist', float('nan')):.3g} "
                    f"med_ratio={mS.get('med_ratio_abs_r_dist', float('nan')):.3g} "
                    f"corr(r,time)={mS.get('corr_r_time', float('nan')):.3g}",
                    flush=True,
                )
    except Exception as e:
        warn(f"Residual scalar metrics failed: {e}", section="DIAG")

    # Optional: shared_event_re tau estimation (method-of-moments).
    try:
        cfg = _get_diag_cfg(state, "shared_event_re_tau")
        if _diag_enabled(cfg, default=False):
            residuals_at = str(cfg.get("residuals_at", "map")).strip().lower()
            if residuals_at not in {"map", "initial", "truth"}:
                residuals_at = "map"
            n_rows = int(cfg.get("n_rows", 200_000))
            seed = int(cfg.get("seed", 0))
            bs = int(cfg.get("batch_size", 50_000))
            sigma_q = float(cfg.get("sigma_quantile", 0.2))
            sigma_source = str(cfg.get("sigma_source", "quantile")).strip().lower()
            min_events_per_cluster = int(cfg.get("min_events_per_cluster", 1))
            min_rows_per_group = int(cfg.get("min_rows_per_group", 500))
            method = str(cfg.get("method", "moments")).strip().lower()
            prefer_hier = cfg.get("hierarchical", None)
            if prefer_hier is None:
                prefer_hier = bool(state.params.get("_shared_event_re_hierarchical", False))

            _log("\nshared_event_re tau estimate (analyze-resid)", flush=True)
            _log(
                f"- residuals_at={residuals_at} n_rows={n_rows:,} seed={seed} batch_size={bs:,} "
                f"sigma_source={sigma_source} sigma_q={sigma_q} method={method}",
                flush=True,
            )

            if bool(prefer_hier):
                est_h = estimate_shared_event_re_hier_tau_s(
                    state=state,
                    n_rows=n_rows,
                    seed=seed,
                    batch_size=bs,
                    residuals_at=residuals_at,
                    sigma_quantile=sigma_q,
                    min_events_per_cluster=min_events_per_cluster,
                )
                if isinstance(est_h, dict):
                    tp = float(est_h.get("tau_event_p", float("nan")))
                    ts = float(est_h.get("tau_event_s", float("nan")))
                    tc_p = float(est_h.get("tau_cluster_p", float("nan")))
                    tc_s = float(est_h.get("tau_cluster_s", float("nan")))
                    sigp = float(est_h.get("sigma_p_est", float("nan")))
                    sigs = float(est_h.get("sigma_s_est", float("nan")))
                    _log(
                        f"  tau_event: P={tp:.4g}s ({1e3*tp:.3g} ms)  S={ts:.4g}s ({1e3*ts:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  tau_cluster: P={tc_p:.4g}s ({1e3*tc_p:.3g} ms)  S={tc_s:.4g}s ({1e3*tc_s:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  sigma_est: P={sigp:.4g}s ({1e3*sigp:.3g} ms)  S={sigs:.4g}s ({1e3*sigs:.3g} ms)",
                        flush=True,
                    )
                else:
                    _log("  (hierarchical) tau estimate skipped: missing cluster ids or feature disabled.", flush=True)
            elif method in {"logdet", "mle", "logdet_small"}:
                # Build tau grid
                grid_raw = cfg.get("tau_grid", None)
                if isinstance(grid_raw, list) and len(grid_raw) > 0:
                    tau_grid = [float(x) for x in grid_raw if np.isfinite(float(x)) and float(x) > 0.0]
                else:
                    tmin = float(cfg.get("tau_grid_min", 1e-3))
                    tmax = float(cfg.get("tau_grid_max", 0.5))
                    tn = int(cfg.get("tau_grid_n", 16))
                    tmin = max(1e-6, tmin)
                    tmax = max(tmin * 1.01, tmax)
                    tn = max(3, tn)
                    tau_grid = list(np.exp(np.linspace(np.log(tmin), np.log(tmax), tn)))
                max_groups = int(cfg.get("logdet_max_groups", 24))
                max_nodes = int(cfg.get("logdet_max_nodes", 400))
                max_edges = int(cfg.get("logdet_max_edges", 20000))
                est = estimate_shared_event_re_tau_logdet(
                    state=state,
                    n_rows=n_rows,
                    seed=seed,
                    batch_size=bs,
                    residuals_at=residuals_at,
                    sigma_source=sigma_source,
                    sigma_quantile=sigma_q,
                    tau_grid=tau_grid,
                    max_groups=max_groups,
                    max_nodes=max_nodes,
                    max_edges=max_edges,
                )
                if isinstance(est, dict):
                    tp = float(est.get("tau_p", float("nan")))
                    ts = float(est.get("tau_s", float("nan")))
                    sigp = float(est.get("sigma_p_used", float("nan")))
                    sigs = float(est.get("sigma_s_used", float("nan")))
                    ngp = int(est.get("n_groups_p", 0))
                    ngs = int(est.get("n_groups_s", 0))
                    _log(
                        f"  tau_logdet: P={tp:.4g}s ({1e3*tp:.3g} ms)  S={ts:.4g}s ({1e3*ts:.3g} ms) "
                        f"(groups P={ngp} S={ngs})",
                        flush=True,
                    )
                    _log(
                        f"  sigma_used[{est.get('sigma_source','?')}]: "
                        f"P={sigp:.4g}s ({1e3*sigp:.3g} ms)  S={sigs:.4g}s ({1e3*sigs:.3g} ms)",
                        flush=True,
                    )
                else:
                    _log("  tau_logdet estimate skipped: insufficient groups or residuals.", flush=True)
            else:
                est = estimate_shared_event_re_tau_s(
                    state=state,
                    n_rows=n_rows,
                    seed=seed,
                    batch_size=bs,
                    residuals_at=residuals_at,
                    sigma_quantile=sigma_q,
                    sigma_source=sigma_source,
                )
                if isinstance(est, dict):
                    tp = float(est.get("tau_p", float("nan")))
                    ts = float(est.get("tau_s", float("nan")))
                    sigp = float(est.get("sigma_p_used", float("nan")))
                    sigs = float(est.get("sigma_s_used", float("nan")))
                    sigp_est = float(est.get("sigma_p_est", float("nan")))
                    sigs_est = float(est.get("sigma_s_est", float("nan")))
                    sigp_param = float(est.get("sigma_p_param", float("nan")))
                    sigs_param = float(est.get("sigma_s_param", float("nan")))
                    sigp_only = float(est.get("resid_std_p", float("nan")))
                    sigs_only = float(est.get("resid_std_s", float("nan")))
                    src = str(est.get("sigma_source", "quantile"))
                    _log(
                        f"  tau: P={tp:.4g}s ({1e3*tp:.3g} ms)  S={ts:.4g}s ({1e3*ts:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  sigma_used[{src}]: P={sigp:.4g}s ({1e3*sigp:.3g} ms)  "
                        f"S={sigs:.4g}s ({1e3*sigs:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  sigma_est(quantile): P={sigp_est:.4g}s ({1e3*sigp_est:.3g} ms)  "
                        f"S={sigs_est:.4g}s ({1e3*sigs_est:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  sigma_param(phase_unc): P={sigp_param:.4g}s ({1e3*sigp_param:.3g} ms)  "
                        f"S={sigs_param:.4g}s ({1e3*sigs_param:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  phase_unc_iid (resid std @ {residuals_at}): P={sigp_only:.4g}s ({1e3*sigp_only:.3g} ms)  "
                        f"S={sigs_only:.4g}s ({1e3*sigs_only:.3g} ms)",
                        flush=True,
                    )
                else:
                    _log("  tau estimate skipped: shared_event_re disabled or no residuals.", flush=True)

            if bool(cfg.get("station_phase_re", False)):
                est_sp = estimate_shared_event_re_station_phase_tau_s(
                    state=state,
                    n_rows=n_rows,
                    seed=seed,
                    batch_size=bs,
                    residuals_at=residuals_at,
                    sigma_quantile=sigma_q,
                    sigma_source=sigma_source,
                    min_rows_per_group=min_rows_per_group,
                )
                if isinstance(est_sp, dict):
                    tp = float(est_sp.get("tau_p", float("nan")))
                    ts = float(est_sp.get("tau_s", float("nan")))
                    sigp = float(est_sp.get("sigma_p_used", float("nan")))
                    sigs = float(est_sp.get("sigma_s_used", float("nan")))
                    sigp_est = float(est_sp.get("sigma_p_est", float("nan")))
                    sigs_est = float(est_sp.get("sigma_s_est", float("nan")))
                    sigp_param = float(est_sp.get("sigma_p_param", float("nan")))
                    sigs_param = float(est_sp.get("sigma_s_param", float("nan")))
                    src = str(est_sp.get("sigma_source", "quantile"))
                    _log(
                        f"  station_phase_re tau: P={tp:.4g}s ({1e3*tp:.3g} ms)  "
                        f"S={ts:.4g}s ({1e3*ts:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  station_phase_re sigma_used[{src}]: P={sigp:.4g}s ({1e3*sigp:.3g} ms)  "
                        f"S={sigs:.4g}s ({1e3*sigs:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  station_phase_re sigma_est(quantile): P={sigp_est:.4g}s ({1e3*sigp_est:.3g} ms)  "
                        f"S={sigs_est:.4g}s ({1e3*sigs_est:.3g} ms)",
                        flush=True,
                    )
                    _log(
                        f"  station_phase_re sigma_param(phase_unc): P={sigp_param:.4g}s ({1e3*sigp_param:.3g} ms)  "
                        f"S={sigs_param:.4g}s ({1e3*sigs_param:.3g} ms)",
                        flush=True,
                    )
    except Exception as e:
        warn(f"shared_event_re tau estimate failed: {e}", section="DIAG")

    # Run the same "end of Phase 1" diagnostics (config-driven).
    # These functions are all safe no-ops when their corresponding features are disabled.
    try:
        from spider.diagnostics.spatial import run_spatial_diag_end_phase1
        run_spatial_diag_end_phase1(state)
    except Exception as e:
        warn(f"Spatial diagnostic failed: {e}", section="DIAG")

    try:
        from spider.diagnostics.pathcorr import run_pathcorr_diag_end_phase1
        run_pathcorr_diag_end_phase1(state)
    except Exception as e:
        warn(f"Path-correlation diagnostic failed: {e}", section="DIAG")

    # Variogram / ell_km diagnostics: for analyze-resid we prefer plotting directly to PNG rather than writing .npz.
    if plot_dir is None:
        try:
            plot_dir = os.path.dirname(os.path.abspath(str(bundle_path))) or "."
        except Exception:
            plot_dir = "."

    # Optional: cross-station correlation for the same event-pair (phase-specific).
    try:
        _maybe_event_pair_station_corr(state=state, plot_dir=str(plot_dir))
    except Exception as e:
        warn(f"event_pair_station_corr failed: {e}", section="DIAG")


