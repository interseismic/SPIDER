from __future__ import annotations

from typing import Optional, Any

import os
import torch
import torch.nn as nn
import numpy as np

from spider.io.phase_bundle import load_phase2_bundle
from spider.core.init_state import _build_initial_state
from spider.utils.console import info, warn


def _get_diag_cfg(state, key: str) -> dict:
    try:
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
) -> None:
    plt, ok = _mpl_pyplot()
    if not ok or plt is None:
        raise RuntimeError("matplotlib is not available")
    edges = np.asarray(edges_km, dtype=np.float64)
    C = np.asarray(corr, dtype=np.float64)
    N = np.asarray(count, dtype=np.float64)

    vals = C[np.isfinite(C)]
    if vals.size:
        v = float(np.nanquantile(np.abs(vals), 0.98))
        v = v if np.isfinite(v) and v > 0 else float(np.nanmax(np.abs(vals)))
    else:
        v = 1.0
    v = float(max(v, 0.05))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    X, Y = np.meshgrid(edges, edges, indexing="xy")
    im = ax0.pcolormesh(X, Y, C, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    ax0.set_title(title)
    ax0.set_ylabel("leg2 length (km)")
    ax0.set_aspect("equal")
    fig.colorbar(im, ax=ax0, shrink=0.85, label="corr")

    im2 = ax1.pcolormesh(X, Y, np.log1p(N), cmap="viridis", shading="auto")
    ax1.set_xlabel("leg1 length (km)")
    ax1.set_ylabel("leg2 length (km)")
    fig.colorbar(im2, ax=ax1, shrink=0.85, label="log(1+n_pairs)")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
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


def _empirical_corr_by_distance_station_basis(
    *,
    coords_km: np.ndarray,  # (S,2)
    vals: np.ndarray,       # (S,)
    n_bins: int,
    xmax_km: Optional[float] = None,
) -> Optional[dict]:
    """
    Direct empirical correlation-by-distance for a scalar station value (e.g., mean residual per station).

    We use a *global* mean/variance (over stations) for normalization:
        ρ_bin = E[(x_i-μ)(x_j-μ) | d_ij in bin] / Var(x)

    This avoids per-bin variance estimates that can be very noisy for small station counts.
    """
    try:
        xy = np.asarray(coords_km, dtype=np.float64)
        x = np.asarray(vals, dtype=np.float64)
        m = np.isfinite(x)
        if m.sum() < 3:
            return None
        xy = xy[m]
        x = x[m]
        S = int(x.shape[0])
        if S < 3:
            return None
        mu = float(np.nanmean(x))
        xc = x - mu
        var = float(np.nanmean(xc * xc))
        if (not np.isfinite(var)) or var <= 0.0:
            return None

        # All pairs (upper triangle) — station counts are small so this is cheap.
        di = xy[:, None, :] - xy[None, :, :]
        dmat = np.sqrt(np.maximum(0.0, np.sum(di * di, axis=2)))
        iu, ju = np.triu_indices(S, k=1)
        d = dmat[iu, ju]
        prod = (xc[iu] * xc[ju])

        if xmax_km is not None and float(xmax_km) > 0:
            md = float(xmax_km)
            mm = d <= md
            d = d[mm]
            prod = prod[mm]
        if d.size < 10:
            return None

        dmax = float(np.nanmax(d))
        if (not np.isfinite(dmax)) or dmax <= 0.0:
            return None
        n_bins = int(max(5, int(n_bins)))
        edges = np.linspace(0.0, dmax, n_bins + 1, dtype=np.float64)
        # For plotting alongside an explicit 0-point (ρ(0)=1), it's more readable to place
        # binned estimates at the *right edge* of each bin: x = Δ, 2Δ, ..., dmax.
        # This makes the x-spacing uniform: 0, Δ, 2Δ, ...
        x_axis = edges[1:]
        bi = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)

        sum_prod = np.zeros((n_bins,), dtype=np.float64)
        cnt = np.zeros((n_bins,), dtype=np.float64)
        np.add.at(sum_prod, bi, prod)
        np.add.at(cnt, bi, 1.0)
        mean_prod = sum_prod / np.maximum(cnt, 1.0)
        rho = mean_prod / var
        # No clipping: negative lobes can be real; user asked for direct empirical corr.
        return {
            "centers_km": x_axis.astype(np.float32, copy=False),
            "rho": rho.astype(np.float32, copy=False),
            "count": cnt.astype(np.float32, copy=False),
            "mu": np.asarray([mu], dtype=np.float32),
            "var": np.asarray([var], dtype=np.float32),
        }
    except Exception:
        return None


def _empirical_corr_by_distance_station_basis_joint(
    *,
    coords_km: np.ndarray,  # (S,2)
    p: np.ndarray,          # (S,)
    s: np.ndarray,          # (S,)
    n_bins: int,
    xmax_km: Optional[float] = None,
) -> Optional[dict]:
    """
    Direct empirical correlation-by-distance for a 2D vector station value v=[p,s].

    ρ_bin = E[(v_i-μ)·(v_j-μ) | d_ij in bin] / E[||v-μ||^2]
    """
    try:
        xy = np.asarray(coords_km, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        m = np.isfinite(p) & np.isfinite(s)
        if m.sum() < 3:
            return None
        xy = xy[m]
        V = np.stack([p[m], s[m]], axis=1)
        S0 = int(V.shape[0])
        if S0 < 3:
            return None
        mu = np.nanmean(V, axis=0)
        Vc = V - mu[None, :]
        var = float(np.nanmean(np.sum(Vc * Vc, axis=1)))
        if (not np.isfinite(var)) or var <= 0.0:
            return None

        di = xy[:, None, :] - xy[None, :, :]
        dmat = np.sqrt(np.maximum(0.0, np.sum(di * di, axis=2)))
        iu, ju = np.triu_indices(S0, k=1)
        d = dmat[iu, ju]
        prod = np.sum(Vc[iu] * Vc[ju], axis=1)

        if xmax_km is not None and float(xmax_km) > 0:
            md = float(xmax_km)
            mm = d <= md
            d = d[mm]
            prod = prod[mm]
        if d.size < 10:
            return None

        dmax = float(np.nanmax(d))
        if (not np.isfinite(dmax)) or dmax <= 0.0:
            return None
        n_bins = int(max(5, int(n_bins)))
        edges = np.linspace(0.0, dmax, n_bins + 1, dtype=np.float64)
        x_axis = edges[1:]
        bi = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)

        sum_prod = np.zeros((n_bins,), dtype=np.float64)
        cnt = np.zeros((n_bins,), dtype=np.float64)
        np.add.at(sum_prod, bi, prod)
        np.add.at(cnt, bi, 1.0)
        mean_prod = sum_prod / np.maximum(cnt, 1.0)
        rho = mean_prod / var
        return {
            "centers_km": x_axis.astype(np.float32, copy=False),
            "rho": rho.astype(np.float32, copy=False),
            "count": cnt.astype(np.float32, copy=False),
            "mu": mu.astype(np.float32, copy=False),
            "var": np.asarray([var], dtype=np.float32),
        }
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
    log_bins: bool = False,
) -> Optional[dict]:
    """
    Direct empirical correlation-by-distance using random pair sampling (for large M).

      ρ_bin = E[(x_i-μ)(x_j-μ) | d_ij in bin] / Var(x)

    Bins are fixed over [0, xmax_km] so `n_bins` controls resolution in the plotted range.
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

        if bool(log_bins):
            eps = float(xmax) * 1e-3
            eps = eps if np.isfinite(eps) and eps > 0 else 1e-3
            e1 = np.logspace(np.log10(eps), np.log10(xmax), n_bins, dtype=np.float64)
            edges = np.concatenate([np.array([0.0], dtype=np.float64), e1])
            centers = np.concatenate([np.array([0.5 * edges[1]], dtype=np.float64), np.sqrt(edges[1:-1] * edges[2:])])
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
        return {
            "centers_km": centers.astype(np.float32, copy=False),
            "rho": rho.astype(np.float32, copy=False),
            "count": cnt.astype(np.float32, copy=False),
            "mu": np.asarray([mu], dtype=np.float32),
            "var": np.asarray([var], dtype=np.float32),
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
    except Exception:
        return

    edges_km = _leg_edges_km(max_leg_km=max_leg_km, n_bins=n_bins, log_bins=log_bins)
    n_leg = int(edges_km.size - 1)
    if n_leg <= 0:
        return

    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N_total, size=n_rows, replace=False).astype(np.int64)
    rows_np.sort()

    # Event positions (km) at MAP
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

    # Residuals for sampled rows (observed - predicted)
    from spider.core.modeling import compute_residuals
    r_chunks = []
    for i0 in range(0, int(rows_t_all.numel()), batch_size):
        i1 = min(i0 + batch_size, int(rows_t_all.numel()))
        rt = rows_t_all[i0:i1]
        II_b = state.II.index_select(0, rt)
        YY_b = state.YY.index_select(0, rt)
        rb = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().cpu().numpy().astype(np.float64, copy=False)
        r_chunks.append(rb)
    resid = np.concatenate(r_chunks, axis=0) if r_chunks else np.zeros((rows_np.size,), dtype=np.float64)

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

    # Leg lengths
    dx = x_ev[e2] - x_ev[e1]
    dy = y_ev[e2] - y_ev[e1]
    dz = z_ev[e2] - z_ev[e1]
    h = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64, copy=False)

    max_leg = float(edges_km[-1])
    keep = np.isfinite(resid) & np.isfinite(h) & (h <= max_leg)
    if np.count_nonzero(keep) < 1000:
        warn("shared_event_legcorr2d: too few sampled rows after filtering; skipping.", section="DIAG")
        return
    resid = resid[keep]
    h = h[keep]
    e1 = e1[keep]
    e2 = e2[keep]
    ph = ph[keep]
    sta = sta[keep]

    # Center residuals per group
    if aggregate_phases:
        gid = sta.astype(np.int64)
    else:
        gid = (sta.astype(np.int64) * 2 + ph.astype(np.int64))
    K = int(gid.max()) + 1
    sum_r = np.bincount(gid, weights=resid, minlength=K).astype(np.float64, copy=False)
    cnt_r = np.bincount(gid, minlength=K).astype(np.float64, copy=False)
    mean_r = sum_r / np.maximum(cnt_r, 1.0)
    r0 = (resid - mean_r[gid]).astype(np.float64, copy=False)

    # Build per-(gid, shared_event) entry lists
    entries: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for j in range(int(r0.size)):
        gk = int(gid[j])
        a = int(e1[j]); b = int(e2[j])
        hj = float(h[j])
        rj = float(r0[j])
        entries.setdefault((gk, a), []).append((rj, hj))
        entries.setdefault((gk, b), []).append((-rj, hj))

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
        warn("shared_event_legcorr2d: no pairs picked; skipping.", section="DIAG")
        return

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
        out_png = os.path.join(str(plot_dir), f"shared_event_legcorr2d_{label}.png")
        _plot_legcorr2d(
            corr=corr,
            count=nmat.astype(np.float64, copy=False),
            edges_km=edges_km,
            title=f"Shared-event corr (leg1,leg2) aggregated={label}",
            out_png=str(out_png),
        )
        print(f"Shared-event legcorr2d -> {out_png} (pairs_picked={picked_total:,})", flush=True)

@torch.no_grad()
def analyze_resid_from_bundle(
    *,
    params: dict,
    bundle_path: str,
    model: nn.Module,
    device: torch.device,
    plot_variograms: bool = True,
    plot_dir: Optional[str] = None,
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
    try:
        state.dX_src.data.copy_(bun.dX_src.to(device=state.device, dtype=torch.float32))
    except Exception as e:
        raise RuntimeError(f"Could not restore MAP dX_src from bundle: {e}") from e
    try:
        if state.log_scale_theta is not None and bun.noise_log_scale is not None:
            state.log_scale_theta.data.copy_(bun.noise_log_scale.to(device=state.device, dtype=torch.float32))
    except Exception:
        pass

    info(
        f"analyze-resid: loaded bundle={bundle_path} events={int(state.X_src.shape[0])} dtimes={int(state.N)}",
        section="DIAG",
    )

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

    # Station-basis ell (and variogram) — only meaningful when station_basis is enabled.
    try:
        if bool(state.params.get("_shared_event_latent_enabled", False)) and bool(
            state.params.get("_shared_event_latent_station_basis_enabled", False)
        ):
            cfg = _get_diag_cfg(state, "station_basis_ell_estimate")
            if _diag_enabled(cfg, default=True):
                from spider.diagnostics.station_basis_ell import estimate_station_basis_ell_km

                curves: dict[str, Any] = {}
                est = estimate_station_basis_ell_km(
                    state=state,
                    n_rows=int(cfg.get("n_rows", 50_000)),
                    seed=int(cfg.get("seed", 0)),
                    batch_size=int(cfg.get("batch_size", 50_000)),
                    n_bins=int(cfg.get("n_bins", 20)),
                    frac_of_plateau=float(cfg.get("frac_of_plateau", 0.5)),
                    curves_out=curves,
                )
                if est is not None:
                    state.params["_station_basis_ell_est_p_km"] = float(est.ell_p_km)
                    state.params["_station_basis_ell_est_s_km"] = float(est.ell_s_km)
                    state.params["_station_basis_ell_est_km"] = float(est.ell_km)
                    print(
                        "Station-basis ell_km estimate (analyze-resid): "
                        f"ell_p≈{est.ell_p_km:.3g} km, ell_s≈{est.ell_s_km:.3g} km, recommend≈{est.ell_km:.3g} km",
                        flush=True,
                    )

                    if plot_variograms:
                        # If cfg provides an outfile path, reuse its basename but write PNG instead.
                        out_cfg = cfg.get("outfile", None)
                        if isinstance(out_cfg, str) and out_cfg:
                            base = os.path.splitext(os.path.basename(out_cfg))[0]
                            out_png = os.path.join(str(plot_dir), f"{base}.png")
                        else:
                            out_png = os.path.join(str(plot_dir), "station_basis_variogram.png")
                        try:
                            _plot_variogram_payload(
                                curves=curves,
                                keys=["p", "s"],
                                title="Station-basis variogram (P/S)",
                                out_png=str(out_png),
                            )
                            print(f"Station-basis variogram plot -> {out_png}", flush=True)
                        except Exception as e:
                            warn(f"Could not plot station-basis variogram: {e}", section="DIAG")
                        # For station-basis, use direct empirical correlation-by-distance instead of converting from variogram.
                        try:
                            coords_km = np.asarray(curves.get("coords_km"))
                            mean_p = np.asarray(curves.get("mean_p"))
                            mean_s = np.asarray(curves.get("mean_s"))
                            nb = int(cfg.get("n_bins", 20))
                            # Aggregate P+S by treating (station,phase) as independent samples at the same location.
                            coords_ps = np.concatenate([coords_km, coords_km], axis=0)
                            vals_ps = np.concatenate([mean_p, mean_s], axis=0)
                            corr_ps = _empirical_corr_by_distance_station_basis(coords_km=coords_ps, vals=vals_ps, n_bins=nb)
                            out_acf = os.path.splitext(str(out_png))[0] + "_acf.png"
                            plt, ok = _mpl_pyplot()
                            if not ok or plt is None:
                                raise RuntimeError("matplotlib is not available")
                            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
                            if isinstance(corr_ps, dict):
                                x = np.asarray(corr_ps.get("centers_km", []), dtype=np.float32)
                                rho = np.asarray(corr_ps.get("rho", []), dtype=np.float32)
                                cnt = np.asarray(corr_ps.get("count", np.zeros_like(x)), dtype=np.float32)
                                if x.size and rho.size:
                                    if float(x[0]) > 0.0:
                                        x = np.concatenate([np.array([0.0], dtype=x.dtype), x])
                                        rho = np.concatenate([np.array([1.0], dtype=rho.dtype), rho])
                                        cnt = np.concatenate([np.array([0.0], dtype=cnt.dtype), cnt])
                                    ax0.plot(x, rho, marker="o", linewidth=1.5, markersize=3, label="ps")
                                    ax1.plot(x, cnt, marker="o", linewidth=1.0, markersize=2, label="ps")
                            ax0.set_title("Station-basis empirical correlation by distance (P+S aggregated)")
                            ax0.set_ylabel("empirical corr ρ")
                            ax0.grid(True, alpha=0.3)
                            ax0.axhline(0.0, linewidth=1.0, alpha=0.4)
                            ax0.legend(loc="best", fontsize=9)
                            ax1.set_xlabel("distance (km)")
                            ax1.set_ylabel("pairs/bin")
                            ax1.grid(True, alpha=0.3)
                            os.makedirs(os.path.dirname(str(out_acf)) or ".", exist_ok=True)
                            fig.tight_layout()
                            fig.savefig(str(out_acf), dpi=150)
                            plt.close(fig)
                            print(f"Station-basis empirical corr plot -> {out_acf}", flush=True)

                            # Print a simple fitted "effective length scale" for convenience.
                            try:
                                if isinstance(corr_ps, dict):
                                    ell_ps = _fit_rbf_ell_from_corr(
                                        x_km=corr_ps.get("centers_km", np.array([])),
                                        rho=corr_ps.get("rho", np.array([])),
                                        count=corr_ps.get("count", None),
                                        rho_min=0.01,
                                    )
                                else:
                                    ell_ps = None
                                if ell_ps is not None:
                                    print(
                                        "Station-basis corr fit (RBF): "
                                        f"ell_ps≈{_fmt_km(ell_ps)} km (P+S aggregated; uses bins with rho>0.01)",
                                        flush=True,
                                    )
                                else:
                                    # If rho is ~0 by the first nonzero bin, ell is below that bin width.
                                    try:
                                        x0 = float(np.asarray(corr_ps.get("centers_km"))[0]) if isinstance(corr_ps, dict) else float("nan")
                                    except Exception:
                                        x0 = float("nan")
                                    msg = (
                                        f"Station-basis corr fit (RBF): insufficient positive rho bins; "
                                        f"likely ell << first bin (~{x0:.3g} km)."
                                        if np.isfinite(x0) else
                                        "Station-basis corr fit (RBF): insufficient data to fit ell."
                                    )
                                    print(msg, flush=True)
                            except Exception:
                                pass
                        except Exception as e:
                            warn(f"Could not plot station-basis empirical corr: {e}", section="DIAG")
    except Exception as e:
        warn(f"Station-basis ell/variogram failed: {e}", section="DIAG")

    # Shared-event-latent event ell (and variogram)
    try:
        if bool(state.params.get("_shared_event_latent_enabled", False)):
            cfg = _get_diag_cfg(state, "shared_event_latent_event_ell_estimate")
            if _diag_enabled(cfg, default=True):
                from spider.diagnostics.shared_event_latent_event_ell import estimate_shared_event_latent_event_ell_km

                curves = {}
                est = estimate_shared_event_latent_event_ell_km(
                    state=state,
                    n_rows=int(cfg.get("n_rows", 100_000)),
                    seed=int(cfg.get("seed", 0)),
                    batch_size=int(cfg.get("batch_size", 50_000)),
                    n_bins=int(cfg.get("n_bins", 20)),
                    frac_of_plateau=float(cfg.get("frac_of_plateau", 0.5)),
                    ridge=float(cfg.get("ridge", 1e-3)),
                    rtol=float(cfg.get("rtol", 1e-6)),
                    maxiter=int(cfg.get("maxiter", 2000)),
                    variogram_pairs=int(cfg.get("variogram_pairs", 200_000)),
                    curves_out=curves,
                    max_dist_km=4.0,
                    log_bins=True,
                )
                if est is not None:
                    state.params["_shared_event_latent_event_ell_est_p_km"] = float(est.ell_p_km)
                    state.params["_shared_event_latent_event_ell_est_s_km"] = float(est.ell_s_km)
                    state.params["_shared_event_latent_event_ell_est_km"] = float(est.ell_km)
                    print(
                        "Shared-event-latent event ell_km estimate (analyze-resid): "
                        f"ell_p≈{est.ell_p_km:.3g} km, ell_s≈{est.ell_s_km:.3g} km, recommend≈{est.ell_km:.3g} km",
                        flush=True,
                    )

                    if plot_variograms:
                        out_cfg = cfg.get("outfile", None)
                        if isinstance(out_cfg, str) and out_cfg:
                            base = os.path.splitext(os.path.basename(out_cfg))[0]
                            out_png = os.path.join(str(plot_dir), f"{base}.png")
                        else:
                            out_png = os.path.join(str(plot_dir), "shared_event_latent_event_variogram.png")
                        try:
                            _plot_variogram_payload(
                                curves=curves,
                                keys=["p", "s"],
                                title="Shared-event-latent event variogram (P/S)",
                                out_png=str(out_png),
                                xmax_km=4.0,
                            )
                            print(f"Shared-event-latent event variogram plot -> {out_png}", flush=True)
                        except Exception as e:
                            warn(f"Could not plot shared-event-latent event variogram: {e}", section="DIAG")
                        # For shared-event-latent event, use direct empirical correlation-by-distance
                        # on the fitted per-event potentials g (P/S/joint).
                        try:
                            cd = curves.get("_corr_data", {})
                            nb = int(cfg.get("n_bins", 20))
                            npairs = int(cfg.get("variogram_pairs", 200_000))
                            corr_ps = None
                            if isinstance(cd, dict):
                                dd = cd.get("joint_ps", None)
                                if isinstance(dd, dict):
                                    coords_km = dd.get("coords_km", None)
                                    vals = dd.get("vals", None)
                                    if coords_km is not None and vals is not None:
                                        corr_ps = _empirical_corr_by_distance_sampled(
                                            coords_km=np.asarray(coords_km),
                                            vals=np.asarray(vals),
                                            n_pairs=npairs,
                                            n_bins=nb,
                                            xmax_km=4.0,
                                            seed=int(cfg.get("seed", 0)) + 41,
                                            log_bins=True,
                                        )

                            out_acf = os.path.splitext(str(out_png))[0] + "_acf.png"
                            plt, ok = _mpl_pyplot()
                            if not ok or plt is None:
                                raise RuntimeError("matplotlib is not available")
                            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
                            if isinstance(corr_ps, dict):
                                x = np.asarray(corr_ps.get("centers_km", []), dtype=np.float32)
                                rho = np.asarray(corr_ps.get("rho", []), dtype=np.float32)
                                cnt = np.asarray(corr_ps.get("count", np.zeros_like(x)), dtype=np.float32)
                                if x.size and rho.size:
                                    if float(x[0]) > 0.0:
                                        x = np.concatenate([np.array([0.0], dtype=x.dtype), x])
                                        rho = np.concatenate([np.array([1.0], dtype=rho.dtype), rho])
                                        cnt = np.concatenate([np.array([0.0], dtype=cnt.dtype), cnt])
                                    ax0.plot(x, rho, marker="o", linewidth=1.5, markersize=3, label="ps")
                                    ax1.plot(x, cnt, marker="o", linewidth=1.0, markersize=2, label="ps")
                            ax0.set_title("Shared-event-latent event empirical correlation by distance (P+S aggregated)")
                            ax0.set_ylabel("empirical corr ρ")
                            ax0.grid(True, alpha=0.3)
                            ax0.axhline(0.0, linewidth=1.0, alpha=0.4)
                            ax0.legend(loc="best", fontsize=9)
                            ax0.set_xlim(0.0, 4.0)
                            ax1.set_xlabel("distance (km)")
                            ax1.set_ylabel("pairs/bin")
                            ax1.grid(True, alpha=0.3)
                            os.makedirs(os.path.dirname(str(out_acf)) or ".", exist_ok=True)
                            fig.tight_layout()
                            fig.savefig(str(out_acf), dpi=150)
                            plt.close(fig)
                            print(f"Shared-event-latent event empirical corr plot -> {out_acf}", flush=True)

                            # Print a simple fitted "effective length scale" for convenience.
                            try:
                                if isinstance(corr_ps, dict):
                                    ell_ps = _fit_rbf_ell_from_corr(
                                        x_km=corr_ps.get("centers_km", np.array([])),
                                        rho=corr_ps.get("rho", np.array([])),
                                        count=corr_ps.get("count", None),
                                        rho_min=0.01,
                                    )
                                else:
                                    ell_ps = None
                                if ell_ps is not None:
                                    print(
                                        "Shared-event-latent event corr fit (RBF): "
                                        f"ell_ps≈{_fmt_km(ell_ps)} km "
                                        f"(P+S aggregated; fit range: 0–4 km; uses bins with rho>0.01)",
                                        flush=True,
                                    )
                                else:
                                    try:
                                        x0 = float(np.asarray(corr_ps.get("centers_km"))[0]) if isinstance(corr_ps, dict) else float("nan")
                                    except Exception:
                                        x0 = float("nan")
                                    msg = (
                                        f"Shared-event-latent event corr fit (RBF): insufficient positive rho bins; "
                                        f"likely ell << first bin (~{x0:.3g} km)."
                                        if np.isfinite(x0) else
                                        "Shared-event-latent event corr fit (RBF): insufficient data to fit ell."
                                    )
                                    print(msg, flush=True)
                            except Exception:
                                pass
                        except Exception as e:
                            warn(f"Could not plot shared-event-latent event empirical corr: {e}", section="DIAG")
    except Exception as e:
        warn(f"Shared-event-latent event ell/variogram failed: {e}", section="DIAG")

    try:
        from spider.diagnostics.shared_event_latent_tau import maybe_estimate_shared_event_latent_tau_after_phase1
        maybe_estimate_shared_event_latent_tau_after_phase1(state=state)
    except Exception as e:
        warn(f"Shared-event-latent tau estimate failed: {e}", section="DIAG")

    # Optional: 2D shared-event cross-correlation binned by both leg lengths (h1,h2).
    try:
        _maybe_shared_event_legcorr2d(state=state, plot_dir=str(plot_dir))
    except Exception as e:
        warn(f"shared_event_legcorr2d failed: {e}", section="DIAG")


