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
        residuals_at: str (default "map")    # "map" | "initial" (ΔX=0)
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
        if residuals_at not in {"map", "initial"}:
            residuals_at = "map"
        out_prefix = str(cfg.get("outfile_prefix", "event_pair_station_corr") or "event_pair_station_corr")
    except Exception:
        return

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
    dX_use = state.dX_src
    if residuals_at == "initial":
        dX_use = torch.zeros_like(state.dX_src, device=state.dX_src.device)

    r_chunks = []
    for i0 in range(0, int(rows_t_all.numel()), batch_size):
        i1 = min(i0 + batch_size, int(rows_t_all.numel()))
        rt = rows_t_all[i0:i1]
        II_b = state.II.index_select(0, rt)
        YY_b = state.YY.index_select(0, rt)
        rb = compute_residuals(II_b, YY_b, state.X_src, dX_use, state.model).detach().cpu().numpy().astype(np.float64, copy=False)
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
        print(f"Event-pair station corr -> {out_png} (phase={phase_name}, stations={K})", flush=True)

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
            from spider.core.modeling import compute_residuals

            residuals_at = str(cfg.get("residuals_at", "map")).strip().lower()
            if residuals_at not in {"map", "initial"}:
                residuals_at = "map"
            n_rows = int(cfg.get("n_rows", 200_000))
            seed = int(cfg.get("seed", 0))
            bs = int(cfg.get("batch_size", 50_000))
            max_groups = int(cfg.get("max_groups", 16))
            min_rows_per_group = int(cfg.get("min_rows_per_group", 500))
            make_plots = bool(cfg.get("make_plots", True))
            qq_sigma = bool(cfg.get("qq_sigma", True))
            qq_mad = bool(cfg.get("qq_mad", True))
            qq_huber = bool(cfg.get("qq_huber", True))
            by_station_phase = bool(cfg.get("by_station_phase", True))
            exceedance_thresholds = cfg.get("exceedance_thresholds", [3, 5, 10])
            huber_fit = bool(cfg.get("huber_fit", True))
            huber_k_min = float(cfg.get("huber_k_min", 0.2))
            huber_k_max = float(cfg.get("huber_k_max", 8.0))
            huber_k_n = int(cfg.get("huber_k_n", 240))

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
            try:
                if getattr(state, "row_station_index", None) is not None:
                    sta_t = state.row_station_index.index_select(0, rows_t)  # type: ignore[union-attr]
                    sta = sta_t.detach().to("cpu").numpy().astype(np.int64, copy=False)
            except Exception:
                sta = None

            # Sigma scales (for z=r/sigma standardization)
            sigma_p: Optional[float]
            sigma_s: Optional[float]
            try:
                if getattr(state, "log_scale_theta", None) is not None:
                    sig_ps = torch.exp(state.log_scale_theta.detach().to("cpu")).numpy().astype(np.float64, copy=False)  # type: ignore[union-attr]
                    sigma_p = float(sig_ps[0]); sigma_s = float(sig_ps[1])
                else:
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
            dX_use = torch.zeros_like(state.dX_src) if residuals_at == "initial" else state.dX_src
            for i0 in range(0, int(II_all.shape[0]), bs):
                i1 = min(i0 + bs, int(II_all.shape[0]))
                II_b = II_all[i0:i1]
                YY_b = YY_all[i0:i1]
                rb = compute_residuals(II_b, YY_b, state.X_src, dX_use, state.model).detach().to("cpu").numpy().astype(np.float64, copy=False)
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

            print("\nResidual distribution diagnostics (analyze-resid)", flush=True)
            print(f"- residuals_at={residuals_at} n_rows_used={int(resid.size):,}", flush=True)
            print(f"- sigma_p={sigma_p} sigma_s={sigma_s}", flush=True)

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
                        print(f"  Huber fit on z_mad: k_hat(P)={kP:.3g}  k_hat(S)={kS:.3g}", flush=True)
                        if np.isfinite(sP_all) and np.isfinite(sS_all) and sP_all > 0 and sS_all > 0:
                            dP = float(kP) * float(sP_all)
                            dS = float(kS) * float(sS_all)
                            print(
                                f"  z_mad scale (sec): P={float(sP_all):.3g} (med={float(medP_all):.3g})  "
                                f"S={float(sS_all):.3g} (med={float(medS_all):.3g})",
                                flush=True,
                            )
                            print(
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
                    print(f"  {lab}: n=0", flush=True)
                    return
                tr = s.get("z_mad_tail_ratio_95_75", float("nan"))
                fit = s.get("z_mad_fit", {})
                print(
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

            print(f"  P exceedance: {_fmt_exceed(exP)}", flush=True)
            print(f"  S exceedance: {_fmt_exceed(exS)}", flush=True)

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
                        print(f"Residual QQ plots -> {str(plot_dir)}", flush=True)
                except Exception as e:
                    warn(f"Could not plot residual QQ: {e}", section="DIAG")

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
                            print(f"\nTop station×phase groups by count (tail_ratio + exceedance on |z_mad|):", flush=True)
                            for gid, n0, tr, ex in rows:
                                sta_id = int(gid // 2)
                                ph_id = int(gid % 2)
                                ph_name = "S" if ph_id == 1 else "P"
                                exs = "  ".join([f">{t:g}:{float(ex.get(float(t), float('nan'))):.3g}" for t in thr])
                                print(f"  sta={sta_id} phase={ph_name} n={n0:,} tail_ratio95/75={tr:.3g} exceed({exs})", flush=True)
                except Exception as e:
                    warn(f"Could not compute station×phase tail summaries: {e}", section="DIAG")

    except Exception as e:
        warn(f"Residual distribution diagnostics failed: {e}", section="DIAG")

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
                # For event-space diagnostics we prefer equal-count (quantile) bins by default
                # so tail bins remain stable (log/linear bins can have very few pairs at long distances).
                try:
                    bmode = str(cfg.get("binning", "equal_count")).strip().lower()
                except Exception:
                    bmode = "equal_count"
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
                    binning=str(bmode),
                )
                if est is not None:
                    state.params["_shared_event_latent_event_ell_est_p_km"] = float(est.ell_p_km)
                    state.params["_shared_event_latent_event_ell_est_s_km"] = float(est.ell_s_km)
                    # Default recommendation is the variogram-based joint estimate.
                    ell_recommend = float(est.ell_km)
                    ell_recommend_src = "variogram_half_plateau"

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
                                            binning=str(bmode),
                                        )

                            out_acf = os.path.splitext(str(out_png))[0] + "_acf.png"
                            # Default: plot per-station × phase curves (mixing stations can wash out structure).
                            try:
                                acf_grouping = str(cfg.get("acf_grouping", "per_station_phase")).strip().lower()
                            except Exception:
                                acf_grouping = "per_station_phase"

                            if acf_grouping in {"agg", "aggregate", "aggregated", "global"}:
                                if isinstance(corr_ps, dict):
                                    _plot_empirical_corr_payload(
                                        corr=corr_ps,
                                        title="Shared-event-latent event empirical correlation by distance (P+S aggregated)",
                                        out_png=str(out_acf),
                                        xmax_km=4.0,
                                    )
                                    print(f"Shared-event-latent event empirical corr plot -> {out_acf}", flush=True)
                            else:
                                # Per-station × phase: solve g(e) per station-phase from that station's DD residual edges.
                                from spider.core.modeling import compute_residuals
                                from spider.diagnostics.shared_event_latent_event_ell import _solve_event_potentials_subgraph

                                # Knobs
                                xmax_km = float(cfg.get("acf_xmax_km", 4.0))
                                max_stations_plot = int(cfg.get("acf_max_stations_plot", 32))
                                min_edges = int(cfg.get("acf_min_edges_per_phase", 500))
                                max_edges = int(cfg.get("acf_max_edges_per_phase", 10_000))
                                n_rows_acf = int(cfg.get("acf_n_rows", cfg.get("n_rows", 100_000)))
                                bs_acf = int(cfg.get("acf_batch_size", cfg.get("batch_size", 50_000)))
                                n_pairs_station = int(cfg.get("acf_pairs_per_station", 80_000))
                                seed0 = int(cfg.get("seed", 0))

                                Ntot = int(getattr(state, "N", 0))
                                if getattr(state, "row_station_index", None) is None or int(getattr(state, "n_stations", 0)) <= 0:
                                    raise RuntimeError("Missing row_station_index/n_stations; cannot do per-station ACF.")
                                n_rows_acf = int(max(1, min(n_rows_acf, Ntot)))
                                bs_acf = int(max(1, bs_acf))
                                max_stations_plot = int(max(1, max_stations_plot))
                                min_edges = int(max(10, min_edges))
                                max_edges = int(max(min_edges, max_edges))
                                n_pairs_station = int(max(1_000, n_pairs_station))

                                rng = np.random.default_rng(int(seed0) + 123)
                                rows_np = rng.choice(Ntot, size=n_rows_acf, replace=False).astype(np.int64, copy=False)
                                rows_np.sort()

                                e1_all = []
                                e2_all = []
                                r_all = []
                                ph_all = []
                                sta_all = []
                                for i0 in range(0, int(rows_np.size), bs_acf):
                                    ii = rows_np[i0 : i0 + bs_acf]
                                    t = torch.from_numpy(ii).to(device=state.device, dtype=torch.int64)
                                    II_b = state.II.index_select(0, t)
                                    YY_b = state.YY.index_select(0, t)
                                    resid = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach()
                                    ph = YY_b[:, 4].detach()
                                    sta = state.row_station_index.index_select(0, t).detach()
                                    e1_all.append(II_b[:, 0].detach().to("cpu", dtype=torch.int64).numpy())
                                    e2_all.append(II_b[:, 1].detach().to("cpu", dtype=torch.int64).numpy())
                                    r_all.append(resid.detach().to("cpu", dtype=torch.float32).numpy())
                                    ph_all.append(ph.detach().to("cpu", dtype=torch.float32).numpy())
                                    sta_all.append(sta.detach().to("cpu", dtype=torch.int64).numpy())

                                e1 = np.concatenate(e1_all, axis=0)
                                e2 = np.concatenate(e2_all, axis=0)
                                r = np.concatenate(r_all, axis=0).astype(np.float32, copy=False)
                                phv = np.concatenate(ph_all, axis=0).astype(np.float32, copy=False)
                                sta = np.concatenate(sta_all, axis=0).astype(np.int64, copy=False)

                                # Filter invalid stations
                                msta = sta >= 0
                                if not np.all(msta):
                                    e1 = e1[msta]; e2 = e2[msta]; r = r[msta]; phv = phv[msta]; sta = sta[msta]
                                if sta.size == 0:
                                    raise RuntimeError("No sampled rows left after station filtering")

                                # Event XY at MAP (CPU)
                                XY = (state.X_src + state.dX_src).detach().cpu()[:, 0:2].to(torch.float32)

                                # Station labels (best-effort)
                                sta_label = {}
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

                                # Choose top stations by sampled row count
                                uniq_sta, counts_sta = np.unique(sta, return_counts=True)
                                ord_sta = np.argsort(-counts_sta)
                                uniq_sta = uniq_sta[ord_sta][: int(min(max_stations_plot, uniq_sta.size))]

                                curves_multi: list[dict] = []
                                for s_id in uniq_sta.tolist():
                                    s_id = int(s_id)
                                    m_s = (sta == s_id)
                                    if int(m_s.sum()) < 2 * min_edges:
                                        continue
                                    # Per-phase masks
                                    mP = m_s & (phv < 0.5)
                                    mS = m_s & (phv > 0.5)

                                    def _do_phase(mask: np.ndarray, phase_label: str, seed_off: int) -> None:
                                        n0 = int(np.sum(mask))
                                        if n0 < min_edges:
                                            return
                                        idx = np.nonzero(mask)[0].astype(np.int64, copy=False)
                                        if idx.size > max_edges:
                                            pick = rng.choice(idx.size, size=max_edges, replace=False)
                                            idx = idx[pick]
                                        nodes = np.unique(np.concatenate([e1[idx], e2[idx]], axis=0))
                                        if int(nodes.size) < 10:
                                            return
                                        nodes.sort()
                                        inv1 = np.searchsorted(nodes, e1[idx]).astype(np.int64, copy=False)
                                        inv2 = np.searchsorted(nodes, e2[idx]).astype(np.int64, copy=False)
                                        g = _solve_event_potentials_subgraph(
                                            n_events=int(nodes.size),
                                            e1=inv1,
                                            e2=inv2,
                                            r=r[idx].astype(np.float64, copy=False),
                                            ridge=float(cfg.get("ridge", 1e-3)),
                                            rtol=float(cfg.get("rtol", 1e-6)),
                                            maxiter=int(cfg.get("maxiter", 2000)),
                                        )
                                        coords = XY.index_select(0, torch.from_numpy(nodes).to(torch.int64)).cpu().numpy()
                                        corr = _empirical_corr_by_distance_sampled(
                                            coords_km=np.asarray(coords),
                                            vals=np.asarray(g),
                                            n_pairs=int(n_pairs_station),
                                            n_bins=int(nb),
                                            xmax_km=float(xmax_km),
                                            seed=int(seed0) + int(seed_off) + int(s_id),
                                            binning=str(bmode),
                                        )
                                        if isinstance(corr, dict):
                                            lab0 = sta_label.get(s_id, f"sta{s_id}")
                                            corr = dict(corr)
                                            corr["label"] = f"{lab0}:{phase_label}"
                                            curves_multi.append(corr)

                                    _do_phase(mP, "P", 1009)
                                    _do_phase(mS, "S", 2009)

                                if not curves_multi:
                                    raise RuntimeError("No station-phase curves computed; increase acf_n_rows or relax thresholds")

                                # If there are too many curves, hide legend (it becomes unreadable).
                                _plot_empirical_corr_payload_multi(
                                    curves=curves_multi,
                                    title="Shared-event-latent event empirical corr by distance (per station × phase)",
                                    out_png=str(out_acf),
                                    xmax_km=float(xmax_km),
                                    show_legend=bool(len(curves_multi) <= 12),
                                )
                                print(f"Shared-event-latent event empirical corr plot (per station×phase) -> {out_acf}", flush=True)

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
                                    # Optional: let the *recommendation* come from the same ACF curve.
                                    # This is useful when users prefer reasoning about correlation decay
                                    # directly rather than semivariogram plateau heuristics.
                                    try:
                                        rec = str(cfg.get("recommend_from", "variogram")).strip().lower()
                                    except Exception:
                                        rec = "variogram"
                                    if rec in {"acf", "corr", "acf_rbf", "corr_rbf", "acf_rbf_fit"}:
                                        ell_recommend = float(ell_ps)
                                        ell_recommend_src = "acf_rbf_fit"
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
                    # Materialize recommendation (after optional ACF fit override above).
                    state.params["_shared_event_latent_event_ell_est_km"] = float(ell_recommend)
                    state.params["_shared_event_latent_event_ell_est_method"] = str(ell_recommend_src)
                    print(
                        "Shared-event-latent event ell_km estimate (analyze-resid): "
                        f"ell_p≈{est.ell_p_km:.3g} km, ell_s≈{est.ell_s_km:.3g} km, "
                        f"recommend≈{ell_recommend:.3g} km (method={ell_recommend_src})",
                        flush=True,
                    )
    except Exception as e:
        warn(f"Shared-event-latent event ell/variogram failed: {e}", section="DIAG")

    try:
        from spider.diagnostics.shared_event_latent_tau import maybe_estimate_shared_event_latent_tau_after_phase1
        maybe_estimate_shared_event_latent_tau_after_phase1(state=state)
    except Exception as e:
        warn(f"Shared-event-latent tau estimate failed: {e}", section="DIAG")

    # Optional: estimate shared_event_latent rho_ps (P/S coupling) from MAP residual structure.
    # This is an empirical tuning hint (writes a small JSON payload to plot_dir).
    try:
        if bool(state.params.get("_shared_event_latent_enabled", False)):
            cfg = _get_diag_cfg(state, "shared_event_latent_rho_ps_estimate")
            if _diag_enabled(cfg, default=True):
                from spider.diagnostics.shared_event_latent_rho import estimate_shared_event_latent_rho_ps
                import json as _json

                est = estimate_shared_event_latent_rho_ps(
                    state=state,
                    n_rows=int(cfg.get("n_rows", 200_000)),
                    seed=int(cfg.get("seed", 0)),
                    batch_size=int(cfg.get("batch_size", 50_000)),
                    max_stations=int(cfg.get("max_stations", 256)),
                    min_edges_per_phase=int(cfg.get("min_edges_per_phase", 500)),
                    max_edges_per_phase=int(cfg.get("max_edges_per_phase", 10_000)),
                    min_events_common=int(cfg.get("min_events_common", 50)),
                    ridge=float(cfg.get("ridge", 1e-3)),
                    rtol=float(cfg.get("rtol", 1e-6)),
                    maxiter=int(cfg.get("maxiter", 2000)),
                    winsor_q=float(cfg.get("winsor_q", 0.01)),
                )
                if est is not None:
                    state.params["_shared_event_latent_rho_ps_estimate"] = float(est.rho_ps)
                    print(
                        "Shared-event rho_ps estimate (analyze-resid): "
                        f"rho_ps≈{est.rho_ps:.3f} "
                        f"(stations={est.n_stations_used}, station_rho p10/p50/p90={est.station_rho_p10:.3f}/{est.station_rho_median:.3f}/{est.station_rho_p90:.3f}, "
                        f"median common events={est.events_common_median:.0f})",
                        flush=True,
                    )
                    try:
                        out_cfg = cfg.get("outfile", None)
                        if isinstance(out_cfg, str) and out_cfg:
                            out_path = str(out_cfg)
                        else:
                            out_path = os.path.join(str(plot_dir), "shared_event_latent_rho_ps_estimate.json")
                        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                        with open(out_path, "w") as f:
                            _json.dump(
                                {
                                    "rho_ps_estimate": float(est.rho_ps),
                                    "n_stations_used": int(est.n_stations_used),
                                    "n_rows_used": int(est.n_rows_used),
                                    "n_rows_used_p": int(est.n_rows_used_p),
                                    "n_rows_used_s": int(est.n_rows_used_s),
                                    "station_rho_p10": float(est.station_rho_p10),
                                    "station_rho_median": float(est.station_rho_median),
                                    "station_rho_p90": float(est.station_rho_p90),
                                    "events_common_median": float(est.events_common_median),
                                    "config_hint": {"model": {"likelihood": {"shared_event_latent": {"rho_ps": float(est.rho_ps)}}}},
                                },
                                f,
                                indent=2,
                            )
                        print(f"Shared-event rho_ps estimate -> {out_path}", flush=True)
                    except Exception as e:
                        warn(f"Could not write rho_ps estimate JSON: {e}", section="DIAG")
    except Exception as e:
        warn(f"Shared-event-latent rho_ps estimate failed: {e}", section="DIAG")

    # Optional: 2D shared-event cross-correlation binned by both leg lengths (h1,h2).
    try:
        _maybe_shared_event_legcorr2d(state=state, plot_dir=str(plot_dir))
    except Exception as e:
        warn(f"shared_event_legcorr2d failed: {e}", section="DIAG")

    # Optional: cross-station correlation for the same event-pair (phase-specific).
    try:
        _maybe_event_pair_station_corr(state=state, plot_dir=str(plot_dir))
    except Exception as e:
        warn(f"event_pair_station_corr failed: {e}", section="DIAG")


