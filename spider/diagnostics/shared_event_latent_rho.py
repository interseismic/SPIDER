from __future__ import annotations

"""
Shared-event-latent diagnostics: estimate rho_ps (P/S coupling).

Note: this module is imported by `spider.core.analyze_resid`, so it must remain
present in the tracked `spider/` package for clean checkouts.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from spider.core.modeling import compute_residuals

# Reuse the same ridge Laplacian solver used by the event-ell diagnostic.
# This keeps the estimator consistent with other shared_event_latent diagnostics.
from spider.diagnostics.shared_event_latent_event_ell import _solve_event_potentials_subgraph


@dataclass
class SharedEventLatentRhoEstimate:
    """Robust empirical estimate for shared_event_latent P/S coupling rho_ps.

    Interpretation:
      - rho_ps is the correlation coefficient between latent P and S components at a station
        (after projecting residual edges onto per-event "potential" fields).
      - This is an empirical-Bayes tuning diagnostic, not a guaranteed-consistent estimator.
    """

    rho_ps: float
    n_stations_used: int
    n_rows_used: int
    n_rows_used_p: int
    n_rows_used_s: int
    # Station-level stats (for debugging / trust)
    station_rho_median: float
    station_rho_p10: float
    station_rho_p90: float
    events_common_median: float


def _winsorize(x: np.ndarray, *, q: float) -> np.ndarray:
    """Winsorize x in-place-ish by clipping to [q, 1-q] quantiles."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float64, copy=False)
    q = float(q)
    if not (0.0 <= q < 0.5):
        q = 0.0
    if q == 0.0:
        return x
    lo = float(np.nanquantile(x, q))
    hi = float(np.nanquantile(x, 1.0 - q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return x
    return np.clip(x, lo, hi)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    mx = float(np.mean(x))
    my = float(np.mean(y))
    xc = x - mx
    yc = y - my
    vx = float(np.mean(xc * xc))
    vy = float(np.mean(yc * yc))
    if not (np.isfinite(vx) and np.isfinite(vy)) or vx <= 0.0 or vy <= 0.0:
        return float("nan")
    c = float(np.mean(xc * yc) / np.sqrt(vx * vy))
    # Guard tiny numerical excursions
    return float(np.clip(c, -0.999, 0.999))


@torch.no_grad()
def estimate_shared_event_latent_rho_ps(
    *,
    state,
    n_rows: int = 200_000,
    seed: int = 0,
    batch_size: int = 50_000,
    max_stations: int = 256,
    min_edges_per_phase: int = 500,
    max_edges_per_phase: int = 10_000,
    min_events_common: int = 50,
    ridge: float = 1e-3,
    rtol: float = 1e-6,
    maxiter: int = 2000,
    winsor_q: float = 0.01,
) -> Optional[SharedEventLatentRhoEstimate]:
    """
    Estimate rho_ps by:
      1) sampling DD rows and computing MAP residuals r
      2) grouping by station (stable station index)
      3) for each station, fitting a per-event potential g_phi (phi in {P,S}) from edges:
             r ≈ g[e2] - g[e1]
         using a ridge-stabilized Laplacian solve on the station subgraph
      4) computing corr(g_P, g_S) over events that appear in both phase subgraphs
      5) aggregating station correlations robustly via Fisher-z weighting.

    This is designed for *very large* datasets: it never iterates over all dtimes;
    it uses a capped random sample and caps per-station edge counts.
    """
    try:
        N = int(getattr(state, "N", 0))
        if N <= 0:
            return None
        if getattr(state, "row_station_index", None) is None:
            return None
        n_rows = int(max(1, min(int(n_rows), N)))
        batch_size = int(max(1, int(batch_size)))
        max_stations = int(max(1, int(max_stations)))
        min_edges_per_phase = int(max(1, int(min_edges_per_phase)))
        max_edges_per_phase = int(max(min_edges_per_phase, int(max_edges_per_phase)))
        min_events_common = int(max(10, int(min_events_common)))
    except Exception:
        return None

    # Sample row indices (CPU) then batch-evaluate residuals + station/phase labels at MAP.
    rng = np.random.default_rng(int(seed))
    rows_np = rng.choice(N, size=n_rows, replace=False).astype(np.int64, copy=False)
    rows_np.sort()

    e1_all: list[np.ndarray] = []
    e2_all: list[np.ndarray] = []
    r_all: list[np.ndarray] = []
    ph_all: list[np.ndarray] = []
    sta_all: list[np.ndarray] = []

    for i0 in range(0, int(rows_np.size), batch_size):
        ii = rows_np[i0 : i0 + batch_size]
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
    ph = np.concatenate(ph_all, axis=0).astype(np.float32, copy=False)
    sta = np.concatenate(sta_all, axis=0)

    # Filter invalid station indices (can happen if join keys were null and filled with -1)
    msta = sta >= 0
    if not np.all(msta):
        e1 = e1[msta]
        e2 = e2[msta]
        r = r[msta]
        ph = ph[msta]
        sta = sta[msta]

    if sta.size == 0:
        return None

    # Sort by station id for grouping
    order = np.argsort(sta, kind="mergesort")
    sta_s = sta[order]
    e1_s = e1[order]
    e2_s = e2[order]
    r_s = r[order]
    ph_s = ph[order]

    # Station boundaries
    cuts = np.nonzero(sta_s[1:] != sta_s[:-1])[0] + 1
    bounds = np.concatenate([np.array([0], dtype=np.int64), cuts.astype(np.int64, copy=False), np.array([sta_s.size], dtype=np.int64)])

    station_rhos: list[float] = []
    station_ns: list[int] = []
    used_rows_p = 0
    used_rows_s = 0

    # Iterate stations in random order (robustness; avoid bias toward low station ids)
    n_groups = int(bounds.size - 1)
    perm_groups = rng.permutation(n_groups)

    for gi in perm_groups:
        if len(station_rhos) >= max_stations:
            break
        i0 = int(bounds[gi])
        i1 = int(bounds[gi + 1])
        if i1 - i0 < 2 * min_edges_per_phase:
            continue

        e1g = e1_s[i0:i1]
        e2g = e2_s[i0:i1]
        rg = r_s[i0:i1].astype(np.float64, copy=False)
        phg = ph_s[i0:i1]

        mP = (phg < 0.5)
        mS = (phg > 0.5)
        nP = int(np.sum(mP))
        nS = int(np.sum(mS))
        if nP < min_edges_per_phase or nS < min_edges_per_phase:
            continue

        # Subsample per phase to cap cost
        def _subsample(mask: np.ndarray) -> np.ndarray:
            idx = np.nonzero(mask)[0].astype(np.int64, copy=False)
            if idx.size <= max_edges_per_phase:
                return idx
            pick = rng.choice(idx.size, size=max_edges_per_phase, replace=False)
            return idx[pick]

        idxP = _subsample(mP)
        idxS = _subsample(mS)
        used_rows_p += int(idxP.size)
        used_rows_s += int(idxS.size)

        # Solve per-phase potentials on subgraphs (event indices are global; relabel to dense)
        def _solve_phase(idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            e1m = e1g[idxs].astype(np.int64, copy=False)
            e2m = e2g[idxs].astype(np.int64, copy=False)
            rm = rg[idxs].astype(np.float64, copy=False)
            nodes = np.unique(np.concatenate([e1m, e2m], axis=0))
            nodes.sort()
            inv1 = np.searchsorted(nodes, e1m).astype(np.int64, copy=False)
            inv2 = np.searchsorted(nodes, e2m).astype(np.int64, copy=False)
            g = _solve_event_potentials_subgraph(
                n_events=int(nodes.size),
                e1=inv1,
                e2=inv2,
                r=rm,
                ridge=float(ridge),
                rtol=float(rtol),
                maxiter=int(maxiter),
            )
            # Center to remove any residual gauge / shrinkage bias (correlation invariant to shifts)
            if g.size:
                g = g.astype(np.float64, copy=False)
                g = g - float(np.mean(g))
            return nodes.astype(np.int64, copy=False), g.astype(np.float64, copy=False)

        try:
            nodesP, gP = _solve_phase(idxP)
            nodesS, gS = _solve_phase(idxS)
        except Exception:
            continue

        # Intersect events that appear in both phase graphs
        common, iP, iS = np.intersect1d(nodesP, nodesS, assume_unique=True, return_indices=True)
        if int(common.size) < min_events_common:
            continue

        x = gP[iP].astype(np.float64, copy=False)
        y = gS[iS].astype(np.float64, copy=False)

        # Winsorize to reduce sensitivity to a handful of extreme events
        x = _winsorize(x, q=float(winsor_q))
        y = _winsorize(y, q=float(winsor_q))

        # Re-center after winsorization
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))

        rho = _pearson_corr(x, y)
        if not np.isfinite(rho):
            continue

        station_rhos.append(float(rho))
        station_ns.append(int(common.size))

    if not station_rhos:
        return None

    rhos = np.asarray(station_rhos, dtype=np.float64)
    ns = np.asarray(station_ns, dtype=np.int64)
    # Fisher-z aggregation (downweights small-n stations)
    w = np.maximum(ns.astype(np.float64) - 3.0, 1.0)
    z = np.arctanh(np.clip(rhos, -0.999, 0.999))
    zbar = float(np.sum(w * z) / np.sum(w))
    rho_hat = float(np.tanh(zbar))

    def _q(p: float) -> float:
        try:
            return float(np.nanquantile(rhos, p))
        except Exception:
            return float("nan")

    return SharedEventLatentRhoEstimate(
        rho_ps=float(rho_hat),
        n_stations_used=int(rhos.size),
        n_rows_used=int(e1.size),
        n_rows_used_p=int(used_rows_p),
        n_rows_used_s=int(used_rows_s),
        station_rho_median=_q(0.5),
        station_rho_p10=_q(0.1),
        station_rho_p90=_q(0.9),
        events_common_median=float(np.nanmedian(ns.astype(np.float64))),
    )


