"""
Utilities for building the corr_error event graph.

This module intentionally contains *only* the event-graph logic so it can be used
from both `locate.py` (initialization) and `epoch_runner.py` (periodic refresh)
without creating circular imports.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

import math
import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore


@dataclass(frozen=True)
class CorrErrorGraphStats:
    n_events: int
    radius_km: float
    k: int
    backend: str
    n_directed_edges: int
    n_undirected_edges: int
    mean_undirected_degree: float
    max_undirected_degree: int
    n_events_with_any_neighbor: int


def build_corr_error_dtimes_graph(
    II_evid: np.ndarray,
    *,
    n_events: int,
    k: int,
    X_km: np.ndarray | None = None,
    radius_km: float | None = None,
    seed: int = 0,
    symmetrize: bool = True,
    weighting: str = "count_degree",
    weight_ell_km: float | None = None,
    weight_eps_km: float = 1e-3,
    normalize_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CorrErrorGraphStats]:
    """
    Build a corr_error event graph directly from observed differential-time pairs.

    This constructs an undirected graph whose candidate edges are the unique event pairs that
    appear in the observed `II` tensor (event index pairs per row). This guarantees that every
    graph edge corresponds to at least one observed differential time.

    Options:
      - `k`: cap the number of neighbors per node by selecting the strongest k neighbors.
             "Strength" is primarily the number of observations for that pair (count),
             with an optional distance tiebreak when `X_km` is provided.
      - `radius_km`: optional distance filter (requires `X_km`), dropping observed pairs whose
                     event-event separation exceeds `radius_km`.
      - `weighting`:
          - 'uniform_degree': per-node uniform weights (sum to 1 per node)
          - 'count_degree'  : per-node weights proportional to pair observation count (sum to 1)
          - 'rbf' / 'inv_dist': distance-weighted (requires `X_km`)
    """
    if not (isinstance(II_evid, np.ndarray) and II_evid.ndim == 2 and II_evid.shape[1] >= 2):
        raise ValueError("II_evid must be a numpy array of shape (N, >=2)")
    n_events = int(n_events)
    if n_events <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            CorrErrorGraphStats(
                n_events=0,
                radius_km=float(radius_km) if radius_km is not None else float("nan"),
                k=int(k),
                backend="dtimes_empty",
                n_directed_edges=0,
                n_undirected_edges=0,
                mean_undirected_degree=0.0,
                max_undirected_degree=0,
                n_events_with_any_neighbor=0,
            ),
        )
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")

    weighting = str(weighting or "count_degree").strip().lower()
    if weighting in {"degree", "deg", "uniform"}:
        weighting = "uniform_degree"
    if weighting in {"count", "countdeg", "count_degree"}:
        weighting = "count_degree"
    if weighting not in {"uniform_degree", "count_degree", "rbf", "inv_dist"}:
        raise ValueError(f"Unsupported weighting={weighting!r} (supported: uniform_degree, count_degree, rbf, inv_dist)")

    xyz = None
    if X_km is not None:
        if not (isinstance(X_km, np.ndarray) and X_km.ndim == 2 and X_km.shape[1] >= 3 and int(X_km.shape[0]) == n_events):
            raise ValueError("X_km must be a numpy array of shape (n_events, >=3) when provided")
        xyz = X_km[:, :3].astype(np.float64, copy=False)

    # --- Unique undirected pairs + counts from II ---
    e1 = II_evid[:, 0].astype(np.int64, copy=False)
    e2 = II_evid[:, 1].astype(np.int64, copy=False)
    # Remove obvious invalids/self loops.
    mask = (e1 >= 0) & (e2 >= 0) & (e1 < n_events) & (e2 < n_events) & (e1 != e2)
    if not bool(mask.any()):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            CorrErrorGraphStats(
                n_events=n_events,
                radius_km=float(radius_km) if radius_km is not None else float("nan"),
                k=int(k),
                backend="dtimes_empty",
                n_directed_edges=0,
                n_undirected_edges=0,
                mean_undirected_degree=0.0,
                max_undirected_degree=0,
                n_events_with_any_neighbor=0,
            ),
        )
    e1 = e1[mask]
    e2 = e2[mask]
    u0 = np.minimum(e1, e2)
    v0 = np.maximum(e1, e2)
    pairs = np.stack([u0, v0], axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    # group-by unique rows
    diff = np.any(pairs[1:] != pairs[:-1], axis=1)
    idx0 = np.concatenate([np.array([0], dtype=np.int64), np.where(diff)[0].astype(np.int64) + 1])
    idx1 = np.concatenate([idx0[1:], np.array([pairs.shape[0]], dtype=np.int64)])
    uniq = pairs[idx0]
    counts = (idx1 - idx0).astype(np.int64, copy=False)
    u = uniq[:, 0].astype(np.int64, copy=False)
    v = uniq[:, 1].astype(np.int64, copy=False)

    # Optional distance filter.
    if (radius_km is not None) and math.isfinite(float(radius_km)) and float(radius_km) > 0.0 and (xyz is not None):
        du = xyz[u] - xyz[v]
        d2 = np.sum(du * du, axis=1)
        keep = d2 <= float(radius_km) * float(radius_km)
        u = u[keep]
        v = v[keep]
        counts = counts[keep]

    if u.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            CorrErrorGraphStats(
                n_events=n_events,
                radius_km=float(radius_km) if radius_km is not None else float("nan"),
                k=int(k),
                backend="dtimes_empty",
                n_directed_edges=0,
                n_undirected_edges=0,
                mean_undirected_degree=0.0,
                max_undirected_degree=0,
                n_events_with_any_neighbor=0,
            ),
        )

    # Build adjacency lists with counts (and optional distance for tie-breaking / weighting).
    adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(n_events)]
    if xyz is not None:
        du = xyz[u] - xyz[v]
        dist = np.sqrt(np.sum(du * du, axis=1)).astype(np.float64, copy=False)
    else:
        dist = None
    for i in range(int(u.size)):
        ui = int(u[i]); vi = int(v[i]); ci = int(counts[i])
        di = float(dist[i]) if dist is not None else float("nan")
        adj[ui].append((vi, ci, di))
        adj[vi].append((ui, ci, di))

    rng = np.random.default_rng(int(seed))
    directed_u: List[int] = []
    directed_v: List[int] = []
    directed_w: List[float] = []
    events_with_neighbor = 0

    if weighting == "rbf":
        ell = float(weight_ell_km) if (weight_ell_km is not None) else float("nan")
        if not (xyz is not None and math.isfinite(ell) and ell > 0.0):
            raise ValueError("weight_ell_km must be finite and > 0 and X_km must be provided when weighting='rbf'")
    else:
        ell = float("nan")
    eps = float(weight_eps_km)
    if not (math.isfinite(eps) and eps > 0.0):
        eps = 1e-3
    if (weighting in {"rbf", "inv_dist"}) and (xyz is None):
        raise ValueError("X_km must be provided when weighting is distance-based ('rbf' or 'inv_dist')")

    for i in range(n_events):
        nbrs = adj[i]
        if not nbrs:
            continue
        events_with_neighbor += 1
        # Select up to k neighbors. Prefer higher count; tie-break by smaller distance when available.
        if len(nbrs) > k:
            # Shuffle first so ties don't always pick the same indices.
            rng.shuffle(nbrs)
            nbrs = sorted(nbrs, key=lambda t: (-int(t[1]), float(t[2]) if math.isfinite(float(t[2])) else float("inf")))[:k]

        js = [int(t[0]) for t in nbrs]
        cs = [float(max(1, int(t[1]))) for t in nbrs]
        ds = [float(t[2]) for t in nbrs]

        if weighting == "uniform_degree":
            raw = [1.0] * len(js)
        elif weighting == "count_degree":
            raw = cs
        elif weighting == "rbf":
            raw = [math.exp(-0.5 * (float(d) / float(ell)) ** 2) for d in ds]
        else:  # inv_dist
            raw = [1.0 / (float(d) + float(eps)) for d in ds]

        if normalize_weights:
            s = float(sum(raw))
            if s > 0.0:
                raw = [float(x) / s for x in raw]

        for j, wj in zip(js, raw):
            directed_u.append(i)
            directed_v.append(int(j))
            directed_w.append(float(wj))

    du = np.asarray(directed_u, dtype=np.int64)
    dv = np.asarray(directed_v, dtype=np.int64)
    dw = np.asarray(directed_w, dtype=np.float32)

    if du.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            CorrErrorGraphStats(
                n_events=n_events,
                radius_km=float(radius_km) if radius_km is not None else float("nan"),
                k=int(k),
                backend="dtimes",
                n_directed_edges=0,
                n_undirected_edges=0,
                mean_undirected_degree=0.0,
                max_undirected_degree=0,
                n_events_with_any_neighbor=int(events_with_neighbor),
            ),
        )

    if symmetrize:
        uu = np.minimum(du, dv)
        vv = np.maximum(du, dv)
        pairs2 = np.stack([uu, vv], axis=1)
        mask2 = (pairs2[:, 0] != pairs2[:, 1])
        pairs2 = pairs2[mask2]
        dw2 = dw[mask2]
        if pairs2.size == 0:
            u_uniq = np.zeros((0,), dtype=np.int64)
            v_uniq = np.zeros((0,), dtype=np.int64)
            w_uniq = np.zeros((0,), dtype=np.float32)
        else:
            uniq2, inv = np.unique(pairs2, axis=0, return_inverse=True)
            w_sum = np.zeros((uniq2.shape[0],), dtype=np.float64)
            np.add.at(w_sum, inv, dw2.astype(np.float64))
            u_uniq = uniq2[:, 0].astype(np.int64, copy=False)
            v_uniq = uniq2[:, 1].astype(np.int64, copy=False)
            w_uniq = w_sum.astype(np.float32, copy=False)
    else:
        u_uniq = du
        v_uniq = dv
        w_uniq = dw

    # Degree stats (undirected interpretation).
    if u_uniq.size > 0:
        deg = np.zeros((n_events,), dtype=np.int64)
        np.add.at(deg, u_uniq, 1)
        np.add.at(deg, v_uniq, 1)
        mean_deg = float(deg.mean())
        max_deg = int(deg.max())
    else:
        mean_deg = 0.0
        max_deg = 0

    stats = CorrErrorGraphStats(
        n_events=n_events,
        radius_km=float(radius_km) if radius_km is not None else float("nan"),
        k=int(k),
        backend="dtimes",
        n_directed_edges=int(du.size),
        n_undirected_edges=int(u_uniq.size),
        mean_undirected_degree=float(mean_deg),
        max_undirected_degree=int(max_deg),
        n_events_with_any_neighbor=int(events_with_neighbor),
    )
    return u_uniq, v_uniq, w_uniq, stats


def build_corr_error_radius_graph(
    X_km: np.ndarray,
    *,
    radius_km: float,
    k: int,
    seed: int = 0,
    cell_size_km: float | None = None,
    cell_hops: int = 2,
    max_tries_per_neighbor: int = 64,
    symmetrize: bool = True,
    weighting: str = "uniform_degree",
    weight_ell_km: float | None = None,
    weight_eps_km: float = 1e-3,
    normalize_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CorrErrorGraphStats]:
    """
    Build a sparse "radius-r with capped-k" undirected graph.

    Fast path (preferred):
      If SciPy is available, uses `scipy.spatial.cKDTree` to connect each node to up to k nearest
      neighbors that fall within `radius_km`. This is fast (C/parallel) and avoids enumerating
      dense radius neighborhoods.

    Fallback path:
      If SciPy is unavailable, uses a pure-Python spatial hash grid with rejection sampling.

    Semantics:
      For each node i, repeatedly sample candidates uniformly from a *superset* U(i) of the
      radius neighborhood (via a spatial hash grid). Accept candidates with ||x_i-x_j||<=r.
      This yields (approximately) uniform samples over the true radius neighborhood without
      enumerating all neighbors (critical for dense clusters).

    Returns:
      u, v: int64 arrays of shape (E,) with u[e] < v[e] (undirected edges).
      w: float32 array of shape (E,) of edge weights.
      stats: basic diagnostics
    """
    if not (isinstance(X_km, np.ndarray) and X_km.ndim == 2 and X_km.shape[1] >= 3):
        raise ValueError("X_km must be a numpy array of shape (n_events, >=3)")
    n = int(X_km.shape[0])
    if n <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            CorrErrorGraphStats(
                n_events=0,
                radius_km=float(radius_km),
                k=int(k),
                backend="empty",
                n_directed_edges=0,
                n_undirected_edges=0,
                mean_undirected_degree=0.0,
                max_undirected_degree=0,
                n_events_with_any_neighbor=0,
            ),
        )
    r = float(radius_km)
    if not (math.isfinite(r) and r > 0.0):
        raise ValueError("radius_km must be finite and > 0")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")

    xyz = X_km[:, :3].astype(np.float64, copy=False)
    weighting = str(weighting or "uniform_degree").strip().lower()
    if weighting in {"degree", "deg", "uniform"}:
        weighting = "uniform_degree"
    if weighting not in {"uniform_degree", "rbf", "inv_dist"}:
        raise ValueError(f"Unsupported weighting={weighting!r} (supported: uniform_degree, rbf, inv_dist)")
    if weighting == "rbf":
        ell = float(weight_ell_km) if (weight_ell_km is not None) else float("nan")
        if not (math.isfinite(ell) and ell > 0.0):
            raise ValueError("weight_ell_km must be finite and > 0 when weighting='rbf'")
    else:
        ell = float("nan")
    eps = float(weight_eps_km)
    if not (math.isfinite(eps) and eps > 0.0):
        eps = 1e-3

    # --- Fast path: cKDTree kNN-within-radius ---
    # This is the critical hot path for large catalogs (e.g., 170k+ events).
    if cKDTree is not None:
        if n < 2:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                CorrErrorGraphStats(
                    n_events=n,
                    radius_km=float(r),
                    k=int(k),
                    backend="ckdtree",
                    n_directed_edges=0,
                    n_undirected_edges=0,
                    mean_undirected_degree=0.0,
                    max_undirected_degree=0,
                    n_events_with_any_neighbor=0,
                ),
            )

        tree = cKDTree(xyz)
        kq = int(min(n, int(k) + 1))  # include self, then drop it
        try:
            dists, nbrs = tree.query(xyz, k=kq, workers=-1)  # SciPy >= 1.6
        except TypeError:
            dists, nbrs = tree.query(xyz, k=kq)  # older SciPy fallback

        dists = np.asarray(dists)
        nbrs = np.asarray(nbrs, dtype=np.int64)
        if dists.ndim == 1:
            # kq == 1 (should only happen when n == 1), but be defensive.
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                CorrErrorGraphStats(
                    n_events=n,
                    radius_km=float(r),
                    k=int(k),
                    backend="ckdtree",
                    n_directed_edges=0,
                    n_undirected_edges=0,
                    mean_undirected_degree=0.0,
                    max_undirected_degree=0,
                    n_events_with_any_neighbor=0,
                ),
            )

        # Drop "self" if present as the first neighbor (typical kNN behavior).
        if int(nbrs.shape[1]) >= 1:
            ii = np.arange(n, dtype=np.int64)[:, None]
            if np.all(nbrs[:, 0:1] == ii):
                nbrs = nbrs[:, 1:]
                dists = dists[:, 1:]

        if nbrs.size == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                CorrErrorGraphStats(
                    n_events=n,
                    radius_km=float(r),
                    k=int(k),
                    backend="ckdtree",
                    n_directed_edges=0,
                    n_undirected_edges=0,
                    mean_undirected_degree=0.0,
                    max_undirected_degree=0,
                    n_events_with_any_neighbor=0,
                ),
            )

        # Keep only neighbors within the radius.
        within = dists <= float(r)
        # Ensure we never connect to self (can happen with duplicate points).
        within = within & (nbrs != np.arange(n, dtype=np.int64)[:, None])

        deg = within.sum(axis=1).astype(np.int64, copy=False)  # (n,)
        events_with_neighbor = int((deg > 0).sum())

        # Directed edges (u -> v) for each row's kept neighbors.
        kk = int(nbrs.shape[1])
        u_all = np.repeat(np.arange(n, dtype=np.int64), kk)
        v_all = nbrs.reshape(-1)
        m_all = within.reshape(-1)
        du = u_all[m_all]
        dv = v_all[m_all]
        dd = dists.reshape(-1)[m_all].astype(np.float64, copy=False)

        if du.size == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                CorrErrorGraphStats(
                    n_events=n,
                    radius_km=float(r),
                    k=int(k),
                    backend="ckdtree",
                    n_directed_edges=0,
                    n_undirected_edges=0,
                    mean_undirected_degree=0.0,
                    max_undirected_degree=0,
                    n_events_with_any_neighbor=events_with_neighbor,
                ),
            )

        # Directed weights.
        if weighting == "uniform_degree":
            deg_inv = np.zeros((n,), dtype=np.float64)
            nz = deg > 0
            deg_inv[nz] = 1.0 / deg[nz].astype(np.float64)
            dw = deg_inv[du].astype(np.float32, copy=False)
        elif weighting == "rbf":
            raw = np.exp(-0.5 * (dd / float(ell)) ** 2)
            if normalize_weights:
                s = np.bincount(du, weights=raw, minlength=n).astype(np.float64, copy=False)
                den = s[du]
                den = np.maximum(den, 1e-30)
                raw = raw / den
            dw = raw.astype(np.float32, copy=False)
        else:  # inv_dist
            raw = 1.0 / (dd + float(eps))
            if normalize_weights:
                s = np.bincount(du, weights=raw, minlength=n).astype(np.float64, copy=False)
                den = s[du]
                den = np.maximum(den, 1e-30)
                raw = raw / den
            dw = raw.astype(np.float32, copy=False)

        if symmetrize:
            u = np.minimum(du, dv)
            v = np.maximum(du, dv)
            pairs = np.stack([u, v], axis=1)
            mask = (pairs[:, 0] != pairs[:, 1])
            pairs = pairs[mask]
            dw2 = dw[mask]
            if pairs.size == 0:
                u_uniq = np.zeros((0,), dtype=np.int64)
                v_uniq = np.zeros((0,), dtype=np.int64)
                w_uniq = np.zeros((0,), dtype=np.float32)
            else:
                uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
                w_sum = np.zeros((uniq.shape[0],), dtype=np.float64)
                np.add.at(w_sum, inv, dw2.astype(np.float64))
                u_uniq = uniq[:, 0].astype(np.int64, copy=False)
                v_uniq = uniq[:, 1].astype(np.int64, copy=False)
                w_uniq = w_sum.astype(np.float32, copy=False)
        else:
            u_uniq = du
            v_uniq = dv
            w_uniq = dw

        if u_uniq.size > 0:
            deg_u = np.zeros((n,), dtype=np.int64)
            np.add.at(deg_u, u_uniq, 1)
            np.add.at(deg_u, v_uniq, 1)
            mean_deg = float(deg_u.mean())
            max_deg = int(deg_u.max())
        else:
            mean_deg = 0.0
            max_deg = 0

        stats = CorrErrorGraphStats(
            n_events=n,
            radius_km=float(r),
            k=int(k),
            backend="ckdtree",
            n_directed_edges=int(du.size),
            n_undirected_edges=int(u_uniq.size),
            mean_undirected_degree=float(mean_deg),
            max_undirected_degree=int(max_deg),
            n_events_with_any_neighbor=int(events_with_neighbor),
        )
        return u_uniq, v_uniq, w_uniq, stats

    # Grid for fast uniform candidate sampling without enumerating all within-radius neighbors.
    h = float(cell_size_km) if (cell_size_km is not None) else float(r)
    if not (math.isfinite(h) and h > 0.0):
        h = float(r)

    # Integer cell coordinates.
    cell = np.floor(xyz / h).astype(np.int64, copy=False)  # (n,3)
    buckets: DefaultDict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx in range(n):
        c = cell[idx]
        buckets[(int(c[0]), int(c[1]), int(c[2]))].append(idx)

    rng = np.random.default_rng(int(seed))
    r2 = float(r * r)
    directed_u: List[int] = []
    directed_v: List[int] = []
    directed_w: List[float] = []
    events_with_neighbor = 0

    # Helper to sample a candidate uniformly from union of bucket lists.
    def _sample_from_union(bin_lists: List[List[int]], bin_counts: List[int], total: int) -> int:
        t = int(rng.integers(0, total))
        # Linear scan is fine: at most (2*cell_hops+1)^3 bins (default 125).
        acc = 0
        for lst, cnt in zip(bin_lists, bin_counts):
            acc2 = acc + cnt
            if t < acc2:
                return int(lst[int(rng.integers(0, cnt))])
            acc = acc2
        # Should be unreachable; fallback defensively.
        return int(bin_lists[-1][int(rng.integers(0, bin_counts[-1]))])

    hops = int(max(0, cell_hops))
    deltas = range(-hops, hops + 1)
    for i in range(n):
        ci0, ci1, ci2 = (int(cell[i, 0]), int(cell[i, 1]), int(cell[i, 2]))

        bin_lists: List[List[int]] = []
        bin_counts: List[int] = []
        total = 0
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    key = (ci0 + dx, ci1 + dy, ci2 + dz)
                    lst = buckets.get(key, None)
                    if lst:
                        bin_lists.append(lst)
                        c = int(len(lst))
                        bin_counts.append(c)
                        total += c

        # If the union contains only i (or is empty), there is no possible neighbor.
        if total <= 1:
            continue

        xi0, xi1, xi2 = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])
        want = int(min(k, max(0, total - 1)))
        chosen: set[int] = set()
        tries_left = int(max(1, max_tries_per_neighbor)) * int(want)
        while (len(chosen) < want) and (tries_left > 0):
            tries_left -= 1
            j = _sample_from_union(bin_lists, bin_counts, total)
            if j == i or (j in chosen):
                continue
            dx0 = float(xyz[j, 0]) - xi0
            dy0 = float(xyz[j, 1]) - xi1
            dz0 = float(xyz[j, 2]) - xi2
            if (dx0 * dx0 + dy0 * dy0 + dz0 * dz0) <= r2:
                chosen.add(j)

        if chosen:
            events_with_neighbor += 1
            # Per-node weights (optionally distance-weighted), optionally normalized to sum to 1.
            js = list(chosen)
            if weighting == "uniform_degree":
                denom = float(max(1, len(js)))  # per-node average normalization
                for j in js:
                    directed_u.append(i)
                    directed_v.append(int(j))
                    directed_w.append(float(1.0 / denom))
            else:
                xi0, xi1, xi2 = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])
                dd = []
                for j in js:
                    dx0 = float(xyz[j, 0]) - xi0
                    dy0 = float(xyz[j, 1]) - xi1
                    dz0 = float(xyz[j, 2]) - xi2
                    dd.append(math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0))
                if weighting == "rbf":
                    raw = [math.exp(-0.5 * (d / float(ell)) ** 2) for d in dd]
                else:  # inv_dist
                    raw = [1.0 / (float(d) + float(eps)) for d in dd]
                if normalize_weights:
                    s = float(sum(raw))
                    if s > 0.0:
                        raw = [float(x) / s for x in raw]
                for j, wj in zip(js, raw):
                    directed_u.append(i)
                    directed_v.append(int(j))
                    directed_w.append(float(wj))

    if len(directed_u) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            CorrErrorGraphStats(
                n_events=n,
                radius_km=float(r),
                k=int(k),
                backend="grid",
                n_directed_edges=0,
                n_undirected_edges=0,
                mean_undirected_degree=0.0,
                max_undirected_degree=0,
                n_events_with_any_neighbor=int(events_with_neighbor),
            ),
        )

    du = np.asarray(directed_u, dtype=np.int64)
    dv = np.asarray(directed_v, dtype=np.int64)
    dw = np.asarray(directed_w, dtype=np.float32)

    if symmetrize:
        u = np.minimum(du, dv)
        v = np.maximum(du, dv)
        pairs = np.stack([u, v], axis=1)
        # Remove self loops (should not exist, but be safe).
        mask = (pairs[:, 0] != pairs[:, 1])
        pairs = pairs[mask]
        dw2 = dw[mask]
        if pairs.size == 0:
            u_uniq = np.zeros((0,), dtype=np.int64)
            v_uniq = np.zeros((0,), dtype=np.int64)
            w_uniq = np.zeros((0,), dtype=np.float32)
        else:
            uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
            w_sum = np.zeros((uniq.shape[0],), dtype=np.float64)
            np.add.at(w_sum, inv, dw2.astype(np.float64))
            u_uniq = uniq[:, 0].astype(np.int64, copy=False)
            v_uniq = uniq[:, 1].astype(np.int64, copy=False)
            w_uniq = w_sum.astype(np.float32, copy=False)
    else:
        # Keep directed edges; represent them as "undirected" pairs without unique'ing.
        u_uniq = du
        v_uniq = dv
        w_uniq = dw

    # Degree stats (undirected interpretation).
    if u_uniq.size > 0:
        deg = np.zeros((n,), dtype=np.int64)
        np.add.at(deg, u_uniq, 1)
        np.add.at(deg, v_uniq, 1)
        mean_deg = float(deg.mean())
        max_deg = int(deg.max())
    else:
        mean_deg = 0.0
        max_deg = 0

    stats = CorrErrorGraphStats(
        n_events=n,
        radius_km=float(r),
        k=int(k),
        backend="grid",
        n_directed_edges=int(du.size),
        n_undirected_edges=int(u_uniq.size),
        mean_undirected_degree=float(mean_deg),
        max_undirected_degree=int(max_deg),
        n_events_with_any_neighbor=int(events_with_neighbor),
    )
    return u_uniq, v_uniq, w_uniq, stats

