"""
Independence-Metropolis resampling for weakly-observed events.

Motivation
----------
Events with only a handful of differential times have posteriors dominated by the
(hierarchical) event prior: kilometers wide. Under SGLD they traverse that width as a slow
random walk (mixing time ~ (posterior width / step)^2), so their sample clouds are lumpy —
which matters when the samples feed density visualizations (KDE fault imaging): an
under-mixed weak event contributes a spuriously tight blob, i.e. fake structure exactly
where the data say least.

Method
------
Every few epochs, for events with at most `max_dtimes` observations:

1. PROPOSE a fresh perturbation from the event's conditional prior, dX_i' ~ N(0, P_k^{-1})
   (P_k = the hierarchical per-cluster precision).
2. ACCEPT/REJECT with Metropolis. For an independence proposal equal to the event prior,
   the prior and proposal densities cancel, leaving the likelihood ratio over the event's
   own rows (cheap: few rows by construction) times the change in the per-component
   centroid prior (moving one event shifts its cluster mean by delta/n_k; the n_k-weighted
   centroid term from compute_prior_loss is reproduced exactly here).
3. On accept the event jumps to the proposal — an (approximately) independent draw from its
   true marginal. On reject it stays.

Weak events that share a differential-time row may not be tested in the same pass (their
acceptance tests would interact), so the weak set is greedily graph-colored on the
"shares a row" adjacency and each sweep processes the colors sequentially. Row sets within
a color are disjoint, so each color needs just one batched residual evaluation for the
proposals (plus one for the current state).

As with the analogous cluster-frame move, the MH likelihood is the independent-Gaussian
data term (`phase_unc`); the shared-event RE term is station/phase-grouped and treated as
invariant to single weak-event moves at this scale.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from spider.core.modeling import compute_residuals_full


@torch.no_grad()
def build_weak_event_cache(
    II: torch.Tensor,
    n_events: int,
    max_dtimes: int,
) -> dict:
    """Identify weak events, greedily color the 'shares a row' adjacency, and precompute
    per-color row sets. Built once per run (the dataset is fixed during sampling)."""
    e1 = II[:, 0].to(torch.int64)
    e2 = II[:, 1].to(torch.int64)
    deg = torch.bincount(e1, minlength=n_events) + torch.bincount(e2, minlength=n_events)
    weak = (deg > 0) & (deg <= int(max_dtimes))
    weak_idx = torch.nonzero(weak).flatten()
    if weak_idx.numel() == 0:
        return {"colors": [], "n_weak": 0}

    # adjacency among weak events: rows whose BOTH endpoints are weak
    both = weak[e1] & weak[e2]
    a = e1[both].cpu().numpy()
    b = e2[both].cpu().numpy()
    adj: dict[int, set] = {}
    for x, y in zip(a.tolist(), b.tolist()):
        adj.setdefault(x, set()).add(y)
        adj.setdefault(y, set()).add(x)

    color_of: dict[int, int] = {}
    n_colors = 1
    for ev in weak_idx.cpu().numpy().tolist():
        used = {color_of[nb] for nb in adj.get(ev, ()) if nb in color_of}
        c = 0
        while c in used:
            c += 1
        color_of[ev] = c
        n_colors = max(n_colors, c + 1)

    device = II.device
    colors = []
    ev_color = torch.full((n_events,), -1, dtype=torch.int64, device=device)
    for ev, c in color_of.items():
        ev_color[ev] = c
    for c in range(n_colors):
        evs = torch.nonzero(ev_color == c).flatten()
        # rows touching an event of this color; owner = which endpoint is being tested.
        own1 = ev_color[e1] == c
        own2 = (ev_color[e2] == c) & ~own1          # coloring guarantees not both, but be safe
        rows = torch.nonzero(own1 | own2).flatten()
        owner_event = torch.where(own1[rows], e1[rows], e2[rows])
        # map owner event id -> position in evs
        pos = torch.full((n_events,), -1, dtype=torch.int64, device=device)
        pos[evs] = torch.arange(evs.numel(), device=device)
        colors.append({"events": evs, "rows": rows, "owner": pos[owner_event]})
    return {"colors": colors, "n_weak": int(weak_idx.numel()), "n_colors": n_colors,
            "degrees": deg, "max_dtimes": int(max_dtimes)}


@torch.no_grad()
def weak_event_mh_sweep(
    *,
    dX: torch.Tensor,                     # (n_ev, 4), modified in place
    X_src: torch.Tensor,
    II: torch.Tensor,
    YY: torch.Tensor,
    model: torch.nn.Module,
    precision: torch.Tensor,              # (4,4) or (K,4,4) hierarchical event-prior precision
    cluster_ids: Optional[torch.Tensor],
    sigma_ps: Sequence[float],
    cache: dict,
    centroid_std: Optional[Sequence[float]] = None,   # per-dim std of the n_k-weighted centroid prior
    batch_size: int = 500_000,
    params: Optional[dict] = None,
) -> dict:
    """One sweep: propose-from-prior + MH for every weak event (color by color)."""
    device = dX.device
    n_ev = int(dX.shape[0])
    if cache.get("n_weak", 0) == 0:
        return {"n_proposed": 0, "n_accepted": 0}

    cid = (cluster_ids.to(torch.int64).reshape(-1) if cluster_ids is not None
           else torch.zeros(n_ev, dtype=torch.int64, device=device))
    K = int(cid.max().item()) + 1
    P = precision.double()
    if P.dim() == 2:
        P = P.unsqueeze(0).expand(K, 4, 4)
    Lch = torch.linalg.cholesky(torch.linalg.inv(P))          # (K,4,4): draws ~ N(0, P^-1)
    n_k = torch.bincount(cid, minlength=K).double()

    use_centroid = centroid_std is not None
    if use_centroid:
        sc2 = torch.tensor([float(s) ** 2 for s in centroid_std], dtype=torch.float64, device=device)
        means = torch.zeros(K, 4, dtype=torch.float64, device=device)
        means.index_add_(0, cid, dX.double())
        means = means / n_k.clamp_min(1.0).unsqueeze(1)

    sp, ss = float(sigma_ps[0]), float(sigma_ps[1])
    n_prop = n_acc = 0
    moved = []
    accepted_events = []
    for col in cache["colors"]:
        evs, rows, owner = col["events"], col["rows"], col["owner"]
        if evs.numel() == 0:
            continue
        ck = cid[evs]
        z = torch.randn(evs.numel(), 4, 1, dtype=torch.float64, device=device)
        prop = (Lch[ck] @ z).squeeze(-1)                       # (n_c, 4)
        delta = prop - dX[evs].double()

        # data term over this color's rows, current vs proposed
        II_s, YY_s = II[rows], YY[rows]
        inv_var = torch.where(YY_s[:, 4] > 0.5,
                              torch.full_like(YY_s[:, 0], 1.0 / (ss * ss)),
                              torch.full_like(YY_s[:, 0], 1.0 / (sp * sp)))
        r_old = compute_residuals_full(II_s, YY_s, X_src, dX, model, batch_size, int(rows.numel()), params=params)
        dX_prop = dX.clone()
        dX_prop[evs] = prop.to(dX.dtype)
        r_new = compute_residuals_full(II_s, YY_s, X_src, dX_prop, model, batch_size, int(rows.numel()), params=params)
        dll = torch.zeros(evs.numel(), dtype=torch.float64, device=device)
        dll.index_add_(0, owner, (-0.5 * (r_new.double() ** 2 - r_old.double() ** 2) * inv_var.double()))

        # n_k-weighted per-component centroid prior: moving one event shifts its cluster
        # mean by delta/n_k; term = n_k * logN(mean; 0, sigma_c) as in compute_prior_loss.
        if use_centroid:
            m0 = means[ck]
            m1 = m0 + delta / n_k[ck].unsqueeze(1)
            dll = dll + (n_k[ck].unsqueeze(1) * (m0**2 - m1**2) / (2.0 * sc2)).sum(dim=1)

        accept = torch.log(torch.rand(evs.numel(), dtype=torch.float64, device=device).clamp_min(1e-300)) < dll
        idx = evs[accept]
        if idx.numel():
            accepted_events.append(idx)
            if use_centroid:
                # keep cluster means current for subsequent colors
                dmean = torch.zeros_like(means)
                dmean.index_add_(0, cid[idx], (prop[accept] - dX[idx].double()))
                means = means + dmean / n_k.clamp_min(1.0).unsqueeze(1)
            moved.append((prop[accept] - dX[idx].double()).norm(dim=1))
            dX[idx] = prop[accept].to(dX.dtype)
        n_prop += int(evs.numel())
        n_acc += int(accept.sum())

    all_moved = torch.cat(moved) if moved else torch.zeros(0, dtype=torch.float64, device=device)
    stats = {
        "n_proposed": n_prop,
        "n_accepted": n_acc,
        "median_jump_m": float(all_moved.median() * 1e3) if all_moved.numel() else 0.0,
    }
    # acceptance by degree bucket: shows which part of the weak population the move actually serves
    deg = cache.get("degrees", None)
    if deg is not None and accepted_events:
        acc_ev = torch.cat(accepted_events)
        d_all = torch.cat([col["events"] for col in cache["colors"]])
        buckets = [(1, 10), (11, 25), (26, 50), (51, 10**9)]
        by = {}
        for lo, hi in buckets:
            m_all = (deg[d_all] >= lo) & (deg[d_all] <= hi)
            m_acc = (deg[acc_ev] >= lo) & (deg[acc_ev] <= hi)
            n_all = int(m_all.sum())
            by[f"{lo}-{min(hi, int(cache['max_dtimes']))}"] = (int(m_acc.sum()), n_all)
        stats["accept_by_degree"] = by
    return stats
