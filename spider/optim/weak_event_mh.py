"""
Metropolis resampling of weakly-observed events with a conditional-Gaussian proposal.

Motivation
----------
Events with only a handful of differential times have posteriors bounded by the (hierarchical) event
prior: hundreds of metres to kilometres wide. Under pSGLD with the RMSprop metric their step is set
by the `eps` floor (tens of metres), so they traverse that width as a slow random walk: on Kumamoto
v1 the integrated autocorrelation time of events with 1-10 rows was 600-2500 SGLD steps against
30-80 for well-observed events. Their sample clouds then look lumpy exactly where the data say
least, which matters when the samples feed density imaging.

Method
------
Every `every_epochs` epochs (Phase 4), for every event with at most `max_dtimes` rows:

1. FIT the event's conditional posterior given the *current* positions of its partners: a few
   Gauss-Newton iterations on 0.5 sum_r w_r r_r(x)^2 + 0.5 x^T P_k x over the event's own rows,
   started from a fixed point (the shift at the first sweep), giving the conditional mode m_i and
   the Laplace precision H_i = F_i + P_k (F_i = sum_r w_r J_r J_r^T).
2. PROPOSE x' ~ t_nu(m_i, s^2 H_i^{-1}) (multivariate Student-t, `t_dof` degrees of freedom, scale
   inflation `scale_inflation`; `t_dof = null` gives a Gaussian). The heavier tails and the
   inflation keep the independence sampler robust where the Laplace fit is narrower than the target.
3. ACCEPT with the independence-Metropolis ratio
       pi(x') q(x) / (pi(x) q(x')),
   pi = data term over the event's rows x event prior x per-component centroid prior (when on).
   The proposal depends only on the partners' current state, the current prior and the fixed start
   point, never on the event's current value, so the move is exact.

Weak events that share a row are never tested in the same pass (greedy graph colouring on the
shares-a-row adjacency); within a colour the row sets are disjoint and everything is batched.
Cost per sweep: (gn_iters + 3) residual/Jacobian evaluations on the weak rows only (0.3 % of the
data on Kumamoto).

The data term is the independent Gaussian with the scales the caller passes in `sigma_ps`; with the
shared-event random-effect likelihood the epoch runner passes sqrt(sigma^2 + 2 tau^2), the
batch-marginal variance of a weak event's row (the effect is not identifiable from one row per
batch), which is what the SGLD gradient effectively applies to these rows.

`proposal = "prior"` keeps the previous independence proposal from the event prior N(0, P_k^{-1})
(acceptance is the posterior-to-prior volume ratio; on Kumamoto ~2.5 %).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch

_D = 4


@torch.no_grad()
def build_weak_event_cache(
    II: torch.Tensor,
    n_events: int,
    max_dtimes: int,
    dX0: Optional[torch.Tensor] = None,
) -> dict:
    """Identify weak events, greedily colour the 'shares a row' adjacency, and precompute per-colour
    row sets. Built once per run (the dataset is fixed during sampling). `dX0` (the shifts at build
    time) is stored as the fixed start point of the conditional fits."""
    e1 = II[:, 0].to(torch.int64)
    e2 = II[:, 1].to(torch.int64)
    deg = torch.bincount(e1, minlength=n_events) + torch.bincount(e2, minlength=n_events)
    weak = (deg > 0) & (deg <= int(max_dtimes))
    weak_idx = torch.nonzero(weak).flatten()
    if weak_idx.numel() == 0:
        return {"colors": [], "n_weak": 0, "n_colors": 0, "degrees": deg, "max_dtimes": int(max_dtimes)}

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
        own1 = ev_color[e1] == c
        own2 = (ev_color[e2] == c) & ~own1          # colouring guarantees not both, but be safe
        rows = torch.nonzero(own1 | own2).flatten()
        is_first = own1[rows]
        owner_event = torch.where(is_first, e1[rows], e2[rows])
        pos = torch.full((n_events,), -1, dtype=torch.int64, device=device)
        pos[evs] = torch.arange(evs.numel(), device=device)
        colors.append({"events": evs, "rows": rows, "owner": pos[owner_event], "is_first": is_first})
    out = {"colors": colors, "n_weak": int(weak_idx.numel()), "n_colors": n_colors,
           "degrees": deg, "max_dtimes": int(max_dtimes)}
    if isinstance(dX0, torch.Tensor):
        out["start_dX"] = dX0.detach().clone()
    return out


def _predict_rows(
    II_s: torch.Tensor,
    YY_s: torch.Tensor,
    X_src: torch.Tensor,
    dX: torch.Tensor,
    model: torch.nn.Module,
    x_rows: torch.Tensor,
    is_first: torch.Tensor,
) -> torch.Tensor:
    """Predicted differential times of the rows, with the tested endpoint's shift taken from `x_rows`
    (one row per dtime row, differentiable) and the partner's from the current `dX`."""
    e1 = II_s[:, 0].to(torch.int64)
    e2 = II_s[:, 1].to(torch.int64)
    src1 = X_src.index_select(0, e1) + dX.index_select(0, e1)
    src2 = X_src.index_select(0, e2) + dX.index_select(0, e2)
    base_t = torch.where(is_first.unsqueeze(1), X_src.index_select(0, e1), X_src.index_select(0, e2))
    src_t = base_t + x_rows
    f = is_first.unsqueeze(1)
    src1 = torch.where(f, src_t, src1)
    src2 = torch.where(f, src2, src_t)
    X_rec = YY_s[:, 1:4]
    phase = YY_s[:, 4:5]
    inp = torch.cat([torch.cat([src1[:, :3], X_rec, phase], dim=1), torch.cat([src2[:, :3], X_rec, phase], dim=1)], dim=0)
    T = model(inp).reshape(-1)
    n = int(src1.shape[0])
    return (T[n:] + src2[:, 3]) - (T[:n] + src1[:, 3])


def _conditional_fit(
    *,
    x0: torch.Tensor,
    owner: torch.Tensor,
    is_first: torch.Tensor,
    II_s: torch.Tensor,
    YY_s: torch.Tensor,
    X_src: torch.Tensor,
    dX: torch.Tensor,
    model: torch.nn.Module,
    w: torch.Tensor,
    P_c: torch.Tensor,
    n_iters: int,
):
    """Gauss-Newton fit of each event's conditional mode from the fixed start `x0`.
    Returns (mode (n_c,4) in dX dtype, precision H (n_c,4,4) float64, n_nonfinite)."""
    n_c = int(x0.shape[0])
    x = x0.detach().clone()
    dt_obs = YY_s[:, 0]
    wd = w.to(torch.float64)
    H = P_c.clone()
    for it in range(int(n_iters) + 1):
        leaf = x.index_select(0, owner).detach().requires_grad_(True)
        with torch.enable_grad():
            pred = _predict_rows(II_s, YY_s, X_src, dX, model, leaf, is_first)
            (J,) = torch.autograd.grad(pred.sum(), leaf)
        r = (dt_obs - pred).detach().to(torch.float64)
        Jd = J.detach().to(torch.float64)
        F = torch.zeros((n_c, _D, _D), dtype=torch.float64, device=x.device)
        F.index_add_(0, owner, wd.view(-1, 1, 1) * Jd.unsqueeze(2) * Jd.unsqueeze(1))
        b = torch.zeros((n_c, _D), dtype=torch.float64, device=x.device)
        b.index_add_(0, owner, (wd * r).unsqueeze(1) * Jd)
        H = F + P_c
        if it == int(n_iters):
            break
        rhs = b - torch.bmm(P_c, x.to(torch.float64).unsqueeze(-1)).squeeze(-1)
        try:
            step = torch.linalg.solve(H, rhs.unsqueeze(-1)).squeeze(-1)
        except Exception:
            step = torch.zeros_like(rhs)
        step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x.to(torch.float64) + step).to(x0.dtype)
    bad = ~(torch.isfinite(x).all(dim=1) & torch.isfinite(H).reshape(n_c, -1).all(dim=1))
    n_bad = int(bad.sum().item())
    if n_bad:
        x[bad] = x0[bad]
        H[bad] = P_c[bad]
    return x, H, n_bad


def _log_t_kernel(y: torch.Tensor, m: torch.Tensor, H: torch.Tensor, scale: float, dof: Optional[float]) -> torch.Tensor:
    """log density (up to a constant common to all evaluations with the same (m, H)) of the proposal
    t_dof(m, scale^2 H^-1) or, for dof=None, N(m, scale^2 H^-1)."""
    d = (y - m).to(torch.float64)
    q = torch.einsum("ni,nij,nj->n", d, H, d) / float(scale) ** 2
    if dof is None:
        return -0.5 * q
    nu = float(dof)
    return -0.5 * (nu + _D) * torch.log1p(q / nu)


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
    batch_size: int = 500_000,            # unused (kept for call-site compatibility)
    params: Optional[dict] = None,        # unused (kept for call-site compatibility)
    proposal: str = "conditional",
    gn_iters: int = 4,
    t_dof: Optional[float] = 5.0,
    scale_inflation: float = 1.2,
    start_dX: Optional[torch.Tensor] = None,
) -> dict:
    """One sweep over the weak events (colour by colour). Returns acceptance statistics."""
    device = dX.device
    n_ev = int(dX.shape[0])
    if cache.get("n_weak", 0) == 0:
        return {"n_proposed": 0, "n_accepted": 0, "median_jump_m": 0.0}
    proposal = str(proposal).strip().lower()
    if proposal not in {"conditional", "prior"}:
        raise ValueError(f"weak_event_mh proposal must be 'conditional' or 'prior' (got {proposal!r})")

    cid = (cluster_ids.to(torch.int64).reshape(-1) if cluster_ids is not None
           else torch.zeros(n_ev, dtype=torch.int64, device=device))
    K = int(cid.max().item()) + 1
    P = precision.to(device=device, dtype=torch.float64)
    if P.dim() == 2:
        P = P.unsqueeze(0).expand(K, _D, _D)
    n_k = torch.bincount(cid, minlength=K).double()

    use_centroid = centroid_std is not None
    if use_centroid:
        sc2 = torch.tensor([float(s) ** 2 for s in centroid_std], dtype=torch.float64, device=device)
        means = torch.zeros(K, _D, dtype=torch.float64, device=device)
        means.index_add_(0, cid, dX.double())
        means = means / n_k.clamp_min(1.0).unsqueeze(1)

    sp, ss = float(sigma_ps[0]), float(sigma_ps[1])
    if start_dX is None:
        start_dX = cache.get("start_dX", None)
    if start_dX is None:
        start_dX = torch.zeros_like(dX)
    Lch_prior = torch.linalg.cholesky(torch.linalg.inv(P)) if proposal == "prior" else None   # (K,4,4)

    n_prop = n_acc = n_bad_total = 0
    moved = []
    accepted_events = []
    for col in cache["colors"]:
        evs, rows, owner, is_first = col["events"], col["rows"], col["owner"], col["is_first"]
        if evs.numel() == 0:
            continue
        n_c = int(evs.numel())
        ck = cid[evs]
        P_c = P.index_select(0, ck)                                          # (n_c,4,4)
        II_s, YY_s = II[rows], YY[rows]
        w = torch.where(YY_s[:, 4] > 0.5,
                        torch.full_like(YY_s[:, 0], 1.0 / (ss * ss)),
                        torch.full_like(YY_s[:, 0], 1.0 / (sp * sp)))
        x_cur = dX[evs].clone()

        def _log_target(x_c: torch.Tensor) -> torch.Tensor:
            pred = _predict_rows(II_s, YY_s, X_src, dX, model, x_c.index_select(0, owner), is_first)
            r = (YY_s[:, 0] - pred).to(torch.float64)
            ll = torch.zeros(n_c, dtype=torch.float64, device=device)
            ll.index_add_(0, owner, -0.5 * w.to(torch.float64) * r * r)
            xd = x_c.to(torch.float64)
            lp = -0.5 * torch.einsum("ni,nij,nj->n", xd, P_c, xd)
            return ll + lp

        if proposal == "conditional":
            m, H, n_bad = _conditional_fit(
                x0=start_dX[evs], owner=owner, is_first=is_first, II_s=II_s, YY_s=YY_s, X_src=X_src, dX=dX,
                model=model, w=w, P_c=P_c, n_iters=int(gn_iters),
            )
            n_bad_total += n_bad
            # draw from t_dof(m, s^2 H^-1): m + s L z / sqrt(u/dof), L L^T = H^-1
            Hs = 0.5 * (H + H.transpose(1, 2))
            try:
                L = torch.linalg.cholesky(torch.linalg.inv(Hs))
            except Exception:
                Hs = Hs + torch.eye(_D, dtype=torch.float64, device=device) * 1e-9
                L = torch.linalg.cholesky(torch.linalg.inv(Hs))
            z = torch.randn(n_c, _D, 1, dtype=torch.float64, device=device)
            draw = (L @ z).squeeze(-1) * float(scale_inflation)
            if t_dof is not None:
                u = torch.distributions.Chi2(torch.tensor(float(t_dof), device=device, dtype=torch.float64)).sample((n_c,))
                draw = draw / torch.sqrt(u / float(t_dof)).unsqueeze(1)
            prop = (m.to(torch.float64) + draw).to(dX.dtype)
            log_q_prop = _log_t_kernel(prop, m, Hs, float(scale_inflation), t_dof)
            log_q_cur = _log_t_kernel(x_cur, m, Hs, float(scale_inflation), t_dof)
            log_alpha = (_log_target(prop) - _log_target(x_cur)) - (log_q_prop - log_q_cur)
        else:
            z = torch.randn(n_c, _D, 1, dtype=torch.float64, device=device)
            prop = (Lch_prior[ck] @ z).squeeze(-1).to(dX.dtype)                # ~ N(0, P^-1): prior and proposal cancel
            # data-only ratio (prior term cancels against the proposal)
            def _log_data(x_c):
                pred = _predict_rows(II_s, YY_s, X_src, dX, model, x_c.index_select(0, owner), is_first)
                r = (YY_s[:, 0] - pred).to(torch.float64)
                ll = torch.zeros(n_c, dtype=torch.float64, device=device)
                ll.index_add_(0, owner, -0.5 * w.to(torch.float64) * r * r)
                return ll
            log_alpha = _log_data(prop) - _log_data(x_cur)

        delta = prop.to(torch.float64) - x_cur.to(torch.float64)
        if use_centroid:
            # n_k-weighted per-component centroid prior: moving one event shifts its cluster mean by delta/n_k
            m0 = means[ck]
            m1 = m0 + delta / n_k[ck].unsqueeze(1)
            log_alpha = log_alpha + (n_k[ck].unsqueeze(1) * (m0 ** 2 - m1 ** 2) / (2.0 * sc2)).sum(dim=1)

        log_alpha = torch.nan_to_num(log_alpha, nan=-float("inf"))
        accept = torch.log(torch.rand(n_c, dtype=torch.float64, device=device).clamp_min(1e-300)) < log_alpha
        idx = evs[accept]
        if idx.numel():
            accepted_events.append(idx)
            if use_centroid:
                dmean = torch.zeros_like(means)
                dmean.index_add_(0, cid[idx], delta[accept])
                means = means + dmean / n_k.clamp_min(1.0).unsqueeze(1)
            moved.append(delta[accept, :3].norm(dim=1))
            dX[idx] = prop[accept]
        n_prop += n_c
        n_acc += int(accept.sum())

    all_moved = torch.cat(moved) if moved else torch.zeros(0, dtype=torch.float64, device=device)
    stats = {
        "n_proposed": n_prop,
        "n_accepted": n_acc,
        "median_jump_m": float(all_moved.median() * 1e3) if all_moved.numel() else 0.0,
        "n_fit_nonfinite": int(n_bad_total),
        "proposal": proposal,
    }
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
            if n_all > 0:
                by[f"{lo}-{min(hi, int(cache['max_dtimes']))}"] = (int(m_acc.sum()), n_all)
        stats["accept_by_degree"] = by
    return stats
