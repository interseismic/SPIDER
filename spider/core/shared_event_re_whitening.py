from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch

from .shared_event_re_gpu import _CollapsedQuadNoGrad


@dataclass
class SharedEventReWhiteningMetrics:
    n_groups_total: int = 0
    n_groups_chol: int = 0
    n_groups_pcg: int = 0
    n_groups_fallback_diag: int = 0
    n_groups_rows_cap: int = 0
    n_groups_nodes_cap: int = 0
    n_groups_tau_zero: int = 0
    max_rows_seen: int = 0
    max_nodes_seen: int = 0
    weight_mean: float = float("nan")
    weight_max: float = float("nan")
    pcg_iters_sum: int = 0
    pcg_iters_max: int = 0
    n_groups_pcg_fail: int = 0


def _edge_to_node(
    u: torch.Tensor,
    v: torch.Tensor,
    edge_vals: torch.Tensor,
    *,
    n_nodes: int,
) -> torch.Tensor:
    b = torch.zeros((n_nodes,), device=edge_vals.device, dtype=edge_vals.dtype)
    b.index_add_(0, u, -edge_vals)
    b.index_add_(0, v, edge_vals)
    return b


def _node_to_edge(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, v) - x.index_select(0, u)


def _laplacian_mv_weighted(
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    x: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    if n_nodes <= 0:
        return torch.zeros_like(x)
    xu = x.index_select(0, u)
    xv = x.index_select(0, v)
    diff = xu - xv
    out = torch.zeros((n_nodes,), device=x.device, dtype=x.dtype)
    out.index_add_(0, u, w * diff)
    out.index_add_(0, v, -w * diff)
    return out


def _pcg_solve_whitening(
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    beta: torch.Tensor,
    deg: torch.Tensor,
    max_iters: int,
    tol: float,
    min_iters: int = 0,
) -> tuple[torch.Tensor, int, bool]:
    n = int(b.numel())
    if n <= 0:
        return b, 0, True
    max_iters = max(1, int(max_iters))
    min_iters = max(0, int(min_iters))
    tol = float(tol)
    if not (tol > 0.0):
        tol = 1e-6
    alpha_t = b.new_tensor(float(alpha))
    # Jacobi preconditioner diag = alpha + beta * deg
    diag = alpha_t + beta * deg.to(dtype=b.dtype, device=b.device)
    diag = diag.clamp_min(1e-12)

    def _A(x: torch.Tensor) -> torch.Tensor:
        return alpha_t * x + beta * _laplacian_mv_weighted(u=u, v=v, w=w, x=x, n_nodes=n)

    x = torch.zeros_like(b)
    r = b - _A(x)
    z = r / diag
    p = z.clone()
    rz = torch.dot(r, z)
    b_norm = torch.sqrt(torch.dot(b, b)).clamp_min(1e-12)
    converged = False
    it = 0
    for it in range(max_iters):
        Ap = _A(p)
        denom = torch.dot(p, Ap).clamp_min(1e-12)
        alpha_cg = rz / denom
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        r_norm = torch.sqrt(torch.dot(r, r))
        if (it + 1) >= min_iters and float(r_norm.item()) <= float(tol) * float(b_norm.item()):
            converged = True
            break
        z = r / diag
        rz_new = torch.dot(r, z)
        beta_cg = rz_new / rz.clamp_min(1e-12)
        p = z + beta_cg * p
        rz = rz_new
    return x, int(it + 1), bool(converged)


def _pcg_solve_whitening_batched(
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    deg: torch.Tensor,
    max_iters: int,
    tol: float,
    min_iters: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched PCG for multiple groups with padded edges/nodes.
    Shapes:
      u,v,w: (B,E) with padding (w=0 for padded edges)
      b,deg: (B,N)
      alpha,beta: (B,) or (B,1)
    Returns (x, iters, ok_mask).
    """
    B = int(b.shape[0])
    N = int(b.shape[1])
    if B <= 0 or N <= 0:
        return b, torch.zeros((B,), device=b.device, dtype=torch.int64), torch.zeros((B,), device=b.device, dtype=torch.bool)
    max_iters = max(1, int(max_iters))
    min_iters = max(0, int(min_iters))
    tol = float(tol) if float(tol) > 0.0 else 1e-6

    alpha = alpha.reshape(-1, 1).to(device=b.device, dtype=b.dtype)
    beta = beta.reshape(-1, 1).to(device=b.device, dtype=b.dtype)
    diag = (alpha + beta * deg.to(dtype=b.dtype, device=b.device)).clamp_min(1e-12)

    # Flattened edge indices for batched scatter
    offsets = (torch.arange(B, device=b.device, dtype=u.dtype) * N).view(-1, 1)
    u_flat = (u + offsets).reshape(-1)
    v_flat = (v + offsets).reshape(-1)
    w_flat = w.reshape(-1).to(device=b.device, dtype=b.dtype)

    def _A(x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1)
        xu = x_flat.index_select(0, u_flat)
        xv = x_flat.index_select(0, v_flat)
        diff = xu - xv
        out_flat = torch.zeros((B * N,), device=b.device, dtype=b.dtype)
        out_flat.index_add_(0, u_flat, w_flat * diff)
        out_flat.index_add_(0, v_flat, -w_flat * diff)
        out = out_flat.view(B, N)
        return alpha * x + beta * out

    x = torch.zeros_like(b)
    r = b - _A(x)
    z = r / diag
    p = z.clone()
    rz = (r * z).sum(dim=1)
    b_norm = torch.sqrt((b * b).sum(dim=1)).clamp_min(1e-12)
    ok = torch.zeros((B,), device=b.device, dtype=torch.bool)
    iters = torch.zeros((B,), device=b.device, dtype=torch.int64)

    for it in range(max_iters):
        Ap = _A(p)
        denom = (p * Ap).sum(dim=1).clamp_min(1e-12)
        alpha_cg = (rz / denom).view(-1, 1)
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        r_norm = torch.sqrt((r * r).sum(dim=1))
        iters = torch.where(ok, iters, torch.full_like(iters, it + 1))
        new_ok = (r_norm <= (tol * b_norm)) & (iters >= min_iters)
        ok = ok | new_ok
        if bool(ok.all()):
            break
        z = r / diag
        rz_new = (r * z).sum(dim=1)
        beta_cg = (rz_new / rz.clamp_min(1e-12)).view(-1, 1)
        p = z + beta_cg * p
        rz = rz_new

    return x, iters, ok


def _build_group_cache(
    *,
    idx: torch.Tensor,
    keys: torch.Tensor,
    ph_id: torch.Tensor,
    sigma_p: torch.Tensor,
    sigma_s: torch.Tensor,
    tau_p: float,
    tau_s: float,
    jitter0: float,
    max_rows_per_group: int,
    max_nodes_per_group: int,
    solver: str,
    edge_weighting: str,
    edge_weight_ell_km: float,
    edge_weight_eps_km: float,
    edge_weight_power: float,
    edge_weight_scale_km: float,
    edge_weight_global_scale: float,
    edge_weight_normalize: bool,
    X_event: Optional[torch.Tensor],
    grouping_cache: Optional[dict] = None,
) -> dict:
    use_precomputed = False
    perm = starts = ends = group_ph = None
    local_u_all = local_v_all = n_nodes_all = None
    if isinstance(grouping_cache, dict):
        try:
            perm = grouping_cache.get("perm", None)
            starts = grouping_cache.get("starts", None)
            ends = grouping_cache.get("ends", None)
            group_ph = grouping_cache.get("ph_group", None)
            local_u_all = grouping_cache.get("local_u", None)
            local_v_all = grouping_cache.get("local_v", None)
            n_nodes_all = grouping_cache.get("n_nodes", None)
            use_precomputed = (
                isinstance(perm, torch.Tensor)
                and isinstance(starts, torch.Tensor)
                and isinstance(ends, torch.Tensor)
                and isinstance(local_u_all, torch.Tensor)
                and isinstance(local_v_all, torch.Tensor)
                and isinstance(n_nodes_all, torch.Tensor)
                and perm.ndim == 1
                and starts.ndim == 1
                and ends.ndim == 1
                and local_u_all.ndim == 1
                and local_v_all.ndim == 1
                and n_nodes_all.ndim == 1
            )
        except Exception:
            use_precomputed = False
    if not use_precomputed:
        keys_sorted, perm = torch.sort(keys)
        if keys_sorted.numel() > 0:
            is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
            is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
            starts = torch.nonzero(is_new, as_tuple=False).reshape(-1)
            ends = torch.cat([starts[1:], torch.tensor([keys_sorted.numel()], device=starts.device, dtype=starts.dtype)])
        else:
            starts = torch.zeros((0,), device=keys.device, dtype=torch.int64)
            ends = torch.zeros((0,), device=keys.device, dtype=torch.int64)

    if not isinstance(group_ph, torch.Tensor):
        group_ph = ph_id.index_select(0, starts) if starts.numel() > 0 else torch.zeros((0,), device=ph_id.device, dtype=ph_id.dtype)
    n_groups = int(starts.numel())
    groups = []
    for g in range(n_groups):
        s = int(starts[g].item())
        e = int(ends[g].item())
        if e <= s:
            groups.append(None)
            continue
        idxs = perm[s:e]
        idx_g = idx.index_select(0, idxs)
        e1 = idx_g[:, 0].to(torch.int64)
        e2 = idx_g[:, 1].to(torch.int64)
        ph_g = float(group_ph[g].item()) if group_ph.numel() > 0 else 0.0
        sigma_g = sigma_p if ph_g < 0.5 else sigma_s
        tau_g = float(tau_p) if ph_g < 0.5 else float(tau_s)

        if not (tau_g > 0.0):
            groups.append(
                {
                    "local_u": None,
                    "local_v": None,
                    "n_nodes": int(0),
                    "chol": None,
                    "sigma": sigma_g,
                    "tau": tau_g,
                    "start": s,
                    "end": e,
                }
            )
            continue

        if use_precomputed and isinstance(n_nodes_all, torch.Tensor):
            n_nodes = int(n_nodes_all[g].item())
        else:
            nodes, inv = torch.unique(torch.cat([e1, e2], dim=0), return_inverse=True)
            n_nodes = int(nodes.numel())
        if (e - s) > int(max_rows_per_group) or n_nodes > int(max_nodes_per_group):
            groups.append(
                {
                    "local_u": None,
                    "local_v": None,
                    "n_nodes": n_nodes,
                    "chol": None,
                    "sigma": sigma_g,
                    "tau": tau_g,
                    "start": s,
                    "end": e,
                }
            )
            continue

        m = int(e1.numel())
        if use_precomputed and isinstance(local_u_all, torch.Tensor) and isinstance(local_v_all, torch.Tensor):
            local_u = local_u_all[s:e]
            local_v = local_v_all[s:e]
        else:
            local_u = inv[:m]
            local_v = inv[m:]
        if edge_weighting in {"distance_rbf", "distance_linear", "distance_power"} and isinstance(X_event, torch.Tensor):
            x_u = X_event.index_select(0, e1)
            x_v = X_event.index_select(0, e2)
            dist = torch.linalg.norm(x_u - x_v, dim=1).clamp_min(0.0)
            if edge_weighting == "distance_rbf":
                ell = float(edge_weight_ell_km)
                w = torch.exp(-((dist / max(ell, 1e-6)) ** 2)) + float(edge_weight_eps_km)
            elif edge_weighting == "distance_linear":
                w = dist + float(edge_weight_eps_km)
            else:
                scale = float(edge_weight_scale_km)
                p = float(edge_weight_power)
                w = torch.pow(dist / max(scale, 1e-6), p) + float(edge_weight_eps_km)
        else:
            w = torch.ones((m,), device=idx.device, dtype=sigma_g.dtype)
        if bool(edge_weight_normalize):
            try:
                w_mean = w.mean().clamp_min(1e-12)
                w = w / w_mean
            except Exception:
                pass
        try:
            w = w * float(edge_weight_global_scale)
        except Exception:
            pass

        deg = torch.zeros((n_nodes,), device=idx.device, dtype=sigma_g.dtype)
        deg.index_add_(0, local_u, w)
        deg.index_add_(0, local_v, w)
        chol = None
        if str(solver).strip().lower() == "chol":
            L = torch.zeros((n_nodes, n_nodes), device=idx.device, dtype=sigma_g.dtype)
            L[local_u, local_v] = L[local_u, local_v] - w
            L[local_v, local_u] = L[local_v, local_u] - w
            L.diagonal().add_(deg)
            beta = (1.0 / sigma_g.square().clamp_min(1e-24)).to(L.dtype)
            alpha = (1.0 / (tau_g * tau_g)) + float(jitter0)
            M = L.mul(beta) + torch.eye(n_nodes, device=L.device, dtype=L.dtype).mul(alpha)
            try:
                chol = torch.linalg.cholesky(M)
            except Exception:
                chol = None

        w_mean = float(w.mean().detach().item()) if w.numel() > 0 else float("nan")
        w_max = float(w.max().detach().item()) if w.numel() > 0 else float("nan")
        w_sqrt = torch.sqrt(w.clamp_min(0.0))
        groups.append(
            {
                "local_u": local_u,
                "local_v": local_v,
                "n_nodes": n_nodes,
                "chol": chol,
                "sigma": sigma_g,
                "tau": tau_g,
                "start": s,
                "end": e,
                "deg": deg,
                "w": w,
                "w_mean": w_mean,
                "w_max": w_max,
                "w_count": int(w.numel()),
                "w_sqrt": w_sqrt,
            }
        )

    return {
        "perm": perm,
        "starts": starts,
        "ends": ends,
        "group_ph": group_ph,
        "groups": groups,
        "edge_weighting": str(edge_weighting),
        "edge_weight_ell_km": float(edge_weight_ell_km),
        "edge_weight_eps_km": float(edge_weight_eps_km),
    }


def _make_cache_key(
    *,
    edge_weighting: str,
    edge_weight_ell_km: float,
    edge_weight_eps_km: float,
    edge_weight_power: float,
    edge_weight_scale_km: float,
    edge_weight_global_scale: float,
    edge_weight_normalize: bool,
    tau_p: float,
    tau_s: float,
    max_rows_per_group: int,
    max_nodes_per_group: int,
    solver: str,
    idx_rows: int,
    cache_key_extra: Optional[tuple],
) -> tuple:
    return (
        str(edge_weighting),
        float(edge_weight_ell_km),
        float(edge_weight_eps_km),
        float(edge_weight_power),
        float(edge_weight_scale_km),
        float(edge_weight_global_scale),
        bool(edge_weight_normalize),
        float(tau_p),
        float(tau_s),
        int(max_rows_per_group),
        int(max_nodes_per_group),
        str(solver),
        int(idx_rows),
        cache_key_extra,
    )


def build_whitening_cache_entry(
    *,
    idx: torch.Tensor,
    keys: torch.Tensor,
    ph_id: torch.Tensor,
    sigma_p: torch.Tensor,
    sigma_s: torch.Tensor,
    tau_p: float,
    tau_s: float,
    jitter0: float,
    max_rows_per_group: int,
    max_nodes_per_group: int,
    solver: str,
    edge_weighting: str,
    edge_weight_ell_km: float,
    edge_weight_eps_km: float,
    edge_weight_power: float,
    edge_weight_scale_km: float,
    edge_weight_global_scale: float,
    edge_weight_normalize: bool,
    X_event: Optional[torch.Tensor],
    grouping_cache: Optional[dict] = None,
) -> dict:
    return _build_group_cache(
        idx=idx,
        keys=keys,
        ph_id=ph_id,
        sigma_p=sigma_p,
        sigma_s=sigma_s,
        tau_p=float(tau_p),
        tau_s=float(tau_s),
        jitter0=float(jitter0),
        max_rows_per_group=int(max_rows_per_group),
        max_nodes_per_group=int(max_nodes_per_group),
        solver=str(solver),
        edge_weighting=str(edge_weighting),
        edge_weight_ell_km=float(edge_weight_ell_km),
        edge_weight_eps_km=float(edge_weight_eps_km),
        edge_weight_power=float(edge_weight_power),
        edge_weight_scale_km=float(edge_weight_scale_km),
        edge_weight_global_scale=float(edge_weight_global_scale),
        edge_weight_normalize=bool(edge_weight_normalize),
        X_event=X_event,
        grouping_cache=grouping_cache,
    )


def compute_quad_whitening(
    *,
    idx: torch.Tensor,
    resid: torch.Tensor,
    keys: torch.Tensor,
    ph_id: torch.Tensor,
    sigma_p: torch.Tensor,
    sigma_s: torch.Tensor,
    tau_p: float,
    tau_s: float,
    jitter0: float,
    max_rows_per_group: int,
    max_nodes_per_group: int,
    solver: str,
    pcg_max_iters: int,
    pcg_tol: float,
    pcg_min_iters: int,
    pcg_batched: bool,
    pcg_bucket_nodes: Optional[list],
    edge_weighting: str,
    edge_weight_ell_km: float,
    edge_weight_eps_km: float,
    edge_weight_power: float,
    edge_weight_scale_km: float,
    edge_weight_global_scale: float,
    edge_weight_normalize: bool,
    X_event: Optional[torch.Tensor],
    cache: Optional[dict],
    grouping_cache: Optional[dict] = None,
    cache_key_extra: Optional[tuple] = None,
) -> Tuple[torch.Tensor, SharedEventReWhiteningMetrics, dict]:
    metrics = SharedEventReWhiteningMetrics()
    if cache is None:
        cache = {}
    cache_key = _make_cache_key(
        edge_weighting=edge_weighting,
        edge_weight_ell_km=edge_weight_ell_km,
        edge_weight_eps_km=edge_weight_eps_km,
        edge_weight_power=edge_weight_power,
        edge_weight_scale_km=edge_weight_scale_km,
        edge_weight_global_scale=edge_weight_global_scale,
        edge_weight_normalize=edge_weight_normalize,
        tau_p=tau_p,
        tau_s=tau_s,
        max_rows_per_group=max_rows_per_group,
        max_nodes_per_group=max_nodes_per_group,
        solver=solver,
        idx_rows=int(idx.shape[0]),
        cache_key_extra=cache_key_extra,
    )
    cache_entry = cache.get(cache_key, None)
    if not isinstance(cache_entry, dict):
        cache_entry = build_whitening_cache_entry(
            idx=idx,
            keys=keys,
            ph_id=ph_id,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
            tau_p=float(tau_p),
            tau_s=float(tau_s),
            jitter0=float(jitter0),
            max_rows_per_group=int(max_rows_per_group),
            max_nodes_per_group=int(max_nodes_per_group),
            solver=str(solver),
            edge_weighting=str(edge_weighting),
            edge_weight_ell_km=float(edge_weight_ell_km),
            edge_weight_eps_km=float(edge_weight_eps_km),
            edge_weight_power=float(edge_weight_power),
            edge_weight_scale_km=float(edge_weight_scale_km),
            edge_weight_global_scale=float(edge_weight_global_scale),
            edge_weight_normalize=bool(edge_weight_normalize),
            X_event=X_event,
            grouping_cache=grouping_cache,
        )
        cache[cache_key] = cache_entry

    perm = cache_entry["perm"]
    resid_perm = resid.index_select(0, perm)
    groups = cache_entry["groups"]
    metrics.n_groups_total = int(len(groups))

    quad_sum = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
    w_sum = 0.0
    w_cnt = 0.0
    w_max_all = float("nan")
    pcg_groups = []
    for g in range(len(groups)):
        gd = groups[g]
        if gd is None:
            continue
        s = int(gd["start"])
        e = int(gd["end"])
        if e <= s:
            continue
        r_g = resid_perm[s:e]
        sigma_g = gd["sigma"].clamp_min(1e-12)
        tau_g = float(gd["tau"])
        n_nodes = int(gd["n_nodes"])
        metrics.max_rows_seen = max(metrics.max_rows_seen, int(e - s))
        metrics.max_nodes_seen = max(metrics.max_nodes_seen, int(n_nodes))
        if (e - s) > int(max_rows_per_group):
            metrics.n_groups_rows_cap += 1
        if n_nodes > int(max_nodes_per_group):
            metrics.n_groups_nodes_cap += 1
        if not (tau_g > 0.0):
            metrics.n_groups_tau_zero += 1
        w_mean = float(gd.get("w_mean", float("nan")))
        w_max = float(gd.get("w_max", float("nan")))
        w_count = float(gd.get("w_count", 0.0) or 0.0)
        if w_count > 0 and math.isfinite(w_mean):
            w_sum += w_mean * w_count
            w_cnt += w_count
        if math.isfinite(w_max):
            w_max_all = max(w_max_all, w_max) if math.isfinite(w_max_all) else w_max
        if (not (tau_g > 0.0)) or (n_nodes <= 0):
            metrics.n_groups_fallback_diag += 1
            quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, r_g / sigma_g.square().clamp_min(1e-24))
            continue

        local_u = gd["local_u"]
        local_v = gd["local_v"]
        w_sqrt = gd.get("w_sqrt", None)
        if not isinstance(w_sqrt, torch.Tensor) or w_sqrt.numel() != r_g.numel():
            w_sqrt = torch.ones_like(r_g)
        beta = (1.0 / sigma_g.square().clamp_min(1e-24)).to(r_g.dtype)

        solver_use = str(solver).strip().lower()
        if solver_use == "pcg" and bool(pcg_batched):
            if (not isinstance(local_u, torch.Tensor)) or (not isinstance(local_v, torch.Tensor)):
                metrics.n_groups_fallback_diag += 1
                quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, r_g / sigma_g.square().clamp_min(1e-24))
                continue
            pcg_groups.append(
                {
                    "r": r_g,
                    "local_u": local_u,
                    "local_v": local_v,
                    "w_sqrt": w_sqrt,
                    "w": gd.get("w", None),
                    "deg": gd.get("deg", None),
                    "beta": beta,
                    "alpha": float((1.0 / (tau_g * tau_g)) + float(jitter0)),
                    "n_nodes": n_nodes,
                    "sigma": sigma_g,
                }
            )
            continue
        if solver_use == "pcg":
            w = gd.get("w", None)
            if not isinstance(w, torch.Tensor) or w.numel() != r_g.numel():
                w = w_sqrt * w_sqrt
            deg = gd.get("deg", None)
            if not isinstance(deg, torch.Tensor) or deg.numel() != n_nodes:
                deg = torch.zeros((n_nodes,), device=r_g.device, dtype=r_g.dtype)
                deg.index_add_(0, local_u, w)
                deg.index_add_(0, local_v, w)
            alpha = (1.0 / (tau_g * tau_g)) + float(jitter0)
            b = _edge_to_node(local_u, local_v, beta * w_sqrt * r_g, n_nodes=n_nodes)
            x, iters, ok = _pcg_solve_whitening(
                u=local_u,
                v=local_v,
                w=w.to(dtype=r_g.dtype, device=r_g.device),
                b=b,
                alpha=float(alpha),
                beta=beta,
                deg=deg,
                max_iters=int(pcg_max_iters),
                tol=float(pcg_tol),
                min_iters=int(pcg_min_iters),
            )
            metrics.n_groups_pcg += 1
            metrics.pcg_iters_sum += int(iters)
            metrics.pcg_iters_max = max(int(metrics.pcg_iters_max), int(iters))
            if not ok:
                metrics.n_groups_pcg_fail += 1
                metrics.n_groups_fallback_diag += 1
                quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, r_g / sigma_g.square().clamp_min(1e-24))
                continue
        else:
            chol = gd["chol"]
            if (not isinstance(chol, torch.Tensor)) or (not (tau_g > 0.0)) or (n_nodes <= 0):
                metrics.n_groups_fallback_diag += 1
                quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, r_g / sigma_g.square().clamp_min(1e-24))
                continue
            metrics.n_groups_chol += 1
            b = _edge_to_node(local_u, local_v, beta * w_sqrt * r_g, n_nodes=n_nodes)
            try:
                x = torch.cholesky_solve(b.unsqueeze(1), chol).squeeze(1)
            except Exception:
                metrics.n_groups_fallback_diag += 1
                quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, r_g / sigma_g.square().clamp_min(1e-24))
                continue
        ax = _node_to_edge(local_u, local_v, x)
        u_edge = beta * (r_g - (w_sqrt * ax))
        quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, u_edge)

    if pcg_groups:
        if not isinstance(pcg_bucket_nodes, list) or not pcg_bucket_nodes:
            pcg_bucket_nodes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        bucket_nodes = sorted({int(x) for x in pcg_bucket_nodes if int(x) > 0})
        # Assign groups to buckets
        bucket_map: dict[int, list[int]] = {}
        leftovers: list[int] = []
        for gi, gd in enumerate(pcg_groups):
            n = int(gd["n_nodes"])
            bsz = None
            for bn in bucket_nodes:
                if n <= bn:
                    bsz = bn
                    break
            if bsz is None:
                leftovers.append(gi)
            else:
                bucket_map.setdefault(int(bsz), []).append(gi)

        def _solve_bucket(bucket_size: int, group_ids: list[int]) -> None:
            if not group_ids:
                return
            B = len(group_ids)
            # Build padded tensors
            max_edges = max(int(pcg_groups[i]["r"].numel()) for i in group_ids)
            max_edges = max(1, max_edges)
            u = torch.zeros((B, max_edges), device=resid.device, dtype=torch.int64)
            v = torch.zeros((B, max_edges), device=resid.device, dtype=torch.int64)
            w = torch.zeros((B, max_edges), device=resid.device, dtype=resid.dtype)
            w_sqrt = torch.zeros((B, max_edges), device=resid.device, dtype=resid.dtype)
            r = torch.zeros((B, max_edges), device=resid.device, dtype=resid.dtype)
            b = torch.zeros((B, bucket_size), device=resid.device, dtype=resid.dtype)
            deg = torch.zeros((B, bucket_size), device=resid.device, dtype=resid.dtype)
            alpha = torch.zeros((B,), device=resid.device, dtype=resid.dtype)
            beta = torch.zeros((B,), device=resid.device, dtype=resid.dtype)
            sigma = torch.zeros((B,), device=resid.device, dtype=resid.dtype)
            lengths = torch.zeros((B,), device=resid.device, dtype=torch.int64)
            n_nodes = torch.zeros((B,), device=resid.device, dtype=torch.int64)

            for bi, gi in enumerate(group_ids):
                gd = pcg_groups[gi]
                r_g = gd["r"]
                m = int(r_g.numel())
                lengths[bi] = m
                n_nodes[bi] = int(gd["n_nodes"])
                sigma[bi] = gd["sigma"].to(dtype=resid.dtype, device=resid.device)
                alpha[bi] = float(gd["alpha"])
                beta[bi] = gd["beta"].to(dtype=resid.dtype, device=resid.device)
                r[bi, :m] = r_g
                w_s = gd["w_sqrt"]
                w_s = w_s if isinstance(w_s, torch.Tensor) else torch.ones_like(r_g)
                w_sqrt[bi, :m] = w_s
                ww = gd.get("w", None)
                if not isinstance(ww, torch.Tensor) or ww.numel() != m:
                    ww = w_s * w_s
                w[bi, :m] = ww.to(dtype=resid.dtype, device=resid.device)
                u[bi, :m] = gd["local_u"].to(device=resid.device, dtype=torch.int64)
                v[bi, :m] = gd["local_v"].to(device=resid.device, dtype=torch.int64)
                dg = gd.get("deg", None)
                if isinstance(dg, torch.Tensor) and dg.numel() == int(gd["n_nodes"]):
                    deg[bi, : int(gd["n_nodes"])] = dg.to(dtype=resid.dtype, device=resid.device)
                else:
                    deg[bi, : int(gd["n_nodes"])].index_add_(0, u[bi, :m], w[bi, :m])
                    deg[bi, : int(gd["n_nodes"])].index_add_(0, v[bi, :m], w[bi, :m])

                # Build RHS b = B^T D^{-1} r
                edge_vals = beta[bi] * w_sqrt[bi, :m] * r[bi, :m]
                b[bi].index_add_(0, u[bi, :m], -edge_vals)
                b[bi].index_add_(0, v[bi, :m], edge_vals)

            x, iters, ok = _pcg_solve_whitening_batched(
                u=u,
                v=v,
                w=w,
                b=b,
                alpha=alpha,
                beta=beta,
                deg=deg,
                max_iters=int(pcg_max_iters),
                tol=float(pcg_tol),
                min_iters=int(pcg_min_iters),
            )
            metrics.n_groups_pcg += int(B)
            metrics.pcg_iters_sum += int(iters.sum().item())
            metrics.pcg_iters_max = max(int(metrics.pcg_iters_max), int(iters.max().item()))
            metrics.n_groups_pcg_fail += int((~ok).sum().item())
            metrics.n_groups_fallback_diag += int((~ok).sum().item())

            # Compute quad per group
            x_flat = x
            ax = x_flat.gather(1, v) - x_flat.gather(1, u)
            for bi, gi in enumerate(group_ids):
                m = int(lengths[bi].item())
                r_g = r[bi, :m]
                if not bool(ok[bi].item()):
                    quad_sum_nonlocal = _CollapsedQuadNoGrad.apply(r_g, r_g / (sigma[bi].square().clamp_min(1e-24)))
                    nonlocal_quad.append(quad_sum_nonlocal)
                    continue
                ax_g = ax[bi, :m]
                u_edge = beta[bi] * (r_g - (w_sqrt[bi, :m] * ax_g))
                quad_sum_nonlocal = _CollapsedQuadNoGrad.apply(r_g, u_edge)
                nonlocal_quad.append(quad_sum_nonlocal)

        nonlocal_quad: list[torch.Tensor] = []
        for bsz, group_ids in bucket_map.items():
            _solve_bucket(int(bsz), group_ids)
        # Process leftovers individually
        for gi in leftovers:
            gd = pcg_groups[gi]
            r_g = gd["r"]
            local_u = gd["local_u"]
            local_v = gd["local_v"]
            w_sqrt = gd["w_sqrt"]
            w = gd.get("w", None)
            if not isinstance(w, torch.Tensor) or w.numel() != r_g.numel():
                w = w_sqrt * w_sqrt
            deg = gd.get("deg", None)
            n_nodes = int(gd["n_nodes"])
            if (not isinstance(local_u, torch.Tensor)) or (not isinstance(local_v, torch.Tensor)):
                metrics.n_groups_fallback_diag += 1
                quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, r_g / (gd["sigma"].square().clamp_min(1e-24)))
                continue
            if not isinstance(deg, torch.Tensor) or deg.numel() != n_nodes:
                deg = torch.zeros((n_nodes,), device=resid.device, dtype=resid.dtype)
                deg.index_add_(0, local_u, w)
                deg.index_add_(0, local_v, w)
            alpha = float(gd["alpha"])
            beta = gd["beta"]
            b = _edge_to_node(local_u, local_v, beta * w_sqrt * r_g, n_nodes=n_nodes)
            x, iters, ok = _pcg_solve_whitening(
                u=local_u,
                v=local_v,
                w=w.to(dtype=resid.dtype, device=resid.device),
                b=b,
                alpha=float(alpha),
                beta=beta,
                deg=deg,
                max_iters=int(pcg_max_iters),
                tol=float(pcg_tol),
                min_iters=int(pcg_min_iters),
            )
            metrics.n_groups_pcg += 1
            metrics.pcg_iters_sum += int(iters)
            metrics.pcg_iters_max = max(int(metrics.pcg_iters_max), int(iters))
            if not ok:
                metrics.n_groups_pcg_fail += 1
                metrics.n_groups_fallback_diag += 1
                quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, r_g / (gd["sigma"].square().clamp_min(1e-24)))
                continue
            ax = _node_to_edge(local_u, local_v, x)
            u_edge = beta * (r_g - (w_sqrt * ax))
            quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_g, u_edge)

        if nonlocal_quad:
            quad_sum = quad_sum + torch.stack(nonlocal_quad).sum()

    if w_cnt > 0:
        metrics.weight_mean = float(w_sum / w_cnt)
    metrics.weight_max = float(w_max_all)
    return quad_sum, metrics, cache
