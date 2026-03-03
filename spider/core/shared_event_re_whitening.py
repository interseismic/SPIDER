from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Optional, Tuple

import math
import time
import torch

from .shared_event_re_gpu import _CollapsedQuadNoGrad, _build_group_ids, _build_local_node_indices, _group_by_keys_gpu


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
    cache_hit: int = 0
    cache_miss: int = 0
    cache_build_ms: float = 0.0
    solve_ms: float = 0.0


def _extract_batch_context_tuple(batch_context: Optional[dict[str, Any]]) -> tuple:
    """
    Produce a stable cache identity component from runtime batch context.
    """
    if not isinstance(batch_context, dict):
        return ("unknown",)
    mode = str(batch_context.get("mode", "unknown"))
    is_shuffled = bool(batch_context.get("is_shuffled", True))
    batch_id = int(batch_context.get("batch_id", -1))
    batch_i0 = int(batch_context.get("batch_i0", -1))
    batch_i1 = int(batch_context.get("batch_i1", -1))
    bucket_id = int(batch_context.get("bucket_id", -1))
    bucket_gen = int(batch_context.get("bucket_gen", -1))
    epoch_index = int(batch_context.get("epoch_index", -1))
    batch_seq = int(batch_context.get("batch_seq", -1))
    if mode == "standard" and (not is_shuffled) and batch_id >= 0:
        return ("standard_fixed", int(batch_id), int(batch_i0), int(batch_i1))
    if mode == "event_bucket" and bucket_id >= 0:
        return ("event_bucket", int(bucket_id), int(bucket_gen))
    # Fallback identity for volatile batches.
    return ("volatile", mode, int(epoch_index), int(batch_seq), int(batch_id), int(bucket_id))


def build_grouping_plan(
    *,
    idx: torch.Tensor,
    keys: torch.Tensor,
    ph_id: torch.Tensor,
    precomputed: Optional[dict] = None,
) -> dict:
    """
    Canonical station_phase grouping plan used by whitening and batched solvers.
    """
    if isinstance(precomputed, dict):
        perm = precomputed.get("perm", None)
        starts = precomputed.get("starts", None)
        ends = precomputed.get("ends", None)
        lengths = precomputed.get("lengths", None)
        group_ids = precomputed.get("group_ids", None)
        local_u = precomputed.get("local_u", None)
        local_v = precomputed.get("local_v", None)
        n_nodes = precomputed.get("n_nodes", None)
        ph_group = precomputed.get("ph_group", None)
        edge_pos = precomputed.get("edge_pos", None)
        ok = (
            isinstance(perm, torch.Tensor)
            and isinstance(starts, torch.Tensor)
            and isinstance(ends, torch.Tensor)
            and isinstance(lengths, torch.Tensor)
            and isinstance(group_ids, torch.Tensor)
            and isinstance(local_u, torch.Tensor)
            and isinstance(local_v, torch.Tensor)
            and isinstance(n_nodes, torch.Tensor)
            and isinstance(ph_group, torch.Tensor)
            and isinstance(edge_pos, torch.Tensor)
            and perm.ndim == 1
            and starts.ndim == 1
            and ends.ndim == 1
            and lengths.ndim == 1
            and group_ids.ndim == 1
            and local_u.ndim == 1
            and local_v.ndim == 1
            and n_nodes.ndim == 1
            and ph_group.ndim == 1
            and edge_pos.ndim == 1
            and int(starts.numel()) == int(ends.numel()) == int(lengths.numel()) == int(n_nodes.numel()) == int(ph_group.numel())
        )
        if ok:
            return {
                "perm": perm,
                "starts": starts,
                "ends": ends,
                "lengths": lengths,
                "group_ids": group_ids,
                "local_u": local_u,
                "local_v": local_v,
                "n_nodes": n_nodes,
                "ph_group": ph_group,
                "edge_pos": edge_pos,
            }

    _, perm, starts, ends = _group_by_keys_gpu(keys)
    lengths = (ends - starts).to(torch.int64)
    group_ids, _ = _build_group_ids(starts, ends)

    idx_perm = idx.index_select(0, perm) if perm.numel() > 0 else idx.new_zeros((0, 2))
    u = idx_perm[:, 0].to(torch.int64) if idx_perm.numel() > 0 else torch.zeros((0,), device=idx.device, dtype=torch.int64)
    v = idx_perm[:, 1].to(torch.int64) if idx_perm.numel() > 0 else torch.zeros((0,), device=idx.device, dtype=torch.int64)
    if u.numel() > 0 and v.numel() > 0:
        max_node_id = int(torch.max(torch.stack([u.max(), v.max()])).item())
    else:
        max_node_id = 0
    local_u, local_v, n_nodes = _build_local_node_indices(
        u,
        v,
        group_ids,
        int(starts.numel()),
        max_node_id=max_node_id,
    )
    ph_perm = ph_id.index_select(0, perm) if perm.numel() > 0 else ph_id.new_zeros((0,))
    ph_group = ph_perm.index_select(0, starts) if starts.numel() > 0 else ph_id.new_zeros((0,))
    if perm.numel() > 0:
        edge_idx = torch.arange(int(perm.numel()), device=perm.device, dtype=starts.dtype)
        edge_pos = edge_idx - starts.index_select(0, group_ids)
    else:
        edge_pos = torch.zeros((0,), device=keys.device, dtype=torch.int64)
    return {
        "perm": perm,
        "starts": starts,
        "ends": ends,
        "lengths": lengths,
        "group_ids": group_ids,
        "local_u": local_u,
        "local_v": local_v,
        "n_nodes": n_nodes,
        "ph_group": ph_group,
        "edge_pos": edge_pos,
    }


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
    x0: Optional[torch.Tensor] = None,
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

    if isinstance(x0, torch.Tensor) and x0.shape == b.shape:
        x = x0.to(device=b.device, dtype=b.dtype).clone()
        r = b - _A(x)
    else:
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
    plan = build_grouping_plan(idx=idx, keys=keys, ph_id=ph_id, precomputed=grouping_cache)
    perm = plan["perm"]
    starts = plan["starts"]
    ends = plan["ends"]
    group_ph = plan["ph_group"]
    local_u_all = plan["local_u"]
    local_v_all = plan["local_v"]
    n_nodes_all = plan["n_nodes"]
    n_groups = int(starts.numel())
    groups = []
    idx_perm_all = idx.index_select(0, perm) if isinstance(perm, torch.Tensor) and int(perm.numel()) > 0 else idx.new_zeros((0, 2))
    for g in range(n_groups):
        s = int(starts[g].item())
        e = int(ends[g].item())
        if e <= s:
            groups.append(None)
            continue
        idx_g = idx_perm_all[s:e, :]
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

        n_nodes = int(n_nodes_all[g].item()) if isinstance(n_nodes_all, torch.Tensor) and int(n_nodes_all.numel()) > g else 0
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
        local_u = local_u_all[s:e] if isinstance(local_u_all, torch.Tensor) else torch.zeros((m,), device=idx.device, dtype=torch.int64)
        local_v = local_v_all[s:e] if isinstance(local_v_all, torch.Tensor) else torch.zeros((m,), device=idx.device, dtype=torch.int64)
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
        "lengths": plan["lengths"],
        "group_ids": plan["group_ids"],
        "local_u_all": local_u_all,
        "local_v_all": local_v_all,
        "n_nodes_all": n_nodes_all,
        "edge_pos": plan["edge_pos"],
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
    batch_context: Optional[dict[str, Any]],
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
        _extract_batch_context_tuple(batch_context),
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
    batch_context: Optional[dict[str, Any]] = None,
    cache_max_entries: int = 0,
    pcg_warm_start: bool = False,
    grouping_cache: Optional[dict] = None,
    cache_key_extra: Optional[tuple] = None,
) -> Tuple[torch.Tensor, SharedEventReWhiteningMetrics, dict]:
    metrics = SharedEventReWhiteningMetrics()
    if cache is None:
        cache = OrderedDict()
    elif not isinstance(cache, OrderedDict):
        try:
            cache = OrderedDict(cache.items())  # type: ignore[arg-type]
        except Exception:
            cache = OrderedDict()
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
        batch_context=batch_context,
        cache_key_extra=cache_key_extra,
    )
    t_build0 = time.perf_counter()
    cache_entry = cache.get(cache_key, None) if isinstance(cache, (dict, OrderedDict)) else None
    if not isinstance(cache_entry, dict):
        metrics.cache_miss = 1
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
        if int(cache_max_entries) > 0:
            while len(cache) > int(cache_max_entries):
                cache.popitem(last=False)
    else:
        metrics.cache_hit = 1
        # LRU touch on hit
        try:
            cache.move_to_end(cache_key)
        except Exception:
            pass
    metrics.cache_build_ms = float(1000.0 * (time.perf_counter() - t_build0))

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
                    "group_index": int(g),
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

    t_solve0 = time.perf_counter()
    if pcg_groups:
        if not isinstance(pcg_bucket_nodes, list) or not pcg_bucket_nodes:
            pcg_bucket_nodes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        bucket_nodes = sorted({int(x) for x in pcg_bucket_nodes if int(x) > 0})
        # Assign groups to node and edge-size buckets to reduce padding waste.
        bucket_map: dict[tuple[int, int], list[int]] = {}
        leftovers: list[int] = []
        for gi, gd in enumerate(pcg_groups):
            n = int(gd["n_nodes"])
            m = max(1, int(gd["r"].numel()))
            bsz = None
            for bn in bucket_nodes:
                if n <= bn:
                    bsz = bn
                    break
            if bsz is None:
                leftovers.append(gi)
            else:
                edge_bin = 1 << int(max(0, int(m - 1)).bit_length())
                bucket_map.setdefault((int(bsz), int(edge_bin)), []).append(gi)

        # Optional warm-start cache (per-group node solution).
        use_warm = bool(pcg_warm_start) and isinstance(cache_entry, dict)
        x0_cache = None
        if use_warm:
            try:
                n_groups_total = int(len(groups))
                max_nodes_all = 0
                for gd in groups:
                    if isinstance(gd, dict):
                        max_nodes_all = max(max_nodes_all, int(gd.get("n_nodes", 0) or 0))
                if max_nodes_all <= 0:
                    use_warm = False
                else:
                    x0_cache_raw = cache_entry.get("pcg_x0", None)
                    ok_x0 = (
                        isinstance(x0_cache_raw, torch.Tensor)
                        and x0_cache_raw.ndim == 2
                        and int(x0_cache_raw.shape[0]) == int(n_groups_total)
                        and int(x0_cache_raw.shape[1]) >= int(max_nodes_all)
                    )
                    if not ok_x0:
                        x0_cache = torch.zeros(
                            (n_groups_total, max_nodes_all),
                            device=resid.device,
                            dtype=resid.dtype,
                        )
                        if isinstance(x0_cache_raw, torch.Tensor) and x0_cache_raw.ndim == 2 and int(x0_cache_raw.shape[0]) == int(n_groups_total):
                            cp = min(int(x0_cache_raw.shape[1]), int(max_nodes_all))
                            if cp > 0:
                                x0_cache[:, :cp] = x0_cache_raw[:, :cp].detach().to(device=resid.device, dtype=resid.dtype)
                        cache_entry["pcg_x0"] = x0_cache.detach()
                    else:
                        x0_cache = x0_cache_raw.detach().to(device=resid.device, dtype=resid.dtype)
                        cache_entry["pcg_x0"] = x0_cache.detach()
            except Exception:
                use_warm = False
                x0_cache = None

        def _solve_bucket(bucket_size: int, group_ids: list[int]) -> None:
            if not group_ids:
                return
            B = len(group_ids)
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
            grp_idx = torch.zeros((B,), device=resid.device, dtype=torch.int64)

            for bi, gi in enumerate(group_ids):
                gd = pcg_groups[gi]
                r_g = gd["r"]
                m = int(r_g.numel())
                lengths[bi] = m
                n_nodes[bi] = int(gd["n_nodes"])
                grp_idx[bi] = int(gd.get("group_index", gi))
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

                edge_vals = beta[bi] * w_sqrt[bi, :m] * r[bi, :m]
                b[bi].index_add_(0, u[bi, :m], -edge_vals)
                b[bi].index_add_(0, v[bi, :m], edge_vals)

            x0_batch = None
            if use_warm and isinstance(x0_cache, torch.Tensor):
                try:
                    x0_batch = torch.zeros((B, bucket_size), device=resid.device, dtype=resid.dtype)
                    for bi in range(B):
                        gix = int(grp_idx[bi].item())
                        nn = int(n_nodes[bi].item())
                        if nn > 0:
                            x0_batch[bi, :nn] = x0_cache[gix, :nn].detach()
                except Exception:
                    x0_batch = None

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
                x0=x0_batch,
            )
            if use_warm and isinstance(x0_cache, torch.Tensor):
                try:
                    for bi in range(B):
                        gix = int(grp_idx[bi].item())
                        nn = int(n_nodes[bi].item())
                        if nn > 0:
                            x0_cache[gix, :nn] = x[bi, :nn].detach()
                    cache_entry["pcg_x0"] = x0_cache.detach()
                except Exception:
                    pass
            metrics.n_groups_pcg += int(B)
            metrics.pcg_iters_sum += int(iters.sum().item())
            metrics.pcg_iters_max = max(int(metrics.pcg_iters_max), int(iters.max().item()))
            metrics.n_groups_pcg_fail += int((~ok).sum().item())
            metrics.n_groups_fallback_diag += int((~ok).sum().item())

            x_flat = x
            ax = x_flat.gather(1, v) - x_flat.gather(1, u)
            for bi in range(B):
                m = int(lengths[bi].item())
                r_g = r[bi, :m]
                if not bool(ok[bi]):
                    quad_sum_nonlocal = _CollapsedQuadNoGrad.apply(r_g, r_g / (sigma[bi].square().clamp_min(1e-24)))
                    nonlocal_quad.append(quad_sum_nonlocal)
                    continue
                ax_g = ax[bi, :m]
                u_edge = beta[bi] * (r_g - (w_sqrt[bi, :m] * ax_g))
                quad_sum_nonlocal = _CollapsedQuadNoGrad.apply(r_g, u_edge)
                nonlocal_quad.append(quad_sum_nonlocal)

        nonlocal_quad: list[torch.Tensor] = []
        for (bsz, _edge_bin), group_ids in bucket_map.items():
            _solve_bucket(int(bsz), group_ids)
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
    metrics.solve_ms = float(1000.0 * (time.perf_counter() - t_solve0))

    if w_cnt > 0:
        metrics.weight_mean = float(w_sum / w_cnt)
    metrics.weight_max = float(w_max_all)
    return quad_sum, metrics, cache
