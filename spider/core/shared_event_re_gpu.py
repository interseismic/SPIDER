from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


class _CollapsedQuadNoGrad(torch.autograd.Function):
    """
    Compute 0.5 * r^T u while defining the gradient w.r.t. r as u.

    We intentionally do NOT backprop through the solver that produced u.
    This mirrors spider.core.modeling._CollapsedQuad without importing it here.
    """

    @staticmethod
    def forward(ctx, r: torch.Tensor, u: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        u_det = u.detach()
        ctx.save_for_backward(u_det)
        return 0.5 * (r * u_det).sum()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (u_det,) = ctx.saved_tensors
        grad_r = grad_out * u_det
        return grad_r, None


@dataclass
class SharedEventReGpuMetrics:
    n_groups_total: int = 0
    n_groups_pcg: int = 0
    n_groups_fallback_diag: int = 0
    n_groups_rows_cap: int = 0
    n_groups_nodes_cap: int = 0
    n_groups_tau_zero: int = 0
    max_rows_seen: int = 0
    max_nodes_seen: int = 0
    max_rows_all: int = 0
    max_nodes_all: int = 0
    ms_total: float = 0.0


def _group_by_keys_gpu(keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys_sorted, perm = torch.sort(keys)
    if keys_sorted.numel() > 0:
        is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
        is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
        starts = torch.nonzero(is_new, as_tuple=False).reshape(-1)
        ends = torch.cat(
            [starts[1:], torch.tensor([keys_sorted.numel()], device=starts.device, dtype=starts.dtype)]
        )
    else:
        starts = torch.zeros((0,), device=keys.device, dtype=torch.int64)
        ends = torch.zeros((0,), device=keys.device, dtype=torch.int64)
    return keys_sorted, perm, starts, ends


def _build_group_ids(starts: torch.Tensor, ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = (ends - starts).to(torch.int64)
    if lengths.numel() == 0:
        return torch.zeros((0,), device=starts.device, dtype=torch.int64), lengths
    group_ids = torch.repeat_interleave(
        torch.arange(int(lengths.numel()), device=starts.device, dtype=torch.int64),
        lengths,
    )
    return group_ids, lengths


def _build_local_node_indices(
    u: torch.Tensor,
    v: torch.Tensor,
    group_ids: torch.Tensor,
    n_groups: int,
    *,
    max_node_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if u.numel() == 0:
        empty = torch.zeros((0,), device=u.device, dtype=torch.int64)
        return empty, empty, torch.zeros((n_groups,), device=u.device, dtype=torch.int64)

    nodes = torch.cat([u, v], dim=0).to(torch.int64)
    g_nodes = torch.cat([group_ids, group_ids], dim=0).to(torch.int64)
    base = int(max(max_node_id + 1, 1))
    key = g_nodes * base + nodes
    order = torch.argsort(key)
    key_sorted = key[order]
    g_sorted = g_nodes[order]

    is_new = torch.ones_like(key_sorted, dtype=torch.bool)
    is_new[1:] = key_sorted[1:] != key_sorted[:-1]
    group_change = torch.ones_like(g_sorted, dtype=torch.bool)
    group_change[1:] = g_sorted[1:] != g_sorted[:-1]

    # local_id_sorted should be 0..(n_unique_in_group-1) for each group.
    c_new = torch.cumsum(is_new.to(torch.int64), dim=0)
    # Track c_new at group starts, then carry forward to compute per-group offsets.
    start_vals = torch.where(group_change, c_new, torch.zeros_like(c_new))
    start_cumsum = torch.cummax(start_vals, dim=0).values
    local_id_sorted = c_new - start_cumsum

    local_id = torch.empty_like(nodes, dtype=torch.int64)
    local_id[order] = local_id_sorted

    local_u = local_id[: u.numel()]
    local_v = local_id[u.numel() :]

    g_unique = g_sorted[is_new]
    counts = torch.zeros((n_groups,), device=nodes.device, dtype=torch.int64)
    if g_unique.numel() > 0:
        counts.index_add_(0, g_unique, torch.ones_like(g_unique, dtype=torch.int64))
    return local_u, local_v, counts


def _laplacian_mv_batched(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    *,
    max_nodes: int,
) -> torch.Tensor:
    batch = int(x.shape[0])
    if batch == 0 or u.numel() == 0:
        return torch.zeros_like(x)
    offsets = torch.arange(batch, device=x.device, dtype=u.dtype).unsqueeze(1) * int(max_nodes)
    u_off = u + offsets
    v_off = v + offsets
    mask = (u >= 0) & (v >= 0)
    u_flat = u_off[mask]
    v_flat = v_off[mask]
    x_flat = x.reshape(-1)
    tmp = x_flat[u_flat] - x_flat[v_flat]
    out_flat = torch.zeros_like(x_flat)
    out_flat.index_add_(0, u_flat, tmp)
    out_flat.index_add_(0, v_flat, -tmp)
    return out_flat.reshape(batch, int(max_nodes))


def _laplacian_mv_batched_flat(
    u_flat: torch.Tensor,
    v_flat: torch.Tensor,
    x: torch.Tensor,
    *,
    batch: int,
    max_nodes: int,
) -> torch.Tensor:
    if batch == 0 or u_flat.numel() == 0:
        return torch.zeros_like(x)
    x_flat = x.reshape(-1)
    xu = x_flat.index_select(0, u_flat)
    xv = x_flat.index_select(0, v_flat)
    diff = xu - xv
    out_flat = torch.zeros((batch * int(max_nodes),), device=x.device, dtype=x.dtype)
    out_flat.index_add_(0, u_flat, diff)
    out_flat.index_add_(0, v_flat, -diff)
    return out_flat.reshape(batch, int(max_nodes))


def _laplacian_w_mv_batched(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    x: torch.Tensor,
    *,
    max_nodes: int,
) -> torch.Tensor:
    batch = int(x.shape[0])
    if batch == 0 or u.numel() == 0:
        return torch.zeros_like(x)
    offsets = torch.arange(batch, device=x.device, dtype=u.dtype).unsqueeze(1) * int(max_nodes)
    u_off = u + offsets
    v_off = v + offsets
    mask = (u >= 0) & (v >= 0)
    u_flat = u_off[mask]
    v_flat = v_off[mask]
    x_flat = x.reshape(-1)
    w_flat = w[mask].to(dtype=x.dtype)
    tmp = w_flat * (x_flat[u_flat] - x_flat[v_flat])
    out_flat = torch.zeros_like(x_flat)
    out_flat.index_add_(0, u_flat, tmp)
    out_flat.index_add_(0, v_flat, -tmp)
    return out_flat.reshape(batch, int(max_nodes))


def _edge_ax_batched(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    *,
    max_nodes: int,
) -> torch.Tensor:
    if u.numel() == 0:
        return torch.zeros_like(u, dtype=x.dtype)
    batch = int(x.shape[0])
    offsets = torch.arange(batch, device=x.device, dtype=u.dtype).unsqueeze(1) * int(max_nodes)
    u_off = u + offsets
    v_off = v + offsets
    mask = (u >= 0) & (v >= 0)
    x_flat = x.reshape(-1)
    ax = torch.zeros_like(u, dtype=x.dtype)
    u_flat = u_off[mask]
    v_flat = v_off[mask]
    ax[mask] = x_flat[v_flat] - x_flat[u_flat]
    return ax


def _edge_ax_from_flat(
    *,
    u_flat: torch.Tensor,
    v_flat: torch.Tensor,
    edge_flat: torch.Tensor,
    x: torch.Tensor,
    max_edges: int,
) -> torch.Tensor:
    if edge_flat.numel() == 0:
        return torch.zeros((int(x.shape[0]), int(max_edges)), device=x.device, dtype=x.dtype)
    x_flat = x.reshape(-1)
    ax_valid = x_flat.index_select(0, v_flat) - x_flat.index_select(0, u_flat)
    out = torch.zeros((int(x.shape[0]) * int(max_edges),), device=x.device, dtype=x.dtype)
    out.index_copy_(0, edge_flat, ax_valid)
    return out.reshape(int(x.shape[0]), int(max_edges))


def _pcg_batched(
    u: torch.Tensor,
    v: torch.Tensor,
    resid: torch.Tensor,
    n_nodes: torch.Tensor,
    *,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    max_iters: int,
    tol: float,
    x0: torch.Tensor | None = None,
    return_x: bool = False,
    u_flat: torch.Tensor | None = None,
    v_flat: torch.Tensor | None = None,
    edge_flat: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Batched PCG solver for (alpha * I + beta * L) x = b.
    Returns u_edge = beta * (resid - A x) for each edge.
    Supports optional flattened indices (u_flat/v_flat/edge_flat) to avoid
    per-iteration offset/mask construction.
    """
    if resid.ndim == 3:
        if isinstance(x0, torch.Tensor) or isinstance(u_flat, torch.Tensor):
            raise ValueError("multi-RHS PCG does not support warm-start or flat indices")
        B, K, E = resid.shape
        resid2 = resid.reshape(B * K, E)
        u2 = u.repeat_interleave(K, dim=0)
        v2 = v.repeat_interleave(K, dim=0)
        n_nodes2 = n_nodes.repeat_interleave(K, dim=0)
        alpha2 = alpha.repeat_interleave(K)
        beta2 = beta.repeat_interleave(K)
        out = _pcg_batched(
            u2, v2, resid2, n_nodes2,
            beta=beta2, alpha=alpha2, max_iters=max_iters, tol=tol,
            x0=None, return_x=False,
        )
        return out.reshape(B, K, E)

    batch = int(resid.shape[0])
    max_edges = int(resid.shape[1])
    if batch == 0 or max_edges == 0:
        return torch.zeros_like(resid)
    max_nodes = int(n_nodes.max().item()) if n_nodes.numel() > 0 else 0
    if max_nodes <= 0:
        return torch.zeros_like(resid)

    # Build b and deg
    resid_scaled = resid * beta.unsqueeze(1)
    ones = torch.ones_like(resid)
    mask = (u >= 0) & (v >= 0)
    resid_scaled = resid_scaled * mask

    b = _scatter_node_accum(u, v, -resid_scaled, resid_scaled, max_nodes=max_nodes)
    deg = _scatter_node_accum(u, v, ones, ones, max_nodes=max_nodes)
    diag = (alpha.unsqueeze(1) + beta.unsqueeze(1) * deg).clamp_min(1e-12)

    def _A(x: torch.Tensor) -> torch.Tensor:
        if isinstance(u_flat, torch.Tensor) and isinstance(v_flat, torch.Tensor):
            lap = _laplacian_mv_batched_flat(u_flat, v_flat, x, batch=batch, max_nodes=max_nodes)
        else:
            lap = _laplacian_mv_batched(u, v, x, max_nodes=max_nodes)
        return alpha.unsqueeze(1) * x + beta.unsqueeze(1) * lap

    # PCG
    if isinstance(x0, torch.Tensor) and x0.shape == (batch, max_nodes):
        x = x0.clone()
        r = b - _A(x)
    else:
        x = torch.zeros((batch, max_nodes), device=resid.device, dtype=resid.dtype)
        r = b.clone()
    z = r / diag
    p = z.clone()
    rz_old = (r * z).sum(dim=1)
    b_norm = b.norm(dim=1).clamp_min(1e-12)

    for _ in range(int(max_iters)):
        Ap = _A(p)
        denom = (p * Ap).sum(dim=1).clamp_min(1e-20)
        r_norm = r.norm(dim=1) / b_norm
        active = r_norm > float(tol)
        if not bool(active.any()):
            break
        a = torch.where(active, rz_old / denom, torch.zeros_like(rz_old))
        x = x + a.unsqueeze(1) * p
        r = r - a.unsqueeze(1) * Ap
        z = r / diag
        rz_new = (r * z).sum(dim=1)
        bcoef = torch.where(active, rz_new / rz_old.clamp_min(1e-30), torch.zeros_like(rz_new))
        p = z + bcoef.unsqueeze(1) * p
        rz_old = torch.where(active, rz_new, rz_old)

    if isinstance(u_flat, torch.Tensor) and isinstance(v_flat, torch.Tensor) and isinstance(edge_flat, torch.Tensor):
        ax = _edge_ax_from_flat(
            u_flat=u_flat,
            v_flat=v_flat,
            edge_flat=edge_flat,
            x=x,
            max_edges=max_edges,
        )
    else:
        ax = _edge_ax_batched(u, v, x, max_nodes=max_nodes)
    u_edge = beta.unsqueeze(1) * (resid - ax)
    if return_x:
        return u_edge, x
    return u_edge


def _pcg_batched_weighted(
    u: torch.Tensor,
    v: torch.Tensor,
    resid: torch.Tensor,
    w: torch.Tensor,
    n_nodes: torch.Tensor,
    *,
    alpha: torch.Tensor,
    max_iters: int,
    tol: float,
) -> torch.Tensor:
    """
    Batched PCG solver for (alpha * I + L_w) x = b.
    Returns x for each group (node space).
    """
    batch = int(resid.shape[0])
    max_edges = int(resid.shape[1])
    if batch == 0 or max_edges == 0:
        return torch.zeros((batch, 0), device=resid.device, dtype=resid.dtype)
    max_nodes = int(n_nodes.max().item()) if n_nodes.numel() > 0 else 0
    if max_nodes <= 0:
        return torch.zeros((batch, 0), device=resid.device, dtype=resid.dtype)

    mask = (u >= 0) & (v >= 0)
    w_mask = w * mask
    wr = w_mask * resid

    b = _scatter_node_accum(u, v, wr, -wr, max_nodes=max_nodes)
    deg = _scatter_node_accum(u, v, w_mask, w_mask, max_nodes=max_nodes)
    diag = (alpha.unsqueeze(1) + deg).clamp_min(1e-12)

    x = torch.zeros((batch, max_nodes), device=resid.device, dtype=resid.dtype)
    r = b.clone()
    z = r / diag
    p = z.clone()
    rz_old = (r * z).sum(dim=1)
    b_norm = b.norm(dim=1).clamp_min(1e-12)

    for _ in range(int(max_iters)):
        Ap = alpha.unsqueeze(1) * p + _laplacian_w_mv_batched(u, v, w_mask, p, max_nodes=max_nodes)
        denom = (p * Ap).sum(dim=1).clamp_min(1e-20)
        r_norm = r.norm(dim=1) / b_norm
        active = r_norm > float(tol)
        if not bool(active.any()):
            break
        a = torch.where(active, rz_old / denom, torch.zeros_like(rz_old))
        x = x + a.unsqueeze(1) * p
        r = r - a.unsqueeze(1) * Ap
        z = r / diag
        rz_new = (r * z).sum(dim=1)
        bcoef = torch.where(active, rz_new / rz_old.clamp_min(1e-30), torch.zeros_like(rz_new))
        p = z + bcoef.unsqueeze(1) * p
        rz_old = torch.where(active, rz_new, rz_old)

    return x


def _edge_sum_batched(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return x[u] + x[v] for each edge (batched)."""
    if u.numel() == 0:
        return torch.zeros_like(u, dtype=x.dtype)
    batch = int(x.shape[0])
    max_nodes = int(x.shape[1])
    offsets = torch.arange(batch, device=x.device, dtype=u.dtype).unsqueeze(1) * int(max_nodes)
    u_off = u + offsets
    v_off = v + offsets
    mask = (u >= 0) & (v >= 0)
    x_flat = x.reshape(-1)
    out = torch.zeros_like(u, dtype=x.dtype)
    u_flat = u_off[mask]
    v_flat = v_off[mask]
    out[mask] = x_flat[u_flat] + x_flat[v_flat]
    return out


def _pcg_batched_sum_weighted(
    u: torch.Tensor,
    v: torch.Tensor,
    b_edge: torch.Tensor,
    w: torch.Tensor,
    n_nodes: torch.Tensor,
    *,
    alpha_nodes: torch.Tensor,
    max_iters: int,
    tol: float,
) -> torch.Tensor:
    """
    Batched PCG solver for (diag(alpha) + A^T W A) x = b,
    where each row has 1 at u and 1 at v.
    Returns x for each group (node space).
    """
    batch = int(b_edge.shape[0])
    max_edges = int(b_edge.shape[1])
    if batch == 0 or max_edges == 0:
        return torch.zeros((batch, 0), device=b_edge.device, dtype=b_edge.dtype)
    max_nodes = int(alpha_nodes.shape[1])
    if max_nodes <= 0:
        return torch.zeros((batch, 0), device=b_edge.device, dtype=b_edge.dtype)

    mask = (u >= 0) & (v >= 0)
    w_mask = w * mask
    b_mask = b_edge * mask

    b = _scatter_node_accum(u, v, b_mask, b_mask, max_nodes=max_nodes)
    deg = _scatter_node_accum(u, v, w_mask, w_mask, max_nodes=max_nodes)
    diag = (alpha_nodes + deg).clamp_min(1e-12)

    x = torch.zeros((batch, max_nodes), device=b_edge.device, dtype=b_edge.dtype)
    r = b.clone()
    z = r / diag
    p = z.clone()
    rz_old = (r * z).sum(dim=1)
    b_norm = b.norm(dim=1).clamp_min(1e-12)

    for _ in range(int(max_iters)):
        px = _edge_sum_batched(u, v, p)
        Ap = (alpha_nodes * p) + _scatter_node_accum(u, v, w_mask * px, w_mask * px, max_nodes=max_nodes)
        denom = (p * Ap).sum(dim=1).clamp_min(1e-20)
        r_norm = r.norm(dim=1) / b_norm
        active = r_norm > float(tol)
        if not bool(active.any()):
            break
        a = torch.where(active, rz_old / denom, torch.zeros_like(rz_old))
        x = x + a.unsqueeze(1) * p
        r = r - a.unsqueeze(1) * Ap
        z = r / diag
        rz_new = (r * z).sum(dim=1)
        bcoef = torch.where(active, rz_new / rz_old.clamp_min(1e-30), torch.zeros_like(rz_new))
        p = z + bcoef.unsqueeze(1) * p
        rz_old = torch.where(active, rz_new, rz_old)

    return x


@dataclass
class SlownessReGpuMetrics:
    n_groups_total: int = 0
    n_groups_pcg: int = 0
    n_groups_fallback_diag: int = 0
    n_groups_rows_cap: int = 0
    n_groups_nodes_cap: int = 0
    n_groups_tau_zero: int = 0
    max_rows_seen: int = 0
    max_nodes_seen: int = 0
    max_rows_all: int = 0
    max_nodes_all: int = 0
    ms_total: float = 0.0
    resid_rms: float = float("nan")
    pred_rms: float = float("nan")
    g_rms: float = float("nan")


def _scatter_node_accum(
    u: torch.Tensor,
    v: torch.Tensor,
    val_u: torch.Tensor,
    val_v: torch.Tensor,
    *,
    max_nodes: int,
) -> torch.Tensor:
    batch = int(u.shape[0])
    offsets = torch.arange(batch, device=u.device, dtype=u.dtype).unsqueeze(1) * int(max_nodes)
    u_off = u + offsets
    v_off = v + offsets
    mask = (u >= 0) & (v >= 0)
    u_flat = u_off[mask]
    v_flat = v_off[mask]
    out_flat = torch.zeros((batch * int(max_nodes),), device=u.device, dtype=val_u.dtype)
    out_flat.index_add_(0, u_flat, val_u[mask])
    out_flat.index_add_(0, v_flat, val_v[mask])
    return out_flat.reshape(batch, int(max_nodes))


def _build_shared_event_re_batch_cache(
    *,
    eligible_groups: torch.Tensor,
    lengths: torch.Tensor,
    n_nodes: torch.Tensor,
    group_ids: torch.Tensor,
    edge_pos: torch.Tensor,
    local_u: torch.Tensor,
    local_v: torch.Tensor,
    n_groups: int,
    max_groups_per_batch: int,
    max_edges_per_batch: int,
    device: torch.device,
) -> list[dict]:
    batches: list[dict] = []
    if eligible_groups.numel() == 0:
        return batches
    max_groups = int(max_groups_per_batch) if int(max_groups_per_batch) > 0 else int(eligible_groups.numel())
    max_edges = int(max_edges_per_batch) if int(max_edges_per_batch) > 0 else 0
    i0 = 0
    while i0 < int(eligible_groups.numel()):
        if max_edges > 0:
            edge_sum = 0
            i1 = i0
            while i1 < int(eligible_groups.numel()):
                g = int(eligible_groups[i1].item())
                g_len = int(lengths[g].item())
                if i1 > i0 and (edge_sum + g_len) > max_edges:
                    break
                edge_sum += g_len
                i1 += 1
                if (i1 - i0) >= max_groups:
                    break
            if i1 <= i0:
                i1 = i0 + 1
        else:
            i1 = min(i0 + max_groups, int(eligible_groups.numel()))

        grp = eligible_groups[i0:i1]
        B = int(grp.numel())
        if B <= 0:
            i0 = i1
            continue
        max_edges_in_batch = int(lengths.index_select(0, grp).max().item())
        max_nodes_in_batch = int(n_nodes.index_select(0, grp).max().item())

        u_pad = torch.full((B, max_edges_in_batch), -1, device=device, dtype=torch.int64)
        v_pad = torch.full((B, max_edges_in_batch), -1, device=device, dtype=torch.int64)

        gmap = torch.full((n_groups,), -1, device=device, dtype=torch.int64)
        gmap.index_copy_(0, grp, torch.arange(B, device=device, dtype=torch.int64))
        row = gmap.index_select(0, group_ids)
        col = edge_pos.to(dtype=torch.int64)
        mask = (row >= 0) & (col >= 0) & (col < int(max_edges_in_batch))
        if bool(mask.any()):
            flat = row[mask] * int(max_edges_in_batch) + col[mask]
            u_pad.view(-1).index_copy_(0, flat, local_u[mask])
            v_pad.view(-1).index_copy_(0, flat, local_v[mask])
            edge_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            row_valid = row[mask].to(torch.int64)
            u_valid = local_u[mask].to(torch.int64)
            v_valid = local_v[mask].to(torch.int64)
            offsets = row_valid * int(max_nodes_in_batch)
            u_flat = u_valid + offsets
            v_flat = v_valid + offsets
        else:
            flat = torch.zeros((0,), device=device, dtype=torch.int64)
            edge_idx = torch.zeros((0,), device=device, dtype=torch.int64)
            row_valid = torch.zeros((0,), device=device, dtype=torch.int64)
            u_flat = torch.zeros((0,), device=device, dtype=torch.int64)
            v_flat = torch.zeros((0,), device=device, dtype=torch.int64)

        batches.append(
            {
                "grp": grp,
                "u_pad": u_pad,
                "v_pad": v_pad,
                "edge_idx": edge_idx,
                "flat": flat,
                "row_valid": row_valid,
                "u_flat": u_flat,
                "v_flat": v_flat,
                "max_edges": max_edges_in_batch,
                "max_nodes": max_nodes_in_batch,
                "n_nodes_b": n_nodes.index_select(0, grp).to(torch.int64),
            }
        )
        i0 = i1
    return batches


def compute_quad_gpu(
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
    pcg_max_iters: int,
    pcg_tol: float,
    max_rows_per_group: int,
    max_nodes_per_group: int,
    fallback_to_diag: bool,
    max_groups_per_batch: int = 64,
    max_edges_per_batch: int = 0,
    reuse_pcg_init: bool = False,
    group_cap: Optional[int] = None,
    enable_profile: bool = False,
    precomputed: Optional[dict] = None,
) -> Tuple[torch.Tensor, SharedEventReGpuMetrics]:
    """
    Compute total quadratic term sum for shared_event_re using a GPU batched PCG prototype.
    Returns (quad_sum, metrics).
    """
    t0 = time.perf_counter()
    if isinstance(precomputed, dict):
        perm = precomputed.get("perm", None)
        starts = precomputed.get("starts", None)
        ends = precomputed.get("ends", None)
        lengths = precomputed.get("lengths", None)
        local_u = precomputed.get("local_u", None)
        local_v = precomputed.get("local_v", None)
        n_nodes = precomputed.get("n_nodes", None)
        ph_group = precomputed.get("ph_group", None)
        group_ids = precomputed.get("group_ids", None)
        edge_pos = precomputed.get("edge_pos", None)
        x0_cache = precomputed.get("x0", None)
        x0_n_nodes = precomputed.get("x0_n_nodes", None)
        ok = (
            isinstance(perm, torch.Tensor) and perm.ndim == 1
            and isinstance(starts, torch.Tensor) and starts.ndim == 1
            and isinstance(ends, torch.Tensor) and ends.ndim == 1
            and int(starts.numel()) == int(ends.numel())
            and isinstance(lengths, torch.Tensor) and lengths.ndim == 1
            and isinstance(local_u, torch.Tensor) and local_u.ndim == 1
            and isinstance(local_v, torch.Tensor) and local_v.ndim == 1
            and isinstance(n_nodes, torch.Tensor) and n_nodes.ndim == 1
            and isinstance(ph_group, torch.Tensor) and ph_group.ndim == 1
            and isinstance(group_ids, torch.Tensor) and group_ids.ndim == 1
            and isinstance(edge_pos, torch.Tensor) and edge_pos.ndim == 1
        )
    else:
        ok = False
        perm = starts = ends = lengths = local_u = local_v = n_nodes = ph_group = group_ids = edge_pos = None
        x0_cache = x0_n_nodes = None

    if not ok:
        keys_sorted, perm, starts, ends = _group_by_keys_gpu(keys)
        lengths = (ends - starts).to(torch.int64)
    if group_cap is not None and int(group_cap) >= 0:
        ncap = int(group_cap)
        ncap = min(ncap, int(starts.numel()))
        if ncap < int(starts.numel()):
            max_edge = int(ends[ncap - 1].item()) if ncap > 0 else 0
            perm = perm[:max_edge]
            starts = starts[:ncap]
            ends = ends[:ncap]
            lengths = lengths[:ncap]
    n_groups = int(starts.numel())
    metrics = SharedEventReGpuMetrics(n_groups_total=n_groups)
    if n_groups == 0 or perm.numel() == 0:
        return torch.tensor(0.0, device=idx.device, dtype=resid.dtype), metrics

    if int(n_comp) <= 0 or int(n_sta) <= 0:
        sigma_perm = sigma.index_select(0, perm).clamp_min(1e-12)
        quad_sum = _CollapsedQuadNoGrad.apply(resid.index_select(0, perm), resid.index_select(0, perm) / sigma_perm.square().clamp_min(1e-24))
        return quad_sum, metrics
    try:
        metrics.max_rows_all = int(lengths.max().item()) if lengths.numel() > 0 else 0
        metrics.max_nodes_all = int(n_nodes.max().item()) if n_nodes.numel() > 0 else 0
    except Exception:
        metrics.max_rows_all = 0
        metrics.max_nodes_all = 0

    idx_perm = idx.index_select(0, perm)
    resid_perm = resid.index_select(0, perm)
    if not ok:
        ph_perm = ph_id.index_select(0, perm)
        group_ids, _ = _build_group_ids(starts, ends)
        u = idx_perm[:, 0].to(torch.int64)
        v = idx_perm[:, 1].to(torch.int64)
        max_node_id = int(torch.max(torch.stack([u.max(), v.max()])).item()) if u.numel() > 0 else 0
        local_u, local_v, n_nodes = _build_local_node_indices(
            u, v, group_ids, n_groups, max_node_id=max_node_id
        )
        ph_group = ph_perm.index_select(0, starts)
        edge_idx = torch.arange(int(perm.numel()), device=perm.device, dtype=starts.dtype)
        edge_pos = edge_idx - starts.index_select(0, group_ids)
    sigma_g = torch.where(ph_group < 0.5, sigma_p, sigma_s).to(resid.dtype)
    tau_g = torch.where(ph_group < 0.5, torch.tensor(float(tau_p), device=ph_group.device), torch.tensor(float(tau_s), device=ph_group.device))

    quad_sum = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
    eligible = (
        (lengths > 0)
        & (lengths <= int(max_rows_per_group))
        & (n_nodes <= int(max_nodes_per_group))
        & (tau_g > 0.0)
    )
    too_many_rows = lengths > int(max_rows_per_group)
    too_many_nodes = n_nodes > int(max_nodes_per_group)
    tau_zero = tau_g <= 0.0
    try:
        metrics.n_groups_rows_cap = int(too_many_rows.sum().item())
        metrics.n_groups_nodes_cap = int(too_many_nodes.sum().item())
        metrics.n_groups_tau_zero = int(tau_zero.sum().item())
    except Exception:
        pass

    # Fallback diag for oversized groups
    fallback_mask = ~eligible
    if bool(fallback_mask.any()):
        metrics.n_groups_fallback_diag = int(fallback_mask.sum().item())
        if not fallback_to_diag:
            raise ValueError("shared_event_re_gpu: oversized group encountered and fallback_to_diag=false")
        for g in torch.nonzero(fallback_mask, as_tuple=False).reshape(-1).tolist():
            s = int(starts[g].item())
            e = int(ends[g].item())
            if e <= s:
                continue
            r_g = resid_perm[s:e]
            sigma = sigma_g[g].clamp_min(1e-12)
            quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(
                r_g, r_g / sigma.square().clamp_min(1e-24)
            )

    # Batched PCG for eligible groups
    eligible_groups = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if eligible_groups.numel() > 0:
        metrics.n_groups_pcg = int(eligible_groups.numel())
        # Sort by group length to reduce padding inside each batch.
        try:
            order = torch.argsort(lengths.index_select(0, eligible_groups))
            eligible_groups = eligible_groups.index_select(0, order)
        except Exception:
            pass

        # Build or reuse cached batch packing to avoid per-epoch packing overhead.
        batch_cache = None
        cache_key = (
            int(max_groups_per_batch),
            int(max_edges_per_batch),
            int(max_rows_per_group),
            int(max_nodes_per_group),
        )
        if isinstance(precomputed, dict):
            try:
                cache_blob = precomputed.get("batch_cache", None)
                if isinstance(cache_blob, dict) and cache_blob.get("cache_key", None) == cache_key:
                    batch_cache = cache_blob.get("batches", None)
            except Exception:
                batch_cache = None
        if not isinstance(batch_cache, list):
            batch_cache = _build_shared_event_re_batch_cache(
                eligible_groups=eligible_groups,
                lengths=lengths,
                n_nodes=n_nodes,
                group_ids=group_ids,
                edge_pos=edge_pos,
                local_u=local_u,
                local_v=local_v,
                n_groups=int(n_groups),
                max_groups_per_batch=int(max_groups_per_batch),
                max_edges_per_batch=int(max_edges_per_batch),
                device=idx.device,
            )
            if isinstance(precomputed, dict):
                precomputed["batch_cache"] = {"cache_key": cache_key, "batches": batch_cache}

        # Optional warm-start cache for PCG (per-group node-space solution).
        use_warm_start = bool(reuse_pcg_init) and isinstance(precomputed, dict)
        if use_warm_start:
            try:
                n_groups_total = int(starts.numel())
                max_nodes_all = int(n_nodes.max().item()) if n_nodes.numel() > 0 else 0
                ok_x0 = (
                    isinstance(x0_cache, torch.Tensor)
                    and isinstance(x0_n_nodes, torch.Tensor)
                    and x0_cache.ndim == 2
                    and int(x0_cache.shape[0]) == n_groups_total
                    and int(x0_n_nodes.numel()) == n_groups_total
                )
                if not ok_x0 or int(x0_cache.shape[1]) < int(max_nodes_all):
                    x0_cache_new = torch.zeros(
                        (n_groups_total, max_nodes_all),
                        device=resid.device,
                        dtype=resid.dtype,
                    )
                    if ok_x0 and int(x0_cache.shape[1]) > 0:
                        x0_cache_new[:, : int(x0_cache.shape[1])] = x0_cache
                    x0_cache = x0_cache_new
                    x0_n_nodes = n_nodes.detach()
                    precomputed["x0"] = x0_cache
                    precomputed["x0_n_nodes"] = x0_n_nodes
                else:
                    x0_n_nodes = n_nodes.detach()
                    precomputed["x0_n_nodes"] = x0_n_nodes
            except Exception:
                use_warm_start = False
                x0_cache = None
        for batch in batch_cache:
            grp = batch["grp"]
            B = int(grp.numel())
            if B <= 0:
                continue
            max_edges_in_batch = int(batch["max_edges"])
            max_nodes_in_batch = int(batch["max_nodes"])
            metrics.max_rows_seen = max(metrics.max_rows_seen, max_edges_in_batch)
            metrics.max_nodes_seen = max(metrics.max_nodes_seen, max_nodes_in_batch)

            u_pad = batch["u_pad"]
            v_pad = batch["v_pad"]
            edge_idx = batch["edge_idx"]
            flat = batch["flat"]
            u_flat = batch["u_flat"]
            v_flat = batch["v_flat"]
            n_nodes_b = batch["n_nodes_b"]

            r_pad = torch.zeros((B, max_edges_in_batch), device=resid.device, dtype=resid.dtype)
            if edge_idx.numel() > 0:
                r_pad.view(-1).index_copy_(0, flat, resid_perm.index_select(0, edge_idx))

            sigma_b = sigma_g.index_select(0, grp)
            tau_b = tau_g.index_select(0, grp)
            beta = (1.0 / sigma_b.square().clamp_min(1e-24)).to(resid.dtype)
            alpha = (1.0 / (tau_b.square().clamp_min(1e-24))) + float(jitter0)
            x0_batch = None
            if use_warm_start and isinstance(x0_cache, torch.Tensor):
                try:
                    x0_batch = x0_cache.index_select(0, grp)[:, :max_nodes_in_batch]
                    if int(max_nodes_in_batch) > 0:
                        mask_nodes = (
                            torch.arange(int(max_nodes_in_batch), device=x0_batch.device).unsqueeze(0)
                            < n_nodes_b.unsqueeze(1)
                        )
                        x0_batch = x0_batch * mask_nodes.to(dtype=x0_batch.dtype)
                except Exception:
                    x0_batch = None
            u_edge = _pcg_batched(
                u_pad, v_pad, r_pad, n_nodes_b,
                beta=beta, alpha=alpha, max_iters=int(pcg_max_iters), tol=float(pcg_tol),
                x0=x0_batch,
                return_x=bool(use_warm_start),
                u_flat=u_flat,
                v_flat=v_flat,
                edge_flat=flat,
            )
            if use_warm_start and isinstance(u_edge, tuple):
                u_edge, x_sol = u_edge
                try:
                    if isinstance(x0_cache, torch.Tensor):
                        x0_cache.index_copy_(0, grp, x_sol)
                except Exception:
                    pass
            quad_sum = quad_sum + _CollapsedQuadNoGrad.apply(r_pad, u_edge)

    if enable_profile:
        metrics.ms_total = float(1000.0 * (time.perf_counter() - t0))
    return quad_sum, metrics
