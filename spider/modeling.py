from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn


def compute_travel_times(idx: torch.Tensor, y: torch.Tensor, X_src: torch.Tensor, dX_src: torch.Tensor, model: nn.Module) -> torch.Tensor:
    src1 = X_src[idx[:, 0]] + dX_src[idx[:, 0]]
    src2 = X_src[idx[:, 1]] + dX_src[idx[:, 1]]
    X_rec = y[:, 1:4]
    phase = y[:, 4:5]
    coords1 = torch.cat([src1[:, :3], X_rec, phase], dim=1)
    coords2 = torch.cat([src2[:, :3], X_rec, phase], dim=1)
    batch_input = torch.cat([coords1, coords2], dim=0)
    T_pred_all = model(batch_input).squeeze()
    n = src1.shape[0]
    T1 = T_pred_all[:n]
    T2 = T_pred_all[n:]
    return (T2 + src2[:, 3]) - (T1 + src1[:, 3])


def compute_residuals(idx: torch.Tensor, y: torch.Tensor, X_src: torch.Tensor, dX_src: torch.Tensor, model: nn.Module) -> torch.Tensor:
    dt_pred = compute_travel_times(idx, y, X_src, dX_src, model)
    return y[:, 0] - dt_pred


def compute_linearization_error_ratio(
    idx: torch.Tensor,
    y: torch.Tensor,
    X_src: torch.Tensor,
    dX_src: torch.Tensor,
    model: nn.Module,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src1 = X_src[idx[:, 0]] + dX_src[idx[:, 0]]
    src2 = X_src[idx[:, 1]] + dX_src[idx[:, 1]]
    x1 = src1[:, :3]
    x2 = src2[:, :3]
    X_rec = y[:, 1:4]
    phase = y[:, 4:5]
    x1_leaf = x1.detach().clone().requires_grad_(True)
    coords1 = torch.cat([x1_leaf, X_rec.detach(), phase.detach()], dim=1)
    T1 = model(coords1).squeeze()
    g1 = torch.autograd.grad(T1.sum(), x1_leaf, retain_graph=False, create_graph=False, allow_unused=False)[0]
    dx = (x2.detach() - x1_leaf.detach())
    dx_norm = torch.linalg.norm(dx, dim=1)
    g_norm = torch.linalg.norm(g1, dim=1)
    dot = (g1 * dx).sum(dim=1)
    with torch.no_grad():
        coords2 = torch.cat([x2.detach(), X_rec.detach(), phase.detach()], dim=1)
        T2 = model(coords2).squeeze()
    e = ((T2 - T1.detach()) - dot).abs()
    denom = (g_norm * dx_norm).clamp_min(float(eps))
    ratio = e / denom
    return e, ratio, dx_norm, g_norm


class _CollapsedQuad(torch.autograd.Function):
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


def _edge_to_node(u: torch.Tensor, v: torch.Tensor, edge_vals: torch.Tensor, *, n_nodes: int) -> torch.Tensor:
    b = torch.zeros((n_nodes,), device=edge_vals.device, dtype=edge_vals.dtype)
    b.index_add_(0, u, -edge_vals)
    b.index_add_(0, v, edge_vals)
    return b


def _node_to_edge(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, v) - x.index_select(0, u)


def _laplacian_mv_weighted(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, x: torch.Tensor, n_nodes: int) -> torch.Tensor:
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
    tol = float(tol) if float(tol) > 0.0 else 1e-6
    alpha_t = b.new_tensor(float(alpha))
    diag = (alpha_t + beta * deg.to(dtype=b.dtype, device=b.device)).clamp_min(1e-12)

    def _A(x: torch.Tensor) -> torch.Tensor:
        return alpha_t * x + beta * _laplacian_mv_weighted(u, v, w, x, n)

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


@dataclass
class SharedEventReMetrics:
    n_groups: int = 0
    n_groups_fallback: int = 0
    n_groups_pcg_fail: int = 0
    max_rows_seen: int = 0
    max_nodes_seen: int = 0
    pcg_iters_sum: int = 0
    pcg_iters_max: int = 0
    edge_weight_sum: float = 0.0
    edge_weight_count: int = 0
    edge_weight_max: float = float("nan")
    pcg_iters_sum: int = 0
    pcg_iters_max: int = 0
    edge_weight_sum: float = 0.0
    edge_weight_count: int = 0
    edge_weight_max: float = float("nan")


def _shared_event_re_quad_whitening(
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
    pcg_max_iters: int,
    pcg_tol: float,
    pcg_min_iters: int,
    edge_weighting: str,
    edge_weight_power: float,
    edge_weight_scale_km: float,
    edge_weight_eps_km: float,
    edge_weight_global_scale: float,
    edge_weight_normalize: bool,
    X_event: Optional[torch.Tensor],
) -> tuple[torch.Tensor, SharedEventReMetrics]:
    metrics = SharedEventReMetrics()
    if resid.numel() == 0:
        return torch.tensor(0.0, device=resid.device, dtype=resid.dtype), metrics

    keys_sorted, perm = torch.sort(keys)
    resid_perm = resid.index_select(0, perm)
    idx_perm = idx.index_select(0, perm)
    ph_perm = ph_id.index_select(0, perm)
    sigma_perm = torch.where(ph_perm < 0.5, sigma_p, sigma_s).clamp_min(1e-12)

    is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
    if keys_sorted.numel() > 1:
        is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
    starts = torch.nonzero(is_new, as_tuple=False).reshape(-1)
    ends = torch.cat([starts[1:], torch.tensor([keys_sorted.numel()], device=starts.device, dtype=starts.dtype)])

    quad_sum = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)

    for s, e in zip(starts.tolist(), ends.tolist()):
        if e <= s:
            continue
        metrics.n_groups += 1
        metrics.max_rows_seen = max(metrics.max_rows_seen, int(e - s))
        r_g = resid_perm[s:e]
        sigma_g = sigma_perm[s:e]
        ph_g = ph_perm[s:e]
        e1 = idx_perm[s:e, 0].to(torch.int64)
        e2 = idx_perm[s:e, 1].to(torch.int64)
        tau_g = float(tau_p if int(ph_g[0].item()) == 0 else tau_s)
        # sigma is constant within a phase/group; use scalar to avoid shape mismatch.
        sigma_val = float(sigma_g[0].item()) if sigma_g.numel() > 0 else 1.0
        sigma_val = max(sigma_val, 1e-12)
        if not (tau_g > 0.0):
            metrics.n_groups_fallback += 1
            quad_sum = quad_sum + _CollapsedQuad.apply(r_g, r_g / (sigma_val * sigma_val))
            continue

        # Local remap
        ev_flat = torch.cat([e1, e2], dim=0)
        nodes, inv = torch.unique(ev_flat, return_inverse=True)
        n_nodes = int(nodes.numel())
        metrics.max_nodes_seen = max(metrics.max_nodes_seen, n_nodes)
        if (e - s) > int(max_rows_per_group) or n_nodes > int(max_nodes_per_group):
            metrics.n_groups_fallback += 1
            quad_sum = quad_sum + _CollapsedQuad.apply(r_g, r_g / (sigma_val * sigma_val))
            continue
        u = inv[: e1.numel()]
        v = inv[e1.numel() :]

        # Edge weights
        if edge_weighting in {"distance_power", "distance_rbf", "distance_linear"} and isinstance(X_event, torch.Tensor):
            x_u = X_event.index_select(0, e1)
            x_v = X_event.index_select(0, e2)
            dist = torch.linalg.norm(x_u - x_v, dim=1).clamp_min(0.0)
            if edge_weighting == "distance_rbf":
                ell = float(max(edge_weight_scale_km, 1e-6))
                w = torch.exp(-((dist / ell) ** 2)) + float(edge_weight_eps_km)
            elif edge_weighting == "distance_linear":
                w = dist + float(edge_weight_eps_km)
            else:
                scale = float(max(edge_weight_scale_km, 1e-6))
                p = float(edge_weight_power)
                w = torch.pow(dist / scale, p) + float(edge_weight_eps_km)
        else:
            w = torch.ones_like(r_g)
        if edge_weight_normalize:
            w_mean = w.mean().clamp_min(1e-12)
            w = w / w_mean
        w = w * float(edge_weight_global_scale)
        try:
            w_mean = float(w.mean().detach().item()) if w.numel() > 0 else float("nan")
            w_max = float(w.max().detach().item()) if w.numel() > 0 else float("nan")
            if math.isfinite(w_mean):
                metrics.edge_weight_sum += float(w_mean) * int(w.numel())
                metrics.edge_weight_count += int(w.numel())
            if math.isfinite(w_max):
                metrics.edge_weight_max = w_max if not math.isfinite(metrics.edge_weight_max) else max(metrics.edge_weight_max, w_max)
        except Exception:
            pass
        w_sqrt = torch.sqrt(w.clamp_min(0.0))
        deg = torch.zeros((n_nodes,), device=r_g.device, dtype=r_g.dtype)
        deg.index_add_(0, u, w)
        deg.index_add_(0, v, w)

        beta = torch.tensor(1.0 / (sigma_val * sigma_val), device=r_g.device, dtype=r_g.dtype)
        alpha = (1.0 / (tau_g * tau_g)) + float(jitter0)
        b = _edge_to_node(u, v, beta * w_sqrt * r_g, n_nodes=n_nodes)
        x, iters, ok = _pcg_solve_whitening(
            u=u,
            v=v,
            w=w.to(dtype=r_g.dtype, device=r_g.device),
            b=b,
            alpha=float(alpha),
            beta=beta,
            deg=deg,
            max_iters=int(pcg_max_iters),
            tol=float(pcg_tol),
            min_iters=int(pcg_min_iters),
        )
        metrics.pcg_iters_sum += int(iters)
        metrics.pcg_iters_max = max(metrics.pcg_iters_max, int(iters))
        if not ok:
            metrics.n_groups_pcg_fail += 1
            metrics.n_groups_fallback += 1
            quad_sum = quad_sum + _CollapsedQuad.apply(r_g, r_g / (sigma_val * sigma_val))
            continue
        ax = _node_to_edge(u, v, x)
        u_edge = beta * (r_g - (w_sqrt * ax))
        quad_sum = quad_sum + _CollapsedQuad.apply(r_g, u_edge)

    return quad_sum, metrics


def compute_likelihood_loss(
    *,
    idx: torch.Tensor,
    y: torch.Tensor,
    X_src: torch.Tensor,
    dX_src: torch.Tensor,
    model: nn.Module,
    sigma_p: torch.Tensor,
    sigma_s: torch.Tensor,
    params: dict,
    row_station_index: Optional[torch.Tensor],
    out_metrics: Optional[dict] = None,
) -> torch.Tensor:
    resid = compute_residuals(idx, y, X_src, dX_src, model)
    phase = y[:, 4]
    is_p = phase < 0.5
    sigma = torch.where(is_p, sigma_p, sigma_s).clamp_min(1e-12)
    data_loss = 0.5 * (resid / sigma).square() + torch.log(sigma)
    nll = data_loss.mean()
    if isinstance(out_metrics, dict):
        out_metrics["loss/data_nll"] = float(nll.detach().item())

    # shared_event_re collapsed quad (correlated gaussian)
    if bool(params.get("_shared_event_re_enabled", False)):
        grouping = str(params.get("_shared_event_re_grouping", "phase")).strip().lower()
        if grouping == "station_phase" and isinstance(row_station_index, torch.Tensor):
            keys = row_station_index.to(dtype=torch.int64) * 2 + phase.to(dtype=torch.int64)
        else:
            keys = phase.to(dtype=torch.int64)
        tau_ps = params.get("_shared_event_re_tau_s", [0.0, 0.0])
        tau_p = float(tau_ps[0])
        tau_s = float(tau_ps[1])
        edge_weighting = str(params.get("_shared_event_re_whitening_edge_weighting", "uniform"))
        quad, metrics = _shared_event_re_quad_whitening(
            idx=idx,
            resid=resid,
            keys=keys,
            ph_id=phase,
            sigma_p=sigma_p,
            sigma_s=sigma_s,
            tau_p=tau_p,
            tau_s=tau_s,
            jitter0=1e-8,
            max_rows_per_group=int(params.get("_shared_event_re_max_rows_per_group", 200000)),
            max_nodes_per_group=int(params.get("_shared_event_re_max_nodes_per_group", 512)),
            pcg_max_iters=int(params.get("_shared_event_re_whitening_pcg_max_iters", 50)),
            pcg_tol=float(params.get("_shared_event_re_whitening_pcg_tol", 1e-4)),
            pcg_min_iters=int(params.get("_shared_event_re_whitening_pcg_min_iters", 0)),
            edge_weighting=edge_weighting,
            edge_weight_power=float(params.get("_shared_event_re_whitening_edge_weight_power", 1.0)),
            edge_weight_scale_km=float(params.get("_shared_event_re_whitening_edge_weight_scale_km", 1.0)),
            edge_weight_eps_km=float(params.get("_shared_event_re_whitening_edge_weight_eps_km", 1e-3)),
            edge_weight_global_scale=float(params.get("_shared_event_re_whitening_edge_weight_global_scale", 1.0)),
            edge_weight_normalize=bool(params.get("_shared_event_re_whitening_edge_weight_normalize", False)),
            X_event=(X_src[:, :3] + dX_src[:, :3]).detach(),
        )
        # Normalize by batch size to keep scale consistent with mean NLL
        nll = nll + (quad / max(int(resid.numel()), 1))
        if isinstance(out_metrics, dict):
            out_metrics["shared_event_re/quad"] = float(quad.detach().item())
            out_metrics["shared_event_re/groups_total"] = int(metrics.n_groups)
            out_metrics["shared_event_re/groups_fallback"] = int(metrics.n_groups_fallback)
            out_metrics["shared_event_re/pcg_fail"] = int(metrics.n_groups_pcg_fail)
            out_metrics["shared_event_re/max_rows"] = int(metrics.max_rows_seen)
            out_metrics["shared_event_re/max_nodes"] = int(metrics.max_nodes_seen)
            if metrics.n_groups > 0:
                out_metrics["shared_event_re/pcg_iters_mean"] = float(metrics.pcg_iters_sum / max(metrics.n_groups, 1))
            out_metrics["shared_event_re/pcg_iters_max"] = int(metrics.pcg_iters_max)
            if metrics.edge_weight_count > 0:
                out_metrics["shared_event_re/edge_weight_mean"] = float(metrics.edge_weight_sum / max(metrics.edge_weight_count, 1))
            out_metrics["shared_event_re/edge_weight_max"] = float(metrics.edge_weight_max)

    return nll


def compute_prior_loss(dX_src: torch.Tensor, *, prior_std: list[float]) -> torch.Tensor:
    std = torch.tensor(prior_std, device=dX_src.device, dtype=dX_src.dtype)
    std = std.clamp_min(1e-12)
    return 0.5 * ((dX_src / std) ** 2).mean()
