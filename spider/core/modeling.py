import numpy as np

import polars as pl
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from spider.utils.console import info, warn

from spider.core.shared_event_re_whitening import compute_quad_whitening



# Standardized stdout helper
def _log(*parts, section: str = "LIKELIHOOD", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

# -----------------------------------------------------------------------------
# Collapsed shared-event random effects (Gaussian; marginalized b; PCG quadratic)
# -----------------------------------------------------------------------------

class _CollapsedQuad(torch.autograd.Function):
    """
    Compute 0.5 * r^T u while defining the gradient w.r.t. r as u.

    For a true quadratic form 0.5 r^T Σ^{-1} r, the gradient w.r.t r is Σ^{-1} r.
    We typically compute u ≈ Σ^{-1} r via an iterative solve, and we do NOT want
    to differentiate through that solver. This custom autograd function makes that
    explicit and stable.
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


def _laplacian_mv(u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Compute (A^T A) x for an undirected edge list (u,v) in local node indexing."""
    if n_nodes <= 0:
        return torch.zeros_like(x)
    out = torch.zeros((n_nodes,), device=x.device, dtype=x.dtype)
    if u.numel() == 0:
        return out
    tmp = x.index_select(0, u) - x.index_select(0, v)
    out.index_add_(0, u, tmp)
    out.index_add_(0, v, -tmp)
    return out


def _laplacian_w_mv(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, x: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Compute weighted Laplacian L_w x for an undirected edge list (u,v) with weights w."""
    if n_nodes <= 0:
        return torch.zeros_like(x)
    out = torch.zeros((n_nodes,), device=x.device, dtype=x.dtype)
    if u.numel() == 0:
        return out
    tmp = w * (x.index_select(0, u) - x.index_select(0, v))
    out.index_add_(0, u, tmp)
    out.index_add_(0, v, -tmp)
    return out


def _sum_incidence_w_mv(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, x: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """
    Compute A^T W A x for an edge list (u,v) where each row is [1 at u, 1 at v].
    This yields: out[u] += w * (x[u] + x[v]), out[v] += w * (x[u] + x[v]).
    """
    if n_nodes <= 0:
        return torch.zeros_like(x)
    out = torch.zeros((n_nodes,), device=x.device, dtype=x.dtype)
    if u.numel() == 0:
        return out
    xu = x.index_select(0, u)
    xv = x.index_select(0, v)
    tmp = w * (xu + xv)
    out.index_add_(0, u, tmp)
    out.index_add_(0, v, tmp)
    return out


def _pcg_solve(
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    deg: torch.Tensor,
    max_iters: int,
    tol: float,
) -> torch.Tensor:
    """
    Solve (alpha*I + beta*L) x = b with (Jacobi-)preconditioned conjugate gradient.
    L = A^T A for the undirected edge list (u,v).
    """
    n = int(b.numel())
    if n == 0:
        return b
    # Initial guess x=0
    x = torch.zeros_like(b)

    def A_mv(z: torch.Tensor) -> torch.Tensor:
        return alpha * z + beta * _laplacian_mv(u, v, z, n)

    r = b - A_mv(x)
    # Jacobi preconditioner: M^{-1} ≈ diag(A)^{-1} where diag(A)=alpha + beta*deg
    diag = (alpha + beta * deg).clamp_min(1e-12)
    z = r / diag
    p = z.clone()
    rz_old = (r * z).sum()
    b_norm = b.norm().clamp_min(1e-12)

    for _ in range(int(max_iters)):
        Ap = A_mv(p)
        denom = (p * Ap).sum().clamp_min(1e-20)
        a = rz_old / denom
        x = x + a * p
        r = r - a * Ap
        if (r.norm() / b_norm).item() <= float(tol):
            break
        z = r / diag
        rz_new = (r * z).sum()
        bcoef = rz_new / rz_old.clamp_min(1e-30)
        p = z + bcoef * p
        rz_old = rz_new

    return x


def _pcg_solve_weighted(
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    max_iters: int,
    tol: float,
) -> torch.Tensor:
    """
    Solve (alpha*I + L_w) x = b with Jacobi-preconditioned CG,
    where L_w is the weighted Laplacian for edges (u,v,w).
    """
    n = int(b.numel())
    if n == 0:
        return b
    x = torch.zeros_like(b)

    def A_mv(z: torch.Tensor) -> torch.Tensor:
        return alpha * z + _laplacian_w_mv(u, v, w, z, n)

    r = b - A_mv(x)
    deg = torch.zeros((n,), device=b.device, dtype=b.dtype)
    if u.numel() > 0:
        deg.index_add_(0, u, w)
        deg.index_add_(0, v, w)
    diag = (alpha + deg).clamp_min(1e-12)
    z = r / diag
    p = z.clone()
    rz_old = (r * z).sum()
    b_norm = b.norm().clamp_min(1e-12)

    for _ in range(int(max_iters)):
        Ap = A_mv(p)
        denom = (p * Ap).sum().clamp_min(1e-20)
        a = rz_old / denom
        x = x + a * p
        r = r - a * Ap
        if (r.norm() / b_norm).item() <= float(tol):
            break
        z = r / diag
        rz_new = (r * z).sum()
        bcoef = rz_new / rz_old.clamp_min(1e-30)
        p = z + bcoef * p
        rz_old = rz_new

    return x


def _pcg_solve_sum_weighted(
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    max_iters: int,
    tol: float,
) -> torch.Tensor:
    """
    Solve (diag(alpha) + A^T W A) x = b with Jacobi-preconditioned CG,
    where each row of A has 1 at u and 1 at v.
    """
    n = int(b.numel())
    if n == 0:
        return b
    x = torch.zeros_like(b)

    def A_mv(z: torch.Tensor) -> torch.Tensor:
        return alpha * z + _sum_incidence_w_mv(u, v, w, z, n)

    r = b - A_mv(x)
    deg = torch.zeros((n,), device=b.device, dtype=b.dtype)
    if u.numel() > 0:
        deg.index_add_(0, u, w)
        deg.index_add_(0, v, w)
    diag = (alpha + deg).clamp_min(1e-12)
    z = r / diag
    p = z.clone()
    rz_old = (r * z).sum()
    b_norm = b.norm().clamp_min(1e-12)

    for _ in range(int(max_iters)):
        Ap = A_mv(p)
        denom = (p * Ap).sum().clamp_min(1e-20)
        a = rz_old / denom
        x = x + a * p
        r = r - a * Ap
        if (r.norm() / b_norm).item() <= float(tol):
            break
        z = r / diag
        rz_new = (r * z).sum()
        bcoef = rz_new / rz_old.clamp_min(1e-30)
        p = z + bcoef * p
        rz_old = rz_new

    return x


def _station_phase_re_quad(
    *,
    resid: torch.Tensor,
    sigma: torch.Tensor,
    ph_id: torch.Tensor,
    sta_idx: Optional[torch.Tensor],
    tau_ps: list,
) -> Tuple[torch.Tensor, int, str]:
    """
    Collapsed station-phase random effects (additive) quadratic term.
    Returns (quad_sum, n_groups, grouping_used).
    """
    if resid.numel() == 0:
        return torch.tensor(0.0, device=resid.device, dtype=resid.dtype), 0, "phase"
    tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
    tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)

    grouping = "station_phase" if isinstance(sta_idx, torch.Tensor) and int(sta_idx.numel()) == int(resid.numel()) else "phase"
    if grouping == "phase":
        keys = ph_id
    else:
        keys = (sta_idx.to(dtype=torch.int64) * 2) + ph_id  # type: ignore[union-attr]

    keys_sorted, perm = torch.sort(keys)
    resid_s = resid.index_select(0, perm)
    sigma_s = sigma.index_select(0, perm)
    ph_s = ph_id.index_select(0, perm)

    quad = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
    n_groups = 0
    if keys_sorted.numel() > 0:
        is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
        is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
        starts = torch.nonzero(is_new, as_tuple=False).flatten()
        ends = torch.cat([starts[1:], torch.tensor([keys_sorted.numel()], device=keys_sorted.device)])
        for s0, e0 in zip(starts.tolist(), ends.tolist()):
            r_g = resid_s[s0:e0]
            s_g = sigma_s[s0:e0].clamp_min(1e-12)
            ph_g = ph_s[s0:e0]
            if r_g.numel() == 0:
                continue
            n_groups += 1
            tau = float(tau_p if int(ph_g[0].item()) == 0 else tau_s)
            if not (tau > 0.0):
                continue
            d_inv = 1.0 / s_g.square().clamp_min(1e-24)
            sum_w = d_inv.sum()
            sum_wr = (d_inv * r_g).sum()
            coef = (tau * tau) / (1.0 + (tau * tau) * float(sum_w))
            u = d_inv * r_g - d_inv * (coef * sum_wr)
            quad = quad + 0.5 * (r_g * u).sum()
    return quad, n_groups, grouping


def _shared_event_re_u_pcg(
    *,
    idx_g: torch.Tensor,
    resid_g: torch.Tensor,
    sigma: torch.Tensor,
    tau: float,
    jitter0: float,
    pcg_max_iters: int,
    pcg_tol: float,
) -> torch.Tensor:
    """
    Compute u ≈ Σ^{-1} r for Σ = sigma^2 I + tau^2 A A^T via node-space PCG.
    """
    # tau <= 0 -> iid
    if not (tau > 0.0):
        s2 = sigma.square().clamp_min(1e-24)
        return resid_g / s2

    # Local remapping of event ids -> [0..n_nodes-1]
    ev_flat = idx_g.reshape(-1)
    nodes, inv_nodes = torch.unique(ev_flat, return_inverse=True)
    m = int(idx_g.shape[0])
    u = inv_nodes[:m]
    v = inv_nodes[m:]
    n_nodes = int(nodes.numel())

    # deg (for Jacobi preconditioner) in local node indexing
    deg = torch.zeros((n_nodes,), device=resid_g.device, dtype=resid_g.dtype)
    if m > 0:
        ones = torch.ones((m,), device=resid_g.device, dtype=resid_g.dtype)
        deg.index_add_(0, u, ones)
        deg.index_add_(0, v, ones)

    # Right-hand side: b = beta * A^T r
    beta = (1.0 / sigma.square().clamp_min(1e-24)).to(device=resid_g.device, dtype=resid_g.dtype)
    b = torch.zeros((n_nodes,), device=resid_g.device, dtype=resid_g.dtype)
    b.index_add_(0, u, -beta * resid_g)
    b.index_add_(0, v, beta * resid_g)

    alpha = torch.tensor((1.0 / (float(tau) * float(tau))) + float(jitter0), device=resid_g.device, dtype=resid_g.dtype)

    # Solve for x (node potentials)
    x = _pcg_solve(
        u=u,
        v=v,
        b=b,
        alpha=alpha,
        beta=beta,
        deg=deg,
        max_iters=int(pcg_max_iters),
        tol=float(pcg_tol),
    )

    # u_edge = beta * (r - A x) with (A x)_e = x[v]-x[u]
    Ax = x.index_select(0, v) - x.index_select(0, u)
    u_edge = beta * (resid_g - Ax)
    return u_edge


# -----------------------------------------------------------------------------
# Core Physics / Travel Time Logic
# -----------------------------------------------------------------------------

def compute_travel_times(idx, y, X_src, ΔX_src, model):
    """
    Compute predicted differential travel times.
    
    Args:
        idx: Event pair indices [B, 2]
        y: Observation data [B, 5] (dt, X_rec, Y_rec, Z_rec, phase)
        X_src: Base source coordinates [M, 4]
        ΔX_src: Source coordinate perturbations [M, 4]
        model: EikoNet model
        
    Returns:
        dt_pred: Predicted differential times [B]
    """
    # 1. Get source positions for pairs
    # idx[:, 0] is ID of first event, idx[:, 1] is ID of second
    # X_src is (x, y, z, t)
    src1 = X_src[idx[:, 0]] + ΔX_src[idx[:, 0]]
    src2 = X_src[idx[:, 1]] + ΔX_src[idx[:, 1]]
    
    # 2. Extract receiver coordinates
    # y is (dt, x_rec, y_rec, z_rec, phase)
    X_rec = y[:, 1:4] # (x, y, z)
    phase = y[:, 4:5] # (phase)
    
    # 3. Prepare batch for EikoNet
    # Input format: (x_src, y_src, z_src, x_rec, y_rec, z_rec, phase)
    # We stack both sources to run one large batch through the model
    # src[:, :3] is spatial (x,y,z)
    coords1 = torch.cat([src1[:, :3], X_rec, phase], dim=1)
    coords2 = torch.cat([src2[:, :3], X_rec, phase], dim=1)
    
    batch_input = torch.cat([coords1, coords2], dim=0)
    
    # 4. Model Forward Pass
    T_pred_all = model(batch_input).squeeze()
    
    # Split back into T1 and T2
    n = src1.shape[0]
    T1 = T_pred_all[:n]
    T2 = T_pred_all[n:]
    
    # 5. Differential Time: (T2 + t2) - (T1 + t1)
    # src[:, 3] is origin time correction
    dt_pred = (T2 + src2[:, 3]) - (T1 + src1[:, 3])
    
    return dt_pred


def compute_residuals(idx, y, X_src, ΔX_src, model):
    """Compute simple residuals (Observed - Predicted)."""
    dt_pred = compute_travel_times(idx, y, X_src, ΔX_src, model)
    dt_obs = y[:, 0]
    return dt_obs - dt_pred


def compute_residuals_full(II, YY, X_src, ΔX_src, model, bs, N):
    """Compute residuals for the entire dataset in batches."""
    residuals = torch.zeros_like(YY[:, 0])
    if N == 0:
        return residuals

    with torch.no_grad():
        # Use a larger batch size for evaluation, but prevent infinite loop if bs=0
        # If input bs is < 1, default to 1024 or N
        bs_safe = max(int(bs), 1024)
        eval_bs = min(bs_safe * 4, N)
        eval_bs = max(eval_bs, 1) # Double safety
        
        for i in range(0, N, eval_bs):
            i_end = min(i + eval_bs, N)
            idx_b = II[i:i_end]
            y_b = YY[i:i_end]
            residuals[i:i_end] = compute_residuals(idx_b, y_b, X_src, ΔX_src, model)
    return residuals


def compute_linearization_error(
    idx: torch.Tensor,
    y: torch.Tensor,
    X_src: torch.Tensor,
    ΔX_src: torch.Tensor,
    model: nn.Module,
) -> torch.Tensor:
    """
    Compute per-row linearization error for the travel-time field T (not including origin time):

        e = | (T(x2) - T(x1)) - ∇T(x1) · (x2 - x1) |

    where x1/x2 are event locations (XYZ), receiver and phase come from `y`.

    Notes:
    - This intentionally computes gradients w.r.t. *inputs* (event location), not model parameters.
    - Because EikoNet is a per-row MLP (no batch coupling like BatchNorm), using
      autograd on sum(T1) yields per-example gradients w.r.t. input coords.
    """
    # Event positions (XYZT); we only use XYZ here
    src1 = X_src[idx[:, 0]] + ΔX_src[idx[:, 0]]
    src2 = X_src[idx[:, 1]] + ΔX_src[idx[:, 1]]
    x1 = src1[:, :3]
    x2 = src2[:, :3]

    # Receiver coords + phase
    X_rec = y[:, 1:4]
    phase = y[:, 4:5]

    # Leaf input for input-gradient
    x1_leaf = x1.detach().clone().requires_grad_(True)
    coords1 = torch.cat([x1_leaf, X_rec.detach(), phase.detach()], dim=1)
    T1 = model(coords1).squeeze()

    # Per-example gradient (see docstring)
    g1 = torch.autograd.grad(
        T1.sum(),
        x1_leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    dot = (g1 * (x2.detach() - x1_leaf.detach())).sum(dim=1)

    # Forward for x2 (no need for grads)
    with torch.no_grad():
        coords2 = torch.cat([x2.detach(), X_rec.detach(), phase.detach()], dim=1)
        T2 = model(coords2).squeeze()

    e = ((T2 - T1.detach()) - dot).abs()
    return e


def compute_linearization_error_ratio(
    idx: torch.Tensor,
    y: torch.Tensor,
    X_src: torch.Tensor,
    ΔX_src: torch.Tensor,
    model: nn.Module,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the same absolute linearization error `e` as `compute_linearization_error`, plus a
    dimensionless "relative Taylor remainder" ratio:

        ratio = e / (||∇T(x1)|| * ||x2-x1|| + eps)

    Motivation:
      Using the (matrix-norm) remainder bound e ≲ 0.5 ||H_T|| ||Δx||^2, we have

        e / (||∇T|| ||Δx||) ≲ 0.5 * (||H_T|| ||Δx|| / ||∇T||)

      so enforcing (||H_T|| ||Δx|| / ||∇T||) < r is approximately equivalent to
      enforcing ratio < 0.5*r.

    Returns:
      e:        (B,) absolute error in seconds
      ratio:    (B,) dimensionless remainder ratio
      dx_norm:  (B,) ||x2-x1|| in km (if inputs are km)
      g_norm:   (B,) ||∇T(x1)|| in s/km (if inputs are km and T is seconds)
    """
    # Event positions (XYZT); we only use XYZ here
    src1 = X_src[idx[:, 0]] + ΔX_src[idx[:, 0]]
    src2 = X_src[idx[:, 1]] + ΔX_src[idx[:, 1]]
    x1 = src1[:, :3]
    x2 = src2[:, :3]

    # Receiver coords + phase
    X_rec = y[:, 1:4]
    phase = y[:, 4:5]

    # Leaf input for input-gradient
    x1_leaf = x1.detach().clone().requires_grad_(True)
    coords1 = torch.cat([x1_leaf, X_rec.detach(), phase.detach()], dim=1)
    T1 = model(coords1).squeeze()

    # Per-example gradient
    g1 = torch.autograd.grad(
        T1.sum(),
        x1_leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

    dx = (x2.detach() - x1_leaf.detach())
    dx_norm = torch.linalg.norm(dx, dim=1)
    g_norm = torch.linalg.norm(g1, dim=1)
    dot = (g1 * dx).sum(dim=1)

    # Forward for x2 (no need for grads)
    with torch.no_grad():
        coords2 = torch.cat([x2.detach(), X_rec.detach(), phase.detach()], dim=1)
        T2 = model(coords2).squeeze()

    e = ((T2 - T1.detach()) - dot).abs()
    denom = (g_norm * dx_norm).clamp_min(float(eps))
    ratio = e / denom
    return e, ratio, dx_norm, g_norm


# -----------------------------------------------------------------------------
# Simplified Loss Components
# -----------------------------------------------------------------------------

def compute_likelihood_loss(
    idx: torch.Tensor,
    y: torch.Tensor,
    X_src: torch.Tensor,
    ΔX_src: torch.Tensor,
    model: nn.Module,
    σ_p: torch.Tensor,
    σ_s: torch.Tensor,
    params: dict,
    nuisance_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the Average Negative Log-Likelihood (per observation).
    
    Loss = Mean( DataLoss(residual / sigma) + log(sigma) )
    """
    def _abort_on_pcg_fallback(params: dict, *, context: str) -> None:
        try:
            if not bool(params.get("_shared_event_re_abort_on_pcg_fallback", False)):
                return
        except Exception:
            return
        g_fb = int(params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
        g_rows = int(params.get("_shared_event_re_runtime_last_groups_rows_cap", 0) or 0)
        g_nodes = int(params.get("_shared_event_re_runtime_last_groups_nodes_cap", 0) or 0)
        g_tau0 = int(params.get("_shared_event_re_runtime_last_groups_tau_zero", 0) or 0)
        w_fb = int(params.get("_shared_event_re_whitening_last_groups_fallback_diag", 0) or 0)
        w_rows = int(params.get("_shared_event_re_whitening_last_groups_rows_cap", 0) or 0)
        w_nodes = int(params.get("_shared_event_re_whitening_last_groups_nodes_cap", 0) or 0)
        w_tau0 = int(params.get("_shared_event_re_whitening_last_groups_tau_zero", 0) or 0)
        w_pcg_fail = int(params.get("_shared_event_re_whitening_last_pcg_fail", 0) or 0)
        g_fb_gpu = int(params.get("_shared_event_re_gpu_last_groups_fallback_diag", 0) or 0)
        if (g_fb + w_fb + w_pcg_fail + g_fb_gpu) <= 0:
            return
        max_rows = int(params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
        max_nodes = int(params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
        msg = (
            f"shared_event_re PCG fallback detected (context={context}).\n"
            f"fallback_counts: runtime={g_fb} whitening={w_fb} whitening_pcg_fail={w_pcg_fail} gpu={g_fb_gpu}\n"
            f"reasons: rows_cap={g_rows or w_rows} nodes_cap={g_nodes or w_nodes} tau_zero={g_tau0 or w_tau0}\n"
            f"max_seen: rows={max_rows} nodes={max_nodes}\n"
            "Suggested fixes:\n"
            "- Increase model.likelihood.shared_event_re.max_rows_per_group / max_nodes_per_group\n"
            "- Reduce inference.batching.standard.warmup/sgld batch sizes\n"
            "- Enable inference.batching.event_batches to limit group sizes\n"
            "- Ensure shared_event_re.tau_s > 0 (tau_zero indicates zero tau)\n"
            "- If whitening is enabled, increase shared_event_re.whitening.pcg_bucket_nodes or max_nodes\n"
        )
        raise RuntimeError(msg)
    # Tempering removed; keep core residual distributions only.
    alpha = 1.0

    # 1. Predict
    dt_pred = compute_travel_times(idx, y, X_src, ΔX_src, model)
    if nuisance_delta is not None:
        dt_pred = dt_pred + nuisance_delta
        
    dt_obs = y[:, 0]
    phase = y[:, 4] # 0 for P, 1 for S (approx)
    
    # 2. Select Sigma
    # phase < 0.5 implies P-wave
    is_p = (phase < 0.5)
    sigma = torch.where(is_p, σ_p, σ_s)

    # Optional: distance-dependent sigma (linear in event-pair separation).
    try:
        if bool(params.get("_sigma_distance_enable", False)):
            slope_ps = params.get("_sigma_distance_slope_ps", [0.0, 0.0])
            min_ps = params.get("_sigma_distance_min_sigma_ps", [0.0, 0.0])
            max_km = params.get("_sigma_distance_max_dist_km", None)
            slope_p = float(slope_ps[0]) if isinstance(slope_ps, (list, tuple)) and len(slope_ps) >= 2 else float(slope_ps)
            slope_s = float(slope_ps[1]) if isinstance(slope_ps, (list, tuple)) and len(slope_ps) >= 2 else float(slope_ps)
            min_p = float(min_ps[0]) if isinstance(min_ps, (list, tuple)) and len(min_ps) >= 2 else float(min_ps)
            min_s = float(min_ps[1]) if isinstance(min_ps, (list, tuple)) and len(min_ps) >= 2 else float(min_ps)
            # Event pair distance in km (XYZ already in km)
            x1 = (X_src + ΔX_src).index_select(0, idx[:, 0])[:, :3]
            x2 = (X_src + ΔX_src).index_select(0, idx[:, 1])[:, :3]
            dist = torch.linalg.norm(x2 - x1, dim=1)
            if isinstance(max_km, (int, float)) and float(max_km) > 0.0:
                dist = dist.clamp_max(float(max_km))
            slope = torch.where(is_p, torch.tensor(float(slope_p), device=dist.device, dtype=dist.dtype),
                                torch.tensor(float(slope_s), device=dist.device, dtype=dist.dtype))
            sigma = sigma + slope * dist
            min_sigma = torch.where(is_p, torch.tensor(float(min_p), device=dist.device, dtype=dist.dtype),
                                    torch.tensor(float(min_s), device=dist.device, dtype=dist.dtype))
            sigma = torch.maximum(sigma, min_sigma)
            # Whitening path currently uses phase-wise sigma scalars; warn once if enabled.
            if bool(params.get("_shared_event_re_whitening_enabled", False)) and not bool(params.get("_sigma_distance_warned_whiten", False)):
                params["_sigma_distance_warned_whiten"] = True
                _log(
                    "Warning: sigma_distance_linear enabled, but shared_event_re whitening uses phase-wise sigma only. "
                    "Distance-dependent sigma is ignored in whitening logdet approximation.",
                    flush=True,
                )
    except Exception:
        pass
    
    # 3. Standardized Residuals
    # Clamp sigma to avoid division by zero
    sigma = sigma.clamp_min(1e-12)
    sigma2 = sigma.square()
    sigma = sigma2.sqrt().clamp_min(1e-12)
    resid = dt_obs - dt_pred
    scaled_resid = resid / sigma
    # Optional: collapsed shared-event random effects (Gaussian; marginalized b).
    # This replaces the independent quadratic term with a correlated quadratic
    # while keeping the overall loss scaled "per observation" (mean over rows).
    try:
        se_enable = bool(params.get("_shared_event_re_enabled", False))
    except Exception:
        se_enable = False
    if se_enable:
        solver = str(params.get("_shared_event_re_solver", "pcg_sparse")).strip().lower()
        if not bool(params.get("_shared_event_re_solver_logged", False)):
            params["_shared_event_re_solver_logged"] = True
            _log(f"[shared_event_re] se_enable={se_enable} solver={solver}", flush=True)
        if solver in {"pcg", "pcg_sparse", "pcg-sparse"}:
            # Current Phase-A implementation: quadratic-only (drop_logdet must be true; enforced by schema).
            # We compute u ≈ Σ^{-1} r per group and return:
            #   mean( 0.5 r^T u ) + mean(log sigma)
            # but with custom autograd so d/dr = u (do not differentiate through the solver).
            grouping = str(params.get("_shared_event_re_grouping", "phase")).strip().lower()
            if grouping in {"stationphase", "station-phase"}:
                grouping = "station_phase"
            if not bool(params.get("_shared_event_re_sizes_logged", False)):
                params["_shared_event_re_sizes_logged"] = True
                _log(
                    f"[shared_event_re] resid_n={int(resid.numel())} idx_n={int(idx.shape[0])} grouping={grouping}",
                    flush=True,
                )
            if resid.numel() == 0 or idx.shape[0] == 0:
                if not bool(params.get("_shared_event_re_empty_batch_logged", False)):
                    params["_shared_event_re_empty_batch_logged"] = True
                    _log("[shared_event_re] WARNING: empty batch passed to shared_event_re", flush=True)
            tau_ps = params.get("_shared_event_re_tau_s", [0.0, 0.0])
            tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            hier_enable = bool(params.get("_shared_event_re_hierarchical", False))
            tau_event_ps = params.get("_shared_event_re_tau_event_s", tau_ps)
            tau_cluster_ps = params.get("_shared_event_re_tau_cluster_s", [0.0, 0.0])
            if hier_enable:
                tau_p = float(tau_event_ps[0]) if isinstance(tau_event_ps, (list, tuple)) and len(tau_event_ps) >= 2 else float(tau_event_ps)
                tau_s = float(tau_event_ps[1]) if isinstance(tau_event_ps, (list, tuple)) and len(tau_event_ps) >= 2 else float(tau_event_ps)
                tau_c_p = float(tau_cluster_ps[0]) if isinstance(tau_cluster_ps, (list, tuple)) and len(tau_cluster_ps) >= 2 else float(tau_cluster_ps)
                tau_c_s = float(tau_cluster_ps[1]) if isinstance(tau_cluster_ps, (list, tuple)) and len(tau_cluster_ps) >= 2 else float(tau_cluster_ps)
            else:
                tau_c_p = 0.0
                tau_c_s = 0.0
            jitter0 = float(params.get("_shared_event_re_jitter0", 1e-8))
            pcg_max_iters = int(params.get("_shared_event_re_pcg_max_iters", 50))
            pcg_tol = float(params.get("_shared_event_re_pcg_tol", 1e-3))
            max_rows_per_group = int(params.get("_shared_event_re_max_rows_per_group", 200000))
            max_nodes_per_group = int(params.get("_shared_event_re_max_nodes_per_group", 512))
            fallback_to_diag = bool(params.get("_shared_event_re_fallback_to_diag", True))


            # Station index is supplied at runtime by the epoch runner when owner-bucket batching is active.
            sta_idx = None
            if grouping == "station_phase":
                sta_idx = params.get("_runtime_bucket_station_index", None)
                if not isinstance(sta_idx, torch.Tensor) or int(sta_idx.numel()) != int(resid.numel()):
                    # Fall back quietly to phase-only; warn once.
                    if not bool(params.get("_shared_event_re_warned_no_station_index", False)):
                        _log(
                            "Warning: shared_event_re.grouping='station_phase' requested but no per-row station index "
                            "was available for this batch. Falling back to grouping='phase'."
                        )
                        params["_shared_event_re_warned_no_station_index"] = True
                    grouping = "phase"
                    sta_idx = None

            # Build group keys (sorted -> contiguous runs)
            ph_id = torch.where(is_p, torch.zeros_like(resid, dtype=torch.int64), torch.ones_like(resid, dtype=torch.int64))
            if grouping == "phase":
                keys = ph_id
            else:
                keys = (sta_idx.to(dtype=torch.int64) * 2) + ph_id  # type: ignore[union-attr]

            # Optional: collapsed station-phase random effects (additive).
            quad_sp = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
            try:
                se_sp_enable = bool(params.get("_shared_event_re_station_phase_enabled", False))
            except Exception:
                se_sp_enable = False
            if se_sp_enable:
                sp_tau_ps = params.get("_shared_event_re_station_phase_tau_s", [0.0, 0.0])
                sp_sta_idx = params.get("_runtime_bucket_station_index", None)
                quad_sp, n_sp_groups, sp_grouping = _station_phase_re_quad(
                    resid=resid,
                    sigma=sigma,
                    ph_id=ph_id,
                    sta_idx=sp_sta_idx,
                    tau_ps=sp_tau_ps,
                )
                params["_shared_event_re_station_phase_last_groups"] = int(n_sp_groups)
                params["_shared_event_re_station_phase_last_grouping"] = str(sp_grouping)
                params["_shared_event_re_station_phase_last_quad"] = float(quad_sp.detach().item())
                if (sp_grouping == "phase") and (not bool(params.get("_shared_event_re_station_phase_warned_no_station_index", False))):
                    params["_shared_event_re_station_phase_warned_no_station_index"] = True
                    _log(
                        "Warning: shared_event_re.station_phase_re enabled but no per-row station index was available; "
                        "falling back to phase-only station_phase_re.",
                        flush=True,
                    )
                if not bool(params.get("_shared_event_re_station_phase_logged", False)):
                    params["_shared_event_re_station_phase_logged"] = True
                    _log(
                        f"[shared_event_re] station_phase_re enabled=True grouping={sp_grouping} "
                        f"tau_p={float(sp_tau_ps[0]):.4g} tau_s={float(sp_tau_ps[1]):.4g} groups={int(n_sp_groups)}",
                        flush=True,
                    )

            # Optional: DD graph k-hop clustering (non-overlapping balls).
            try:
                cluster_mode = str(params.get("_shared_event_re_cluster_mode", "none")).strip().lower()
            except Exception:
                cluster_mode = "none"
            cluster_ids = params.get("_shared_event_re_cluster_ids", None)
            m_tot_full = float(max(int(resid.numel()), 1))
            quad_diag_extra = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
            if cluster_mode in {"dd_khop", "component"} and isinstance(cluster_ids, torch.Tensor) and int(cluster_ids.numel()) > 0:
                try:
                    e1 = idx[:, 0].to(torch.int64)
                    e2 = idx[:, 1].to(torch.int64)
                    c1 = cluster_ids.index_select(0, e1)
                    c2 = cluster_ids.index_select(0, e2)
                    same = (c1 == c2)
                    if hier_enable:
                        # Hierarchical: keep all edges; add extra variance for cross-cluster pairs.
                        tau_c = torch.where(is_p, resid.new_tensor(float(tau_c_p)), resid.new_tensor(float(tau_c_s)))
                        extra = torch.where(same, torch.zeros_like(tau_c), 2.0 * tau_c.square())
                        sigma = (sigma.square() + extra).sqrt().clamp_min(1e-12)
                    else:
                        if not bool(same.all()):
                            # Diagonal contribution for cross-cluster edges.
                            resid_x = resid[~same]
                            sigma_x = sigma[~same]
                            quad_diag_extra = 0.5 * (resid_x * (resid_x / sigma_x.square().clamp_min(1e-24))).sum()
                            # Keep only within-cluster edges for correlated solve.
                            resid = resid[same]
                            idx = idx[same]
                            sigma = sigma[same]
                            is_p = is_p[same]
                            ph_id = ph_id[same]
                            keys = keys[same]
                except Exception:
                    pass
            log_sigma_mean = torch.log(sigma).mean()

            if not bool(params.get("_shared_event_re_entered", False)):
                params["_shared_event_re_entered"] = True
                _log(f"[shared_event_re] entered block solver={solver} grouping={grouping}", flush=True)

            # Build a shared grouping cache once so whitening and PCG can reuse it.
            grouping_cache = None
            try:
                from spider.core import shared_event_re_gpu

                keys_sorted, perm0, starts0, ends0 = shared_event_re_gpu._group_by_keys_gpu(keys)
                lengths0 = (ends0 - starts0).to(torch.int64)
                group_ids0, _ = shared_event_re_gpu._build_group_ids(starts0, ends0)
                idx_perm0 = idx.index_select(0, perm0)
                u0 = idx_perm0[:, 0].to(torch.int64)
                v0 = idx_perm0[:, 1].to(torch.int64)
                if u0.numel() > 0 and v0.numel() > 0:
                    max_node_id0 = int(torch.max(torch.stack([u0.max(), v0.max()])).item())
                else:
                    max_node_id0 = 0
                local_u0, local_v0, n_nodes0 = shared_event_re_gpu._build_local_node_indices(
                    u0, v0, group_ids0, int(starts0.numel()), max_node_id=max_node_id0
                )
                ph_perm0 = ph_id.index_select(0, perm0) if perm0.numel() > 0 else ph_id.new_zeros((0,))
                ph_group0 = ph_perm0.index_select(0, starts0) if starts0.numel() > 0 else ph_id.new_zeros((0,))
                if perm0.numel() > 0:
                    edge_idx0 = torch.arange(int(perm0.numel()), device=perm0.device, dtype=starts0.dtype)
                    edge_pos0 = edge_idx0 - starts0.index_select(0, group_ids0)
                else:
                    edge_pos0 = torch.zeros((0,), device=perm0.device, dtype=starts0.dtype)
                grouping_cache = {
                    "perm": perm0.detach(),
                    "starts": starts0.detach(),
                    "ends": ends0.detach(),
                    "lengths": lengths0.detach(),
                    "group_ids": group_ids0.detach(),
                    "local_u": local_u0.detach(),
                    "local_v": local_v0.detach(),
                    "n_nodes": n_nodes0.detach(),
                    "ph_group": ph_group0.detach(),
                    "edge_pos": edge_pos0.detach(),
                }
            except Exception:
                grouping_cache = None

            # One-time runtime log so users can confirm activation and grouping.
            # Resolve GPU enable: if unset, default to CUDA availability for this batch.
            try:
                gpu_enable_raw = params.get("_shared_event_re_gpu_enable", None)
            except Exception:
                gpu_enable_raw = None
            if gpu_enable_raw is None:
                gpu_enable = bool(resid.is_cuda)
            else:
                gpu_enable = bool(gpu_enable_raw)

            if not bool(params.get("_shared_event_re_logged", False)):
                params["_shared_event_re_logged"] = True
                try:
                    wflag = bool(params.get("_shared_event_re_whitening_enabled", False))
                    wmode = str(params.get("_shared_event_re_whitening_edge_weighting", "none"))
                except Exception:
                    wflag = False
                    wmode = "none"
                if hier_enable:
                    _log(
                        f"[shared_event_re] enabled=True grouping={grouping} "
                        f"tau_event_p={float(tau_p):.4g} tau_event_s={float(tau_s):.4g} "
                        f"tau_cluster_p={float(tau_c_p):.4g} tau_cluster_s={float(tau_c_s):.4g} "
                        f"max_rows={int(max_rows_per_group)} max_nodes={int(max_nodes_per_group)} "
                        f"gpu={gpu_enable} whitening={wflag} weight={wmode}",
                        flush=True,
                    )
                else:
                    _log(
                        f"[shared_event_re] enabled=True grouping={grouping} "
                        f"tau_p={float(tau_p):.4g} tau_s={float(tau_s):.4g} "
                        f"max_rows={int(max_rows_per_group)} max_nodes={int(max_nodes_per_group)} "
                        f"gpu={gpu_enable} whitening={wflag} weight={wmode}",
                        flush=True,
                    )

            # Optional: whitening operator path (static covariance).
            try:
                whiten_enable = bool(params.get("_shared_event_re_whitening_enabled", False))
            except Exception:
                whiten_enable = False
            if whiten_enable:
                # Quick diagnostic: compare whitening quadratic to diagonal quadratic once.
                quad_diag = None
                try:
                    with torch.no_grad():
                        quad_diag = 0.5 * (resid.square() / sigma.square().clamp_min(1e-24)).sum()
                except Exception:
                    quad_diag = None
                edge_weighting = str(params.get("_shared_event_re_whitening_edge_weighting", "uniform")).strip().lower()
                edge_weight_ell_km = float(params.get("_shared_event_re_whitening_edge_weight_ell_km", 1.0))
                edge_weight_eps_km = float(params.get("_shared_event_re_whitening_edge_weight_eps_km", 1e-3))
                edge_weight_power = float(params.get("_shared_event_re_whitening_edge_weight_power", 1.0))
                edge_weight_scale_km = float(params.get("_shared_event_re_whitening_edge_weight_scale_km", 1.0))
                edge_weight_global_scale = float(params.get("_shared_event_re_whitening_edge_weight_global_scale", 1.0))
                edge_weight_normalize = bool(params.get("_shared_event_re_whitening_edge_weight_normalize", False))
                whiten_solver = str(params.get("_shared_event_re_whitening_solver", "pcg")).strip().lower()
                whiten_pcg_max_iters = int(params.get("_shared_event_re_whitening_pcg_max_iters", 200))
                whiten_pcg_tol = float(params.get("_shared_event_re_whitening_pcg_tol", 1e-6))
                whiten_pcg_min_iters = int(params.get("_shared_event_re_whitening_pcg_min_iters", 0))
                whiten_pcg_batched = bool(params.get("_shared_event_re_whitening_pcg_batched", False))
                whiten_pcg_bucket_nodes = params.get("_shared_event_re_whitening_pcg_bucket_nodes", None)
                cache = params.get("_shared_event_re_whitening_cache", None)
                if not isinstance(cache, dict):
                    cache = {}
                cache_max = int(params.get("_shared_event_re_whitening_cache_max_entries", 0) or 0)
                X_event = None
                if edge_weighting in {"distance_rbf", "distance_linear", "distance_power"}:
                    X_event = params.get("_shared_event_re_whitening_X_event", None)
                    if not isinstance(X_event, torch.Tensor) or int(X_event.shape[0]) != int(X_src.shape[0]):
                        X_event = (X_src + ΔX_src)[:, :3].detach().to(device=X_src.device, dtype=X_src.dtype)
                        params["_shared_event_re_whitening_X_event"] = X_event
                cache_key_extra = None
                try:
                    precompute = bool(params.get("_shared_event_re_whitening_precompute", False))
                    batch_shuffle = bool(params.get("_runtime_batch_shuffle", True))
                    batch_id = int(params.get("_runtime_batch_id", -1))
                    if precompute and (not batch_shuffle) and batch_id >= 0:
                        cache_key_extra = ("batch", int(batch_id))
                except Exception:
                    cache_key_extra = None
                quad_w, metrics_w, cache = compute_quad_whitening(
                    idx=idx,
                    resid=resid,
                    keys=keys,
                    ph_id=ph_id,
                    sigma_p=σ_p,
                    sigma_s=σ_s,
                    tau_p=float(tau_p),
                    tau_s=float(tau_s),
                    jitter0=float(jitter0),
                    max_rows_per_group=int(max_rows_per_group),
                    max_nodes_per_group=int(max_nodes_per_group),
                    solver=str(whiten_solver),
                    pcg_max_iters=int(whiten_pcg_max_iters),
                    pcg_tol=float(whiten_pcg_tol),
                    pcg_min_iters=int(whiten_pcg_min_iters),
                    pcg_batched=bool(whiten_pcg_batched),
                    pcg_bucket_nodes=whiten_pcg_bucket_nodes if isinstance(whiten_pcg_bucket_nodes, list) else None,
                    edge_weighting=edge_weighting,
                    edge_weight_ell_km=float(edge_weight_ell_km),
                    edge_weight_eps_km=float(edge_weight_eps_km),
                    edge_weight_power=float(edge_weight_power),
                    edge_weight_scale_km=float(edge_weight_scale_km),
                        edge_weight_global_scale=float(edge_weight_global_scale),
                        edge_weight_normalize=bool(edge_weight_normalize),
                    X_event=X_event,
                    cache=cache,
                    grouping_cache=grouping_cache,
                    cache_key_extra=cache_key_extra,
                )
                if cache_max > 0 and len(cache) > cache_max:
                    try:
                        cache.pop(next(iter(cache)))
                    except Exception:
                        pass
                params["_shared_event_re_whitening_cache"] = cache
                params["_shared_event_re_whitening_last_groups"] = int(metrics_w.n_groups_total)
                params["_shared_event_re_whitening_last_groups_chol"] = int(metrics_w.n_groups_chol)
                params["_shared_event_re_whitening_last_groups_pcg"] = int(metrics_w.n_groups_pcg)
                params["_shared_event_re_whitening_last_groups_fallback_diag"] = int(metrics_w.n_groups_fallback_diag)
                params["_shared_event_re_whitening_last_groups_rows_cap"] = int(metrics_w.n_groups_rows_cap)
                params["_shared_event_re_whitening_last_groups_nodes_cap"] = int(metrics_w.n_groups_nodes_cap)
                params["_shared_event_re_whitening_last_groups_tau_zero"] = int(metrics_w.n_groups_tau_zero)
                params["_shared_event_re_whitening_last_max_rows"] = int(metrics_w.max_rows_seen)
                params["_shared_event_re_whitening_last_max_nodes"] = int(metrics_w.max_nodes_seen)
                params["_shared_event_re_whitening_last_pcg_iters_sum"] = int(metrics_w.pcg_iters_sum)
                params["_shared_event_re_whitening_last_pcg_iters_max"] = int(metrics_w.pcg_iters_max)
                params["_shared_event_re_whitening_last_pcg_fail"] = int(metrics_w.n_groups_pcg_fail)
                # Mirror whitening stats into runtime stats so epoch_runner logs remain consistent.
                params["_shared_event_re_runtime_last_grouping"] = str(grouping)
                params["_shared_event_re_runtime_last_groups"] = int(metrics_w.n_groups_total)
                params["_shared_event_re_runtime_last_groups_pcg"] = int(metrics_w.n_groups_pcg)
                params["_shared_event_re_runtime_last_groups_fallback_diag"] = int(metrics_w.n_groups_fallback_diag)
                params["_shared_event_re_runtime_last_groups_rows_cap"] = int(metrics_w.n_groups_rows_cap)
                params["_shared_event_re_runtime_last_groups_nodes_cap"] = int(metrics_w.n_groups_nodes_cap)
                params["_shared_event_re_runtime_last_groups_tau_zero"] = int(metrics_w.n_groups_tau_zero)
                params["_shared_event_re_runtime_last_max_rows"] = int(metrics_w.max_rows_seen)
                params["_shared_event_re_runtime_last_max_nodes"] = int(metrics_w.max_nodes_seen)
                # Whitening path does not expose "all" stats; reuse max seen.
                params["_shared_event_re_runtime_last_max_rows_all"] = int(metrics_w.max_rows_seen)
                params["_shared_event_re_runtime_last_max_nodes_all"] = int(metrics_w.max_nodes_seen)
                if not bool(params.get("_shared_event_re_whitening_logged", False)):
                    params["_shared_event_re_whitening_logged"] = True
                    _log(
                        "[shared_event_re] whitening "
                        f"groups={int(metrics_w.n_groups_total)} "
                        f"chol={int(metrics_w.n_groups_chol)} "
                        f"pcg={int(metrics_w.n_groups_pcg)} "
                        f"fallback={int(metrics_w.n_groups_fallback_diag)} "
                        f"max_rows={int(metrics_w.max_rows_seen)} "
                        f"max_nodes={int(metrics_w.max_nodes_seen)} "
                        f"w_mean={float(metrics_w.weight_mean):.3g} "
                        f"w_max={float(metrics_w.weight_max):.3g} "
                        f"solver={str(whiten_solver)}",
                        flush=True,
                    )
                    if isinstance(quad_diag, torch.Tensor):
                        try:
                            qd = float(quad_diag.detach().cpu().item())
                            qw = float(quad_w.detach().cpu().item())
                            ratio = qw / max(qd, 1e-12)
                            _log(
                                f"[shared_event_re] whitening quad_diag={qd:.3g} quad_whiten={qw:.3g} ratio={ratio:.3g}",
                                flush=True,
                            )
                        except Exception:
                            pass
                try:
                    if not torch.isfinite(quad_w).all():
                        raise ValueError("shared_event_re whitening produced non-finite quad")
                except Exception:
                    raise
                quad = quad_w + quad_diag_extra + quad_sp
                loss_like = (quad / m_tot_full) + log_sigma_mean
                _abort_on_pcg_fallback(params, context="whitening")
                return float(alpha) * loss_like

            # Optional GPU-native prototype path (batched PCG + grouping).
            perm = None
            starts = None
            ends = None
            if gpu_enable:
                if not bool(params.get("_shared_event_re_gpu_logged", False)):
                    params["_shared_event_re_gpu_logged"] = True
                    _log("[shared_event_re] GPU path enabled", flush=True)
                try:
                    from spider.core import shared_event_re_gpu
                    # Optional GPU-side cache (avoid per-batch grouping/index remap when batch is stable).
                    cache_ok = False
                    cache_entry = None
                    cache_key = None
                    try:
                        cache_ok = int(params.get("_shared_event_re_cache_max_entries", 0) or 0) > 0
                    except Exception:
                        cache_ok = False
                    # Prefer stable standard batching; fall back to event-bucket ids if present.
                    try:
                        bid = int(params.get("_runtime_batch_id", -1))
                    except Exception:
                        bid = -1
                    try:
                        bucket_id = int(params.get("_runtime_bucket_id", -1))
                    except Exception:
                        bucket_id = -1
                    try:
                        bucket_gen = int(params.get("_runtime_bucket_gen", -1))
                    except Exception:
                        bucket_gen = -1
                    if cache_ok:
                        try:
                            bsz = int(keys.numel())
                            if (not bool(params.get("_runtime_batch_shuffle", True))) and bid >= 0:
                                cache_key = ("shared_event_re_gpu", "batch", str(grouping), int(bsz), int(bid))
                            elif bucket_id >= 0:
                                cache_key = ("shared_event_re_gpu", "bucket", str(grouping), int(bsz), int(bucket_id), int(bucket_gen))
                            if cache_key is not None:
                                cache = params.setdefault("_shared_event_re_gpu_cache", {})
                                cache_entry = cache.get(cache_key, None) if isinstance(cache, dict) else None
                        except Exception:
                            cache_entry = None
                    if (cache_entry is None) and isinstance(grouping_cache, dict):
                        cache_entry = grouping_cache
                        if cache_ok and cache_key is not None:
                            try:
                                cache = params.setdefault("_shared_event_re_gpu_cache", {})
                                if isinstance(cache, dict):
                                    cache[cache_key] = cache_entry
                            except Exception:
                                pass
                    if cache_ok and not isinstance(cache_entry, dict):
                        try:
                            keys_sorted, perm0, starts0, ends0 = shared_event_re_gpu._group_by_keys_gpu(keys)
                            lengths0 = (ends0 - starts0).to(torch.int64)
                            group_ids0, _ = shared_event_re_gpu._build_group_ids(starts0, ends0)
                            idx_perm0 = idx.index_select(0, perm0)
                            u0 = idx_perm0[:, 0].to(torch.int64)
                            v0 = idx_perm0[:, 1].to(torch.int64)
                            max_node_id0 = int(torch.max(torch.stack([u0.max(), v0.max()])).item()) if u0.numel() > 0 else 0
                            local_u0, local_v0, n_nodes0 = shared_event_re_gpu._build_local_node_indices(
                                u0, v0, group_ids0, int(starts0.numel()), max_node_id=max_node_id0
                            )
                            ph_perm0 = ph_id.index_select(0, perm0)
                            ph_group0 = ph_perm0.index_select(0, starts0)
                            edge_idx0 = torch.arange(int(perm0.numel()), device=perm0.device, dtype=starts0.dtype)
                            edge_pos0 = edge_idx0 - starts0.index_select(0, group_ids0)
                            cache_entry = {
                                "perm": perm0.detach(),
                                "starts": starts0.detach(),
                                "ends": ends0.detach(),
                                "lengths": lengths0.detach(),
                                "local_u": local_u0.detach(),
                                "local_v": local_v0.detach(),
                                "n_nodes": n_nodes0.detach(),
                                "ph_group": ph_group0.detach(),
                                "group_ids": group_ids0.detach(),
                                "edge_pos": edge_pos0.detach(),
                            }
                            if cache_ok:
                                cache = params.setdefault("_shared_event_re_gpu_cache", {})
                                if isinstance(cache, dict):
                                    if cache_key is not None:
                                        cache[cache_key] = cache_entry
                                    max_entries = int(params.get("_shared_event_re_cache_max_entries", 0) or 0)
                                    if max_entries > 0 and len(cache) > max_entries:
                                        cache.pop(next(iter(cache)))
                        except Exception:
                            cache_entry = None
                    t0_gpu = time.perf_counter()
                    quad, metrics = shared_event_re_gpu.compute_quad_gpu(
                        idx=idx,
                        resid=resid,
                        keys=keys,
                        ph_id=ph_id,
                        sigma_p=σ_p,
                        sigma_s=σ_s,
                        tau_p=float(tau_p),
                        tau_s=float(tau_s),
                        jitter0=float(jitter0),
                        pcg_max_iters=int(pcg_max_iters),
                        pcg_tol=float(pcg_tol),
                        max_rows_per_group=int(max_rows_per_group),
                        max_nodes_per_group=int(max_nodes_per_group),
                        fallback_to_diag=bool(fallback_to_diag),
                        max_groups_per_batch=int(params.get("_shared_event_re_gpu_max_groups_per_batch", 64)),
                        max_edges_per_batch=int(params.get("_shared_event_re_gpu_max_edges_per_batch", 0) or 0),
                        enable_profile=bool(params.get("_shared_event_re_gpu_profile", False)),
                        reuse_pcg_init=bool(params.get("_shared_event_re_gpu_reuse_pcg_init", False)),
                        precomputed=cache_entry,
                    )
                    quad = quad + quad_diag_extra + quad_sp
                    loss_like = (quad / m_tot_full) + log_sigma_mean
                    _abort_on_pcg_fallback(params, context="gpu")
                    if bool(params.get("_shared_event_re_gpu_profile", False)):
                        dt_ms = float(1000.0 * (time.perf_counter() - t0_gpu))
                        params["_shared_event_re_gpu_time_ms_sum"] = float(params.get("_shared_event_re_gpu_time_ms_sum", 0.0) or 0.0) + dt_ms
                        params["_shared_event_re_gpu_time_ms_count"] = int(params.get("_shared_event_re_gpu_time_ms_count", 0) or 0) + 1
                    try:
                        params["_shared_event_re_gpu_last_groups"] = int(metrics.n_groups_total)
                        params["_shared_event_re_gpu_last_groups_pcg"] = int(metrics.n_groups_pcg)
                        params["_shared_event_re_gpu_last_groups_fallback_diag"] = int(metrics.n_groups_fallback_diag)
                        params["_shared_event_re_gpu_last_groups_rows_cap"] = int(metrics.n_groups_rows_cap)
                        params["_shared_event_re_gpu_last_groups_nodes_cap"] = int(metrics.n_groups_nodes_cap)
                        params["_shared_event_re_gpu_last_groups_tau_zero"] = int(metrics.n_groups_tau_zero)
                        params["_shared_event_re_gpu_last_max_rows"] = int(metrics.max_rows_seen)
                        params["_shared_event_re_gpu_last_max_nodes"] = int(metrics.max_nodes_seen)
                        params["_shared_event_re_gpu_last_max_rows_all"] = int(metrics.max_rows_all)
                        params["_shared_event_re_gpu_last_max_nodes_all"] = int(metrics.max_nodes_all)
                        # Mirror to runtime_last_* so epoch_runner logs remain consistent.
                        params["_shared_event_re_runtime_last_grouping"] = str(grouping)
                        params["_shared_event_re_runtime_last_groups"] = int(metrics.n_groups_total)
                        params["_shared_event_re_runtime_last_groups_pcg"] = int(metrics.n_groups_pcg)
                        params["_shared_event_re_runtime_last_groups_fallback_diag"] = int(metrics.n_groups_fallback_diag)
                        params["_shared_event_re_runtime_last_groups_rows_cap"] = int(metrics.n_groups_rows_cap)
                        params["_shared_event_re_runtime_last_groups_nodes_cap"] = int(metrics.n_groups_nodes_cap)
                        params["_shared_event_re_runtime_last_groups_tau_zero"] = int(metrics.n_groups_tau_zero)
                        params["_shared_event_re_runtime_last_max_rows"] = int(metrics.max_rows_seen)
                        params["_shared_event_re_runtime_last_max_nodes"] = int(metrics.max_nodes_seen)
                        params["_shared_event_re_runtime_last_max_rows_all"] = int(metrics.max_rows_all)
                        params["_shared_event_re_runtime_last_max_nodes_all"] = int(metrics.max_nodes_all)
                    except Exception:
                        pass

                    # Optional parity check: compare GPU vs CPU on a capped number of groups.
                    debug_cap = int(params.get("_shared_event_re_gpu_debug_max_groups", 0) or 0)
                    if debug_cap > 0:
                        quad_dbg, _ = shared_event_re_gpu.compute_quad_gpu(
                            idx=idx,
                            resid=resid,
                            keys=keys,
                            ph_id=ph_id,
                            sigma_p=σ_p,
                            sigma_s=σ_s,
                            tau_p=float(tau_p),
                            tau_s=float(tau_s),
                            jitter0=float(jitter0),
                            pcg_max_iters=int(pcg_max_iters),
                            pcg_tol=float(pcg_tol),
                            max_rows_per_group=int(max_rows_per_group),
                            max_nodes_per_group=int(max_nodes_per_group),
                            fallback_to_diag=bool(fallback_to_diag),
                            max_groups_per_batch=int(params.get("_shared_event_re_gpu_max_groups_per_batch", 64)),
                            reuse_pcg_init=bool(params.get("_shared_event_re_gpu_reuse_pcg_init", False)),
                            group_cap=int(debug_cap),
                        )
                        # CPU reference on the same capped subset
                        keys_sorted_dbg, perm_dbg = torch.sort(keys)
                        is_new_dbg = torch.ones_like(keys_sorted_dbg, dtype=torch.bool)
                        if keys_sorted_dbg.numel() > 0:
                            is_new_dbg[1:] = keys_sorted_dbg[1:] != keys_sorted_dbg[:-1]
                        starts_dbg = torch.nonzero(is_new_dbg, as_tuple=False).reshape(-1)
                        if starts_dbg.numel() > 0:
                            ends_dbg = torch.cat(
                                [starts_dbg[1:], torch.tensor([keys_sorted_dbg.numel()], device=starts_dbg.device, dtype=starts_dbg.dtype)]
                            )
                        else:
                            ends_dbg = torch.zeros((0,), device=resid.device, dtype=torch.int64)
                        ncap = min(int(debug_cap), int(starts_dbg.numel()))
                        quad_cpu = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
                        for si, ei in zip(starts_dbg[:ncap].tolist(), ends_dbg[:ncap].tolist()):
                            idxs = perm_dbg[si:ei]
                            if idxs.numel() == 0:
                                continue
                            idx_g = idx.index_select(0, idxs)
                            resid_g = resid.index_select(0, idxs)
                            ph_g = int(ph_id.index_select(0, idxs[:1]).item())
                            tau = tau_p if ph_g == 0 else tau_s
                            sigma_g = σ_p if ph_g == 0 else σ_s
                            if not (float(tau) > 0.0):
                                u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                            else:
                                u_g = _shared_event_re_u_pcg(
                                    idx_g=idx_g,
                                    resid_g=resid_g,
                                    sigma=sigma_g.clamp_min(1e-12),
                                    tau=float(tau),
                                    jitter0=float(jitter0),
                                    pcg_max_iters=int(pcg_max_iters),
                                    pcg_tol=float(pcg_tol),
                                )
                            quad_cpu = quad_cpu + _CollapsedQuad.apply(resid_g, u_g)
                        denom = quad_cpu.abs().clamp_min(1e-12)
                        params["_shared_event_re_gpu_debug_rel_err"] = float((quad_dbg - quad_cpu).abs().item() / denom.item())
                        params["_shared_event_re_gpu_debug_groups"] = int(ncap)

                    return float(alpha) * loss_like
                except Exception as e:
                    params["_shared_event_re_gpu_failed"] = str(e)
                    # Fall back to the CPU path below.
                    gpu_enable = False

            # --- Optional caching of station_phase/phase grouping (CPU path) ---
            cache_ok = False
            cache_entry = None
            group_n_nodes = None
            cache_key = None
            try:
                cache_ok = int(params.get("_shared_event_re_cache_max_entries", 0) or 0) > 0
            except Exception:
                cache_ok = False
            try:
                bid = int(params.get("_runtime_batch_id", -1))
            except Exception:
                bid = -1
            try:
                bucket_id = int(params.get("_runtime_bucket_id", -1))
            except Exception:
                bucket_id = -1
            try:
                bucket_gen = int(params.get("_runtime_bucket_gen", -1))
            except Exception:
                bucket_gen = -1
            if cache_ok:
                try:
                    bsz = int(keys.numel())
                    if (not bool(params.get("_runtime_batch_shuffle", True))) and bid >= 0:
                        cache_key = ("shared_event_re", "batch", str(grouping), int(bsz), int(bid))
                    elif bucket_id >= 0:
                        cache_key = ("shared_event_re", "bucket", str(grouping), int(bsz), int(bucket_id), int(bucket_gen))
                    if cache_key is not None:
                        cache = params.setdefault("_shared_event_re_group_cache", {})
                        cache_entry = cache.get(cache_key, None) if isinstance(cache, dict) else None
                except Exception:
                    cache_entry = None
                    cache_key = None

            if isinstance(cache_entry, dict):
                perm = cache_entry.get("perm", None)
                starts = cache_entry.get("starts", None)
                ends = cache_entry.get("ends", None)
                group_n_nodes = cache_entry.get("n_nodes", None)
                ok = (
                    isinstance(perm, torch.Tensor) and perm.ndim == 1 and int(perm.numel()) == int(keys.numel())
                    and isinstance(starts, torch.Tensor) and starts.ndim == 1
                    and isinstance(ends, torch.Tensor) and ends.ndim == 1
                    and int(starts.numel()) == int(ends.numel())
                )
                if not ok:
                    perm = None
                    starts = None
                    ends = None
                    group_n_nodes = None

            if not (isinstance(perm, torch.Tensor) and isinstance(starts, torch.Tensor) and isinstance(ends, torch.Tensor)):
                keys_sorted, perm = torch.sort(keys)
                # Run boundaries
                if keys_sorted.numel() > 0:
                    is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
                    is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
                    starts = torch.nonzero(is_new, as_tuple=False).reshape(-1)
                    ends = torch.cat([starts[1:], torch.tensor([keys_sorted.numel()], device=starts.device, dtype=starts.dtype)])
                else:
                    starts = torch.zeros((0,), device=resid.device, dtype=torch.int64)
                    ends = torch.zeros((0,), device=resid.device, dtype=torch.int64)

                # Precompute node counts per group (one-time per cached batch) to avoid torch.unique later.
                if cache_ok and isinstance(starts, torch.Tensor) and int(starts.numel()) > 0:
                    try:
                        group_n_nodes = []
                        for si, ei in zip(starts.tolist(), ends.tolist()):
                            idxs = perm[si:ei]
                            if idxs.numel() <= 0:
                                group_n_nodes.append(0)
                                continue
                            idx_g = idx.index_select(0, idxs)
                            group_n_nodes.append(int(torch.unique(idx_g.reshape(-1)).numel()))
                    except Exception:
                        group_n_nodes = None

                # Store cache entry (best-effort)
                if cache_ok and isinstance(cache_key, tuple):
                    try:
                        cache = params.setdefault("_shared_event_re_group_cache", {})
                        if isinstance(cache, dict):
                            cache[cache_key] = {
                                "perm": perm.detach(),
                                "starts": starts.detach(),
                                "ends": ends.detach(),
                                "n_nodes": group_n_nodes,
                            }
                            max_entries = int(params.get("_shared_event_re_cache_max_entries", 0) or 0)
                            if max_entries > 0 and len(cache) > max_entries:
                                cache.pop(next(iter(cache)))
                    except Exception:
                        pass

            quad = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
            # Optional: lightweight per-call workload summary (used by epoch_runner profiling/logging).
            # We keep this extremely cheap (just counters) so it can be enabled in long runs.
            n_groups_total = 0
            n_groups_pcg = 0
            n_groups_fallback_diag = 0
            max_rows_seen = 0
            max_nodes_seen = 0
            # Process each group
            for gi, (si, ei) in enumerate(zip(starts.tolist(), ends.tolist())):
                idxs = perm[si:ei]
                m_g = int(idxs.numel())
                if m_g <= 0:
                    continue
                n_groups_total += 1
                if m_g > max_rows_seen:
                    max_rows_seen = m_g
                # Decode phase for this group (0=P, 1=S)
                ph_g = int(ph_id.index_select(0, idxs[:1]).item())
                tau = tau_p if ph_g == 0 else tau_s
                sigma_g = σ_p if ph_g == 0 else σ_s

                if m_g > int(max_rows_per_group):
                    if not fallback_to_diag:
                        raise ValueError(
                            f"shared_event_re: group too large (rows={m_g} > max_rows_per_group={max_rows_per_group}). "
                            f"Set max_rows_per_group higher or enable fallback_to_diag."
                        )
                    n_groups_fallback_diag += 1
                    u_g = resid.index_select(0, idxs) / sigma_g.square().clamp_min(1e-24)
                    quad = quad + _CollapsedQuad.apply(resid.index_select(0, idxs), u_g)
                    continue

                idx_g = idx.index_select(0, idxs)
                # Enforce node-count limit (avoid pathological buckets)
                if isinstance(group_n_nodes, list) and gi < len(group_n_nodes):
                    n_nodes = int(group_n_nodes[gi])
                else:
                    try:
                        n_nodes = int(torch.unique(idx_g.reshape(-1)).numel())
                    except Exception:
                        n_nodes = m_g * 2
                if n_nodes > max_nodes_seen:
                    max_nodes_seen = n_nodes
                if n_nodes > int(max_nodes_per_group):
                    if not fallback_to_diag:
                        raise ValueError(
                            f"shared_event_re: group too large (nodes={n_nodes} > max_nodes_per_group={max_nodes_per_group}). "
                            f"Set max_nodes_per_group higher or enable fallback_to_diag."
                        )
                    n_groups_fallback_diag += 1
                    u_g = resid.index_select(0, idxs) / sigma_g.square().clamp_min(1e-24)
                    quad = quad + _CollapsedQuad.apply(resid.index_select(0, idxs), u_g)
                    continue

                resid_g = resid.index_select(0, idxs)
                u_g = _shared_event_re_u_pcg(
                    idx_g=idx_g,
                    resid_g=resid_g,
                    sigma=sigma_g.clamp_min(1e-12),
                    tau=float(tau),
                    jitter0=float(jitter0),
                    pcg_max_iters=int(pcg_max_iters),
                    pcg_tol=float(pcg_tol),
                )
                if float(tau) > 0.0:
                    n_groups_pcg += 1
                quad = quad + _CollapsedQuad.apply(resid_g, u_g)

            # Stash stats for the caller (epoch_runner) to optionally log.
            # Note: params is a mutable dict shared across calls; we keep keys private/prefixed.
            try:
                params["_shared_event_re_runtime_last_grouping"] = str(grouping)
                params["_shared_event_re_runtime_last_groups"] = int(n_groups_total)
                params["_shared_event_re_runtime_last_groups_pcg"] = int(n_groups_pcg)
                params["_shared_event_re_runtime_last_groups_fallback_diag"] = int(n_groups_fallback_diag)
                params["_shared_event_re_runtime_last_max_rows"] = int(max_rows_seen)
                params["_shared_event_re_runtime_last_max_nodes"] = int(max_nodes_seen)
            except Exception:
                pass
            if not bool(params.get("_shared_event_re_logged_runtime", False)):
                params["_shared_event_re_logged_runtime"] = True
                _log(
                    f"[shared_event_re] runtime grouping={grouping} groups={int(n_groups_total)} "
                    f"pcg={int(n_groups_pcg)} fallback={int(n_groups_fallback_diag)} "
                    f"max_rows={int(max_rows_seen)} max_nodes={int(max_nodes_seen)}",
                    flush=True,
                )
            if not bool(params.get("_shared_event_re_logged_delta", False)):
                params["_shared_event_re_logged_delta"] = True
                try:
                    with torch.no_grad():
                        sigma_diag = sigma
                        r = resid
                        quad_diag = 0.5 * (r * (r / sigma_diag.square().clamp_min(1e-24))).sum()
                    m_tot = float(max(int(resid.numel()), 1))
                    loss_re = (quad / m_tot) + torch.log(sigma).mean()
                    loss_diag = (quad_diag / m_tot) + torch.log(sigma).mean()
                    delta = float((loss_re - loss_diag).detach().item())
                    _log(f"[shared_event_re] loss_delta_vs_diag={delta:.6e}", flush=True)
                except Exception as e:
                    _log(f"[shared_event_re] loss_delta_vs_diag failed: {e}", flush=True)

            # Mean over observations (to match the rest of SPIDER)
            quad = quad + quad_diag_extra + quad_sp
            # Keep the independent log(sigma) term for compatibility (with learn_noise_scale=false it's a constant anyway).
            loss_like = (quad / m_tot_full) + log_sigma_mean
            _abort_on_pcg_fallback(params, context="cpu")
            return float(alpha) * loss_like
        else:
            raise NotImplementedError(
                f"model.likelihood.shared_event_re: solver='{solver}' is not implemented. "
                f"Supported in Phase-A: solver='pcg_sparse'."
            )
    # 4. Loss Function
    loss_type = str(params.get("likelihood", "huber")).strip().lower()

    if loss_type in {"gaussian", "mse", "l2"}:
        # NLL ~ 0.5 * r^2
        data_loss = 0.5 * (scaled_resid ** 2)
    elif loss_type in {"student_t", "student-t", "studentt"}:
        # Student-t NLL:
        #   log(sigma) + 0.5*log(nu*pi) + lgamma(nu/2) - lgamma((nu+1)/2)
        #   + (nu+1)/2 * log(1 + (r/sigma)^2 / nu)
        #
        # nu is treated as fixed (configured) here.
        try:
            nu_f = float(params.get("_student_t_nu", 4.0))
        except Exception:
            nu_f = 4.0
        if not (nu_f > 0.0):
            nu_f = 4.0
        nu = scaled_resid.new_tensor(nu_f)
        pi = scaled_resid.new_tensor(float(np.pi))
        t_const = 0.5 * torch.log(nu * pi) + torch.lgamma(0.5 * nu) - torch.lgamma(0.5 * (nu + 1.0))
        data_loss = 0.5 * (nu + 1.0) * torch.log1p((scaled_resid ** 2) / nu) + t_const
    elif loss_type in {"laplace", "l1", "mae"}:
        # NLL ~ |r|
        data_loss = torch.abs(scaled_resid)
    else:
        # Huber (Smooth L1)
        huber_delta = float(params["model"]["likelihood"].get("huber_delta", 1.0))
        data_loss = F.huber_loss(
            scaled_resid, 
            torch.zeros_like(scaled_resid), 
            reduction='none', 
            delta=huber_delta
        )
        
    # 5. Log-determinant term (log sigma)
    # Total NLL = DataLoss + log(sigma)
    total_nll = data_loss + torch.log(sigma)

    # Return MEAN (Average Loss)
    return total_nll.mean()


def compute_prior_loss(
    ΔX_src: torch.Tensor,
    prior_event: torch.distributions.Distribution,
    prior_centroid: torch.distributions.Distribution,
    σ_p: torch.Tensor,
    σ_s: torch.Tensor,
    N_total: int,
    params: dict,
    cluster_ids: torch.Tensor | None = None,
    cluster_counts: torch.Tensor | None = None,
    event_precision_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the Total Prior Negative Log-Probability, scaled by 1/N_total.
    
    This ensures the prior is weighted consistently with the Average Likelihood.
    """
    # Explicit enable flags (defaults preserve legacy behavior)
    event_prior_enable = bool(params.get("prior_event_enable", True))
    centroid_prior_enable = bool(params.get("prior_centroid_enable", False))
    # Runtime gates for other priors (set by the epoch runner). Defaults keep legacy behavior.
    event_runtime_enable = bool(params.get("_prior_event_runtime_enable", True))
    centroid_runtime_enable = bool(params.get("_prior_centroid_runtime_enable", True))

    # 1. Event Location Prior (sum over M events)
    # P(ΔX)
    if (not event_prior_enable) or (not event_runtime_enable):
        log_prob_events = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    elif event_precision_matrix is not None:
        # Hierarchical prior: dX_i ~ N(0, P0^-1)
        # Log Prob ~ -0.5 * dX^T P0 dX + 0.5 log|P0|
        
        # event_precision_matrix is now (K, 4, 4) if cluster_ids provided, else (4, 4)
        if cluster_ids is not None and event_precision_matrix.ndim == 3:
            # Gather correct P0 for each event: (M, 4, 4)
            P0_per_event = event_precision_matrix.index_select(0, cluster_ids)
            
            # term1 = sum_i ( dX_i^T P0_i dX_i )
            # Vectorized: (dX @ P0 * dX).sum()
            # dX_src: (M, 4) -> (M, 1, 4)
            # P0_per_event: (M, 4, 4)
            # matmul: (M, 1, 4) @ (M, 4, 4) -> (M, 1, 4)
            # dot dX: (M, 1, 4) * (M, 1, 4) -> sum last dim -> sum over M
            dX_expanded = ΔX_src.unsqueeze(1)
            term1 = torch.matmul(dX_expanded, P0_per_event).squeeze(1) # (M, 4)
            term1 = (term1 * ΔX_src).sum()
            
            # Log determinant term: sum_i logdet(P0_i)
            # Instead of computing for every event, we can sum per cluster
            # sum_k (M_k * logdet(P0_k))
            if cluster_counts is not None:
                log_det_k = torch.logdet(event_precision_matrix) # (K,)
                # Ensure cluster_counts has shape (K,) or squeeze properly
                cc = cluster_counts
                if cc.ndim > 1:
                    cc = cc.squeeze() # Flatten to 1D
                sum_log_det = (log_det_k * cc).sum()
            else:
                # Fallback if counts not passed (unlikely if ids passed)
                sum_log_det = torch.logdet(P0_per_event).sum()
                
            log_prob_events = -0.5 * term1 + 0.5 * sum_log_det
            
        else:
            # Single global P0 (4, 4)
            P0 = event_precision_matrix
            term1 = torch.sum((ΔX_src @ P0) * ΔX_src) # Scalar sum over M events
            
            log_det = torch.logdet(P0)
            M = ΔX_src.shape[0]
            
            log_prob_events = -0.5 * term1 + 0.5 * M * log_det
        
    else:
        # Standard fixed diagonal prior
        # We treat prior_event as a batch distribution or independent
        log_prob_events = prior_event.log_prob(ΔX_src).sum()
    
    # 2. Centroid prior (scaled by number of events for comparable strength)
    if (not centroid_prior_enable) or (not centroid_runtime_enable):
        log_prob_centroid = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    else:
        if ΔX_src.numel() == 0:
            log_prob_centroid = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
        else:
            centroid = ΔX_src.mean(dim=0)
            log_prob_centroid = prior_centroid.log_prob(centroid)
            if isinstance(log_prob_centroid, torch.Tensor) and log_prob_centroid.ndim > 0:
                log_prob_centroid = log_prob_centroid.sum()
            n_events = int(ΔX_src.shape[0])
            log_prob_centroid = log_prob_centroid * float(n_events)

    # 3. Noise prior removed: SPIDER uses fixed `phase_unc` only (no σ learning).
    log_prob_noise = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    
    # Total Log Prior
    total_log_prior = log_prob_events + log_prob_centroid + log_prob_noise
    
    base_prior_loss = -total_log_prior / float(N_total)
    return base_prior_loss


def total_loss(
    idx, y, X_src, ΔX_src, model,
    prior_event, prior_centroid, σ_p, σ_s,
    N_total, params, nuisance_delta=None,
    cluster_ids=None, cluster_counts=None,
    event_precision_matrix=None,
):
    """
    Compute Total Unified Loss (Average Negative Log Posterior).
    
    Objective = AverageNLL(Data) + (1/N) * NegativeLogPrior(Params)
    
    This objective is independent of dataset size N (as N->inf), 
    stabilizing gradients/hyperparams.
    """
    
    # 1. Likelihood (Average over batch)
    loss_like = compute_likelihood_loss(
        idx, y, X_src, ΔX_src, model, σ_p, σ_s, params, nuisance_delta
    )
    
    loss_prior = compute_prior_loss(
        ΔX_src, prior_event, prior_centroid, σ_p, σ_s, N_total, params,
        cluster_ids, cluster_counts, event_precision_matrix,
    )
    
    return loss_like + loss_prior


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def med_abs_dev_torch(x):
    return torch.median(torch.abs(x - torch.median(x)))

def write_output(origins0, X_src1, unc_src, projector):
    """
    Convert projected coordinates back to lat/lon and write DataFrame.
    """
    origins = origins0.clone()
    
    # Update columns
    new_cols = [
        pl.Series("T_src", X_src1[:, 3]),
        pl.Series("depth", X_src1[:, 2]),
        pl.Series("X", X_src1[:, 0]),
        pl.Series("Y", X_src1[:, 1]),
        pl.Series("unc_x", np.full(len(origins0), np.nan)),
        pl.Series("unc_y", np.full(len(origins0), np.nan)),
        pl.Series("unc_z", np.full(len(origins0), np.nan)),
    ]
    origins = origins.with_columns(new_cols)

    # Project back to Lat/Lon
    X = origins["X"].to_numpy()
    Y = origins["Y"].to_numpy()
    lon, lat = projector(X, Y, inverse=True)
    
    origins = origins.with_columns([
        pl.Series("longitude", lon),
        pl.Series("latitude", lat)
    ])
    
    return origins

# Legacy alias for compatibility
posterior_loss = total_loss
likelihood_loss = compute_likelihood_loss
prior_loss = compute_prior_loss

def prior_loss_event(ΔX_src, prior_event):
    """Legacy wrapper for event prior loss (unscaled)."""
    return -prior_event.log_prob(ΔX_src).sum()


def med_abs_dev(x):
    """Compute median absolute deviation."""
    return np.median(np.abs(x - np.median(x)))

def shuffle_data(x, y):
    """Shuffle data arrays together."""
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]

