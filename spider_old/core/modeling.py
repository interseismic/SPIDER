import numpy as np
import polars as pl
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from spider.core.shared_event_re_whitening import compute_quad_whitening


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
    sigma_extra_var: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the Average Negative Log-Likelihood (per observation).
    
    Loss = Mean( DataLoss(residual / sigma) + log(sigma) )
    """
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
    
    # 3. Standardized Residuals
    # Clamp sigma to avoid division by zero
    sigma = sigma.clamp_min(1e-12)
    sigma2 = sigma.square()
    # Optional: add extra per-observation variance (e.g., FITC diagonal correction for inducing GP).
    # This leaves noise priors (which depend on base σ_p/σ_s) unchanged.
    if sigma_extra_var is not None:
        try:
            sev = sigma_extra_var
            if not isinstance(sev, torch.Tensor):
                sev = torch.tensor(sev, device=sigma.device, dtype=sigma.dtype)
            sev = sev.to(device=sigma.device, dtype=sigma.dtype).clamp_min(0.0)
            sigma2 = sigma2 + sev
        except Exception:
            pass
    # Optional: Student-t scale-mixture (per-row precision lambda).
    try:
        if bool(params.get("_student_t_scale_enabled", False)):
            lam_full = params.get("_student_t_lambda", None)
            rows = params.get("_runtime_batch_rows", None)
            if isinstance(lam_full, torch.Tensor):
                if lam_full.numel() == sigma2.numel():
                    lam_b = lam_full
                elif isinstance(rows, torch.Tensor) and int(rows.numel()) == int(sigma2.numel()):
                    lam_b = lam_full.index_select(0, rows.to(torch.int64))
                else:
                    lam_b = None
                if isinstance(lam_b, torch.Tensor):
                    lam_min = float(params.get("_student_t_scale_min_lambda", 1e-6))
                    lam_max = float(params.get("_student_t_scale_max_lambda", 1e6))
                    lam_b = lam_b.to(device=sigma2.device, dtype=sigma2.dtype).clamp_min(lam_min).clamp_max(lam_max)
                    sigma2 = sigma2 / lam_b
            else:
                lam_b = None
            if (not isinstance(lam_full, torch.Tensor)) or (not isinstance(rows, torch.Tensor) and lam_full.numel() != sigma2.numel()):
                if not bool(params.get("_student_t_scale_warned_missing_rows", False)):
                    params["_student_t_scale_warned_missing_rows"] = True
                    print("[student_t_scale] missing per-row indices; skipping lambda scaling for this batch", flush=True)
    except Exception:
        pass
    sigma = sigma2.sqrt().clamp_min(1e-12)
    resid = dt_obs - dt_pred
    scaled_resid = resid / sigma
    # Cache baseline loss inputs for optional dd_graph_re loss delta.
    resid_base = resid
    scaled_resid_base = scaled_resid

    # Optional: DD-graph random effects (explicit event latents, per phase).
    if bool(params.get("_dd_graph_re_enabled", False)):
        b_p = params.get("_dd_graph_re_b_p", None)
        b_s = params.get("_dd_graph_re_b_s", None)
        if not (isinstance(b_p, torch.Tensor) and isinstance(b_s, torch.Tensor)):
            raise ValueError("dd_graph_re enabled but latents are missing.")
        bi_p = b_p.index_select(0, idx[:, 0].to(torch.int64))
        bj_p = b_p.index_select(0, idx[:, 1].to(torch.int64))
        bi_s = b_s.index_select(0, idx[:, 0].to(torch.int64))
        bj_s = b_s.index_select(0, idx[:, 1].to(torch.int64))
        pred_dd = torch.where(is_p, bi_p - bj_p, bi_s - bj_s)
        resid0_dd = resid
        resid = resid0_dd - pred_dd
        scaled_resid = resid / sigma
        try:
            epoch_idx = int(params.get("_runtime_epoch_index", -1))
            if int(params.get("_dd_graph_re_logged_epoch", -2)) != int(epoch_idx):
                params["_dd_graph_re_logged_epoch"] = int(epoch_idx)
                rms_pred = float((pred_dd.square().mean().sqrt()).detach().item())
                rms_resid = float((resid.square().mean().sqrt()).detach().item())
                pred_p = pred_dd[is_p] if isinstance(is_p, torch.Tensor) else None
                pred_s = pred_dd[~is_p] if isinstance(is_p, torch.Tensor) else None
                resid_p = resid[is_p] if isinstance(is_p, torch.Tensor) else None
                resid_s = resid[~is_p] if isinstance(is_p, torch.Tensor) else None
                def _rms(t: torch.Tensor | None) -> float:
                    if not isinstance(t, torch.Tensor) or t.numel() == 0:
                        return float("nan")
                    return float(t.square().mean().sqrt().detach().item())
                params["_dd_graph_re_pred_rms"] = float(rms_pred)
                params["_dd_graph_re_resid_rms"] = float(rms_resid)
                params["_dd_graph_re_pred_rms_p"] = _rms(pred_p)
                params["_dd_graph_re_pred_rms_s"] = _rms(pred_s)
                params["_dd_graph_re_resid_rms_p"] = _rms(resid_p)
                params["_dd_graph_re_resid_rms_s"] = _rms(resid_s)
        except Exception:
            pass

    # Optional: collapsed slowness inducing-GP covariance likelihood (Gaussian; marginalized; no latent state).
    #
    # Phase-A implementation: quadratic-only (drop logdet). We compute u ≈ Σ^{-1} r per group and return:
    #   mean( 0.5 r^T u ) + mean(log sigma)
    # with custom autograd so d/dr = u (do not differentiate through the solver).
    try:
        sl_enable = bool(params.get("_slowness_re_enabled", False))
    except Exception:
        sl_enable = False
    if not bool(params.get("_shared_event_re_seen_in_loss", False)):
        params["_shared_event_re_seen_in_loss"] = True
        try:
            se_flag = bool(params.get("_shared_event_re_enabled", False))
        except Exception:
            se_flag = False
        try:
            sl_flag = bool(params.get("_slowness_re_enabled", False))
        except Exception:
            sl_flag = False
        print(f"[shared_event_re] compute_likelihood_loss se_enabled={se_flag} slowness_re_enabled={sl_flag}", flush=True)
    if sl_enable:
        sl_mode = str(params.get("_slowness_re_mode", "scalar_sep")).strip().lower()
        if sl_mode in {"scalar", "scalar_sep", "scalar_separation", "event_sep"}:
            # Scalar slowness random effects: r_ij ≈ g_ij (s_i - s_j), with g_ij = ||x_i - x_j||.
            grouping = str(params.get("_slowness_re_grouping", "station_phase")).strip().lower()
            if grouping in {"stationphase", "station-phase"}:
                grouping = "station_phase"
            tau_ps = params.get("_slowness_re_tau_s", [0.0, 0.0])
            tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            tau_units = str(params.get("_slowness_re_tau_units", "abs")).strip().lower()
            max_rows_per_group = int(params.get("_slowness_re_max_rows_per_group", 200000))
            max_nodes_per_group = int(params.get("_slowness_re_max_nodes_per_group", 2048))
            fallback_to_diag = bool(params.get("_slowness_re_fallback_to_diag", True))
            pcg_max_iters = int(params.get("_slowness_re_pcg_max_iters", 30))
            pcg_tol = float(params.get("_slowness_re_pcg_tol", 1e-4))
            jitter0 = float(params.get("_slowness_re_jitter0", 1e-8))
            sep_cap_km = float(params.get("_slowness_re_sep_cap_km", 0.0))

            # tau units: if vel_frac, convert with a simple Vp/Vs scaling.
            if tau_units == "vel_frac":
                vp = float(params.get("_slowness_re_vp_km_s", 6.0))
                vs = float(params.get("_slowness_re_vs_km_s", 3.5))
                tau_p = tau_p / max(vp, 1e-6)
                tau_s = tau_s / max(vs, 1e-6)

            # Phase id per row: 0=P, 1=S.
            ph_id = torch.where(
                is_p,
                torch.zeros_like(resid, dtype=torch.int64),
                torch.ones_like(resid, dtype=torch.int64),
            )

            # Station index (optional) for station_phase grouping.
            sta_idx = None
            if grouping == "station_phase":
                sta_idx = params.get("_runtime_bucket_station_index", None)
                if not isinstance(sta_idx, torch.Tensor) or int(sta_idx.numel()) != int(resid.numel()):
                    if not bool(params.get("_slowness_re_warned_no_station_index", False)):
                        print(
                            "Warning: slowness_re.grouping='station_phase' requested but no per-row station index "
                            "was available for this batch. Falling back to grouping='phase'."
                        )
                        params["_slowness_re_warned_no_station_index"] = True
                    grouping = "phase"
                    sta_idx = None

            # Separation magnitude g_ij (optionally frozen at MAP locations).
            freeze_g = bool(params.get("_slowness_re_freeze_g_at_map", False))
            if freeze_g:
                X_map = params.get("_slowness_re_g_x_map", None)
                if not isinstance(X_map, torch.Tensor) or int(X_map.shape[0]) != int(X_src.shape[0]):
                    X_map = (X_src + ΔX_src)[:, :3].detach().to(device=X_src.device, dtype=X_src.dtype)
                    params["_slowness_re_g_x_map"] = X_map
                x1 = X_map.index_select(0, idx[:, 0].to(torch.int64))
                x2 = X_map.index_select(0, idx[:, 1].to(torch.int64))
            else:
                x1 = X_src[idx[:, 0], :3] + ΔX_src[idx[:, 0], :3]
                x2 = X_src[idx[:, 1], :3] + ΔX_src[idx[:, 1], :3]
            g = torch.linalg.norm(x1 - x2, dim=1).clamp_min(1e-6)
            if sep_cap_km > 0.0 and math.isfinite(sep_cap_km):
                g = g.clamp_max(float(sep_cap_km))

            # Group keys (sorted -> contiguous runs).
            if grouping == "phase":
                keys = ph_id
            else:
                keys = (sta_idx.to(dtype=torch.int64) * 2) + ph_id  # type: ignore[union-attr]

            quad = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
            m_tot = float(max(int(resid.numel()), 1))
            log_sigma_mean = torch.log(sigma).mean()

            # Optional GPU path (batched PCG).
            gpu_enable_raw = params.get("_slowness_re_gpu_enable", None)
            gpu_enable = bool(resid.is_cuda) if gpu_enable_raw is None else bool(gpu_enable_raw)
            if gpu_enable:
                try:
                    from spider.core import shared_event_re_gpu
                    debug_cap = int(params.get("_slowness_re_gpu_debug_max_groups", 0) or 0)
                    quad_gpu, metrics = shared_event_re_gpu.compute_slowness_quad_gpu(
                        idx=idx,
                        resid=resid,
                        g=g,
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
                        max_groups_per_batch=int(params.get("_slowness_re_gpu_max_groups_per_batch", 64)),
                        max_edges_per_batch=int(params.get("_slowness_re_gpu_max_edges_per_batch", 0) or 0),
                        group_cap=(debug_cap if debug_cap > 0 else None),
                        enable_profile=bool(params.get("_slowness_re_gpu_profile", False)),
                    )
                    quad = quad_gpu
                    try:
                        params["_slowness_re_runtime_last_grouping"] = str(grouping)
                        params["_slowness_re_runtime_last_groups"] = int(metrics.n_groups_total)
                        params["_slowness_re_runtime_last_groups_woodbury"] = int(metrics.n_groups_pcg)
                        params["_slowness_re_runtime_last_groups_fallback_diag"] = int(metrics.n_groups_fallback_diag)
                        params["_slowness_re_runtime_last_max_rows"] = int(metrics.max_rows_seen)
                        params["_slowness_re_runtime_last_max_nodes"] = int(metrics.max_nodes_seen)
                        params["_slowness_re_gpu_last_groups_rows_cap"] = int(metrics.n_groups_rows_cap)
                        params["_slowness_re_gpu_last_groups_nodes_cap"] = int(metrics.n_groups_nodes_cap)
                        params["_slowness_re_gpu_last_groups_tau_zero"] = int(metrics.n_groups_tau_zero)
                        params["_slowness_re_gpu_last_max_rows_all"] = int(metrics.max_rows_all)
                        params["_slowness_re_gpu_last_max_nodes_all"] = int(metrics.max_nodes_all)
                        params["_slowness_re_gpu_last_ms_total"] = float(metrics.ms_total)
                    except Exception:
                        pass
                    # Optional parity check on a capped subset of groups.
                    if (debug_cap > 0) and (not bool(params.get("_slowness_re_gpu_debug_logged", False))):
                        params["_slowness_re_gpu_debug_logged"] = True
                        try:
                            keys_sorted, perm_dbg = torch.sort(keys)
                            if keys_sorted.numel() > 0:
                                is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
                                is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
                                starts = torch.nonzero(is_new, as_tuple=False).flatten()
                                ends = torch.cat([starts[1:], torch.tensor([keys_sorted.numel()], device=keys_sorted.device)])
                            else:
                                starts = torch.zeros((0,), device=keys.device, dtype=torch.int64)
                                ends = torch.zeros((0,), device=keys.device, dtype=torch.int64)
                            ncap = min(int(debug_cap), int(starts.numel()))
                            if ncap > 0:
                                max_edge = int(ends[ncap - 1].item())
                                perm_dbg = perm_dbg[:max_edge]
                                starts = starts[:ncap]
                                ends = ends[:ncap]
                            resid_s = resid.index_select(0, perm_dbg)
                            sigma_s = sigma.index_select(0, perm_dbg)
                            idx_s = idx.index_select(0, perm_dbg)
                            g_s = g.index_select(0, perm_dbg)
                            ph_s = ph_id.index_select(0, perm_dbg)
                            quad_cpu = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
                            for s0, e0 in zip(starts.tolist(), ends.tolist()):
                                resid_g = resid_s[s0:e0]
                                sigma_g = sigma_s[s0:e0]
                                idx_g = idx_s[s0:e0]
                                g_g = g_s[s0:e0]
                                m = int(resid_g.numel())
                                if m == 0:
                                    continue
                                ev_flat = idx_g.reshape(-1)
                                nodes, inv_nodes = torch.unique(ev_flat, return_inverse=True)
                                n_nodes = int(nodes.numel())
                                tau = float(tau_p if int(ph_s[s0].item()) == 0 else tau_s)
                                if (m > max_rows_per_group) or (n_nodes > max_nodes_per_group) or (not (tau > 0.0)):
                                    u_edge = resid_g / sigma_g.square().clamp_min(1e-24)
                                    quad_cpu = quad_cpu + _CollapsedQuad.apply(resid_g, u_edge)
                                    continue
                                u = inv_nodes[:m]
                                v = inv_nodes[m:]
                                w = (g_g * g_g) / sigma_g.square().clamp_min(1e-24)
                                b = torch.zeros((n_nodes,), device=resid.device, dtype=resid.dtype)
                                b.index_add_(0, u, w * resid_g)
                                b.index_add_(0, v, -w * resid_g)
                                alpha_pcg = torch.tensor((1.0 / (tau * tau)) + float(jitter0), device=resid.device, dtype=resid.dtype)
                                x = _pcg_solve_weighted(
                                    u=u, v=v, w=w, b=b, alpha=alpha_pcg,
                                    max_iters=int(pcg_max_iters), tol=float(pcg_tol),
                                )
                                x_i = x.index_select(0, u)
                                x_j = x.index_select(0, v)
                                Hx = g_g * (x_i - x_j)
                                u_edge = (resid_g - Hx) / sigma_g.square().clamp_min(1e-24)
                                quad_cpu = quad_cpu + _CollapsedQuad.apply(resid_g, u_edge)
                            diff = float((quad_gpu - quad_cpu).detach().item())
                            print(f"[slowness_re_gpu] quad diff (gpu-cpu) on {int(ncap)} groups = {diff:.6e}", flush=True)
                        except Exception as e:
                            print(f"[slowness_re_gpu] debug compare failed: {e}", flush=True)

                    if not bool(params.get("_slowness_re_logged_runtime", False)):
                        params["_slowness_re_logged_runtime"] = True
                        try:
                            g = int(params.get("_slowness_re_runtime_last_groups", 0) or 0)
                            g_pcg = int(params.get("_slowness_re_runtime_last_groups_woodbury", 0) or 0)
                            g_fb = int(params.get("_slowness_re_runtime_last_groups_fallback_diag", 0) or 0)
                            mr = int(params.get("_slowness_re_runtime_last_max_rows", 0) or 0)
                            mn = int(params.get("_slowness_re_runtime_last_max_nodes", 0) or 0)
                            grp = str(params.get("_slowness_re_runtime_last_grouping", grouping))
                            print(
                                f"[slowness_re] runtime grouping={grp} groups={g} pcg={g_pcg} "
                                f"fallback={g_fb} max_rows={mr} max_nodes={mn}",
                                flush=True,
                            )
                        except Exception:
                            pass
                    if not bool(params.get("_slowness_re_logged_delta", False)):
                        params["_slowness_re_logged_delta"] = True
                        try:
                            quad_diag = _CollapsedQuad.apply(resid, resid / sigma.square().clamp_min(1e-24))
                            delta = float(((quad - quad_diag) / m_tot).detach().item())
                            print(f"[slowness_re] loss delta vs diag = {delta:.6e}", flush=True)
                        except Exception as e:
                            print(f"[slowness_re] loss delta vs diag failed: {e}", flush=True)
                    loss_like = (quad / m_tot) + log_sigma_mean
                    return loss_like
                except Exception:
                    pass

            keys_sorted, perm = torch.sort(keys)
            resid_s = resid.index_select(0, perm)
            sigma_s = sigma.index_select(0, perm)
            idx_s = idx.index_select(0, perm)
            g_s = g.index_select(0, perm)
            ph_s = ph_id.index_select(0, perm)

            n_groups_total = 0
            n_groups_pcg = 0
            n_groups_fallback_diag = 0
            max_rows_seen = 0
            max_nodes_seen = 0

            if keys_sorted.numel() > 0:
                is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
                is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
                starts = torch.nonzero(is_new, as_tuple=False).flatten()
                ends = torch.cat([starts[1:], torch.tensor([keys_sorted.numel()], device=keys_sorted.device)])
                for s0, e0 in zip(starts.tolist(), ends.tolist()):
                    n_groups_total += 1
                    resid_g = resid_s[s0:e0]
                    sigma_g = sigma_s[s0:e0]
                    idx_g = idx_s[s0:e0]
                    g_g = g_s[s0:e0]
                    ph_g = ph_s[s0:e0]
                    m = int(resid_g.numel())
                    if m == 0:
                        continue
                    max_rows_seen = max(max_rows_seen, m)
                    ev_flat = idx_g.reshape(-1)
                    nodes, inv_nodes = torch.unique(ev_flat, return_inverse=True)
                    n_nodes = int(nodes.numel())
                    max_nodes_seen = max(max_nodes_seen, n_nodes)
                    tau = float(tau_p if int(ph_g[0].item()) == 0 else tau_s)
                    if (m > max_rows_per_group) or (n_nodes > max_nodes_per_group) or (not (tau > 0.0)):
                        if fallback_to_diag:
                            u_edge = resid_g / sigma_g.square().clamp_min(1e-24)
                            quad = quad + _CollapsedQuad.apply(resid_g, u_edge)
                            n_groups_fallback_diag += 1
                            continue
                        raise ValueError(
                            f"slowness_re group too large (rows={m}, nodes={n_nodes}); "
                            "increase max_rows_per_group/max_nodes_per_group or enable fallback_to_diag."
                        )

                    u = inv_nodes[:m]
                    v = inv_nodes[m:]
                    w = (g_g * g_g) / sigma_g.square().clamp_min(1e-24)
                    b = torch.zeros((n_nodes,), device=resid.device, dtype=resid.dtype)
                    b.index_add_(0, u, w * resid_g)
                    b.index_add_(0, v, -w * resid_g)
                    alpha_pcg = torch.tensor((1.0 / (tau * tau)) + float(jitter0), device=resid.device, dtype=resid.dtype)
                    x = _pcg_solve_weighted(
                        u=u,
                        v=v,
                        w=w,
                        b=b,
                        alpha=alpha_pcg,
                        max_iters=int(pcg_max_iters),
                        tol=float(pcg_tol),
                    )
                    x_i = x.index_select(0, u)
                    x_j = x.index_select(0, v)
                    Hx = g_g * (x_i - x_j)
                    u_edge = (resid_g - Hx) / sigma_g.square().clamp_min(1e-24)
                    quad = quad + _CollapsedQuad.apply(resid_g, u_edge)
                    n_groups_pcg += 1

            # Stash stats for caller (epoch_runner profiling).
            try:
                params["_slowness_re_runtime_last_grouping"] = str(grouping)
                params["_slowness_re_runtime_last_groups"] = int(n_groups_total)
                params["_slowness_re_runtime_last_groups_woodbury"] = int(n_groups_pcg)
                params["_slowness_re_runtime_last_groups_fallback_diag"] = int(n_groups_fallback_diag)
                params["_slowness_re_runtime_last_max_rows"] = int(max_rows_seen)
                params["_slowness_re_runtime_last_max_nodes"] = int(max_nodes_seen)
            except Exception:
                pass
            if not bool(params.get("_slowness_re_logged_runtime", False)):
                params["_slowness_re_logged_runtime"] = True
                print(
                    f"[slowness_re] runtime grouping={grouping} groups={int(n_groups_total)} "
                    f"pcg={int(n_groups_pcg)} fallback={int(n_groups_fallback_diag)} "
                    f"max_rows={int(max_rows_seen)} max_nodes={int(max_nodes_seen)}",
                    flush=True,
                )

            if not bool(params.get("_slowness_re_logged_delta", False)):
                params["_slowness_re_logged_delta"] = True
                try:
                    quad_diag = _CollapsedQuad.apply(resid, resid / sigma.square().clamp_min(1e-24))
                    delta = float(((quad - quad_diag) / m_tot).detach().item())
                    print(f"[slowness_re] loss delta vs diag = {delta:.6e}", flush=True)
                except Exception as e:
                    print(f"[slowness_re] loss delta vs diag failed: {e}", flush=True)
            loss_like = (quad / m_tot) + log_sigma_mean
            return loss_like

        if sl_mode in {"component_station_explicit"}:
            # Explicit component + station slowness latents:
            # r_ij = g_ij * (s_c + a_k) + eps, with s_c per component and a_k per station (both per phase).
            grouping = str(params.get("_slowness_re_grouping", "phase")).strip().lower()
            if grouping in {"stationphase", "station-phase"}:
                grouping = "station_phase"
            tau_ps = params.get("_slowness_re_tau_s", [0.0, 0.0])
            tau_sta_ps = params.get("_slowness_re_tau_station_s", [0.0, 0.0])
            tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            tau_sta_p = float(tau_sta_ps[0]) if isinstance(tau_sta_ps, (list, tuple)) and len(tau_sta_ps) >= 2 else float(tau_sta_ps)
            tau_sta_s = float(tau_sta_ps[1]) if isinstance(tau_sta_ps, (list, tuple)) and len(tau_sta_ps) >= 2 else float(tau_sta_ps)
            tau_units = str(params.get("_slowness_re_tau_units", "abs")).strip().lower()
            sep_cap_km = float(params.get("_slowness_re_sep_cap_km", 0.0))

            # tau units: if vel_frac, convert with a simple Vp/Vs scaling.
            if tau_units == "vel_frac":
                vp = float(params.get("_slowness_re_vp_km_s", 6.0))
                vs = float(params.get("_slowness_re_vs_km_s", 3.5))
                tau_p = tau_p / max(vp, 1e-6)
                tau_s = tau_s / max(vs, 1e-6)
                tau_sta_p = tau_sta_p / max(vp, 1e-6)
                tau_sta_s = tau_sta_s / max(vs, 1e-6)

            # Station index (required)
            sta_idx = params.get("_runtime_bucket_station_index", None)
            if (not isinstance(sta_idx, torch.Tensor)) or (int(sta_idx.numel()) != int(resid.numel())):
                raise ValueError("slowness_re.component_station_explicit requires per-row station index.")

            # Component ids per event (required)
            cid_ev = params.get("_runtime_event_cluster_ids", None)
            if not isinstance(cid_ev, torch.Tensor):
                cid_ev = params.get("_shared_event_re_cluster_ids", None)
            if not isinstance(cid_ev, torch.Tensor):
                raise ValueError("slowness_re.component_station_explicit requires event component ids.")

            # Separation magnitude g_ij (optionally frozen at MAP locations).
            freeze_g = bool(params.get("_slowness_re_freeze_g_at_map", False))
            if freeze_g:
                X_map = params.get("_slowness_re_g_x_map", None)
                if not isinstance(X_map, torch.Tensor) or int(X_map.shape[0]) != int(X_src.shape[0]):
                    X_map = (X_src + ΔX_src)[:, :3].detach().to(device=X_src.device, dtype=X_src.dtype)
                    params["_slowness_re_g_x_map"] = X_map
                x1 = X_map.index_select(0, idx[:, 0].to(torch.int64))
                x2 = X_map.index_select(0, idx[:, 1].to(torch.int64))
            else:
                x1 = X_src[idx[:, 0], :3] + ΔX_src[idx[:, 0], :3]
                x2 = X_src[idx[:, 1], :3] + ΔX_src[idx[:, 1], :3]
            g = torch.linalg.norm(x1 - x2, dim=1).clamp_min(1e-6)
            if sep_cap_km > 0.0 and math.isfinite(sep_cap_km):
                g = g.clamp_max(float(sep_cap_km))

            # Per-row component id (assume within-component pairs; if not, use event-1 id).
            comp1 = cid_ev.index_select(0, idx[:, 0].to(torch.int64))
            comp2 = cid_ev.index_select(0, idx[:, 1].to(torch.int64))
            comp_row = comp1
            if bool((comp1 != comp2).any()) and (not bool(params.get("_slowness_re_warned_cross_component", False))):
                params["_slowness_re_warned_cross_component"] = True
                print("Warning: slowness_re.component_station_explicit saw cross-component pairs; using component of event-1.", flush=True)

            # Latents (required)
            s_comp_p = params.get("_slowness_re_explicit_comp_p", None)
            s_comp_s = params.get("_slowness_re_explicit_comp_s", None)
            a_sta_p = params.get("_slowness_re_explicit_station_p", None)
            a_sta_s = params.get("_slowness_re_explicit_station_s", None)
            if not all(isinstance(x, torch.Tensor) for x in (s_comp_p, s_comp_s, a_sta_p, a_sta_s)):
                raise ValueError("slowness_re.component_station_explicit requires explicit latents in params.")

            s_comp = torch.where(is_p, s_comp_p.index_select(0, comp_row), s_comp_s.index_select(0, comp_row))
            a_sta = torch.where(is_p, a_sta_p.index_select(0, sta_idx), a_sta_s.index_select(0, sta_idx))
            pred = g * (s_comp + a_sta)
            resid0 = resid
            resid = resid0 - pred
            scaled_resid = resid / sigma

            epoch_idx = int(params.get("_runtime_epoch_index", -1))
            if int(params.get("_slowness_re_explicit_logged_epoch", -2)) != int(epoch_idx):
                params["_slowness_re_explicit_logged_epoch"] = int(epoch_idx)
                try:
                    rms_pred = float((pred.square().mean().sqrt()).detach().item())
                    rms_resid = float((resid.square().mean().sqrt()).detach().item())
                    g_rms = float((g.square().mean().sqrt()).detach().item())
                    print(
                        f"[slowness_re] explicit rms: resid={rms_resid:.4f}s pred={rms_pred:.4f}s "
                        f"g={g_rms:.4f}km tau_comp_p={tau_p:.3g} tau_comp_s={tau_s:.3g} "
                        f"tau_sta_p={tau_sta_p:.3g} tau_sta_s={tau_sta_s:.3g} units={tau_units}",
                        flush=True,
                    )
                    # Extra diagnostics: latent scales + correlation with residuals.
                    s_sum = s_comp + a_sta
                    s_rms = float((s_sum.square().mean().sqrt()).detach().item()) if s_sum.numel() > 0 else float("nan")
                    sc_rms = float((s_comp.square().mean().sqrt()).detach().item()) if s_comp.numel() > 0 else float("nan")
                    sa_rms = float((a_sta.square().mean().sqrt()).detach().item()) if a_sta.numel() > 0 else float("nan")
                    r0 = resid0 - resid0.mean()
                    p0 = pred - pred.mean()
                    denom = float((r0.square().mean().sqrt() * p0.square().mean().sqrt()).detach().item())
                    if denom > 0.0 and torch.isfinite(r0).all() and torch.isfinite(p0).all():
                        corr = float((r0 * p0).mean().detach().item() / denom)
                    else:
                        corr = float("nan")
                    print(
                        f"[slowness_re] explicit stats: rms_s={s_rms:.4f} rms_sc={sc_rms:.4f} "
                        f"rms_sa={sa_rms:.4f} corr(resid0,pred)={corr:.4f}",
                        flush=True,
                    )
                except Exception:
                    pass
            # Explicit mode uses standard per-row loss on adjusted residuals.
            loss_type = str(params.get("likelihood", "huber")).strip().lower()
            if loss_type in {"gaussian", "mse", "l2"}:
                data_loss = 0.5 * (scaled_resid ** 2)
            elif loss_type in {"student_t", "student-t", "studentt"}:
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
                data_loss = torch.abs(scaled_resid)
            else:
                huber_delta = float(params["model"]["likelihood"].get("huber_delta", 1.0))
                data_loss = F.huber_loss(
                    scaled_resid,
                    torch.zeros_like(scaled_resid),
                    reduction="none",
                    delta=huber_delta,
                )
            total_nll = data_loss + torch.log(sigma)
            if int(params.get("_slowness_re_explicit_loss_logged_epoch", -2)) != int(epoch_idx):
                params["_slowness_re_explicit_loss_logged_epoch"] = int(epoch_idx)
                try:
                    # Baseline loss without explicit correction (same loss family).
                    scaled_resid0 = resid0 / sigma
                    if loss_type in {"gaussian", "mse", "l2"}:
                        data_loss0 = 0.5 * (scaled_resid0 ** 2)
                    elif loss_type in {"student_t", "student-t", "studentt"}:
                        data_loss0 = 0.5 * (nu + 1.0) * torch.log1p((scaled_resid0 ** 2) / nu) + t_const
                    elif loss_type in {"laplace", "l1", "mae"}:
                        data_loss0 = torch.abs(scaled_resid0)
                    else:
                        data_loss0 = F.huber_loss(
                            scaled_resid0,
                            torch.zeros_like(scaled_resid0),
                            reduction="none",
                            delta=huber_delta,
                        )
                    total_nll0 = data_loss0 + torch.log(sigma)
                    delta = float((total_nll.mean() - total_nll0.mean()).detach().item())
                    print(f"[slowness_re] explicit loss delta vs baseline = {delta:.6e}", flush=True)
                except Exception:
                    pass
            return total_nll.mean()

        if sl_mode in {"component_station"}:
            # Component + station slowness random effects:
            # r_ij = g_ij * (s_c + a_k) + eps, with s_c per component and a_k per station (both per phase).
            grouping = str(params.get("_slowness_re_grouping", "phase")).strip().lower()
            if grouping in {"stationphase", "station-phase"}:
                grouping = "station_phase"
            tau_ps = params.get("_slowness_re_tau_s", [0.0, 0.0])
            tau_sta_ps = params.get("_slowness_re_tau_station_s", [0.0, 0.0])
            tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            tau_sta_p = float(tau_sta_ps[0]) if isinstance(tau_sta_ps, (list, tuple)) and len(tau_sta_ps) >= 2 else float(tau_sta_ps)
            tau_sta_s = float(tau_sta_ps[1]) if isinstance(tau_sta_ps, (list, tuple)) and len(tau_sta_ps) >= 2 else float(tau_sta_ps)
            tau_units = str(params.get("_slowness_re_tau_units", "abs")).strip().lower()
            max_rows_per_group = int(params.get("_slowness_re_max_rows_per_group", 200000))
            max_nodes_per_group = int(params.get("_slowness_re_max_nodes_per_group", 2048))
            fallback_to_diag = bool(params.get("_slowness_re_fallback_to_diag", True))
            pcg_max_iters = int(params.get("_slowness_re_pcg_max_iters", 30))
            pcg_tol = float(params.get("_slowness_re_pcg_tol", 1e-4))
            jitter0 = float(params.get("_slowness_re_jitter0", 1e-8))
            sep_cap_km = float(params.get("_slowness_re_sep_cap_km", 0.0))

            # tau units: if vel_frac, convert with a simple Vp/Vs scaling.
            if tau_units == "vel_frac":
                vp = float(params.get("_slowness_re_vp_km_s", 6.0))
                vs = float(params.get("_slowness_re_vs_km_s", 3.5))
                tau_p = tau_p / max(vp, 1e-6)
                tau_s = tau_s / max(vs, 1e-6)
                tau_sta_p = tau_sta_p / max(vp, 1e-6)
                tau_sta_s = tau_sta_s / max(vs, 1e-6)

            ph_id = torch.where(
                is_p,
                torch.zeros_like(resid, dtype=torch.int64),
                torch.ones_like(resid, dtype=torch.int64),
            )

            # Station index (required)
            sta_idx = params.get("_runtime_bucket_station_index", None)
            if (not isinstance(sta_idx, torch.Tensor)) or (int(sta_idx.numel()) != int(resid.numel())):
                raise ValueError("slowness_re.component_station requires per-row station index.")

            # Component ids per event (required)
            cid_ev = params.get("_runtime_event_cluster_ids", None)
            if not isinstance(cid_ev, torch.Tensor):
                cid_ev = params.get("_shared_event_re_cluster_ids", None)
            if not isinstance(cid_ev, torch.Tensor):
                raise ValueError("slowness_re.component_station requires event component ids.")

            # Separation magnitude g_ij (optionally frozen at MAP locations).
            freeze_g = bool(params.get("_slowness_re_freeze_g_at_map", False))
            if freeze_g:
                X_map = params.get("_slowness_re_g_x_map", None)
                if not isinstance(X_map, torch.Tensor) or int(X_map.shape[0]) != int(X_src.shape[0]):
                    X_map = (X_src + ΔX_src)[:, :3].detach().to(device=X_src.device, dtype=X_src.dtype)
                    params["_slowness_re_g_x_map"] = X_map
                x1 = X_map.index_select(0, idx[:, 0].to(torch.int64))
                x2 = X_map.index_select(0, idx[:, 1].to(torch.int64))
            else:
                x1 = X_src[idx[:, 0], :3] + ΔX_src[idx[:, 0], :3]
                x2 = X_src[idx[:, 1], :3] + ΔX_src[idx[:, 1], :3]
            g = torch.linalg.norm(x1 - x2, dim=1).clamp_min(1e-6)
            if sep_cap_km > 0.0 and math.isfinite(sep_cap_km):
                g = g.clamp_max(float(sep_cap_km))

            if not bool(params.get("_slowness_re_comp_station_logged", False)):
                params["_slowness_re_comp_station_logged"] = True
                try:
                    g_cpu = g.detach().float().cpu()
                    g_med = float(torch.median(g_cpu).item()) if g_cpu.numel() > 0 else float("nan")
                    g_max = float(g_cpu.max().item()) if g_cpu.numel() > 0 else float("nan")
                    g_p90 = float(torch.quantile(g_cpu, 0.9).item()) if g_cpu.numel() > 0 else float("nan")
                    print(
                        f"[slowness_re] component_station g_km median={g_med:.4g} p90={g_p90:.4g} max={g_max:.4g} "
                        f"tau_comp_p={tau_p:.3g} tau_comp_s={tau_s:.3g} "
                        f"tau_sta_p={tau_sta_p:.3g} tau_sta_s={tau_sta_s:.3g} units={tau_units}",
                        flush=True,
                    )
                except Exception:
                    pass

            # Per-row component id (assume within-component pairs; if not, use event-1 id).
            comp1 = cid_ev.index_select(0, idx[:, 0].to(torch.int64))
            comp2 = cid_ev.index_select(0, idx[:, 1].to(torch.int64))
            comp_row = comp1
            if bool((comp1 != comp2).any()) and (not bool(params.get("_slowness_re_warned_cross_component", False))):
                params["_slowness_re_warned_cross_component"] = True
                print("Warning: slowness_re.component_station saw cross-component pairs; using component of event-1.", flush=True)

            try:
                n_comp = int(comp_row.max().item()) + 1 if comp_row.numel() > 0 else 0
                n_sta = int(sta_idx.max().item()) + 1 if sta_idx.numel() > 0 else 0
            except Exception:
                n_comp = 0
                n_sta = 0
            if n_comp <= 0 or n_sta <= 0:
                u_edge = resid / sigma.square().clamp_min(1e-24)
                quad = _CollapsedQuad.apply(resid, u_edge)
                loss_like = (quad / float(max(int(resid.numel()), 1))) + torch.log(sigma).mean()
                return loss_like

            quad = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
            m_tot = float(max(int(resid.numel()), 1))
            log_sigma_mean = torch.log(sigma).mean()

            # Optional GPU path (batched PCG).
            gpu_enable_raw = params.get("_slowness_re_gpu_enable", None)
            gpu_enable = bool(resid.is_cuda) if gpu_enable_raw is None else bool(gpu_enable_raw)
            if gpu_enable:
                try:
                    from spider.core import shared_event_re_gpu
                    debug_cap = int(params.get("_slowness_re_gpu_debug_max_groups", 0) or 0)
                    quad_gpu, metrics = shared_event_re_gpu.compute_slowness_comp_station_quad_gpu(
                        idx=idx,
                        resid=resid,
                        g=g,
                        comp_id=comp_row,
                        sta_id=sta_idx,
                        ph_id=ph_id,
                        sigma=sigma,
                        tau_comp_p=float(tau_p),
                        tau_comp_s=float(tau_s),
                        tau_sta_p=float(tau_sta_p),
                        tau_sta_s=float(tau_sta_s),
                        jitter0=float(jitter0),
                        pcg_max_iters=int(pcg_max_iters),
                        pcg_tol=float(pcg_tol),
                        max_rows_per_group=int(max_rows_per_group),
                        max_nodes_per_group=int(max_nodes_per_group),
                        fallback_to_diag=bool(fallback_to_diag),
                        max_groups_per_batch=int(params.get("_slowness_re_gpu_max_groups_per_batch", 64)),
                        max_edges_per_batch=int(params.get("_slowness_re_gpu_max_edges_per_batch", 0) or 0),
                        group_cap=(debug_cap if debug_cap > 0 else None),
                        enable_profile=bool(params.get("_slowness_re_gpu_profile", False)),
                        n_comp=int(n_comp),
                        n_sta=int(n_sta),
                    )
                    quad = quad_gpu
                    try:
                        params["_slowness_re_runtime_last_grouping"] = "phase"
                        params["_slowness_re_runtime_last_groups"] = int(metrics.n_groups_total)
                        params["_slowness_re_runtime_last_groups_woodbury"] = int(metrics.n_groups_pcg)
                        params["_slowness_re_runtime_last_groups_fallback_diag"] = int(metrics.n_groups_fallback_diag)
                        params["_slowness_re_runtime_last_max_rows"] = int(metrics.max_rows_seen)
                        params["_slowness_re_runtime_last_max_nodes"] = int(metrics.max_nodes_seen)
                        params["_slowness_re_gpu_last_groups_rows_cap"] = int(metrics.n_groups_rows_cap)
                        params["_slowness_re_gpu_last_groups_nodes_cap"] = int(metrics.n_groups_nodes_cap)
                        params["_slowness_re_gpu_last_groups_tau_zero"] = int(metrics.n_groups_tau_zero)
                        params["_slowness_re_gpu_last_max_rows_all"] = int(metrics.max_rows_all)
                        params["_slowness_re_gpu_last_max_nodes_all"] = int(metrics.max_nodes_all)
                        params["_slowness_re_gpu_last_ms_total"] = float(metrics.ms_total)
                    except Exception:
                        pass
                    if not bool(params.get("_slowness_re_logged_runtime", False)):
                        params["_slowness_re_logged_runtime"] = True
                        try:
                            print(
                                f"[slowness_re] runtime grouping=phase groups={int(metrics.n_groups_total)} "
                                f"pcg={int(metrics.n_groups_pcg)} fallback={int(metrics.n_groups_fallback_diag)} "
                                f"max_rows={int(metrics.max_rows_seen)} max_nodes={int(metrics.max_nodes_seen)}",
                                flush=True,
                            )
                        except Exception:
                            pass
                    if not bool(params.get("_slowness_re_logged_pred_rms", False)):
                        params["_slowness_re_logged_pred_rms"] = True
                        try:
                            if np.isfinite(metrics.pred_rms) and np.isfinite(metrics.resid_rms):
                                print(
                                    f"[slowness_re] rms: resid={metrics.resid_rms:.4g}s "
                                    f"pred={metrics.pred_rms:.4g}s g={metrics.g_rms:.4g}km",
                                    flush=True,
                                )
                        except Exception:
                            pass
                    if not bool(params.get("_slowness_re_logged_delta", False)):
                        params["_slowness_re_logged_delta"] = True
                        try:
                            quad_diag = _CollapsedQuad.apply(resid, resid / sigma.square().clamp_min(1e-24))
                            delta = float(((quad - quad_diag) / m_tot).detach().item())
                            print(f"[slowness_re] loss delta vs diag = {delta:.6e}", flush=True)
                        except Exception as e:
                            print(f"[slowness_re] loss delta vs diag failed: {e}", flush=True)
                    loss_like = (quad / m_tot) + log_sigma_mean
                    return loss_like
                except Exception:
                    pass

            # CPU path (phase grouping; station slowness per phase).
            keys = ph_id  # group by phase only
            keys_sorted, perm = torch.sort(keys)
            resid_s = resid.index_select(0, perm)
            sigma_s = sigma.index_select(0, perm)
            comp_s = comp_row.index_select(0, perm)
            sta_s = sta_idx.index_select(0, perm)
            g_s = g.index_select(0, perm)
            ph_s = ph_id.index_select(0, perm)

            n_groups_total = 0
            n_groups_pcg = 0
            n_groups_fallback_diag = 0
            max_rows_seen = 0
            max_nodes_seen = 0

            if keys_sorted.numel() > 0:
                is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
                is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
                starts = torch.nonzero(is_new, as_tuple=False).flatten()
                ends = torch.cat([starts[1:], torch.tensor([keys_sorted.numel()], device=keys_sorted.device)])
                for s0, e0 in zip(starts.tolist(), ends.tolist()):
                    n_groups_total += 1
                    r_g = resid_s[s0:e0]
                    s_g = sigma_s[s0:e0].clamp_min(1e-12)
                    g_g = g_s[s0:e0]
                    c_g = comp_s[s0:e0].to(torch.int64)
                    k_g = sta_s[s0:e0].to(torch.int64)
                    ph_g = ph_s[s0:e0]
                    m = int(r_g.numel())
                    if m == 0:
                        continue
                    max_rows_seen = max(max_rows_seen, m)
                    n_nodes = int(n_comp + n_sta)
                    max_nodes_seen = max(max_nodes_seen, n_nodes)
                    tau_c = float(tau_p if int(ph_g[0].item()) == 0 else tau_s)
                    tau_a = float(tau_sta_p if int(ph_g[0].item()) == 0 else tau_sta_s)

                    if (m > max_rows_per_group) or (n_nodes > max_nodes_per_group) or (not ((tau_c > 0.0) or (tau_a > 0.0))):
                        if fallback_to_diag:
                            u_edge = r_g / s_g.square().clamp_min(1e-24)
                            quad = quad + _CollapsedQuad.apply(r_g, u_edge)
                            n_groups_fallback_diag += 1
                            continue
                        raise ValueError(
                            f"slowness_re.component_station group too large (rows={m}, nodes={n_nodes}); "
                            "increase max_rows_per_group/max_nodes_per_group or enable fallback_to_diag."
                        )

                    u = c_g
                    v = (int(n_comp) + k_g).to(torch.int64)
                    w = (g_g * g_g) / s_g.square().clamp_min(1e-24)
                    b_edge = (g_g * r_g) / s_g.square().clamp_min(1e-24)
                    b = torch.zeros((n_nodes,), device=resid.device, dtype=resid.dtype)
                    b.index_add_(0, u, b_edge)
                    b.index_add_(0, v, b_edge)

                    tau_c_eff = max(float(tau_c), 1e-12)
                    tau_a_eff = max(float(tau_a), 1e-12)
                    alpha = torch.full((n_nodes,), 1.0 / (tau_a_eff * tau_a_eff), device=resid.device, dtype=resid.dtype)
                    if n_comp > 0:
                        alpha[: int(n_comp)] = 1.0 / (tau_c_eff * tau_c_eff)
                    alpha = alpha + float(jitter0)

                    x = _pcg_solve_sum_weighted(
                        u=u, v=v, w=w, b=b, alpha=alpha,
                        max_iters=int(pcg_max_iters), tol=float(pcg_tol),
                    )
                    x_u = x.index_select(0, u)
                    x_v = x.index_select(0, v)
                    Hx = g_g * (x_u + x_v)
                    u_edge = (r_g - Hx) / s_g.square().clamp_min(1e-24)
                    quad = quad + _CollapsedQuad.apply(r_g, u_edge)
                    if not bool(params.get("_slowness_re_logged_pred_rms", False)):
                        params["_slowness_re_logged_pred_rms"] = True
                        try:
                            pred_rms = float(torch.sqrt(torch.mean(Hx * Hx)).detach().item())
                            r_rms = float(torch.sqrt(torch.mean(r_g * r_g)).detach().item())
                            g_rms = float(torch.sqrt(torch.mean(g_g * g_g)).detach().item())
                            print(
                                f"[slowness_re] rms: resid={r_rms:.4g}s pred={pred_rms:.4g}s g={g_rms:.4g}km",
                                flush=True,
                            )
                        except Exception:
                            pass
                    n_groups_pcg += 1

            try:
                params["_slowness_re_runtime_last_grouping"] = "phase"
                params["_slowness_re_runtime_last_groups"] = int(n_groups_total)
                params["_slowness_re_runtime_last_groups_woodbury"] = int(n_groups_pcg)
                params["_slowness_re_runtime_last_groups_fallback_diag"] = int(n_groups_fallback_diag)
                params["_slowness_re_runtime_last_max_rows"] = int(max_rows_seen)
                params["_slowness_re_runtime_last_max_nodes"] = int(max_nodes_seen)
            except Exception:
                pass

            if not bool(params.get("_slowness_re_logged_runtime", False)):
                params["_slowness_re_logged_runtime"] = True
                print(
                    f"[slowness_re] runtime grouping=phase groups={int(n_groups_total)} "
                    f"pcg={int(n_groups_pcg)} fallback={int(n_groups_fallback_diag)} "
                    f"max_rows={int(max_rows_seen)} max_nodes={int(max_nodes_seen)}",
                    flush=True,
                )

            if not bool(params.get("_slowness_re_logged_delta", False)):
                params["_slowness_re_logged_delta"] = True
                try:
                    quad_diag = _CollapsedQuad.apply(resid, resid / sigma.square().clamp_min(1e-24))
                    delta = float(((quad - quad_diag) / m_tot).detach().item())
                    print(f"[slowness_re] loss delta vs diag = {delta:.6e}", flush=True)
                except Exception as e:
                    print(f"[slowness_re] loss delta vs diag failed: {e}", flush=True)
            loss_like = (quad / m_tot) + log_sigma_mean
            return loss_like

        # All supported slowness_re modes are handled above.

        # Optional profiling (off by default). Enable via inference.diagnostics.profile_slowness_re=true.
        prof_sl = bool(params.get("_profile_slowness_re", False))
        prof_sl_use_cuda_events = False
        prof_sl_max_groups = 0
        if prof_sl:
            try:
                prof_sl_max_groups = int(params.get("_profile_slowness_re_max_groups", 2))
            except Exception:
                prof_sl_max_groups = 2
            if prof_sl_max_groups < 0:
                prof_sl_max_groups = 0
            try:
                prof_sl_use_cuda_events = bool(params.get("_profile_slowness_re_use_cuda_events", True))
            except Exception:
                prof_sl_use_cuda_events = True
            prof_sl_use_cuda_events = bool(prof_sl_use_cuda_events and resid.is_cuda)
            # Per-call state
            params["_sl_re__groups_profiled_in_call"] = 0
            if prof_sl_use_cuda_events:
                try:
                    e_total0 = torch.cuda.Event(enable_timing=True)
                    e_total1 = torch.cuda.Event(enable_timing=True)
                    e_total0.record()
                except Exception:
                    e_total0 = None
                    e_total1 = None
            else:
                t_total0 = time.perf_counter()

        grouping = str(params.get("_slowness_re_grouping", "station_phase")).strip().lower()
        if grouping in {"stationphase", "station-phase"}:
            grouping = "station_phase"
        solver = str(params.get("_slowness_re_solver", "cholesky")).strip().lower()
        if solver in {"chol"}:
            solver = "cholesky"
        tau_ps = params.get("_slowness_re_tau_s", [0.0, 0.0])
        tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
        tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
        tau_units = str(params.get("_slowness_re_tau_units", "abs")).strip().lower()
        ell_km = float(params.get("_slowness_re_ell_km", 0.0))
        max_rows_per_group = int(params.get("_slowness_re_max_rows_per_group", 200000))
        max_nodes_per_group = int(params.get("_slowness_re_max_nodes_per_group", 2048))
        fallback_to_diag = bool(params.get("_slowness_re_fallback_to_diag", True))
        pcg_max_iters = int(params.get("_slowness_re_pcg_max_iters", 30))
        pcg_tol = float(params.get("_slowness_re_pcg_tol", 1e-4))
        pcg_check_every = int(params.get("_slowness_re_pcg_check_every", 0))
        pcg_min_inducing = int(params.get("_slowness_re_pcg_min_inducing", 128))
        pcg_min_rows = int(params.get("_slowness_re_pcg_min_rows", 2000))

        # Phase id per row: 0=P, 1=S. Needed by both grouping and station-basis formulations.
        ph_id = torch.where(
            is_p,
            torch.zeros_like(resid, dtype=torch.int64),
            torch.ones_like(resid, dtype=torch.int64),
        )

        # Optional: station basis for slowness_re (receiver-dependent, low-rank).
        # NOTE: This requires a per-row station index for the current batch.
        W_sta = params.get("_slowness_re_station_basis_W", None)
        try:
            use_sta_basis = bool(params.get("_slowness_re_station_basis_enabled", False)) and isinstance(W_sta, torch.Tensor)
        except Exception:
            use_sta_basis = False
        sta_idx_basis = params.get("_runtime_bucket_station_index", None)
        if use_sta_basis:
            if (not isinstance(sta_idx_basis, torch.Tensor)) or (int(sta_idx_basis.numel()) != int(resid.numel())):
                raise ValueError("slowness_re.station_basis enabled but no per-row station index is available for this batch.")

        # We currently rely on a homoscedastic base sigma and incorporate only our own diagonal correction (FITC).
        # sigma_extra_var may already have been applied above (e.g., by shared_event_latent); we accept it here.

        # Station index (optional) for station_phase grouping
        sta_idx = None
        if grouping == "station_phase":
            sta_idx = params.get("_runtime_bucket_station_index", None)
            if not isinstance(sta_idx, torch.Tensor) or int(sta_idx.numel()) != int(resid.numel()):
                if not bool(params.get("_slowness_re_warned_no_station_index", False)):
                    print(
                        "Warning: slowness_re.grouping='station_phase' requested but no per-row station index "
                        "was available for this batch. Falling back to grouping='phase'."
                    )
                    params["_slowness_re_warned_no_station_index"] = True
                grouping = "phase"
                sta_idx = None

        # NOTE: We intentionally do NOT require cluster_ids here for performance. If inducing points are
        # selected per connected component and K_UU is block diagonal, solving a single system per
        # station-phase group still results in zero cross-component coupling.
        cid_ev = params.get("_runtime_event_cluster_ids", None)

        if sl_mode == "component_station_explicit":
            # Defensive: explicit mode should have returned above; fall back to standard loss.
            loss_type = str(params.get("likelihood", "huber")).strip().lower()
            if loss_type in {"gaussian", "mse", "l2"}:
                data_loss = 0.5 * (scaled_resid ** 2)
            elif loss_type in {"student_t", "student-t", "studentt"}:
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
                data_loss = torch.abs(scaled_resid)
            else:
                huber_delta = float(params["model"]["likelihood"].get("huber_delta", 1.0))
                data_loss = F.huber_loss(
                    scaled_resid,
                    torch.zeros_like(scaled_resid),
                    reduction="none",
                    delta=huber_delta,
                )
            total_nll = data_loss + torch.log(sigma)
            return total_nll.mean()

        # Inducing artifacts (built once at MAP by locate.py)
        nei_idx_ev = params.get("_slowness_re_inducing_neighbor_idx", None)
        if not isinstance(nei_idx_ev, torch.Tensor):
            try:
                a = np.asarray(nei_idx_ev, dtype=np.int64)
                if a.ndim == 2:
                    nei_idx_ev = torch.from_numpy(a)
            except Exception:
                pass
        inducing_xyz = params.get("_slowness_re_inducing_xyz_km_t", None)
        if not isinstance(inducing_xyz, torch.Tensor):
            inducing_xyz0 = params.get("_slowness_re_inducing_xyz_km", None)
            if isinstance(inducing_xyz0, torch.Tensor):
                inducing_xyz = inducing_xyz0
            else:
                try:
                    a = np.asarray(inducing_xyz0, dtype=np.float32)
                    if a.ndim == 2 and a.shape[1] >= 3:
                        inducing_xyz = torch.from_numpy(a[:, :3])
                except Exception:
                    inducing_xyz = None
        offs = params.get("_slowness_re_inducing_offsets", None)
        K_full = params.get("_slowness_re_inducing_K_full", None)
        K_full3 = params.get("_slowness_re_inducing_K_full_3", None)
        K_blocks = params.get("_slowness_re_inducing_K_blocks", None)
        comp_to_block = params.get("_slowness_re_inducing_comp_to_block", None)
        fitc_resid_ev = params.get("_slowness_re_inducing_fitc_resid", None)

        have_full = isinstance(K_full, torch.Tensor) and K_full.ndim == 2
        have_inducing = (
            isinstance(nei_idx_ev, torch.Tensor)
            and isinstance(inducing_xyz, torch.Tensor)
            and isinstance(offs, torch.Tensor)
            and (have_full or (isinstance(comp_to_block, torch.Tensor) and isinstance(K_blocks, list) and len(K_blocks) > 0))
        )
        if (sl_mode != "component_station_explicit") and (not have_inducing):
            raise ValueError(
                "slowness_re.enabled=true but inducing artifacts are missing. "
                "Run MAP initialization (Phase 1) with slowness_re enabled so locate.py can build them."
            )
        # Normalize devices/dtypes
        dev = resid.device
        nei_idx_ev = nei_idx_ev.to(device=dev, dtype=torch.int64)
        inducing_xyz = inducing_xyz.to(device=dev, dtype=torch.float32)
        offs = offs.to(device=dev, dtype=torch.int64)
        if have_full:
            K_full = K_full.to(device=dev, dtype=torch.float32)
            if isinstance(K_full3, torch.Tensor) and K_full3.ndim == 2:
                K_full3 = K_full3.to(device=dev, dtype=torch.float32)
            else:
                K_full3 = None
        else:
            comp_to_block = comp_to_block.to(device=dev, dtype=torch.int64)
        if isinstance(fitc_resid_ev, torch.Tensor):
            fitc_resid_ev = fitc_resid_ev.to(device=dev, dtype=torch.float32)

        # Current event XYZ in km (detach so covariance does not participate in autograd).
        X_cur = (X_src + ΔX_src)[:, :3].detach().to(device=dev, dtype=torch.float32)

        # Optional v(z) for vel_frac units (cache on-device in params via epoch_runner).
        zc = params.get("_runtime_eikonet_v1d_z_cent_km_t", None)
        vp = params.get("_runtime_eikonet_v1d_vp_km_s_t", None)
        vs = params.get("_runtime_eikonet_v1d_vs_km_s_t", None)
        if tau_units == "vel_frac" and (not (isinstance(zc, torch.Tensor) and isinstance(vp, torch.Tensor) and isinstance(vs, torch.Tensor))):
            # Fallback: build tensors from serialized lists (if available).
            try:
                z_arr = np.asarray(params.get("_eikonet_v1d_depth_centers_km", []), dtype=np.float32).reshape(-1)
                vp_arr = np.asarray(params.get("_eikonet_v1d_vp_km_s", []), dtype=np.float32).reshape(-1)
                vs_arr = np.asarray(params.get("_eikonet_v1d_vs_km_s", []), dtype=np.float32).reshape(-1)
                if z_arr.size > 0 and vp_arr.size == z_arr.size and vs_arr.size == z_arr.size:
                    zc = torch.from_numpy(z_arr).to(device=dev, dtype=torch.float32)
                    vp = torch.from_numpy(vp_arr).to(device=dev, dtype=torch.float32)
                    vs = torch.from_numpy(vs_arr).to(device=dev, dtype=torch.float32)
                    params["_runtime_eikonet_v1d_z_cent_km_t"] = zc
                    params["_runtime_eikonet_v1d_vp_km_s_t"] = vp
                    params["_runtime_eikonet_v1d_vs_km_s_t"] = vs
            except Exception:
                pass
        if tau_units == "vel_frac" and (not (isinstance(zc, torch.Tensor) and isinstance(vp, torch.Tensor) and isinstance(vs, torch.Tensor))):
            if not bool(params.get("_slowness_re_warned_no_v1d", False)):
                print(
                    "Warning: slowness_re.tau_s uses vel_frac but no EikoNet v(z) curve was available. "
                    "Falling back to constant vP=6.0 km/s, vS=3.5 km/s for scaling."
                )
                params["_slowness_re_warned_no_v1d"] = True
            zc = None
            vp = None
            vs = None

        def _matern32(d: torch.Tensor, ell: float) -> torch.Tensor:
            a = float(np.sqrt(3.0) / float(max(ell, 1e-12)))
            x = (a * d).to(torch.float32)
            return (1.0 + x) * torch.exp(-x)

        def _interp1d_linear(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # x must be sorted ascending; returns linear interpolation with endpoint clamping.
            if int(x.numel()) <= 1:
                return y.reshape(1).expand_as(z)
            idx = torch.bucketize(z, x)
            idx1 = idx.clamp(0, int(x.numel()) - 1)
            idx0 = (idx1 - 1).clamp(0, int(x.numel()) - 1)
            x0 = x.index_select(0, idx0)
            x1 = x.index_select(0, idx1)
            y0 = y.index_select(0, idx0)
            y1 = y.index_select(0, idx1)
            denom = (x1 - x0).clamp_min(1e-6)
            t = ((z - x0) / denom).clamp(0.0, 1.0)
            return y0 + t * (y1 - y0)

        def _pcg_station_basis_solve(
            *,
            resid_g: torch.Tensor,          # (B,)
            sta_g: torch.Tensor,            # (B,) int64
            w_g: torch.Tensor,              # (B,3) float32
            nei_loc: torch.Tensor,          # (B,K) int64 local inducing indices [0..M-1] with invalid->0
            kbar: torch.Tensor,             # (B,K) float32 with invalid->0
            alpha_d: torch.Tensor,          # (B,) float32 = D^{-1}
            Kprior: torch.Tensor,           # (M,M) float32
            W: torch.Tensor,                # (S,R) float32
            max_iters: int,
            tol: float,
            check_every: int,
        ) -> tuple[torch.Tensor, int]:
            """
            Basis-space PCG for station-dependent slowness_re:
              (Kprior ⊗ I_{3R} + B^T D^{-1} B) y = B^T D^{-1} r
            with station weights A = W[sta,:]. Returns u = D^{-1}(r - B y).
            """
            Bn = int(resid_g.numel())
            K = int(nei_loc.shape[1])
            M = int(Kprior.shape[0])
            R = int(W.shape[1])
            if Bn <= 0 or K <= 0 or M <= 0 or R <= 0:
                return resid_g / torch.ones_like(resid_g).clamp_min(1e-24), 0

            A = W.index_select(0, sta_g.to(torch.int64)).to(torch.float32)  # (B,R)
            A_T = A.transpose(0, 1).contiguous()                            # (R,B)

            idx_flat = nei_loc.reshape(-1).to(torch.int64)                  # (B*K,)
            idx_exp = idx_flat.unsqueeze(0).expand(R, -1)                   # (R,B*K)
            wx = w_g[:, 0].contiguous()
            wy = w_g[:, 1].contiguous()
            wz = w_g[:, 2].contiguous()

            # rhs = B^T D^{-1} r
            yr = (alpha_d * resid_g.detach().to(torch.float32)).contiguous()  # (B,)
            rhs_x = torch.zeros((R, M), device=dev, dtype=torch.float32)
            rhs_y = torch.zeros((R, M), device=dev, dtype=torch.float32)
            rhs_z = torch.zeros((R, M), device=dev, dtype=torch.float32)
            base_x = (kbar * (wx * yr).unsqueeze(1)).reshape(-1)  # (B*K,)
            base_y = (kbar * (wy * yr).unsqueeze(1)).reshape(-1)
            base_z = (kbar * (wz * yr).unsqueeze(1)).reshape(-1)
            wv_x = (A_T.unsqueeze(2) * base_x.view(1, Bn, K)).reshape(R, -1)
            wv_y = (A_T.unsqueeze(2) * base_y.view(1, Bn, K)).reshape(R, -1)
            wv_z = (A_T.unsqueeze(2) * base_z.view(1, Bn, K)).reshape(R, -1)
            rhs_x.scatter_add_(1, idx_exp, wv_x)
            rhs_y.scatter_add_(1, idx_exp, wv_y)
            rhs_z.scatter_add_(1, idx_exp, wv_z)
            b_vec = torch.cat([rhs_x.reshape(-1), rhs_y.reshape(-1), rhs_z.reshape(-1)], dim=0)  # (3*R*M,)

            # Jacobi preconditioner diag
            k2 = (kbar * kbar).to(torch.float32)
            a2_T = (A_T * A_T).contiguous()  # (R,B)
            diag_x = torch.zeros((R, M), device=dev, dtype=torch.float32)
            diag_y = torch.zeros((R, M), device=dev, dtype=torch.float32)
            diag_z = torch.zeros((R, M), device=dev, dtype=torch.float32)
            base_dx = (k2 * (alpha_d * (wx * wx)).unsqueeze(1)).reshape(-1)
            base_dy = (k2 * (alpha_d * (wy * wy)).unsqueeze(1)).reshape(-1)
            base_dz = (k2 * (alpha_d * (wz * wz)).unsqueeze(1)).reshape(-1)
            dv_x = (a2_T.unsqueeze(2) * base_dx.view(1, Bn, K)).reshape(R, -1)
            dv_y = (a2_T.unsqueeze(2) * base_dy.view(1, Bn, K)).reshape(R, -1)
            dv_z = (a2_T.unsqueeze(2) * base_dz.view(1, Bn, K)).reshape(R, -1)
            diag_x.scatter_add_(1, idx_exp, dv_x)
            diag_y.scatter_add_(1, idx_exp, dv_y)
            diag_z.scatter_add_(1, idx_exp, dv_z)
            kd = torch.diagonal(Kprior).contiguous().view(1, M).expand(R, M)
            diag_x.add_(kd)
            diag_y.add_(kd)
            diag_z.add_(kd)
            diag_inv = torch.cat(
                [
                    (1.0 / diag_x.clamp_min(1e-12)).reshape(-1),
                    (1.0 / diag_y.clamp_min(1e-12)).reshape(-1),
                    (1.0 / diag_z.clamp_min(1e-12)).reshape(-1),
                ],
                dim=0,
            )  # (3*R*M,)

            def _A_mul(p_vec: torch.Tensor) -> torch.Tensor:
                p = p_vec.view(3, R, M)
                px = p[0]
                py = p[1]
                pz = p[2]
                # Prior term
                ax = (Kprior @ px.transpose(0, 1)).transpose(0, 1).contiguous()
                ay = (Kprior @ py.transpose(0, 1)).transpose(0, 1).contiguous()
                az = (Kprior @ pz.transpose(0, 1)).transpose(0, 1).contiguous()
                # Data term: Bt D^{-1} B p
                px_g = px.index_select(1, idx_flat).view(R, Bn, K)
                py_g = py.index_select(1, idx_flat).view(R, Bn, K)
                pz_g = pz.index_select(1, idx_flat).view(R, Bn, K)
                dot = (wx.view(1, Bn, 1) * px_g) + (wy.view(1, Bn, 1) * py_g) + (wz.view(1, Bn, 1) * pz_g)
                tmp = (kbar.view(1, Bn, K) * dot).sum(dim=2)  # (R,B)
                t = (tmp * A_T).sum(dim=0)                   # (B,)
                yb = (alpha_d * t).contiguous()              # (B,)
                base_x2 = (kbar * (wx * yb).unsqueeze(1)).reshape(-1)
                base_y2 = (kbar * (wy * yb).unsqueeze(1)).reshape(-1)
                base_z2 = (kbar * (wz * yb).unsqueeze(1)).reshape(-1)
                sv_x = (A_T.unsqueeze(2) * base_x2.view(1, Bn, K)).reshape(R, -1)
                sv_y = (A_T.unsqueeze(2) * base_y2.view(1, Bn, K)).reshape(R, -1)
                sv_z = (A_T.unsqueeze(2) * base_z2.view(1, Bn, K)).reshape(R, -1)
                ax.scatter_add_(1, idx_exp, sv_x)
                ay.scatter_add_(1, idx_exp, sv_y)
                az.scatter_add_(1, idx_exp, sv_z)
                return torch.cat([ax.reshape(-1), ay.reshape(-1), az.reshape(-1)], dim=0)

            # PCG (optionally check every N iterations; checks incur device->host sync)
            x = torch.zeros_like(b_vec)
            r0 = b_vec.clone()
            z0 = diag_inv * r0
            p = z0.clone()
            rz = (r0 * z0).sum()
            bnorm = torch.sqrt((b_vec * b_vec).sum()).clamp_min(1e-12)
            it_done = 0
            for it in range(int(max_iters)):
                Ap = _A_mul(p)
                denom = (p * Ap).sum().clamp_min(1e-24)
                a = rz / denom
                x = x + a * p
                r0 = r0 - a * Ap
                it_done = it + 1
                if int(check_every) > 0 and ((it_done % int(check_every)) == 0):
                    rel = torch.sqrt((r0 * r0).sum()) / bnorm
                    if float(rel) < float(tol):
                        break
                z1 = diag_inv * r0
                rz_new = (r0 * z1).sum()
                beta = rz_new / rz.clamp_min(1e-24)
                p = z1 + beta * p
                rz = rz_new

            y = x.view(3, R, M)
            yx = y[0]; yy2 = y[1]; yz2 = y[2]
            yx_g = yx.index_select(1, idx_flat).view(R, Bn, K)
            yy_g = yy2.index_select(1, idx_flat).view(R, Bn, K)
            yz_g = yz2.index_select(1, idx_flat).view(R, Bn, K)
            dot_y = (wx.view(1, Bn, 1) * yx_g) + (wy.view(1, Bn, 1) * yy_g) + (wz.view(1, Bn, 1) * yz_g)
            tmp_y = (kbar.view(1, Bn, K) * dot_y).sum(dim=2)  # (R,B)
            by = (tmp_y * A_T).sum(dim=0)                     # (B,)
            u = alpha_d * (resid_g.detach().to(torch.float32) - by)
            return u.to(dtype=resid_g.dtype), int(it_done)

        # If station basis is enabled and W is available, use the basis formulation.
        if use_sta_basis:
            # Move/cached W to device.
            W_t = params.get("_slowness_re_station_basis_W_t", None)
            if not isinstance(W_t, torch.Tensor):
                if not isinstance(W_sta, torch.Tensor):
                    raise ValueError("slowness_re.station_basis enabled but W was not a torch.Tensor")
                W_t = W_sta.to(device=dev, dtype=torch.float32).contiguous()
                params["_slowness_re_station_basis_W_t"] = W_t
            else:
                W_t = W_t.to(device=dev, dtype=torch.float32)

            if not bool(params.get("_slowness_re_warned_station_basis_grouping", False)):
                if grouping == "station_phase":
                    print("Info: slowness_re.station_basis enabled; solving per (phase,component) in station-basis space (not per-station groups).")
                params["_slowness_re_warned_station_basis_grouping"] = True

            # Determine whether to split by connected component blocks (mirrors the station_phase grouping logic).
            n_comp = params.get("_runtime_n_components", 1)
            try:
                n_comp_i = int(n_comp) if n_comp is not None else 1
            except Exception:
                n_comp_i = 1
            prefer_componentwise = (
                n_comp_i > 1
                and isinstance(cid_ev, torch.Tensor)
                and isinstance(comp_to_block, torch.Tensor)
                and isinstance(K_blocks, list) and len(K_blocks) > 0
                and isinstance(offs, torch.Tensor)
            )

            # Component id per row (for blockwise Kuu)
            if isinstance(cid_ev, torch.Tensor):
                comp_row_all = cid_ev.index_select(0, idx[:, 0].to(dtype=torch.int64))
            else:
                comp_row_all = torch.zeros((int(idx.shape[0]),), device=dev, dtype=torch.int64)

            quad = torch.tensor(0.0, device=dev, dtype=resid.dtype)
            n_groups_total = 0
            n_groups_woodbury = 0
            n_groups_fallback_diag = 0
            max_rows_seen = 0
            max_nodes_seen = 0
            max_M_seen = 0

            # Indices per phase
            for ph_g in (0, 1):
                tau = float(tau_p if ph_g == 0 else tau_s)
                sigma_g = σ_p if ph_g == 0 else σ_s
                idxs_ph = torch.nonzero(ph_id == int(ph_g), as_tuple=False).reshape(-1)
                if int(idxs_ph.numel()) == 0:
                    continue
                # For basis solve we keep all stations but still split by component blocks when available.
                comps = torch.unique(comp_row_all.index_select(0, idxs_ph))
                for comp_id in comps.tolist():
                    idxs = idxs_ph[comp_row_all.index_select(0, idxs_ph) == int(comp_id)]
                    m_g = int(idxs.numel())
                    if m_g <= 0:
                        continue
                    n_groups_total += 1
                    max_rows_seen = max(max_rows_seen, m_g)

                    idx_g = idx.index_select(0, idxs)
                    try:
                        n_nodes = int(torch.unique(idx_g.reshape(-1)).numel())
                    except Exception:
                        n_nodes = m_g * 2
                    max_nodes_seen = max(max_nodes_seen, int(n_nodes))
                    if m_g > int(max_rows_per_group) or n_nodes > int(max_nodes_per_group):
                        if not fallback_to_diag:
                            raise ValueError("slowness_re.station_basis: group exceeded max_rows/max_nodes and fallback_to_diag is false.")
                        n_groups_fallback_diag += 1
                        resid_g = resid.index_select(0, idxs)
                        u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                        quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                        continue

                    resid_g = resid.index_select(0, idxs)
                    if not (tau > 0.0):
                        u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                        quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                        continue

                    # Select inducing block for this component (or global).
                    bi = -1
                    if prefer_componentwise and isinstance(comp_to_block, torch.Tensor) and isinstance(K_blocks, list) and K_blocks:
                        try:
                            bi = int(comp_to_block[int(comp_id)].item())
                        except Exception:
                            bi = -1
                    if have_full and isinstance(K_full, torch.Tensor):
                        Kuu = K_full
                        off0 = 0
                        off1 = int(Kuu.shape[0])
                    else:
                        if bi < 0 or bi >= int(len(K_blocks)):
                            if not fallback_to_diag:
                                raise ValueError("slowness_re.station_basis: missing inducing block for component.")
                            n_groups_fallback_diag += 1
                            u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                            quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                            continue
                        Kuu = K_blocks[bi].to(device=dev, dtype=torch.float32)
                        off0 = int(offs[bi].item())
                        off1 = int(offs[bi + 1].item())
                    M = int(Kuu.shape[0])
                    max_M_seen = max(max_M_seen, int(M))

                    # Endpoints and geometry
                    e1 = idx_g[:, 0].to(dtype=torch.int64)
                    e2 = idx_g[:, 1].to(dtype=torch.int64)
                    x1 = X_cur.index_select(0, e1)
                    x2 = X_cur.index_select(0, e2)
                    dx = x2 - x1
                    if tau_units == "vel_frac":
                        zbar = 0.5 * (x1[:, 2] + x2[:, 2])
                        if isinstance(zc, torch.Tensor) and isinstance(vp, torch.Tensor) and isinstance(vs, torch.Tensor):
                            v = _interp1d_linear(zbar.to(torch.float32), zc.to(torch.float32), (vp if ph_g == 0 else vs).to(torch.float32))
                        else:
                            v = torch.full((int(zbar.numel()),), 6.0 if ph_g == 0 else 3.5, device=dev, dtype=torch.float32)
                        w = (dx * (1.0 / v.clamp_min(1e-3)).unsqueeze(1)).to(torch.float32)
                    else:
                        w = dx.to(torch.float32)

                    # Neighbor rows
                    nei1 = nei_idx_ev.index_select(0, e1)
                    nei2 = nei_idx_ev.index_select(0, e2)
                    nei_g = torch.cat([nei1, nei2], dim=1)
                    mask = (nei_g >= 0)
                    nei_clamped = torch.where(mask, nei_g, torch.zeros_like(nei_g))
                    in_block = (nei_clamped >= int(off0)) & (nei_clamped < int(off1))
                    mask = mask & in_block
                    nei_loc = (nei_clamped - int(off0)).to(torch.int64)
                    nei_loc = torch.where(mask, nei_loc, torch.zeros_like(nei_loc))
                    Bg = int(nei_loc.shape[0])
                    Kg = int(nei_loc.shape[1])
                    if Kg <= 0:
                        n_groups_fallback_diag += 1
                        u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                        quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                        continue

                    # Kernel weights (event->inducing)
                    k_map = params.get("_slowness_re_inducing_neighbor_k_matern32", None)
                    if isinstance(k_map, torch.Tensor) and k_map.ndim == 2 and int(k_map.shape[0]) == int(X_src.shape[0]):
                        k_map = k_map.to(device=dev, dtype=torch.float32)
                        k1 = k_map.index_select(0, e1)
                        k2 = k_map.index_select(0, e2)
                        k_eu = torch.cat([k1, k2], dim=1)
                    else:
                        U = inducing_xyz.index_select(0, nei_clamped.reshape(-1)).reshape(Bg, Kg, 3)
                        xe = torch.cat(
                            [x1.unsqueeze(1).expand(Bg, Kg // 2, 3), x2.unsqueeze(1).expand(Bg, Kg // 2, 3)],
                            dim=1,
                        )
                        d = torch.linalg.norm(U - xe, dim=2)
                        k_eu = _matern32(d, ell_km)
                    k_eu = torch.where(mask, k_eu, torch.zeros_like(k_eu))
                    kbar = 0.5 * k_eu

                    # Diagonal D and alpha
                    s2 = sigma_g.square().clamp_min(1e-24).to(torch.float32)
                    if isinstance(fitc_resid_ev, torch.Tensor):
                        lam1 = fitc_resid_ev.index_select(0, e1).to(torch.float32)
                        lam2 = fitc_resid_ev.index_select(0, e2).to(torch.float32)
                        lam_bar = 0.5 * (lam1 + lam2)
                        w2 = (w * w).sum(dim=1).to(torch.float32)
                        diag_extra = (float(tau) * float(tau)) * (lam_bar.clamp_min(0.0)) * w2
                        D = (s2 + diag_extra).clamp_min(1e-24)
                    else:
                        D = s2.expand(Bg).clamp_min(1e-24)
                    alpha_d = (1.0 / D).to(torch.float32)

                    # Prior precision Kprior = (1/tau^2)Kuu
                    inv_tau2 = float(1.0 / max(float(tau) * float(tau), 1e-24))
                    Kprior = (inv_tau2 * Kuu).to(device=dev, dtype=torch.float32)

                    sta_g = sta_idx_basis.index_select(0, idxs).to(torch.int64)  # (B,)
                    with torch.no_grad():
                        u_float, it_done = _pcg_station_basis_solve(
                            resid_g=resid_g,
                            sta_g=sta_g,
                            w_g=w,
                            nei_loc=nei_loc,
                            kbar=kbar.to(torch.float32),
                            alpha_d=alpha_d,
                            Kprior=Kprior,
                            W=W_t,
                            max_iters=int(pcg_max_iters),
                            tol=float(pcg_tol),
                            check_every=int(pcg_check_every),
                        )
                    n_groups_woodbury += 1
                    quad = quad + _CollapsedQuad.apply(resid_g, u_float)
                    if prof_sl:
                        try:
                            params["_sl_re_pcg_iters_sum"] = int(params.get("_sl_re_pcg_iters_sum", 0) or 0) + int(it_done)
                        except Exception:
                            pass

            # Stash stats for caller + profiler
            try:
                params["_slowness_re_runtime_last_grouping"] = "station_basis"
                params["_slowness_re_runtime_last_groups"] = int(n_groups_total)
                params["_slowness_re_runtime_last_groups_woodbury"] = int(n_groups_woodbury)
                params["_slowness_re_runtime_last_groups_fallback_diag"] = int(n_groups_fallback_diag)
                params["_slowness_re_runtime_last_max_rows"] = int(max_rows_seen)
                params["_slowness_re_runtime_last_max_nodes"] = int(max_nodes_seen)
            except Exception:
                pass

            m_tot = float(max(int(resid.numel()), 1))
            loss_like = (quad / m_tot) + torch.log(sigma).mean()
            # Profiling finalize (total time)
            if prof_sl:
                try:
                    params["_sl_re_time_ms_count"] = int(params.get("_sl_re_time_ms_count", 0) or 0) + 1
                except Exception:
                    pass
                if prof_sl_use_cuda_events and "e_total0" in locals() and e_total0 is not None and e_total1 is not None:
                    try:
                        e_total1.record()
                        e_total1.synchronize()
                        dt_ms = float(e_total0.elapsed_time(e_total1))
                        params["_sl_re_time_ms_sum"] = float(params.get("_sl_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                    except Exception:
                        pass
                else:
                    try:
                        dt_ms = 1000.0 * float(time.perf_counter() - t_total0)
                        params["_sl_re_time_ms_sum"] = float(params.get("_sl_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                    except Exception:
                        pass
            return float(alpha) * loss_like

        # Build group keys.
        if prof_sl:
            if prof_sl_use_cuda_events:
                try:
                    e_grp0 = torch.cuda.Event(enable_timing=True)
                    e_grp1 = torch.cuda.Event(enable_timing=True)
                    e_grp0.record()
                except Exception:
                    e_grp0 = None
                    e_grp1 = None
            else:
                t_grp0 = time.perf_counter()
        #
        # IMPORTANT performance note:
        # If there are many connected components, using K_full (sum of all inducing points) can make the
        # per-group linear solve enormous (3*M_total). Even though K_full is block-diagonal and each row
        # only touches its own component's inducing points, the dense solve cost scales with M_total.
        #
        # For many components (common in practice), we therefore prefer *component-wise* grouping even
        # when K_full is available: group by (component, station, phase) and solve against the component's
        # K_UU block only. This is mathematically equivalent when K_full is block-diagonal and neighbor
        # supports are disjoint across components, but it is vastly faster.
        ph_id = torch.where(is_p, torch.zeros_like(resid, dtype=torch.int64), torch.ones_like(resid, dtype=torch.int64))
        # Decide whether to force component-wise grouping.
        n_comp = params.get("_runtime_n_components", 1)
        try:
            n_comp_i = int(n_comp) if n_comp is not None else 1
        except Exception:
            n_comp_i = 1
        prefer_componentwise = (
            n_comp_i > 1
            and isinstance(cid_ev, torch.Tensor)
            and isinstance(comp_to_block, torch.Tensor)
            and isinstance(K_blocks, list) and len(K_blocks) > 0
            and isinstance(offs, torch.Tensor)
        )

        # Component id per row (for component-wise grouping and/or component-indexed inducing blocks).
        if cid_ev is None:
            comp_row = torch.zeros_like(ph_id, dtype=torch.int64)
        else:
            comp_row = cid_ev.index_select(0, idx[:, 0].to(dtype=torch.int64))

        if grouping == "phase":
            # For phase grouping, include component id when componentwise to keep solves small.
            keys = (comp_row * 2) + ph_id if prefer_componentwise else ph_id
        else:
            # station_phase
            n_sta = params.get("_runtime_n_stations", None)
            if (not isinstance(n_sta, int)) or n_sta <= 0:
                # fall back to local max+1 (best-effort)
                try:
                    n_sta = int(sta_idx.max().item()) + 1  # type: ignore[union-attr]
                except Exception:
                    n_sta = 1
            if prefer_componentwise:
                keys = ((comp_row * int(n_sta)) + sta_idx.to(dtype=torch.int64)) * 2 + ph_id  # type: ignore[union-attr]
            else:
                keys = (sta_idx.to(dtype=torch.int64) * 2) + ph_id  # type: ignore[union-attr]

        # --- Bucket fast-path (event_batches + reorder_all): avoid torch.sort(keys) entirely ---
        #
        # If owner-buckets are built with reorder_all=true, each bucket slice is already partitioned
        # into P then S (via _runtime_bucket_p_count). If additionally rows are ordered by
        # (station, component) within each phase, we can form groups by a simple boundary scan.
        use_bucket_fastpath = False
        try:
            use_bucket_fastpath = (
                grouping != "phase"
                and int(params.get("_runtime_bucket_id", -1)) >= 0
                and bool(params.get("event_bucket_reorder_all", False))
                and int(params.get("_runtime_bucket_p_count", -1)) >= 0
                and isinstance(params.get("_runtime_bucket_station_index", None), torch.Tensor)
                and isinstance(params.get("_runtime_bucket_comp_index", None), torch.Tensor)
            )
        except Exception:
            use_bucket_fastpath = False

        if use_bucket_fastpath:
            sta_rt = params.get("_runtime_bucket_station_index", None)
            comp_rt = params.get("_runtime_bucket_comp_index", None)
            B_rt = int(resid.numel())
            p_cnt = int(params.get("_runtime_bucket_p_count", -1))
            if not (isinstance(sta_rt, torch.Tensor) and isinstance(comp_rt, torch.Tensor)):
                use_bucket_fastpath = False
            elif int(sta_rt.numel()) != B_rt or int(comp_rt.numel()) != B_rt:
                use_bucket_fastpath = False
            elif p_cnt < 0 or p_cnt > B_rt:
                use_bucket_fastpath = False
            else:
                # Build group boundaries for P block and S block separately.
                # Groups are constant in (station, component). Phase is implicit by block.
                starts_list: list[torch.Tensor] = []
                ends_list: list[torch.Tensor] = []
                ph_list: list[int] = []
                comp_list: list[int] = []

                def _scan_block(i0: int, i1: int, ph_val: int) -> None:
                    n = int(i1 - i0)
                    if n <= 0:
                        return
                    sta = sta_rt[i0:i1].to(dtype=torch.int64)
                    if prefer_componentwise:
                        comp = comp_rt[i0:i1].to(dtype=torch.int64)
                    else:
                        # If not splitting by component, treat comp as constant to group by station only.
                        comp = torch.zeros_like(sta)
                    is_new = torch.ones((n,), device=sta.device, dtype=torch.bool)
                    if n > 1:
                        is_new[1:] = (sta[1:] != sta[:-1]) | (comp[1:] != comp[:-1])
                    starts = torch.nonzero(is_new, as_tuple=False).reshape(-1) + int(i0)
                    ends = torch.cat(
                        [
                            starts[1:],
                            torch.tensor([int(i1)], device=starts.device, dtype=starts.dtype),
                        ],
                        dim=0,
                    )
                    starts_list.append(starts)
                    ends_list.append(ends)
                    # Decode representative comp id per group (CPU one-time sync).
                    try:
                        c0 = comp_rt.index_select(0, starts).detach().cpu().tolist()
                    except Exception:
                        c0 = [0 for _ in range(int(starts.numel()))]
                    ph_list.extend([int(ph_val) for _ in range(int(starts.numel()))])
                    comp_list.extend([int(x) for x in c0])

                _scan_block(0, p_cnt, 0)
                _scan_block(p_cnt, B_rt, 1)

                if starts_list:
                    starts = torch.cat(starts_list, dim=0)
                    ends = torch.cat(ends_list, dim=0)
                else:
                    starts = torch.zeros((0,), device=resid.device, dtype=torch.int64)
                    ends = torch.zeros((0,), device=resid.device, dtype=torch.int64)
                perm = torch.arange(B_rt, device=resid.device, dtype=torch.int64)
                group_ph = ph_list
                group_comp = comp_list if prefer_componentwise else None

        # --- Optional caching of station_phase grouping ---
        #
        # Sorting `keys` each batch can be expensive when batch sizes are huge (100k–1M+).
        # When standard batching is deterministic (batch_shuffle=false), we can cache the sort perm
        # and group boundaries per batch id to avoid re-sorting every epoch.
        cache_ok = False
        try:
            cache_ok = (not bool(params.get("_runtime_batch_shuffle", True))) and (int(params.get("_runtime_batch_id", -1)) >= 0)
        except Exception:
            cache_ok = False
        cache_key = None
        cache_entry = None
        # If bucket fast-path was used, skip standard cache/sort.
        if use_bucket_fastpath:
            cache_ok = False
            cache_key = None
            cache_entry = {"perm": perm, "starts": starts, "ends": ends, "ph": group_ph, "comp": group_comp}
        if cache_ok:
            try:
                bid = int(params.get("_runtime_batch_id", -1))
                bsz = int(keys.numel())
                # Include prefer_componentwise and grouping in the cache key to avoid collisions.
                cache_key = ("slowness_re", str(grouping), int(prefer_componentwise), int(bsz), int(bid))
                cache = params.setdefault("_slowness_re_group_cache", {})
                cache_entry = cache.get(cache_key, None) if isinstance(cache, dict) else None
            except Exception:
                cache_key = None
                cache_entry = None

        if isinstance(cache_entry, dict):
            perm = cache_entry.get("perm", None)
            starts = cache_entry.get("starts", None)
            ends = cache_entry.get("ends", None)
            group_ph = cache_entry.get("ph", None)
            group_comp = cache_entry.get("comp", None)
            # Validate shapes (best-effort); fall back if invalid.
            ok = (
                isinstance(perm, torch.Tensor) and perm.ndim == 1 and int(perm.numel()) == int(keys.numel())
                and isinstance(starts, torch.Tensor) and starts.ndim == 1
                and isinstance(ends, torch.Tensor) and ends.ndim == 1 and int(ends.numel()) == int(starts.numel())
                and isinstance(group_ph, list) and len(group_ph) == int(starts.numel())
            )
            if not ok:
                perm = None
                starts = None
                ends = None
                group_ph = None
                group_comp = None
                cache_entry = None

        if cache_entry is None:
            # Compute grouping fresh
            keys_sorted, perm = torch.sort(keys)
            if keys_sorted.numel() > 0:
                is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
                is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
                starts = torch.nonzero(is_new, as_tuple=False).reshape(-1)
                ends = torch.cat(
                    [starts[1:], torch.tensor([keys_sorted.numel()], device=starts.device, dtype=starts.dtype)]
                )
            else:
                starts = torch.zeros((0,), device=resid.device, dtype=torch.int64)
                ends = torch.zeros((0,), device=resid.device, dtype=torch.int64)

            # Decode (phase, component) per group on CPU to avoid per-group GPU syncs later.
            group_ph = []
            group_comp = [] if prefer_componentwise else None
            try:
                if starts.numel() > 0:
                    gk = keys_sorted.index_select(0, starts).detach().cpu().numpy().astype(np.int64, copy=False)
                    for kk in gk.tolist():
                        ph_g = int(kk & 1)
                        group_ph.append(ph_g)
                        if prefer_componentwise and group_comp is not None:
                            if grouping == "phase":
                                group_comp.append(int(kk >> 1))
                            else:
                                # key = ((comp*n_sta + sta)*2 + ph)
                                comp_id = int((kk >> 1) // int(n_sta)) if isinstance(n_sta, int) and n_sta > 0 else 0
                                group_comp.append(int(comp_id))
                else:
                    group_ph = []
                    group_comp = [] if prefer_componentwise else None
            except Exception:
                # Fallback: keep decoded lists aligned with number of groups.
                group_ph = [0 for _ in range(int(starts.numel()))]
                group_comp = ([0 for _ in range(int(starts.numel()))] if prefer_componentwise else None)

            # Insert into cache (best-effort) if enabled.
            if cache_ok and cache_key is not None:
                try:
                    cache = params.setdefault("_slowness_re_group_cache", {})
                    order = params.setdefault("_slowness_re_group_cache_order", [])
                    max_entries = int(params.get("_slowness_re_group_cache_max_entries", 32))
                    if max_entries < 0:
                        max_entries = 0
                    if isinstance(cache, dict) and max_entries > 0:
                        cache[cache_key] = {"perm": perm, "starts": starts, "ends": ends, "ph": group_ph, "comp": group_comp}
                        if isinstance(order, list):
                            order.append(cache_key)
                            # Evict oldest entries
                            while len(order) > int(max_entries):
                                old = order.pop(0)
                                try:
                                    if old in cache:
                                        del cache[old]
                                except Exception:
                                    pass
                except Exception:
                    pass

        # Profiling: grouping timing (keys/build/sort/cache). We count this once per slowness_re call.
        if prof_sl:
            try:
                if prof_sl_use_cuda_events and e_grp0 is not None and e_grp1 is not None:
                    e_grp1.record()
                    e_grp1.synchronize()
                    dt_ms = float(e_grp0.elapsed_time(e_grp1))
                else:
                    dt_ms = 1000.0 * float(time.perf_counter() - t_grp0)
                params["_sl_re_grouping_ms_sum"] = float(params.get("_sl_re_grouping_ms_sum", 0.0) or 0.0) + float(dt_ms)
            except Exception:
                pass

        quad = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
        n_groups_total = 0
        n_groups_woodbury = 0
        n_groups_fallback_diag = 0
        max_rows_seen = 0
        max_nodes_seen = 0
        max_M_seen = 0
        # Use CPU-side boundaries for group iteration (one-time sync), then run group work on GPU.
        try:
            starts_cpu = starts.detach().cpu().numpy()
            ends_cpu = ends.detach().cpu().numpy()
        except Exception:
            starts_cpu = None
            ends_cpu = None

        # Cache small lookup tables on CPU to avoid .item() syncs in the group loop.
        comp_to_block_cpu = None
        offs_cpu = None
        try:
            if prefer_componentwise and isinstance(comp_to_block, torch.Tensor):
                comp_to_block_cpu = comp_to_block.detach().cpu().numpy()
            if prefer_componentwise and isinstance(offs, torch.Tensor):
                offs_cpu = offs.detach().cpu().numpy()
        except Exception:
            comp_to_block_cpu = None
            offs_cpu = None

        # Cache device-resident Kuu blocks and per-phase Kprior=(1/tau^2)Kuu to avoid repeated host->device
        # copies and repeated scaling inside the group loop.
        kuu_dev_cache = None
        kprior_cache = None
        try:
            kuu_dev_cache = params.setdefault("_slowness_re_kuu_device_cache", {})
            kprior_cache = params.setdefault("_slowness_re_kprior_cache", {})
        except Exception:
            kuu_dev_cache = None
            kprior_cache = None

        # Optional fast path for small M: precompute per-batch neighbor->dense Kbar once and reuse across groups.
        #
        # When grouping='station_phase' and there is a single inducing block (no component splitting),
        # each row belongs to exactly one group but the neighbor lists + kernel weights are shared.
        # Precomputing avoids per-group allocations and many small kernel launches.
        precompute_ok = False
        Kuu_global = None
        M_global = 0
        pre_max_M = int(params.get("_slowness_re_precompute_kbar_dense_max_M", 64))
        try:
            if pre_max_M < 1:
                pre_max_M = 0
        except Exception:
            pre_max_M = 64
        try:
            if grouping == "station_phase" and (not prefer_componentwise) and int(pre_max_M) > 0:
                if have_full and isinstance(K_full, torch.Tensor):
                    Kuu_global = K_full
                    M_global = int(Kuu_global.shape[0])
                    precompute_ok = True
                elif isinstance(K_blocks, list) and len(K_blocks) == 1 and isinstance(offs, torch.Tensor) and int(offs.numel()) >= 2:
                    Kuu0 = K_blocks[0]
                    if isinstance(Kuu0, torch.Tensor):
                        Kuu_global = Kuu0.to(device=dev, dtype=torch.float32) if (Kuu0.device != dev or Kuu0.dtype != torch.float32) else Kuu0
                        M_global = int(Kuu_global.shape[0])
                        precompute_ok = True
        except Exception:
            precompute_ok = False
        if precompute_ok and (int(M_global) > int(pre_max_M)):
            precompute_ok = False

        # Per-batch precomputes (only valid when precompute_ok)
        Kbar_dense_all = None
        nei_loc_all = None
        kbar_all = None
        w_all = None
        if precompute_ok:
            try:
                # Row endpoints
                e1_all = idx[:, 0].to(dtype=torch.int64)
                e2_all = idx[:, 1].to(dtype=torch.int64)
                x1_all = X_cur.index_select(0, e1_all)
                x2_all = X_cur.index_select(0, e2_all)
                dx_all = x2_all - x1_all  # (B,3)
                # w depends on tau_units; for vel_frac, scale by v(z) per phase.
                if tau_units == "vel_frac":
                    zbar_all = 0.5 * (x1_all[:, 2] + x2_all[:, 2])
                    if isinstance(zc, torch.Tensor) and isinstance(vp, torch.Tensor) and isinstance(vs, torch.Tensor):
                        vP = _interp1d_linear(zbar_all.to(torch.float32), zc.to(torch.float32), vp.to(torch.float32))
                        vS = _interp1d_linear(zbar_all.to(torch.float32), zc.to(torch.float32), vs.to(torch.float32))
                        v_all = torch.where(ph_id == 0, vP, vS).clamp_min(1e-3)
                    else:
                        v_all = torch.where(ph_id == 0, torch.full_like(zbar_all, 6.0), torch.full_like(zbar_all, 3.5)).clamp_min(1e-3)
                    w_all = (dx_all * (1.0 / v_all).unsqueeze(1)).to(torch.float32)
                else:
                    w_all = dx_all.to(torch.float32)

                # Neighbor indices (B,2m)
                nei1_all = nei_idx_ev.index_select(0, e1_all)
                nei2_all = nei_idx_ev.index_select(0, e2_all)
                nei_g_all = torch.cat([nei1_all, nei2_all], dim=1)
                mask_all = (nei_g_all >= 0)
                nei_clamped_all = torch.where(mask_all, nei_g_all, torch.zeros_like(nei_g_all))
                # Since we only allow this fast-path when there's a single global block, local == global.
                nei_loc_all = nei_clamped_all.to(torch.int64)
                nei_loc_all = torch.where(mask_all, nei_loc_all, torch.zeros_like(nei_loc_all))

                # Kernel weights (B,2m)
                k_map = params.get("_slowness_re_inducing_neighbor_k_matern32", None)
                if isinstance(k_map, torch.Tensor) and k_map.ndim == 2 and int(k_map.shape[0]) == int(X_src.shape[0]):
                    k_map = k_map.to(device=dev, dtype=torch.float32)
                    k1_all = k_map.index_select(0, e1_all)
                    k2_all = k_map.index_select(0, e2_all)
                    k_eu_all = torch.cat([k1_all, k2_all], dim=1)
                else:
                    # Dynamic kernel eval (depends on current X_cur)
                    B_all = int(nei_loc_all.shape[0])
                    K_all = int(nei_loc_all.shape[1])
                    U_all = inducing_xyz.index_select(0, nei_clamped_all.reshape(-1)).reshape(B_all, K_all, 3)
                    xe_all = torch.cat(
                        [x1_all.unsqueeze(1).expand(B_all, K_all // 2, 3), x2_all.unsqueeze(1).expand(B_all, K_all // 2, 3)],
                        dim=1,
                    )
                    d_all = torch.linalg.norm(U_all - xe_all, dim=2)
                    k_eu_all = _matern32(d_all, ell_km)
                k_eu_all = torch.where(mask_all, k_eu_all, torch.zeros_like(k_eu_all))
                kbar_all = (0.5 * k_eu_all).to(torch.float32)

                # Dense Kbar rows (B,M)
                Kbar_dense_all = torch.zeros((int(idx.shape[0]), int(M_global)), device=dev, dtype=torch.float32)
                Kbar_dense_all.scatter_add_(1, nei_loc_all, kbar_all)
            except Exception:
                Kbar_dense_all = None
                nei_loc_all = None
                kbar_all = None
                w_all = None

        # main group loop
        if starts_cpu is not None and ends_cpu is not None and isinstance(group_ph, list) and len(group_ph) == int(starts.numel()):
            if prefer_componentwise and isinstance(group_comp, list) and len(group_comp) == int(starts.numel()):
                group_iter = zip(starts_cpu.tolist(), ends_cpu.tolist(), group_ph, group_comp)
            else:
                group_iter = zip(starts_cpu.tolist(), ends_cpu.tolist(), group_ph, [None] * int(starts.numel()))
        else:
            # Fallback (rare): sync starts/ends and infer phase from tensor
            group_iter = zip(starts.tolist(), ends.tolist(), [None] * int(starts.numel()), [None] * int(starts.numel()))

        for si, ei, ph_g0, comp_id0 in group_iter:
            idxs = perm[si:ei]
            m_g = int(idxs.numel())
            if m_g <= 0:
                continue
            n_groups_total += 1
            if m_g > max_rows_seen:
                max_rows_seen = m_g
            # Phase/component for this group (prefer cached decoded values to avoid syncs).
            if ph_g0 is not None:
                ph_g = int(ph_g0)
            else:
                ph_g = int(ph_id.index_select(0, idxs[:1]).item())
            comp_id = int(comp_id0) if (prefer_componentwise and comp_id0 is not None) else None
            tau0 = tau_p if ph_g == 0 else tau_s
            tau = float(tau0)
            sigma_g = σ_p if ph_g == 0 else σ_s

            if m_g > int(max_rows_per_group):
                if not fallback_to_diag:
                    raise ValueError(
                        f"slowness_re: group too large (rows={m_g} > max_rows_per_group={max_rows_per_group}). "
                        f"Set max_rows_per_group higher or enable fallback_to_diag."
                    )
                n_groups_fallback_diag += 1
                u_g = resid.index_select(0, idxs) / sigma_g.square().clamp_min(1e-24)
                quad = quad + _CollapsedQuad.apply(resid.index_select(0, idxs), u_g)
                continue

            idx_g = idx.index_select(0, idxs)
            try:
                n_nodes = int(torch.unique(idx_g.reshape(-1)).numel())
            except Exception:
                n_nodes = m_g * 2
            if n_nodes > max_nodes_seen:
                max_nodes_seen = n_nodes
            if n_nodes > int(max_nodes_per_group):
                if not fallback_to_diag:
                    raise ValueError(
                        f"slowness_re: group too large (nodes={n_nodes} > max_nodes_per_group={max_nodes_per_group}). "
                        f"Set max_nodes_per_group higher or enable fallback_to_diag."
                    )
                n_groups_fallback_diag += 1
                u_g = resid.index_select(0, idxs) / sigma_g.square().clamp_min(1e-24)
                quad = quad + _CollapsedQuad.apply(resid.index_select(0, idxs), u_g)
                continue

            resid_g = resid.index_select(0, idxs)
            # tau <= 0 -> iid
            if not (tau > 0.0):
                u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                continue

            # Choose K_UU block and inducing index window.
            # Prefer component-wise blocks when there are multiple components.
            bi = -1
            if prefer_componentwise:
                # Ensure we have a component id (decode from key when possible; else fall back to GPU).
                if comp_id is None:
                    try:
                        comp_id = int(cid_ev.index_select(0, idx_g[:1, 0].to(dtype=torch.int64)).item()) if isinstance(cid_ev, torch.Tensor) else 0
                    except Exception:
                        comp_id = 0
                try:
                    if comp_to_block_cpu is not None:
                        bi = int(comp_to_block_cpu[int(comp_id)])
                    else:
                        bi = int(comp_to_block[int(comp_id)].item())  # may sync (fallback)
                except Exception:
                    bi = -1
                if bi < 0 or bi >= int(len(K_blocks)):
                    n_groups_fallback_diag += 1
                    u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                    quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                    continue
                # Get device-resident Kuu for this block (cache to avoid repeated .to()).
                Kuu = None
                try:
                    if isinstance(kuu_dev_cache, dict) and int(bi) in kuu_dev_cache:
                        Kuu = kuu_dev_cache[int(bi)]
                except Exception:
                    Kuu = None
                if not (isinstance(Kuu, torch.Tensor) and (Kuu.device == dev) and (Kuu.dtype == torch.float32)):
                    Kuu0 = K_blocks[bi]
                    if not isinstance(Kuu0, torch.Tensor):
                        n_groups_fallback_diag += 1
                        u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                        quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                        continue
                    if (Kuu0.device != dev) or (Kuu0.dtype != torch.float32):
                        Kuu = Kuu0.to(device=dev, dtype=torch.float32)
                    else:
                        Kuu = Kuu0
                    try:
                        if isinstance(kuu_dev_cache, dict):
                            kuu_dev_cache[int(bi)] = Kuu
                    except Exception:
                        pass
                try:
                    if offs_cpu is not None:
                        off0 = int(offs_cpu[int(bi)])
                        off1 = int(offs_cpu[int(bi) + 1])
                    else:
                        off0 = int(offs[int(bi)].item())
                        off1 = int(offs[int(bi) + 1].item())
                except Exception:
                    off0 = 0
                    off1 = int(Kuu.shape[0])
                M = int(Kuu.shape[0])
                if M <= 0 or (off1 - off0) != M:
                    n_groups_fallback_diag += 1
                    u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                    quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                    continue
            else:
                # Single-component (or explicitly not splitting): use K_full if available, else fall back to blocks.
                if have_full and isinstance(K_full, torch.Tensor):
                    Kuu = K_full
                    M = int(Kuu.shape[0])
                    off0 = 0
                    off1 = M
                    if M <= 0:
                        n_groups_fallback_diag += 1
                        u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                        quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                        continue
                else:
                    # Fallback to blocks (component id from GPU).
                    try:
                        comp_id2 = int(cid_ev.index_select(0, idx_g[:1, 0].to(dtype=torch.int64)).item()) if isinstance(cid_ev, torch.Tensor) else 0
                        bi = int(comp_to_block[int(comp_id2)].item()) if isinstance(comp_to_block, torch.Tensor) else -1
                    except Exception:
                        bi = -1
                    if bi < 0 or bi >= int(len(K_blocks)):
                        n_groups_fallback_diag += 1
                        u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                        quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                        continue

                    # Cache device-resident Kuu
                    Kuu = None
                    try:
                        if isinstance(kuu_dev_cache, dict) and int(bi) in kuu_dev_cache:
                            Kuu = kuu_dev_cache[int(bi)]
                    except Exception:
                        Kuu = None
                    if not (isinstance(Kuu, torch.Tensor) and (Kuu.device == dev) and (Kuu.dtype == torch.float32)):
                        Kuu0 = K_blocks[bi]
                        if not isinstance(Kuu0, torch.Tensor):
                            n_groups_fallback_diag += 1
                            u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                            quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                            continue
                        if (Kuu0.device != dev) or (Kuu0.dtype != torch.float32):
                            Kuu = Kuu0.to(device=dev, dtype=torch.float32)
                        else:
                            Kuu = Kuu0
                        try:
                            if isinstance(kuu_dev_cache, dict):
                                kuu_dev_cache[int(bi)] = Kuu
                        except Exception:
                            pass
                    off0 = int(offs[bi].item())
                    off1 = int(offs[bi + 1].item())
                    M = int(Kuu.shape[0])
                    if M <= 0 or (off1 - off0) != M:
                        n_groups_fallback_diag += 1
                        u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                        quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                        continue

            # Track maximum inducing size seen in this batch (useful for profiling / tuning solver thresholds).
            try:
                max_M_seen = max(int(max_M_seen), int(M))
            except Exception:
                pass

            # Endpoints and geometry (X_cur is already detached by construction).
            e1 = idx_g[:, 0].to(dtype=torch.int64)
            e2 = idx_g[:, 1].to(dtype=torch.int64)
            x1 = X_cur.index_select(0, e1)
            x2 = X_cur.index_select(0, e2)
            dx = x2 - x1  # (B,3)

            # Scale w for vel_frac units: w = dx / v(z)
            if tau_units == "vel_frac":
                zbar = 0.5 * (x1[:, 2] + x2[:, 2])
                if isinstance(zc, torch.Tensor) and isinstance(vp, torch.Tensor) and isinstance(vs, torch.Tensor):
                    v = _interp1d_linear(zbar.to(torch.float32), zc.to(torch.float32), (vp if ph_g == 0 else vs).to(torch.float32))
                else:
                    v = torch.full((int(zbar.numel()),), 6.0 if ph_g == 0 else 3.5, device=dev, dtype=torch.float32)
                inv_v = (1.0 / v.clamp_min(1e-3)).to(torch.float32)
                w = dx * inv_v.unsqueeze(1)
            else:
                w = dx

            # Build Kbar sparse rows (union of endpoint neighbor lists)
            if Kbar_dense_all is not None and nei_loc_all is not None and kbar_all is not None and w_all is not None and (int(off0) == 0) and (int(M) == int(M_global)):
                # Fast path: use precomputed neighbor/kbar and dense Kbar
                nei_loc = nei_loc_all.index_select(0, idxs)
                kbar = kbar_all.index_select(0, idxs)
                Kbar_dense = Kbar_dense_all.index_select(0, idxs)
                w = w_all.index_select(0, idxs).to(torch.float32)
                # Mask is implicit in kbar/nei_loc construction; use a cheap valid-mask derived from kbar.
                mask = (kbar != 0.0)
            else:
                nei1 = nei_idx_ev.index_select(0, e1)  # (B,m)
                nei2 = nei_idx_ev.index_select(0, e2)  # (B,m)
                nei_g = torch.cat([nei1, nei2], dim=1)  # (B,2m) global inducing idx
                mask = (nei_g >= 0)
                # clamp for gather
                nei_clamped = torch.where(mask, nei_g, torch.zeros_like(nei_g))
                # enforce component block bounds
                in_block = (nei_clamped >= int(off0)) & (nei_clamped < int(off1))
                mask = mask & in_block
                nei_loc = (nei_clamped - int(off0)).to(torch.int64)
                nei_loc = torch.where(mask, nei_loc, torch.zeros_like(nei_loc))

            B = int(nei_loc.shape[0])
            K = int(nei_loc.shape[1])
            if K <= 0:
                n_groups_fallback_diag += 1
                u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                continue
            if (K % 2) != 0:
                # We assume K = 2*m (neighbors for each endpoint). If it's not even, degrade safely.
                n_groups_fallback_diag += 1
                u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                continue

            # Kernel weights k(e,U). Prefer precomputed Matérn(3/2) at MAP for speed; fall back to dynamic eval.
            prof_this_group = False
            if prof_sl:
                try:
                    prof_this_group = int(params.get("_sl_re__groups_profiled_in_call", 0) or 0) < int(prof_sl_max_groups)
                except Exception:
                    prof_this_group = False
            if prof_this_group:
                if prof_sl_use_cuda_events:
                    try:
                        e_k0 = torch.cuda.Event(enable_timing=True)
                        e_k1 = torch.cuda.Event(enable_timing=True)
                        e_a0 = torch.cuda.Event(enable_timing=True)
                        e_a1 = torch.cuda.Event(enable_timing=True)
                        e_s0 = torch.cuda.Event(enable_timing=True)
                        e_s1 = torch.cuda.Event(enable_timing=True)
                        e_k0.record()
                    except Exception:
                        prof_this_group = False
                else:
                    t_k0 = time.perf_counter()

            if kbar_all is None:
                k_map = params.get("_slowness_re_inducing_neighbor_k_matern32", None)
                if isinstance(k_map, torch.Tensor) and k_map.ndim == 2 and int(k_map.shape[0]) == int(X_src.shape[0]):
                    k_map = k_map.to(device=dev, dtype=torch.float32)
                    k1 = k_map.index_select(0, e1)
                    k2 = k_map.index_select(0, e2)
                    k_eu = torch.cat([k1, k2], dim=1)
                else:
                    U = inducing_xyz.index_select(0, nei_clamped.reshape(-1)).reshape(B, K, 3)
                    xe = torch.cat(
                        [x1.unsqueeze(1).expand(B, K // 2, 3), x2.unsqueeze(1).expand(B, K // 2, 3)],
                        dim=1,
                    )
                    d = torch.linalg.norm(U - xe, dim=2)
                    k_eu = _matern32(d, ell_km)
                k_eu = torch.where(mask, k_eu, torch.zeros_like(k_eu))
                # 0.5 * (k(x1,U) + k(x2,U)) is represented by concatenation with a 0.5 scale
                kbar = 0.5 * k_eu  # (B,K)
            else:
                # Using cached kbar from the fast path above.
                kbar = kbar
            if prof_this_group:
                try:
                    if prof_sl_use_cuda_events:
                        e_k1.record()
                        e_a0.record()
                    else:
                        dt_ms = 1000.0 * float(time.perf_counter() - t_k0)
                        params["_sl_re_kernel_ms_sum"] = float(params.get("_sl_re_kernel_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        t_a0 = time.perf_counter()
                except Exception:
                    pass

            # Diagonal D = sigma^2 + FITC_diag (optional)
            s2 = sigma_g.square().clamp_min(1e-24).to(torch.float32)
            if isinstance(fitc_resid_ev, torch.Tensor):
                lam1 = fitc_resid_ev.index_select(0, e1).to(torch.float32)
                lam2 = fitc_resid_ev.index_select(0, e2).to(torch.float32)
                lam_bar = 0.5 * (lam1 + lam2)
                w2 = (w * w).sum(dim=1).to(torch.float32)
                diag_extra = (float(tau) * float(tau)) * (lam_bar.clamp_min(0.0)) * w2
                D = (s2 + diag_extra).clamp_min(1e-24)
            else:
                D = s2.expand(B).clamp_min(1e-24)
            alpha_d = (1.0 / D).to(torch.float32)  # (B,)

            # Assemble S = (1/tau^2) * (Kuu ⊗ I3) + B^T D^{-1} B, and rhs = B^T D^{-1} r.
            #
            # IMPORTANT performance note:
            # The earlier implementation built B^T D^{-1} B via explicit K×K outer products with python loops,
            # which is extremely slow for many small groups. Here we instead form a tiny dense B (shape B×(3M))
            # using scatter_add into (B×M), then use a weighted matmul to get BtDB and rhs. For the common case
            # in compact clusters with long ell, M is tiny (1–8) and this is very fast.
            w_f = w.to(torch.float32)
            # Solve path: do NOT build autograd graphs here (covariance depends on detached X_cur anyway,
            # and _CollapsedQuad applies the correct d/dr=u without differentiating through the solve).
            with torch.no_grad():
                # Optional PCG solver in 3M that avoids forming BtDB / cholesky.
                # This is typically beneficial when M is moderate/large (e.g. 100–300) and group size B is large.
                use_pcg = bool(solver == "pcg") and (int(M) >= int(pcg_min_inducing)) and (int(B) >= int(pcg_min_rows))
                if use_pcg:
                    if prof_this_group and prof_sl_use_cuda_events:
                        try:
                            e_s0.record()
                        except Exception:
                            pass
                    # Kprior is cached for this block+phase later in the cholesky path; reuse that cache here too.
                    tau2 = float(tau) * float(tau)
                    inv_tau2 = float(1.0 / max(tau2, 1e-24))
                    Kprior = None
                    try:
                        if isinstance(kprior_cache, dict):
                            Kprior = kprior_cache.get((int(bi), int(ph_g)), None)
                    except Exception:
                        Kprior = None
                    if not (isinstance(Kprior, torch.Tensor) and (Kprior.device == dev) and (Kprior.dtype == torch.float32) and int(Kprior.shape[0]) == int(M)):
                        Kprior = (inv_tau2 * Kuu).to(device=dev, dtype=torch.float32)
                        try:
                            if isinstance(kprior_cache, dict):
                                kprior_cache[(int(bi), int(ph_g))] = Kprior
                        except Exception:
                            pass

                    idx_flat = nei_loc.reshape(-1).to(dtype=torch.int64)
                    wx = w_f[:, 0].contiguous()
                    wy = w_f[:, 1].contiguous()
                    wz = w_f[:, 2].contiguous()

                    # rhs = B^T D^{-1} r
                    yr = (alpha_d * resid_g.detach().to(torch.float32)).contiguous()  # (B,)
                    rhs_x = torch.zeros((M,), device=dev, dtype=torch.float32)
                    rhs_y = torch.zeros((M,), device=dev, dtype=torch.float32)
                    rhs_z = torch.zeros((M,), device=dev, dtype=torch.float32)
                    rhs_x.scatter_add_(0, idx_flat, (kbar * (wx * yr).unsqueeze(1)).reshape(-1))
                    rhs_y.scatter_add_(0, idx_flat, (kbar * (wy * yr).unsqueeze(1)).reshape(-1))
                    rhs_z.scatter_add_(0, idx_flat, (kbar * (wz * yr).unsqueeze(1)).reshape(-1))
                    b_vec = torch.cat([rhs_x, rhs_y, rhs_z], dim=0)  # (3M,)

                    # Jacobi preconditioner: diag(Kprior) + diag(B^T D^{-1} B) per dimension.
                    k2 = (kbar * kbar).to(torch.float32)
                    diag_x = torch.zeros((M,), device=dev, dtype=torch.float32)
                    diag_y = torch.zeros((M,), device=dev, dtype=torch.float32)
                    diag_z = torch.zeros((M,), device=dev, dtype=torch.float32)
                    diag_x.scatter_add_(0, idx_flat, (k2 * (alpha_d * (wx * wx)).unsqueeze(1)).reshape(-1))
                    diag_y.scatter_add_(0, idx_flat, (k2 * (alpha_d * (wy * wy)).unsqueeze(1)).reshape(-1))
                    diag_z.scatter_add_(0, idx_flat, (k2 * (alpha_d * (wz * wz)).unsqueeze(1)).reshape(-1))
                    kd = torch.diagonal(Kprior).contiguous()
                    diag_x.add_(kd)
                    diag_y.add_(kd)
                    diag_z.add_(kd)
                    diag_inv = torch.cat(
                        [
                            (1.0 / diag_x.clamp_min(1e-12)),
                            (1.0 / diag_y.clamp_min(1e-12)),
                            (1.0 / diag_z.clamp_min(1e-12)),
                        ],
                        dim=0,
                    )

                    def _A_mul(p_vec: torch.Tensor) -> torch.Tensor:
                        px = p_vec[0:M]
                        py = p_vec[M : 2 * M]
                        pz = p_vec[2 * M : 3 * M]
                        # Kprior ⊗ I3 term
                        ax = Kprior @ px
                        ay = Kprior @ py
                        az = Kprior @ pz
                        # BtDB term via sparse neighbor matvecs
                        px_g = torch.index_select(px, 0, idx_flat).reshape(B, K)
                        py_g = torch.index_select(py, 0, idx_flat).reshape(B, K)
                        pz_g = torch.index_select(pz, 0, idx_flat).reshape(B, K)
                        t = (kbar * (wx.unsqueeze(1) * px_g + wy.unsqueeze(1) * py_g + wz.unsqueeze(1) * pz_g)).sum(dim=1)  # (B,)
                        yb = (alpha_d * t).contiguous()
                        ax.scatter_add_(0, idx_flat, (kbar * (wx * yb).unsqueeze(1)).reshape(-1))
                        ay.scatter_add_(0, idx_flat, (kbar * (wy * yb).unsqueeze(1)).reshape(-1))
                        az.scatter_add_(0, idx_flat, (kbar * (wz * yb).unsqueeze(1)).reshape(-1))
                        return torch.cat([ax, ay, az], dim=0)

                    # Fixed-iteration PCG (no early-exit by default to avoid per-iter device sync).
                    x = torch.zeros_like(b_vec)
                    r = b_vec.clone()
                    z = diag_inv * r
                    p = z.clone()
                    rz = (r * z).sum()
                    bnorm = torch.sqrt((b_vec * b_vec).sum()).clamp_min(1e-12)
                    it_done = 0
                    for it in range(int(pcg_max_iters)):
                        Ap = _A_mul(p)
                        denom = (p * Ap).sum().clamp_min(1e-24)
                        a = rz / denom
                        x = x + a * p
                        r = r - a * Ap
                        it_done = it + 1
                        if int(pcg_check_every) > 0 and ((it_done % int(pcg_check_every)) == 0):
                            rel = torch.sqrt((r * r).sum()) / bnorm
                            if float(rel) < float(pcg_tol):
                                break
                        z = diag_inv * r
                        rz_new = (r * z).sum()
                        beta = rz_new / rz.clamp_min(1e-24)
                        p = z + beta * p
                        rz = rz_new

                    y = x
                    # u = D^{-1}(r - B y)
                    yx = y[0:M]
                    yy = y[M : 2 * M]
                    yz = y[2 * M : 3 * M]
                    yx_g = torch.index_select(yx, 0, idx_flat).reshape(B, K)
                    yy_g = torch.index_select(yy, 0, idx_flat).reshape(B, K)
                    yz_g = torch.index_select(yz, 0, idx_flat).reshape(B, K)
                    by = (kbar * (wx.unsqueeze(1) * yx_g + wy.unsqueeze(1) * yy_g + wz.unsqueeze(1) * yz_g)).sum(dim=1)  # (B,)
                    u_g = alpha_d * (resid_g.detach().to(torch.float32) - by)
                    u_g = u_g.to(dtype=resid_g.dtype)
                    n_groups_woodbury += 1
                    quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                    try:
                        params["_slowness_re_runtime_last_solver"] = "pcg"
                        params["_slowness_re_runtime_last_pcg_iters"] = int(it_done)
                    except Exception:
                        pass
                    # Profiling finalize (PCG group)
                    if prof_this_group:
                        try:
                            params["_sl_re_pcg_iters_sum"] = int(params.get("_sl_re_pcg_iters_sum", 0) or 0) + int(it_done)
                        except Exception:
                            pass
                        try:
                            if prof_sl_use_cuda_events:
                                e_s1.record()
                                e_s1.synchronize()
                                params["_sl_re_kernel_ms_sum"] = float(params.get("_sl_re_kernel_ms_sum", 0.0) or 0.0) + float(e_k0.elapsed_time(e_k1))
                                params["_sl_re_assemble_ms_sum"] = float(params.get("_sl_re_assemble_ms_sum", 0.0) or 0.0) + float(e_a0.elapsed_time(e_s0))
                                params["_sl_re_solve_ms_sum"] = float(params.get("_sl_re_solve_ms_sum", 0.0) or 0.0) + float(e_s0.elapsed_time(e_s1))
                            else:
                                params["_sl_re_assemble_ms_sum"] = float(params.get("_sl_re_assemble_ms_sum", 0.0) or 0.0) + (1000.0 * float(time.perf_counter() - t_a0))
                        except Exception:
                            pass
                        try:
                            params["_sl_re__groups_profiled_in_call"] = int(params.get("_sl_re__groups_profiled_in_call", 0) or 0) + 1
                            params["_sl_re_profiled_groups_sum"] = int(params.get("_sl_re_profiled_groups_sum", 0) or 0) + 1
                        except Exception:
                            pass
                    continue

                # Dense Kbar: (B,M) with Kbar[b,u] = sum_k kbar[b,k] for neighbors mapping to u.
                if Kbar_dense_all is None:
                    Kbar_dense = torch.zeros((B, M), device=dev, dtype=torch.float32)
                    # nei_loc is (B,K) with invalid entries mapped to 0 and kbar already zeroed for invalid => safe scatter_add
                    Kbar_dense.scatter_add_(1, nei_loc, kbar)
                # B_dense: (B,3M)
                B0 = (w_f[:, 0:1] * Kbar_dense)
                B1 = (w_f[:, 1:2] * Kbar_dense)
                B2 = (w_f[:, 2:3] * Kbar_dense)
                B_dense = torch.cat([B0, B1, B2], dim=1).contiguous()

                # Weighted system using sqrt(alpha_d): BtDB = (sqrtA*B)^T (sqrtA*B), rhs = (sqrtA*B)^T (sqrtA*r)
                sA = alpha_d.sqrt().to(torch.float32)  # (B,)
                WB = B_dense * sA.unsqueeze(1)         # (B,3M)
                wr = resid_g.detach().to(torch.float32) * sA    # (B,)
                BtDB = WB.transpose(0, 1) @ WB         # (3M,3M)
                rhs_vec = WB.transpose(0, 1) @ wr      # (3M,)

                # Add prior term A^{-1} = (1/tau^2) * (Kuu ⊗ I3)
                tau2 = float(tau) * float(tau)
                inv_tau2 = float(1.0 / max(tau2, 1e-24))
                # Cache Kprior=(1/tau^2)Kuu per (block, phase). For K_full mode use bi=-1.
                Kprior = None
                try:
                    if isinstance(kprior_cache, dict):
                        Kprior = kprior_cache.get((int(bi), int(ph_g)), None)
                except Exception:
                    Kprior = None
                if not (isinstance(Kprior, torch.Tensor) and (Kprior.device == dev) and (Kprior.dtype == torch.float32) and int(Kprior.shape[0]) == int(M)):
                    Kprior = (inv_tau2 * Kuu).to(device=dev, dtype=torch.float32)
                    try:
                        if isinstance(kprior_cache, dict):
                            kprior_cache[(int(bi), int(ph_g))] = Kprior
                    except Exception:
                        pass
                # Avoid allocating a separate block-diagonal matrix: add Kprior into the 3 diagonal blocks in-place.
                G = BtDB
                G[0:M, 0:M].add_(Kprior)
                G[M : 2 * M, M : 2 * M].add_(Kprior)
                G[2 * M : 3 * M, 2 * M : 3 * M].add_(Kprior)

                # Solve S y = rhs (best-effort; fall back to diag)
                if prof_this_group and prof_sl_use_cuda_events:
                    try:
                        e_s0.record()
                    except Exception:
                        pass
                try:
                    # Use cholesky with exception fallback. This avoids per-group `.item()` GPU syncs.
                    L = torch.linalg.cholesky(G)
                    y = torch.cholesky_solve(rhs_vec.reshape(-1, 1), L).reshape(-1)  # (3M,)
                except Exception:
                    # Escalate jitter on diagonal and retry once (rare).
                    try:
                        j = float(params.get("_slowness_re_cholesky_jitter", 1e-4))
                    except Exception:
                        j = 1e-4
                    try:
                        if j > 0.0:
                            G.diagonal().add_(float(j))
                        L = torch.linalg.cholesky(G)
                        y = torch.cholesky_solve(rhs_vec.reshape(-1, 1), L).reshape(-1)
                    except Exception:
                        y = None
                # Profiling finalize (Cholesky group)
                if prof_this_group:
                    try:
                        if prof_sl_use_cuda_events:
                            e_s1.record()
                            e_s1.synchronize()
                            params["_sl_re_kernel_ms_sum"] = float(params.get("_sl_re_kernel_ms_sum", 0.0) or 0.0) + float(e_k0.elapsed_time(e_k1))
                            params["_sl_re_assemble_ms_sum"] = float(params.get("_sl_re_assemble_ms_sum", 0.0) or 0.0) + float(e_a0.elapsed_time(e_s0))
                            params["_sl_re_solve_ms_sum"] = float(params.get("_sl_re_solve_ms_sum", 0.0) or 0.0) + float(e_s0.elapsed_time(e_s1))
                        else:
                            params["_sl_re_assemble_ms_sum"] = float(params.get("_sl_re_assemble_ms_sum", 0.0) or 0.0) + (1000.0 * float(time.perf_counter() - t_a0))
                    except Exception:
                        pass
                    try:
                        params["_sl_re__groups_profiled_in_call"] = int(params.get("_sl_re__groups_profiled_in_call", 0) or 0) + 1
                        params["_sl_re_profiled_groups_sum"] = int(params.get("_sl_re_profiled_groups_sum", 0) or 0) + 1
                    except Exception:
                        pass

                if y is None:
                    n_groups_fallback_diag += 1
                    u_g = resid_g / sigma_g.square().clamp_min(1e-24)
                    quad = quad + _CollapsedQuad.apply(resid_g, u_g)
                    continue

                by = (B_dense @ y).to(torch.float32)  # (B,)
                u_g = alpha_d * (resid_g.detach().to(torch.float32) - by)
                u_g = u_g.to(dtype=resid_g.dtype)
                n_groups_woodbury += 1
                quad = quad + _CollapsedQuad.apply(resid_g, u_g)

        # Stash stats for caller
        try:
            params["_slowness_re_runtime_last_grouping"] = str(grouping)
            params["_slowness_re_runtime_last_groups"] = int(n_groups_total)
            params["_slowness_re_runtime_last_groups_woodbury"] = int(n_groups_woodbury)
            params["_slowness_re_runtime_last_groups_fallback_diag"] = int(n_groups_fallback_diag)
            params["_slowness_re_runtime_last_max_rows"] = int(max_rows_seen)
            params["_slowness_re_runtime_last_max_nodes"] = int(max_nodes_seen)
        except Exception:
            pass

        # If profiling enabled, accumulate per-batch workload stats for epoch-level reporting.
        if prof_sl:
            try:
                params["_sl_re_groups_sum"] = int(params.get("_sl_re_groups_sum", 0) or 0) + int(n_groups_total)
                params["_sl_re_groups_woodbury_sum"] = int(params.get("_sl_re_groups_woodbury_sum", 0) or 0) + int(n_groups_woodbury)
                params["_sl_re_groups_fallback_sum"] = int(params.get("_sl_re_groups_fallback_sum", 0) or 0) + int(n_groups_fallback_diag)
                params["_sl_re_max_rows_max"] = max(int(params.get("_sl_re_max_rows_max", 0) or 0), int(max_rows_seen))
                params["_sl_re_max_nodes_max"] = max(int(params.get("_sl_re_max_nodes_max", 0) or 0), int(max_nodes_seen))
                params["_sl_re_max_M_max"] = max(int(params.get("_sl_re_max_M_max", 0) or 0), int(max_M_seen))
            except Exception:
                pass

        m_tot = float(max(int(resid.numel()), 1))
        loss_like = (quad / m_tot) + torch.log(sigma).mean()
        # Per-call (batch) profiling finalize.
        if prof_sl:
            try:
                params["_sl_re_time_ms_count"] = int(params.get("_sl_re_time_ms_count", 0) or 0) + 1
            except Exception:
                pass
            if prof_sl_use_cuda_events and "e_total0" in locals() and e_total0 is not None and e_total1 is not None:
                try:
                    e_total1.record()
                    e_total1.synchronize()
                    dt_ms = float(e_total0.elapsed_time(e_total1))
                    params["_sl_re_time_ms_sum"] = float(params.get("_sl_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                except Exception:
                    pass
            else:
                try:
                    dt_ms = 1000.0 * float(time.perf_counter() - t_total0)
                    params["_sl_re_time_ms_sum"] = float(params.get("_sl_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                except Exception:
                    pass
        return float(alpha) * loss_like

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
            print(f"[shared_event_re] se_enable={se_enable} solver={solver}", flush=True)
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
                print(
                    f"[shared_event_re] resid_n={int(resid.numel())} idx_n={int(idx.shape[0])} grouping={grouping}",
                    flush=True,
                )
            if resid.numel() == 0 or idx.shape[0] == 0:
                if not bool(params.get("_shared_event_re_empty_batch_logged", False)):
                    params["_shared_event_re_empty_batch_logged"] = True
                    print("[shared_event_re] WARNING: empty batch passed to shared_event_re", flush=True)
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

            if sigma_extra_var is not None:
                raise ValueError(
                    "model.likelihood.shared_event_re: sigma_extra_var is not supported with pcg_sparse "
                    "(would require a weighted Laplacian / heteroscedastic diagonal; implement in a future phase)"
                )

            # Station index is supplied at runtime by the epoch runner when owner-bucket batching is active.
            sta_idx = None
            if grouping == "station_phase":
                sta_idx = params.get("_runtime_bucket_station_index", None)
                if not isinstance(sta_idx, torch.Tensor) or int(sta_idx.numel()) != int(resid.numel()):
                    # Fall back quietly to phase-only; warn once.
                    if not bool(params.get("_shared_event_re_warned_no_station_index", False)):
                        print(
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
                    print(
                        "Warning: shared_event_re.station_phase_re enabled but no per-row station index was available; "
                        "falling back to phase-only station_phase_re.",
                        flush=True,
                    )
                if not bool(params.get("_shared_event_re_station_phase_logged", False)):
                    params["_shared_event_re_station_phase_logged"] = True
                    print(
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
                print(f"[shared_event_re] entered block solver={solver} grouping={grouping}", flush=True)

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
                    print(
                        f"[shared_event_re] enabled=True grouping={grouping} "
                        f"tau_event_p={float(tau_p):.4g} tau_event_s={float(tau_s):.4g} "
                        f"tau_cluster_p={float(tau_c_p):.4g} tau_cluster_s={float(tau_c_s):.4g} "
                        f"max_rows={int(max_rows_per_group)} max_nodes={int(max_nodes_per_group)} "
                        f"gpu={gpu_enable} whitening={wflag} weight={wmode}",
                        flush=True,
                    )
                else:
                    print(
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
                    print(
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
                            print(
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
                return float(alpha) * loss_like

            # Optional GPU-native prototype path (batched PCG + grouping).
            perm = None
            starts = None
            ends = None
            if gpu_enable:
                if not bool(params.get("_shared_event_re_gpu_logged", False)):
                    params["_shared_event_re_gpu_logged"] = True
                    print("[shared_event_re] GPU path enabled", flush=True)
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
                print(
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
                    print(f"[shared_event_re] loss_delta_vs_diag={delta:.6e}", flush=True)
                except Exception as e:
                    print(f"[shared_event_re] loss_delta_vs_diag failed: {e}", flush=True)

            # Mean over observations (to match the rest of SPIDER)
            quad = quad + quad_diag_extra + quad_sp
            # Keep the independent log(sigma) term for compatibility (with learn_noise_scale=false it's a constant anyway).
            loss_like = (quad / m_tot_full) + log_sigma_mean
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

    # Optional: dd_graph_re loss delta vs baseline (per epoch)
    if bool(params.get("_dd_graph_re_enabled", False)):
        try:
            epoch_idx = int(params.get("_runtime_epoch_index", -1))
            if int(params.get("_dd_graph_re_loss_logged_epoch", -2)) != int(epoch_idx):
                params["_dd_graph_re_loss_logged_epoch"] = int(epoch_idx)
                if loss_type in {"gaussian", "mse", "l2"}:
                    data_loss0 = 0.5 * (scaled_resid_base ** 2)
                elif loss_type in {"student_t", "student-t", "studentt"}:
                    data_loss0 = 0.5 * (nu + 1.0) * torch.log1p((scaled_resid_base ** 2) / nu) + t_const
                elif loss_type in {"laplace", "l1", "mae"}:
                    data_loss0 = torch.abs(scaled_resid_base)
                else:
                    data_loss0 = F.huber_loss(
                        scaled_resid_base,
                        torch.zeros_like(scaled_resid_base),
                        reduction="none",
                        delta=huber_delta,
                    )
                total_nll0 = data_loss0 + torch.log(sigma)
                params["_dd_graph_re_loss_delta"] = float((total_nll.mean() - total_nll0.mean()).detach().item())
        except Exception:
            pass
    
    # Return MEAN (Average Loss)
    return total_nll.mean()


def compute_prior_loss(
    ΔX_src: torch.Tensor,
    prior_event: torch.distributions.Distribution,
    σ_p: torch.Tensor,
    σ_s: torch.Tensor,
    N_total: int,
    params: dict,
    cluster_ids: torch.Tensor | None = None,
    cluster_counts: torch.Tensor | None = None,
    event_precision_matrix: torch.Tensor | None = None,
    corr_error_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the Total Prior Negative Log-Probability, scaled by 1/N_total.
    
    This ensures the prior is weighted consistently with the Average Likelihood.
    """
    # Explicit enable flags (defaults preserve legacy behavior)
    event_prior_enable = bool(params.get("prior_event_enable", True))
    # Runtime gates for other priors (set by the epoch runner). Defaults keep legacy behavior.
    event_runtime_enable = bool(params.get("_prior_event_runtime_enable", True))

    # Correlated forward-model error prior p(b) (optional).
    # This uses an event-graph precision Q (radius-r subsampled Laplacian + q_diag I).
    log_prob_b_corr = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    try:
        if bool(params.get("_corr_error_enabled", False)) and isinstance(corr_error_b, torch.Tensor):
            # Expect b shape (n_events, R, 2)
            b = corr_error_b
            if not (b.ndim == 3 and int(b.shape[2]) == 2):
                raise RuntimeError("corr_error prior: expected b shape (n_events, R, 2)")
            u = params.get("_corr_error_u", None)
            v = params.get("_corr_error_v", None)
            w = params.get("_corr_error_w", None)
            q_diag = float(params.get("_corr_error_q_diag", 0.0))
            if not (isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor)):
                raise RuntimeError("corr_error prior: missing event graph (u,v,w)")

            u_i = u.to(torch.int64)
            v_i = v.to(torch.int64)
            w_f = w.to(dtype=b.dtype, device=b.device)

            # Apply Q to a [N,R] tensor: y = q_diag*x + L_w x
            def _apply_Q(xNR: torch.Tensor) -> torch.Tensor:
                y = xNR * float(max(0.0, q_diag))
                if int(u_i.numel()) > 0:
                    xu = xNR.index_select(0, u_i)
                    xv = xNR.index_select(0, v_i)
                    diff = xu - xv  # [E,R]
                    dw = diff * w_f.unsqueeze(1)
                    y.index_add_(0, u_i, dw)
                    y.index_add_(0, v_i, -dw)
                return y

            bP = b[:, :, 0]
            bS = b[:, :, 1]
            qP = _apply_Q(bP)
            qS = _apply_Q(bS)
            
            # Energy calculation
            e00 = (bP * qP).sum()
            e11 = (bS * qS).sum()
            e01 = (bP * qS).sum()

            tau_ps = params.get("_corr_error_tau_s", [0.0, 0.0])
            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
            rho = float(params.get("_corr_error_rho_ps", 0.0))
            if (tau_p > 0.0) and (tau_s > 0.0) and (abs(rho) < 1.0):
                det = (tau_p * tau_p) * (tau_s * tau_s) * (1.0 - rho * rho)
                inv00 = (tau_s * tau_s) / det
                inv11 = (tau_p * tau_p) / det
                inv01 = (-rho * tau_p * tau_s) / det
            else:
                inv00 = 0.0
                inv11 = 0.0
                inv01 = 0.0
            energy = 0.5 * (float(inv00) * e00 + float(inv11) * e11 + 2.0 * float(inv01) * e01)
            
            log_prob_b_corr = (-energy).to(dtype=ΔX_src.dtype)
    except Exception:
        log_prob_b_corr = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)

    # Explicit slowness_re latents prior (IID Gaussian).
    log_prob_slowness = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    try:
        if bool(params.get("_slowness_re_explicit_enabled", False)):
            s_comp_p = params.get("_slowness_re_explicit_comp_p", None)
            s_comp_s = params.get("_slowness_re_explicit_comp_s", None)
            a_sta_p = params.get("_slowness_re_explicit_station_p", None)
            a_sta_s = params.get("_slowness_re_explicit_station_s", None)
            if all(isinstance(x, torch.Tensor) for x in (s_comp_p, s_comp_s, a_sta_p, a_sta_s)):
                tau_ps = params.get("_slowness_re_tau_s", [0.0, 0.0])
                tau_sta_ps = params.get("_slowness_re_tau_station_s", [0.0, 0.0])
                tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
                tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
                tau_sta_p = float(tau_sta_ps[0]) if isinstance(tau_sta_ps, (list, tuple)) and len(tau_sta_ps) >= 2 else float(tau_sta_ps)
                tau_sta_s = float(tau_sta_ps[1]) if isinstance(tau_sta_ps, (list, tuple)) and len(tau_sta_ps) >= 2 else float(tau_sta_ps)
                tau_units = str(params.get("_slowness_re_tau_units", "abs")).strip().lower()
                if tau_units == "vel_frac":
                    vp = float(params.get("_slowness_re_vp_km_s", 6.0))
                    vs = float(params.get("_slowness_re_vs_km_s", 3.5))
                    tau_p = tau_p / max(vp, 1e-6)
                    tau_s = tau_s / max(vs, 1e-6)
                    tau_sta_p = tau_sta_p / max(vp, 1e-6)
                    tau_sta_s = tau_sta_s / max(vs, 1e-6)
                energy = torch.zeros((), device=ΔX_src.device, dtype=ΔX_src.dtype)
                if tau_p > 0.0:
                    energy = energy + 0.5 * (s_comp_p / float(tau_p)).square().sum()
                if tau_s > 0.0:
                    energy = energy + 0.5 * (s_comp_s / float(tau_s)).square().sum()
                if tau_sta_p > 0.0:
                    energy = energy + 0.5 * (a_sta_p / float(tau_sta_p)).square().sum()
                if tau_sta_s > 0.0:
                    energy = energy + 0.5 * (a_sta_s / float(tau_sta_s)).square().sum()
                log_prob_slowness = -energy
    except Exception:
        log_prob_slowness = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)

    # DD-graph random effects prior (event latents with Laplacian/GMRF prior).
    log_prob_dd_graph = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    try:
        if bool(params.get("_dd_graph_re_enabled", False)):
            b_p = params.get("_dd_graph_re_b_p", None)
            b_s = params.get("_dd_graph_re_b_s", None)
            u = params.get("_dd_graph_re_u", None)
            v = params.get("_dd_graph_re_v", None)
            w = params.get("_dd_graph_re_w", None)
            if all(isinstance(x, torch.Tensor) for x in (b_p, b_s, u, v, w)):
                tau_ps = params.get("_dd_graph_re_tau_s", [0.0, 0.0])
                tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
                tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
                q_diag = float(params.get("_dd_graph_re_q_diag", 0.0))
                u_i = u.to(torch.int64)
                v_i = v.to(torch.int64)
                w_f = w.to(device=b_p.device, dtype=b_p.dtype)
                def _energy(b: torch.Tensor) -> torch.Tensor:
                    if u_i.numel() == 0:
                        return (b.square().sum() * float(max(0.0, q_diag)))
                    diff = b.index_select(0, u_i) - b.index_select(0, v_i)
                    return (w_f * diff.square()).sum() + (b.square().sum() * float(max(0.0, q_diag)))
                energy = torch.zeros((), device=ΔX_src.device, dtype=ΔX_src.dtype)
                if tau_p > 0.0:
                    energy = energy + 0.5 * _energy(b_p) / float(tau_p * tau_p)
                if tau_s > 0.0:
                    energy = energy + 0.5 * _energy(b_s) / float(tau_s * tau_s)
                log_prob_dd_graph = -energy
    except Exception:
        log_prob_dd_graph = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)

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
    
    # 2. Noise prior removed: SPIDER uses fixed `phase_unc` only (no σ learning).
    log_prob_noise = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    
    # (shared_event_latent prior removed)

    # Total Log Prior
    total_log_prior = log_prob_events + log_prob_noise + log_prob_b_corr + log_prob_slowness + log_prob_dd_graph
    
    base_prior_loss = -total_log_prior / float(N_total)
    return base_prior_loss


def total_loss(
    idx, y, X_src, ΔX_src, model,
    prior_event, σ_p, σ_s,
    N_total, params, nuisance_delta=None,
    sigma_extra_var=None,
    cluster_ids=None, cluster_counts=None,
    event_precision_matrix=None,
    corr_error_b=None,
):
    """
    Compute Total Unified Loss (Average Negative Log Posterior).
    
    Objective = AverageNLL(Data) + (1/N) * NegativeLogPrior(Params)
    
    This objective is independent of dataset size N (as N->inf), 
    stabilizing gradients/hyperparams.
    """
    
    # 1. Likelihood (Average over batch)
    loss_like = compute_likelihood_loss(
        idx, y, X_src, ΔX_src, model, σ_p, σ_s, params, nuisance_delta, sigma_extra_var
    )
    
    loss_prior = compute_prior_loss(
        ΔX_src, prior_event, σ_p, σ_s, N_total, params,
        cluster_ids, cluster_counts, event_precision_matrix,
        corr_error_b,
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

