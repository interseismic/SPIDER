import numpy as np
import polars as pl
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # Optional: add extra per-observation variance (e.g., FITC diagonal correction for inducing GP).
    # This leaves noise priors (which depend on base σ_p/σ_s) unchanged.
    if sigma_extra_var is not None:
        try:
            sev = sigma_extra_var
            if not isinstance(sev, torch.Tensor):
                sev = torch.tensor(sev, device=sigma.device, dtype=sigma.dtype)
            sev = sev.to(device=sigma.device, dtype=sigma.dtype).clamp_min(0.0)
            sigma2 = sigma.square() + sev
            sigma = sigma2.sqrt().clamp_min(1e-12)
        except Exception:
            pass
    resid = dt_obs - dt_pred
    scaled_resid = resid / sigma

    # Structured likelihood components were removed (start fresh). If an old bundle/checkpoint
    # carried internal enable flags, fail fast with a clear message.
    if bool(params.get("_slowness_re_enabled", False)) or bool(params.get("_shared_event_re_enabled", False)) or bool(params.get("_shared_event_latent_enabled", False)):
        raise ValueError(
            "Structured likelihood components (slowness_re/shared_event_re/shared_event_latent) have been removed from SPIDER. "
            "Delete these blocks from your config and regenerate any bundles/checkpoints from a clean run."
        )

    # Optional: collapsed slowness inducing-GP covariance likelihood (Gaussian; marginalized; no latent state).
    #
    # Phase-A implementation: quadratic-only (drop logdet). We compute u ≈ Σ^{-1} r per group and return:
    #   mean( 0.5 r^T u ) + mean(log sigma)
    # with custom autograd so d/dr = u (do not differentiate through the solver).
    try:
        sl_enable = bool(params.get("_slowness_re_enabled", False))
    except Exception:
        sl_enable = False
    if sl_enable:
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
        if not have_inducing:
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

        if cid_ev is None:
            comp_row = torch.zeros_like(ph_id, dtype=torch.int64)
        else:
            comp_row = cid_ev.index_select(0, idx[:, 0].to(dtype=torch.int64))

        if grouping == "phase":
            # For phase grouping, include component id when componentwise to keep solves small.
            keys = (comp_row * 2) + ph_id if prefer_componentwise else ph_id
            n_sta = None
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
                ends = torch.cat([starts[1:], torch.tensor([keys_sorted.numel()], device=starts.device, dtype=starts.dtype)])
            else:
                starts = torch.zeros((0,), device=resid.device, dtype=torch.int64)
                ends = torch.zeros((0,), device=resid.device, dtype=torch.int64)

            # Decode (phase, component) per group on CPU to avoid per-group GPU syncs later.
            group_ph: list[int] = []
            group_comp: list[int] | None = [] if prefer_componentwise else None
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
                # Fallback: keep empty decoded lists; group loop can still recover phase from tensor.
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
        if solver in {"pcg", "pcg_sparse", "pcg-sparse"}:
            # Current Phase-A implementation: quadratic-only (drop_logdet must be true; enforced by schema).
            # We compute u ≈ Σ^{-1} r per group and return:
            #   mean( 0.5 r^T u ) + mean(log sigma)
            # but with custom autograd so d/dr = u (do not differentiate through the solver).
            grouping = str(params.get("_shared_event_re_grouping", "phase")).strip().lower()
            if grouping in {"stationphase", "station-phase"}:
                grouping = "station_phase"
            tau_ps = params.get("_shared_event_re_tau_s", [0.0, 0.0])
            tau_p = float(tau_ps[0]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
            tau_s = float(tau_ps[1]) if isinstance(tau_ps, (list, tuple)) and len(tau_ps) >= 2 else float(tau_ps)
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

            quad = torch.tensor(0.0, device=resid.device, dtype=resid.dtype)
            # Optional: lightweight per-call workload summary (used by epoch_runner profiling/logging).
            # We keep this extremely cheap (just counters) so it can be enabled in long runs.
            n_groups_total = 0
            n_groups_pcg = 0
            n_groups_fallback_diag = 0
            max_rows_seen = 0
            max_nodes_seen = 0
            # Process each group
            for si, ei in zip(starts.tolist(), ends.tolist()):
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

            # Mean over observations (to match the rest of SPIDER)
            m_tot = float(max(int(resid.numel()), 1))
            # Keep the independent log(sigma) term for compatibility (with learn_noise_scale=false it's a constant anyway).
            loss_like = (quad / m_tot) + torch.log(sigma).mean()
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
    corr_error_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the Total Prior Negative Log-Probability, scaled by 1/N_total.
    
    This ensures the prior is weighted consistently with the Average Likelihood.
    """
    # Explicit enable flags (defaults preserve legacy behavior)
    event_prior_enable = bool(params.get("prior_event_enable", True))
    centroid_prior_enable = bool(params.get("prior_centroid_enable", True))
    # Runtime gates for other priors (set by the epoch runner). Defaults keep legacy behavior.
    event_runtime_enable = bool(params.get("_prior_event_runtime_enable", True))
    centroid_runtime_enable = bool(params.get("_prior_centroid_runtime_enable", True))

    # Correlated forward-model error prior p(b) (optional).
    # This uses a fixed event-graph precision Q (kNN Laplacian + q_diag I) built from MAP.
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
            q_diag = float(params.get("_corr_error_q_diag", params.get("_corr_error_event_graph_q_diag", 0.0)))
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
    
    # 2. Centroid Prior
    # If cluster information is available, apply prior to each cluster centroid individually.
    # Otherwise, fall back to global centroid.
    if (not centroid_prior_enable) or (not centroid_runtime_enable):
        log_prob_centroid = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    elif cluster_ids is not None and cluster_counts is not None:
        # Number of clusters
        K = cluster_counts.shape[0]
        # Sum dX per cluster
        # cluster_sums: (K, 4)
        cluster_sums = torch.zeros(K, 4, device=ΔX_src.device, dtype=ΔX_src.dtype)
        cluster_sums.index_add_(0, cluster_ids, ΔX_src)
        
        # Compute centroids
        # cluster_counts: (K, 1)
        cluster_centroids = cluster_sums / cluster_counts.view(-1, 1)
        
        # Apply prior to all K centroids and sum
        # Scaling: We scale by the number of events in each cluster? 
        # Or just by total events M like before?
        # Legacy behavior was M * log_prob(global_mean).
        # This is equivalent to sum_k (M_k * log_prob(centroid_k)) if we want each event to contribute.
        # Yes, let's scale each cluster's prior term by its size M_k.
        # This effectively treats the centroid constraint as having strength proportional to cluster size.
        
        # log_prob shape: (K,)
        lp_clusters = prior_centroid.log_prob(cluster_centroids)
        # Weight by cluster size M_k
        cc = cluster_counts
        if cc.ndim > 1:
            cc = cc.squeeze()
        log_prob_centroid = (lp_clusters * cc).sum()
        
    else:
        # P(mean(ΔX))
        # We apply a scaling factor of M (number of events) to match legacy behavior.
        # The total prior is divided by N later, so this term becomes (M/N) * log_prob_centroid in the average loss.
        # In the total sum (N * loss), it acts as M * log_prob_centroid.
        global_centroid = ΔX_src.mean(dim=0)
        log_prob_centroid = prior_centroid.log_prob(global_centroid).sum() * float(ΔX_src.shape[0])
    
    # 3. Noise prior removed: SPIDER uses fixed `phase_unc` only (no σ learning).
    log_prob_noise = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    
    # (shared_event_latent prior removed)

    # Total Log Prior
    total_log_prior = log_prob_events + log_prob_centroid + log_prob_noise + log_prob_b_corr
    
    base_prior_loss = -total_log_prior / float(N_total)
    return base_prior_loss


def total_loss(
    idx, y, X_src, ΔX_src, model, 
    prior_event, prior_centroid, σ_p, σ_s, 
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
        ΔX_src, prior_event, prior_centroid, σ_p, σ_s, N_total, params,
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

def prior_loss_centroid(ΔX_src, prior_centroid):
    """Legacy wrapper for centroid prior loss (scaled by M to match old behavior)."""
    global_centroid = ΔX_src.mean(dim=0)
    return -prior_centroid.log_prob(global_centroid).sum() * ΔX_src.shape[0]

def med_abs_dev(x):
    """Compute median absolute deviation."""
    return np.median(np.abs(x - np.median(x)))

def shuffle_data(x, y):
    """Shuffle data arrays together."""
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]

