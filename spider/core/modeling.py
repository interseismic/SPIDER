import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    # 4. Loss Function
    loss_type = str(params.get("likelihood", "huber")).strip().lower()
    # Likelihood-only tempering (power posterior): posterior ∝ prior * likelihood^alpha
    # This scales ONLY the likelihood term, leaving priors unchanged.
    try:
        alpha = float(params.get("_likelihood_tempering_alpha", 1.0))
        if not (alpha > 0.0) or not np.isfinite(alpha):
            alpha = 1.0
    except Exception:
        alpha = 1.0

    if loss_type in {"gaussian", "mse", "l2"}:
        # NLL ~ 0.5 * r^2
        data_loss = 0.5 * (scaled_resid ** 2)
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
    return float(alpha) * total_nll.mean()


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
    shared_event_latent_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the Total Prior Negative Log-Probability, scaled by 1/N_total.
    
    This ensures the prior is weighted consistently with the Average Likelihood.
    """
    # Explicit enable flags (defaults preserve legacy behavior)
    event_prior_enable = bool(params.get("prior_event_enable", True))
    centroid_prior_enable = bool(params.get("prior_centroid_enable", True))
    noise_prior_enable = bool(params.get("prior_noise_enable", True))
    # Runtime gates for other priors (set by the epoch runner). Defaults keep legacy behavior.
    event_runtime_enable = bool(params.get("_prior_event_runtime_enable", True))
    centroid_runtime_enable = bool(params.get("_prior_centroid_runtime_enable", True))
    noise_runtime_enable = bool(params.get("_prior_noise_runtime_enable", True))

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
    
    # 3. Noise Prior (P(sigma))
    if noise_prior_enable and noise_runtime_enable:
        log_prob_noise = _compute_noise_prior_log_prob(σ_p, σ_s, params)
    else:
        log_prob_noise = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    
    # 4. Optional uncollapsed shared-event latent prior p(b)
    # This uses a fixed event-kernel precision Q (kNN Laplacian + q_diag I) built from MAP.
    # Prior: for each station s, b_s (N x 2) ~ N(0, Σ ⊗ Kevent), implemented via precision (Σ^{-1} ⊗ Q).
    # We omit logdet constants (they don't affect gradients w.r.t. b or ΔX when Kevent is fixed).
    log_prob_b = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)
    try:
        if bool(params.get("_shared_event_latent_enabled", False)) and isinstance(shared_event_latent_b, torch.Tensor):
            mode = str(params.get("_shared_event_latent_parameterization", "full")).strip().lower()
            if mode not in {"full", "inducing_gp", "graph_gmrf"}:
                mode = "full"
            u = params.get("_shared_event_latent_u", None)
            v = params.get("_shared_event_latent_v", None)
            w = params.get("_shared_event_latent_w", None)
            q_diag = float(params.get("_shared_event_latent_q_diag_runtime", params.get("_shared_event_latent_q_diag", 0.0)))
            b = shared_event_latent_b
            if b.ndim == 3 and int(b.shape[2]) == 2:
                # Σ^{-1} for joint (P,S) coupling
                tau_ps = params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                rho = float(params.get("_shared_event_latent_rho_ps", 0.0))
                if (tau_p > 0.0) and (tau_s > 0.0) and (abs(rho) < 1.0):
                    det = (tau_p * tau_p) * (tau_s * tau_s) * (1.0 - rho * rho)
                    inv00 = (tau_s * tau_s) / det
                    inv11 = (tau_p * tau_p) / det
                    inv01 = (-rho * tau_p * tau_s) / det

                    bP = b[:, :, 0]
                    bS = b[:, :, 1]

            if mode == "inducing_gp":
                # Inducing-point GP coefficients prior (predictive-process mean):
                # For each connected component block, coefficients c (per station, per inducing point) have prior
                #   c ~ N(0, K_UU^{-1})  ⇔  log p(c) ∝ -0.5 * c^T K_UU c
                # where K_UU is the inducing kernel matrix for that component (RBF with ell_km).
                offs = params.get("_shared_event_latent_inducing_offsets", None)
                K_blocks = params.get("_shared_event_latent_inducing_K_blocks", None)
                if isinstance(offs, torch.Tensor) and isinstance(K_blocks, list) and K_blocks:
                    e00 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                    e11 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                    e01 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                    # offs length = n_blocks+1
                    nb = int(max(0, int(offs.numel()) - 1))
                    for bi in range(nb):
                        i0 = int(offs[bi].item())
                        i1 = int(offs[bi + 1].item())
                        if i1 <= i0:
                            continue
                        try:
                            K = K_blocks[bi]
                        except Exception:
                            continue
                        if not isinstance(K, torch.Tensor) or K.numel() == 0:
                            continue
                        cP = bP[:, i0:i1]
                        cS = bS[:, i0:i1]
                        # Ensure kernel on same device/dtype
                        Kt = K.to(device=b.device, dtype=b.dtype)
                        # Quadratic forms summed over stations:
                        # sum_s c_s^T K c_s = sum_s sum_i c_{s,i} ( (c_s @ K)_i )
                        KP = torch.matmul(cP, Kt)
                        KS = torch.matmul(cS, Kt)
                        e00 = e00 + (cP * KP).sum()
                        e11 = e11 + (cS * KS).sum()
                        e01 = e01 + (cP * KS).sum()
                    energy = 0.5 * (float(inv00) * e00 + float(inv11) * e11 + 2.0 * float(inv01) * e01)
                    log_prob_b = (-energy).to(dtype=ΔX_src.dtype)
            else:
                # Full Laplacian-GMRF prior over events:
                # Apply Q to a [S,N] tensor: y = q_diag*x + L_w x
                if isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor):
                    u_i = u.to(torch.int64)
                    v_i = v.to(torch.int64)
                    w_f = w.to(dtype=b.dtype)

                    def _apply_Q(xSN: torch.Tensor) -> torch.Tensor:
                        y = xSN * float(max(0.0, q_diag))
                        if int(u_i.numel()) > 0:
                            xu = xSN.index_select(1, u_i)
                            xv = xSN.index_select(1, v_i)
                            diff = xu - xv  # [S,E]
                            dw = diff * w_f.unsqueeze(0)
                            y.index_add_(1, u_i, dw)
                            y.index_add_(1, v_i, -dw)
                        return y

                    qP = _apply_Q(bP)
                    qS = _apply_Q(bS)
                    # Energy = 0.5 * sum_s [ inv00 bP·qP + inv11 bS·qS + 2 inv01 bP·qS ]
                    e00 = (bP * qP).sum()
                    e11 = (bS * qS).sum()
                    e01 = (bP * qS).sum()
                    energy = 0.5 * (float(inv00) * e00 + float(inv11) * e11 + 2.0 * float(inv01) * e01)
                    log_prob_b = (-energy).to(dtype=ΔX_src.dtype)
    except Exception:
        log_prob_b = torch.tensor(0.0, device=ΔX_src.device, dtype=ΔX_src.dtype)

    # Total Log Prior
    total_log_prior = log_prob_events + log_prob_centroid + log_prob_noise + log_prob_b
    
    base_prior_loss = -total_log_prior / float(N_total)
    return base_prior_loss


def _compute_noise_prior_log_prob(σ_p, σ_s, params) -> torch.Tensor:
    """Helper to compute log_prob for noise scales based on config."""
    prior_type = str(params.get("noise_prior", "none")).strip().lower()
    if prior_type in {"none", "", "false", "off"}:
        return torch.tensor(0.0, device=σ_p.device)

    # Ensure valid inputs
    s_p = σ_p.clamp_min(1e-12)
    s_s = σ_s.clamp_min(1e-12)
    
    # Extract params
    def get_loc_scale():
        l = params.get("noise_prior_loc", [0.0, 0.0])
        s = params.get("noise_prior_scale", [1.0, 1.0])
        l_t = torch.tensor(l, device=σ_p.device)
        s_t = torch.tensor(s, device=σ_p.device).abs().clamp_min(1e-12)
        return l_t, s_t

    loc, scale = get_loc_scale()
    weight = float(params.get("noise_prior_weight", 1.0))
    
    lp = 0.0
    
    if prior_type in {"lognormal", "log_normal"}:
        # LogNormal(loc, scale)
        # log_prob(x) = -log(x) - log(scale*sqrt(2pi)) - (log(x)-loc)^2 / (2*scale^2)
        d_p = torch.distributions.LogNormal(loc[0], scale[0])
        d_s = torch.distributions.LogNormal(loc[1], scale[1])
        lp = d_p.log_prob(s_p) + d_s.log_prob(s_s)
        
    elif prior_type in {"half_normal", "halfnormal"}:
        # HalfNormal(scale)
        d_p = torch.distributions.HalfNormal(scale[0])
        d_s = torch.distributions.HalfNormal(scale[1])
        lp = d_p.log_prob(s_p) + d_s.log_prob(s_s)
        
    elif prior_type in {"half_cauchy", "halfcauchy"}:
        # HalfCauchy(scale)
        d_p = torch.distributions.HalfCauchy(scale[0])
        d_s = torch.distributions.HalfCauchy(scale[1])
        lp = d_p.log_prob(s_p) + d_s.log_prob(s_s)
        
    return lp * weight


def total_loss(
    idx, y, X_src, ΔX_src, model, 
    prior_event, prior_centroid, σ_p, σ_s, 
    N_total, params, nuisance_delta=None,
    sigma_extra_var=None,
    cluster_ids=None, cluster_counts=None,
    event_precision_matrix=None,
    shared_event_latent_b=None,
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
        shared_event_latent_b,
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

