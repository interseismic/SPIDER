import torch

import math

from .gauge import project_event_mean_inplace
from spider.utils.console import info, warn


# Standardized stdout helper
def _log(*parts, section: str = "SGLD", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

class pSGLD(torch.optim.Optimizer):
    """
    Preconditioned Stochastic Gradient Langevin Dynamics (pSGLD) optimizer.

    This optimizer combines stochastic gradient descent with Langevin dynamics
    for Bayesian sampling. It supports preconditioning using RMSprop-style
    adaptive learning rates.

    Optionally includes the Γ(θ) correction term from the original pSGLD
    paper to account for the drift induced by a state-dependent preconditioner.
    We use a diagonal, low-cost approximation suitable for RMSprop-style
    diagonal metrics.

    Note: for the matrix-valued `blockdiag_fisher` preconditioner we do NOT compute
    the exact matrix divergence Γ(θ). Optionally, SPIDER can add a cheap diagonal
    proxy based on diag(EMA[ g g^T ]) for blockdiag_fisher (see group key
    `blockdiag_fisher_include_gamma_proxy`), which can reduce drift when the
    preconditioner is adapting.
    """

    def __init__(self, params, n_obs, lr=1e-3, beta=0.99, eps=1e-5,
                 preconditioning=True, add_noise=True,
                 preconditioner: str = "rmsprop",
                 include_gamma: bool = True,
                 freeze_preconditioner: bool = False):
        """
        Initialize SGLD optimizer.

        Args:
            params (iterable): Parameters to optimize
            n_obs (int): Number of observations (for noise scaling)
            lr (float): Learning rate (step size)
            beta (float): Exponential decay rate for moving average
            eps (float): Small constant for numerical stability
            preconditioning (bool): Whether to use adaptive preconditioning
            add_noise (bool): Whether to add Langevin noise
            preconditioner (str): 'rmsprop' (default) to control G(θ)
            include_gamma (bool): If True, add a diagonal approximation to the
                Γ(θ) correction term from pSGLD. This adds a small extra drift
                accounting for state-dependent G. The original paper notes Γ can
                be omitted with negligible bias when beta≈1.
            freeze_preconditioner (bool): If True, keep the preconditioner
                statistics fixed (typically enabled only during sampling when
                requested via higher-level config).
            
            Note on noise temperature:
            You can control the injected Langevin noise temperature per param-group by setting
            group['temperature'] (default 1.0). The noise standard deviation scales as sqrt(temperature).
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon parameter: {eps}")
        if n_obs <= 0:
            raise ValueError(f"Invalid n_obs: {n_obs}")

        preconditioner = str(preconditioner).strip().lower()
        # Allow blockdiag_fisher (alias: matrix_ema)
        if preconditioner == "matrix_ema":
            preconditioner = "blockdiag_fisher"
        if preconditioner in {"none", "false", ""}:
            if preconditioning:
                raise ValueError("preconditioner cannot be 'none' when preconditioning=True")
            # Keep a valid label even when preconditioning is disabled.
            preconditioner = "rmsprop"
        if preconditioner not in {"rmsprop", "matrix", "blockdiag_fisher", "monge", "shampoo"}:
            raise ValueError(
                "preconditioner must be 'rmsprop', 'matrix', 'blockdiag_fisher', 'monge', or 'shampoo' "
                f"(alias: 'matrix_ema'); got '{preconditioner}'"
            )

        defaults = dict(lr=lr, beta=beta, eps=eps, n_obs=n_obs,
                        preconditioning=preconditioning, add_noise=add_noise,
                        preconditioner=preconditioner,
                        include_gamma=include_gamma,
                        grad_ema_beta=0.99,
                        temperature=1.0,  # default temperature
                        noise_scale=1.0,  # default ramp scale
                        freeze_preconditioner=freeze_preconditioner)
        super().__init__(params, defaults)

    def set_lr(self, new_lr):
        """Set learning rate for all parameter groups."""
        for group in self.param_groups:
            group['lr'] = new_lr
    
    def set_temperature(self, new_temperature: float):
        """Set temperature for all parameter groups (scales noise std by sqrt(T))."""
        if new_temperature < 0.0 or not math.isfinite(new_temperature):
            raise ValueError(f"Invalid temperature: {new_temperature}")
        for group in self.param_groups:
            group['temperature'] = float(new_temperature)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            n_obs = group['n_obs']
            preconditioning = group['preconditioning']
            add_noise = group['add_noise']
            preconditioner = group.get('preconditioner', 'rmsprop')
            include_gamma = group.get('include_gamma', True)
            noise_scale = float(group.get('noise_scale', 1.0))
            temperature = float(group.get('temperature', 1.0))
            grad_ema_beta = float(group.get('grad_ema_beta', 0.99))
            freeze_preconditioner = group.get('freeze_preconditioner', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                raw_grad = p.grad  # minibatch-mean grad ḡ

                # --- Optional gauge projection: remove translation mode before preconditioner stats update ---
                try:
                    gauge_enable = bool(getattr(self, "_gauge_project_enable", False))
                    gauge_param = getattr(self, "_gauge_project_param", None)
                    if gauge_enable and (gauge_param is p):
                        if isinstance(raw_grad, torch.Tensor) and raw_grad.ndim == 2 and int(raw_grad.shape[1]) >= 4:
                            dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                            mode = str(getattr(self, "_gauge_project_mode", "global"))
                            cid = getattr(self, "_gauge_cluster_ids", None)
                            cc = getattr(self, "_gauge_cluster_counts", None)
                            project_event_mean_inplace(raw_grad, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                except Exception:
                    pass

                grad_for_drift = raw_grad.mul(n_obs)  # sum-loglik convention (N*ḡ)
                
                # Define ḡ for preconditioner stats (minibatch mean, optionally DD-normalized)
                dd_degree = getattr(p, "_dd_degree", None)
                if dd_degree is not None:
                    grad_for_precond = raw_grad / dd_degree
                else:
                    grad_for_precond = raw_grad

                state = self.state[p]

                # Initialize state if not present
                if "ema_g" not in state:
                    state['step'] = 0
                    if preconditioning:
                        if preconditioner == "rmsprop":
                            state.setdefault('exp_avg_sq', torch.zeros_like(p))
                        elif preconditioner == "blockdiag_fisher" and p.ndim == 2 and p.shape[1] == 4:
                            state.setdefault('exp_avg_outer', torch.zeros((p.shape[0], 4, 4), device=p.device, dtype=p.dtype))
                    state['ema_g'] = torch.zeros_like(p)
                    state['ema_g2'] = torch.zeros_like(p)

                # Ensure step exists
                if 'step' not in state:
                    state['step'] = 0
                state['step'] += 1

                # --- MATRIX PRECONDITIONER PATHS ---
                if preconditioner in {"matrix", "blockdiag_fisher"}:
                    # User-provided matrix (FIM) path
                    M_inv = state.get('matrix_inv', None)
                    L_fac = state.get('matrix_L', None)

                    # If blockdiag_fisher: optionally use static disjoint blocks (size <= max_cluster_size)
                    # specified in the param-group. This is separate from connected-component detection.
                    block_members = group.get("blockdiag_fisher_block_members", None)
                    block_sizes = group.get("blockdiag_fisher_block_sizes", None)
                    if preconditioner == "blockdiag_fisher" and (block_members is not None) and (block_sizes is not None) and (p.ndim == 2 and p.shape[1] == 4):
                        # Optimized, batched blockwise Fisher-like preconditioner:
                        # - gather block gradients once (K,S,4)
                        # - update EMA outer-products in batch (K,D,D)
                        # - apply drift/noise via triangular solves (no explicit inverses)
                        bm = block_members.to(device=p.device)
                        bs = block_sizes.to(device=p.device)
                        K = int(bm.shape[0])
                        Smax = int(bm.shape[1])
                        Dmax = int(4 * Smax)

                        # EMA storage (K,D,D)
                        Vb = state.get("block_exp_avg_outer", None)
                        if (Vb is None) or (not isinstance(Vb, torch.Tensor)) or (Vb.shape != (K, Dmax, Dmax)) or (Vb.device != p.device) or (Vb.dtype != p.dtype):
                            Vb = torch.zeros((K, Dmax, Dmax), device=p.device, dtype=p.dtype)
                            state["block_exp_avg_outer"] = Vb

                        # Update EMA gradient stats (using drift-scaled gradients) for gnoise diagnostics.
                        ema_g = state.get('ema_g', None)
                        ema_g2 = state.get('ema_g2', None)
                        if ema_g is None:
                            ema_g = torch.zeros_like(p)
                            state['ema_g'] = ema_g
                        if ema_g2 is None:
                            ema_g2 = torch.zeros_like(p)
                            state['ema_g2'] = ema_g2
                        ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                        ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))

                        # Gather block indices and grads (pad entries -> masked to 0)
                        mask = (bm >= 0)
                        idx_clamped = bm.clamp_min(0).reshape(-1)
                        g_pre = grad_for_precond.index_select(0, idx_clamped).view(K, Smax, 4)
                        g_drift = grad_for_drift.index_select(0, idx_clamped).view(K, Smax, 4)
                        g_pre = g_pre * mask.unsqueeze(-1)
                        g_drift = g_drift * mask.unsqueeze(-1)

                        gv = g_pre.reshape(K, Dmax)  # (K,D)
                        if preconditioning and (not freeze_preconditioner):
                            outer = torch.bmm(gv.unsqueeze(-1), gv.unsqueeze(-2))  # (K,D,D)
                            Vb.mul_(beta).add_(outer, alpha=(1.0 - beta))

                        # Build damped SPD matrix M = V + eps I (batched)
                        I = torch.eye(Dmax, device=p.device, dtype=p.dtype).unsqueeze(0)  # (1,D,D)
                        M = Vb + eps * I
                        M = 0.5 * (M + M.transpose(-1, -2))

                        # Cholesky (batched) with jitter escalation on failing blocks
                        L, chol_info = torch.linalg.cholesky_ex(M)  # (K,D,D), (K,)
                        fail = (chol_info != 0)
                        if torch.any(fail):
                            base_jitter = float(max(eps, 1e-10))
                            max_tries = int(group.get("blockdiag_fisher_cholesky_max_tries", 6))
                            for t in range(max_tries):
                                if not torch.any(fail):
                                    break
                                jitter = base_jitter * (10.0 ** t)
                                M = M + (jitter * I) * fail.to(M.dtype).view(-1, 1, 1)
                                M = 0.5 * (M + M.transpose(-1, -2))
                                L, chol_info = torch.linalg.cholesky_ex(M)
                                fail = (chol_info != 0)

                        # Apply drift: x = M^{-1} g using triangular solves (batched)
                        gd = g_drift.reshape(K, Dmax, 1)
                        # Good blocks: solve; bad blocks: diagonal fallback
                        y = torch.linalg.solve_triangular(L, gd, upper=False)
                        x = torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)

                        if torch.any(fail):
                            base_jitter = float(max(eps, 1e-10))
                            diag = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(base_jitter)  # (K,D)
                            inv_diag = 1.0 / diag
                            x_bad = inv_diag.unsqueeze(-1) * gd
                            x = torch.where(fail.view(-1, 1, 1), x_bad, x)

                        # Noise: corr = L^{-T} epsn has covariance M^{-1}
                        temp = max(0.0, temperature)
                        std = math.sqrt(2.0 * lr * temp) * noise_scale
                        if add_noise and std > 0.0:
                            epsn = torch.randn((K, Dmax, 1), device=p.device, dtype=p.dtype)
                            corr = torch.linalg.solve_triangular(L.transpose(-1, -2), epsn, upper=True)
                            if torch.any(fail):
                                base_jitter = float(max(eps, 1e-10))
                                diag = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(base_jitter)
                                inv_sqrt = torch.sqrt(1.0 / diag).unsqueeze(-1)
                                corr_bad = inv_sqrt * epsn
                                corr = torch.where(fail.view(-1, 1, 1), corr_bad, corr)
                        else:
                            corr = None

                        upd_vec = (lr * x)
                        if corr is not None:
                            upd_vec = upd_vec + (std * corr)

                        # Optional: cheap diagonal Γ proxy for blockdiag_fisher (NOT exact matrix divergence).
                        # Uses diag(Vb) as a second-moment proxy for each coordinate in the block.
                        if bool(group.get("blockdiag_fisher_include_gamma_proxy", False)) and preconditioning:
                            try:
                                vdiag = torch.diagonal(Vb, dim1=-2, dim2=-1)  # (K,D)
                                sqrt_v = vdiag.clamp_min(0.0).sqrt()
                                denom = (eps + sqrt_v)
                                # Gather raw minibatch-mean grad for gamma term (same convention as diagonal path)
                                g_raw = raw_grad.index_select(0, idx_clamped).view(K, Smax, 4)
                                g_raw = g_raw * mask.unsqueeze(-1)
                                g_raw_vec = g_raw.reshape(K, Dmax)
                                gamma_vec = - (1.0 - beta) * g_raw_vec * (sqrt_v / (denom * denom))
                                upd_vec = upd_vec + (lr * gamma_vec.unsqueeze(-1))
                            except Exception:
                                pass

                        upd = upd_vec.view(K, Smax, 4) * mask.unsqueeze(-1)

                        # Scatter-add updates to parameters (disjoint blocks => no overlap)
                        if mask.any():
                            idx_real = bm[mask]
                            upd_real = upd[mask]
                            p.index_add_(0, idx_real, -upd_real)

                        # Keep the existing per-event 4x4 `matrix_inv`/`matrix_L` stats path for diagnostics/logging.
                        # This is cheap (batched over N) and avoids a heavy per-block inverse just for stats.
                        V = state.get('exp_avg_outer', None)
                        if V is None:
                            V = torch.zeros((p.shape[0], 4, 4), device=p.device, dtype=p.dtype)
                            state['exp_avg_outer'] = V
                        if preconditioning and (not freeze_preconditioner):
                            g_ev = grad_for_precond  # (N,4)
                            outer_ev = torch.matmul(g_ev.unsqueeze(-1), g_ev.unsqueeze(-2))  # (N,4,4)
                            V.mul_(beta).add_(outer_ev, alpha=(1.0 - beta))
                        I4 = torch.eye(4, device=p.device, dtype=p.dtype).unsqueeze(0)
                        M4 = V + eps * I4
                        M4 = 0.5 * (M4 + M4.transpose(-1, -2))
                        L4, chol_info4 = torch.linalg.cholesky_ex(M4)
                        if torch.any(chol_info4 != 0):
                            base_jitter = float(max(eps, 1e-10))
                            max_tries4 = int(group.get("blockdiag_fisher_cholesky_max_tries", 6))
                            info_mask = (chol_info4 != 0)
                            for t in range(max_tries4):
                                if not torch.any(info_mask):
                                    break
                                jitter = base_jitter * (10.0 ** t)
                                M4 = M4 + (jitter * I4) * info_mask.to(M4.dtype).view(-1, 1, 1)
                                M4 = 0.5 * (M4 + M4.transpose(-1, -2))
                                L4, chol_info4 = torch.linalg.cholesky_ex(M4)
                                info_mask = (chol_info4 != 0)
                        # For diagnostics, approximate M^{-1} via cholesky inverse (4x4 is cheap)
                        L4_inv = torch.linalg.inv(L4)
                        L_fac4 = L4_inv.mT
                        M_inv4 = L_fac4 @ L_fac4.mT
                        state['matrix_inv'] = M_inv4
                        state['matrix_L'] = L_fac4

                        continue

                    # If blockdiag_fisher (legacy per-event 4x4): compute/update block metric from EMA of ḡ ḡ^T
                    if preconditioner == "blockdiag_fisher" and (p.ndim == 2 and p.shape[1] == 4):
                        V = state.get('exp_avg_outer', None)
                        if V is None:
                            V = torch.zeros((p.shape[0], 4, 4), device=p.device, dtype=p.dtype)
                            state['exp_avg_outer'] = V

                        if preconditioning and (not freeze_preconditioner):
                            g = grad_for_precond  # (N,4)
                            outer = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-2))  # (N,4,4)
                            V.mul_(beta).add_(outer, alpha=(1.0 - beta))

                        # Damped precision-like matrix
                        I = torch.eye(4, device=p.device, dtype=p.dtype).unsqueeze(0)  # (1,4,4)
                        M = V + eps * I  # (N,4,4)
                        # Numerical safety: enforce symmetry and add jitter to any non-PD blocks.
                        # V should be PSD in theory (EMA of outer-products), but float error / NaNs can break PD.
                        M = 0.5 * (M + M.transpose(-1, -2))

                        # Build M_inv and its factor for noise: L_fac L_fac^T = M^{-1}
                        # Use cholesky_ex so we can recover gracefully instead of crashing.
                        L_M, chol_info = torch.linalg.cholesky_ex(M)  # (N,4,4), (N,)
                        if torch.any(chol_info != 0):
                            # Adaptive jitter on failing blocks only.
                            # Start from eps, but allow escalation since eps may be extremely small.
                            base_jitter = float(max(eps, 1e-10))
                            max_tries = int(group.get("blockdiag_fisher_cholesky_max_tries", group.get("matrix_ema_cholesky_max_tries", 6)))
                            info_mask = (chol_info != 0)
                            for k in range(max_tries):
                                if not torch.any(info_mask):
                                    break
                                jitter_k = base_jitter * (10.0 ** k)
                                M = M + (jitter_k * I) * info_mask.to(M.dtype).view(-1, 1, 1)
                                M = 0.5 * (M + M.transpose(-1, -2))
                                L_M, chol_info = torch.linalg.cholesky_ex(M)
                                info_mask = (chol_info != 0)

                            if torch.any(chol_info != 0):
                                # Final fallback: diagonal-only inverse for the remaining bad blocks.
                                bad = (chol_info != 0)
                                diag = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(base_jitter)  # (N,4)
                                inv_diag = 1.0 / diag
                                # L_fac such that L_fac @ eps ~ N(0, M_inv): for diagonal M_inv, L_fac = diag(sqrt(inv_diag))
                                L_fac_fallback = torch.diag_embed(torch.sqrt(inv_diag))
                                M_inv_fallback = torch.diag_embed(inv_diag)

                                # For good blocks, compute from Cholesky.
                                good = ~bad
                                L_M_inv = torch.linalg.inv(L_M)
                                L_fac_good = L_M_inv.mT
                                M_inv_good = L_fac_good @ L_fac_good.mT

                                # Merge
                                L_fac = torch.where(good.view(-1, 1, 1), L_fac_good, L_fac_fallback)
                                M_inv = torch.where(good.view(-1, 1, 1), M_inv_good, M_inv_fallback)
                            else:
                                L_M_inv = torch.linalg.inv(L_M)
                                L_fac = L_M_inv.mT
                                M_inv = L_fac @ L_fac.mT
                        else:
                            L_M_inv = torch.linalg.inv(L_M)
                            L_fac = L_M_inv.mT
                            M_inv = L_fac @ L_fac.mT

                        state['matrix_inv'] = M_inv
                        state['matrix_L'] = L_fac

                    # Use matrix if available
                    if M_inv is not None and L_fac is not None:
                        g_u = grad_for_drift.unsqueeze(-1)                      # (N,4,1)
                        precond_grad = torch.matmul(M_inv, g_u).squeeze(-1)     # (N,4)
                        update = lr * precond_grad

                        # Optional: cheap diagonal Γ proxy for blockdiag_fisher (NOT exact matrix divergence).
                        if (preconditioner == "blockdiag_fisher") and bool(group.get("blockdiag_fisher_include_gamma_proxy", False)) and preconditioning:
                            try:
                                V = state.get('exp_avg_outer', None)
                                if isinstance(V, torch.Tensor) and (V.ndim == 3 and V.shape[-2:] == (4, 4)):
                                    vdiag = torch.diagonal(V, dim1=-2, dim2=-1)  # (N,4)
                                    sqrt_v = vdiag.clamp_min(0.0).sqrt()
                                    denom = (eps + sqrt_v)
                                    gamma = - (1.0 - beta) * raw_grad * (sqrt_v / (denom * denom))
                                    update = update + lr * gamma
                            except Exception:
                                pass

                        if add_noise:
                            temp = max(0.0, temperature)
                            std = math.sqrt(2.0 * lr * temp) * noise_scale
                            eps_noise = torch.randn_like(grad_for_drift).unsqueeze(-1)
                            corr_noise = torch.matmul(L_fac, eps_noise).squeeze(-1)
                            noise_u = std * corr_noise
                            # Optional gauge projection of injected noise.
                            try:
                                if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                                    if bool(getattr(self, "_gauge_project_apply_noise", True)):
                                        dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                                        mode = str(getattr(self, "_gauge_project_mode", "global"))
                                        cid = getattr(self, "_gauge_cluster_ids", None)
                                        cc = getattr(self, "_gauge_cluster_counts", None)
                                        project_event_mean_inplace(noise_u, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                            except Exception:
                                pass
                            update += noise_u
                        
                        # {{ edit }} Update EMA gradient stats (using drift-scaled gradients) even in matrix path
                        ema_g = state.get('ema_g', None)
                        ema_g2 = state.get('ema_g2', None)
                        if ema_g is None:
                            ema_g = torch.zeros_like(p)
                            state['ema_g'] = ema_g
                        if ema_g2 is None:
                            ema_g2 = torch.zeros_like(p)
                            state['ema_g2'] = ema_g2
                        ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                        ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))

                        # Optional gauge projection of total update (protects against drift of translation mode).
                        try:
                            if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                                dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                                mode = str(getattr(self, "_gauge_project_mode", "global"))
                                cid = getattr(self, "_gauge_cluster_ids", None)
                                cc = getattr(self, "_gauge_cluster_counts", None)
                                project_event_mean_inplace(update, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                        except Exception:
                            pass
                        p.add_(-update)
                        continue
                # ----------------------------------

                if preconditioning and preconditioner in {"monge", "shampoo"}:
                    # Non-diagonal preconditioners (Monge / Shampoo)
                    ema_g = state['ema_g']
                    ema_g2 = state['ema_g2']
                    ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                    ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))
                    state['ema_g'] = ema_g
                    state['ema_g2'] = ema_g2

                    if preconditioner == "monge":
                        alpha = float(group.get("monge_alpha", 1.0))
                        if not (math.isfinite(alpha) and alpha > 0.0):
                            alpha = 1.0
                        monge_beta = float(group.get("monge_beta", grad_ema_beta))
                        if not (math.isfinite(monge_beta) and 0.0 <= monge_beta < 1.0):
                            monge_beta = grad_ema_beta
                        # Use EMA of minibatch-mean gradient (paper's v_t / ghat_t).
                        monge_ema = state.get("monge_ema", None)
                        if monge_ema is None or not isinstance(monge_ema, torch.Tensor):
                            monge_ema = torch.zeros_like(p)
                            state["monge_ema"] = monge_ema
                        if not freeze_preconditioner:
                            monge_ema.mul_(monge_beta).add_(grad_for_precond, alpha=(1.0 - monge_beta))
                        # Monge rank-1 vector u = alpha * v_t
                        u = monge_ema.mul(alpha)
                        u_dot_u = float((u * u).sum().item())
                        denom = 1.0 + u_dot_u
                        if u_dot_u > 0.0:
                            u_dot_g = (u * grad_for_drift).sum()
                            precond_grad = grad_for_drift - u * (u_dot_g / denom)
                            # Diagonal proxy for diagnostics
                            diag_g = (1.0 - (u * u) / denom).clamp_min(0.0)
                            state["precond_diag"] = diag_g
                        else:
                            precond_grad = grad_for_drift
                            state["precond_diag"] = torch.ones_like(p)

                        update = lr * precond_grad
                        if add_noise:
                            temp = max(0.0, temperature)
                            std = math.sqrt(2.0 * lr * temp) * noise_scale
                            z = torch.randn_like(p) * std
                            if u_dot_u > 0.0:
                                c = (1.0 - (1.0 / math.sqrt(1.0 + u_dot_u))) / max(u_dot_u, 1e-12)
                                u_dot_z = (u * z).sum()
                                z = z - u * (u_dot_z * c)
                            update = update + z
                        # One-time early-step diagnostic print to compare scaling vs RMSprop.
                        if int(state.get('step', 0)) <= 3 and (p is group.get("params", [None])[0]):
                            try:
                                var_g = (ema_g2 - ema_g * ema_g).clamp_min(0.0)
                                diag_g = state.get("precond_diag", torch.ones_like(p))
                                var_noise = (2.0 * lr * max(temperature, 0.0)) * (noise_scale * noise_scale) * diag_g
                                num = (lr * lr) * (diag_g * diag_g) * var_g
                                ratio = (num / var_noise.clamp_min(1e-30)).clamp_min(1e-30)
                                g_pre = grad_for_precond
                                g_drift = grad_for_drift
                                _log(
                                    "[monge_debug]"
                                    f" step={int(state.get('step', 0))}"
                                    f" n_obs={int(n_obs)}"
                                    f" lr={float(lr):.3e}"
                                    f" alpha={float(alpha):.3e}"
                                    f" monge_beta={float(monge_beta):.3e}"
                                    f" u_dot_u={float(u_dot_u):.3e}"
                                    f" monge_ema_norm={float(monge_ema.norm().item()):.3e}"
                                    f" g_pre_norm={float(g_pre.norm().item()):.3e}"
                                    f" g_drift_norm={float(g_drift.norm().item()):.3e}"
                                    f" diag_g_med={float(diag_g.median().item()):.3e}"
                                    f" var_g_med={float(var_g.median().item()):.3e}"
                                    f" var_noise_med={float(var_noise.median().item()):.3e}"
                                    f" ratio_med={float(ratio.median().item()):.3e}"
                                )
                            except Exception:
                                pass
                        p.add_(-update)
                        continue

                    # Shampoo (Kronecker) for small 2D tensors
                    if preconditioner == "shampoo":
                        if p.ndim != 2:
                            # Fallback to RMSprop for unsupported shapes
                            v = state['exp_avg_sq']
                            if not freeze_preconditioner:
                                v.mul_(beta).addcmul_(grad_for_precond, grad_for_precond, value=1 - beta)
                            G = 1.0 / (eps + v.sqrt())
                        else:
                            n0, n1 = int(p.shape[0]), int(p.shape[1])
                            max_dim = int(group.get("shampoo_max_dim", 512))
                            if n0 > max_dim or n1 > max_dim:
                                v = state['exp_avg_sq']
                                if not freeze_preconditioner:
                                    v.mul_(beta).addcmul_(grad_for_precond, grad_for_precond, value=1 - beta)
                                G = 1.0 / (eps + v.sqrt())
                            else:
                                beta_s = float(group.get("shampoo_beta", beta))
                                eps_s = float(group.get("shampoo_eps", eps))
                                update_every = int(group.get("shampoo_update_every", 10))
                                L = state.get("shampoo_L", None)
                                R = state.get("shampoo_R", None)
                                if L is None or L.shape != (n0, n0):
                                    L = torch.zeros((n0, n0), device=p.device, dtype=p.dtype)
                                    state["shampoo_L"] = L
                                if R is None or R.shape != (n1, n1):
                                    R = torch.zeros((n1, n1), device=p.device, dtype=p.dtype)
                                    state["shampoo_R"] = R
                                if not freeze_preconditioner:
                                    L.mul_(beta_s).add_(grad_for_precond @ grad_for_precond.mT, alpha=(1.0 - beta_s))
                                    R.mul_(beta_s).add_(grad_for_precond.mT @ grad_for_precond, alpha=(1.0 - beta_s))
                                # Cache inverse sqrt
                                if (state['step'] % update_every) == 0 or ("shampoo_L_inv_sqrt" not in state):
                                    eye0 = torch.eye(n0, device=p.device, dtype=p.dtype)
                                    eye1 = torch.eye(n1, device=p.device, dtype=p.dtype)
                                    evals0, evecs0 = torch.linalg.eigh(L + eps_s * eye0)
                                    evals1, evecs1 = torch.linalg.eigh(R + eps_s * eye1)
                                    inv0 = evecs0 @ torch.diag(1.0 / torch.sqrt(evals0.clamp_min(0.0))) @ evecs0.mT
                                    inv1 = evecs1 @ torch.diag(1.0 / torch.sqrt(evals1.clamp_min(0.0))) @ evecs1.mT
                                    state["shampoo_L_inv_sqrt"] = inv0
                                    state["shampoo_R_inv_sqrt"] = inv1
                                inv0 = state.get("shampoo_L_inv_sqrt")
                                inv1 = state.get("shampoo_R_inv_sqrt")
                                if inv0 is None or inv1 is None:
                                    G = torch.ones_like(p)
                                else:
                                    precond_grad = inv0 @ grad_for_drift @ inv1
                                    update = lr * precond_grad
                                    if add_noise:
                                        temp = max(0.0, temperature)
                                        std = math.sqrt(2.0 * lr * temp) * noise_scale
                                        z = torch.randn_like(p) * std
                                        z = inv0 @ z @ inv1
                                        update = update + z
                                    # Diagonal proxy for diagnostics
                                    diag_g = (inv0.diagonal().unsqueeze(1) * inv1.diagonal().unsqueeze(0)).clamp_min(0.0)
                                    state["precond_diag"] = diag_g
                                    p.add_(-update)
                                    continue

                    # Shampoo fallback uses diagonal G computed above
                    if preconditioning:
                        update = lr * (G * grad_for_drift)
                        if add_noise:
                            temp = max(0.0, temperature)
                            std = math.sqrt(2.0 * lr * temp) * noise_scale
                            noise = torch.randn_like(p) * std * G.sqrt()
                            update = update + noise
                        p.add_(-update)
                        continue

                if preconditioning:
                    v = state['exp_avg_sq']
                    if not freeze_preconditioner:
                        v.mul_(beta).addcmul_(grad_for_precond, grad_for_precond, value=1 - beta)
                    
                    # Preconditioner choice (RMSprop)
                    G = 1.0 / (eps + v.sqrt())
                else:
                    G = torch.ones_like(p)

                # Update EMA gradient stats (using drift-scaled gradients)
                ema_g = state['ema_g']
                ema_g2 = state['ema_g2']
                # ema_g = β * ema_g + (1-β) * g
                ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                # ema_g2 = β * ema_g2 + (1-β) * g^2
                ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))
                state['ema_g'] = ema_g
                state['ema_g2'] = ema_g2

                # Compute update step
                update = lr * G * grad_for_drift
                # One-time early-step diagnostic print for RMSprop-scale comparisons.
                if int(state.get('step', 0)) <= 3 and (p is group.get("params", [None])[0]):
                    try:
                        var_g = (ema_g2 - ema_g * ema_g).clamp_min(0.0)
                        var_noise = (2.0 * lr * max(temperature, 0.0)) * (noise_scale * noise_scale) * G
                        num = (lr * lr) * (G * G) * var_g
                        ratio = (num / var_noise.clamp_min(1e-30)).clamp_min(1e-30)
                        _log(
                            "[rmsprop_debug]"
                            f" step={int(state.get('step', 0))}"
                            f" n_obs={int(n_obs)}"
                            f" lr={float(lr):.3e}"
                            f" beta={float(beta):.3e}"
                            f" eps={float(eps):.3e}"
                            f" g_pre_norm={float(grad_for_precond.norm().item()):.3e}"
                            f" g_drift_norm={float(grad_for_drift.norm().item()):.3e}"
                            f" v_med={float(v.median().item()):.3e}"
                            f" G_med={float(G.median().item()):.3e}"
                            f" var_g_med={float(var_g.median().item()):.3e}"
                            f" var_noise_med={float(var_noise.median().item()):.3e}"
                            f" ratio_med={float(ratio.median().item()):.3e}"
                        )
                    except Exception:
                        pass

                # Gamma correction term (approximate, diagonal case)
                # Γ_i ≈ - (1-β) * g_i * sqrt(v_i) / (eps + sqrt(v_i))^2
                if include_gamma and preconditioning:
                    sqrt_v = v.sqrt().clamp_min(0.0)
                    denom = (eps + sqrt_v)
                    gamma = - (1.0 - beta) * raw_grad * (sqrt_v / (denom * denom))
                    update = update + lr * gamma

                # Add Langevin noise if requested:
                # std = sqrt(2 * lr * temperature) * sqrt(G) * noise_scale
                if add_noise:
                    # Guard temperature
                    temp = max(0.0, temperature)
                    std = math.sqrt(2.0 * lr * temp) * noise_scale
                    noise = torch.randn_like(p) * std * G.sqrt()
                    # Optional gauge projection of injected noise.
                    try:
                        if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                            if bool(getattr(self, "_gauge_project_apply_noise", True)):
                                dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                                mode = str(getattr(self, "_gauge_project_mode", "global"))
                                cid = getattr(self, "_gauge_cluster_ids", None)
                                cc = getattr(self, "_gauge_cluster_counts", None)
                                project_event_mean_inplace(noise, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                    except Exception:
                        pass
                    update += noise

                # Update parameter
                # Optional gauge projection of total update.
                try:
                    if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                        dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                        mode = str(getattr(self, "_gauge_project_mode", "global"))
                        cid = getattr(self, "_gauge_cluster_ids", None)
                        cc = getattr(self, "_gauge_cluster_counts", None)
                        project_event_mean_inplace(update, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                except Exception:
                    pass
                p.add_(-update)

        return loss

    @torch.no_grad()
    def grad_vs_noise_geomean(self) -> float:
        stats = self.grad_vs_noise_stats()
        return stats.get("gm", float("nan"))

    @torch.no_grad()
    def grad_vs_noise_stats(self) -> dict:
        """
        Compute statistics (GeoMean, Median, P10, P90, Min, Max) of the ratio:
        (update variance from minibatch gradient noise) / (injected Langevin noise variance).
        """
        eps = 1e-30

        def _summarize(cat_ratios: torch.Tensor) -> dict:
            if cat_ratios is None or (not isinstance(cat_ratios, torch.Tensor)) or cat_ratios.numel() == 0:
                return {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            # Geometric Mean (clamped to avoid log(0))
            log_mean = torch.log(cat_ratios.clamp_min(1e-20)).mean()
            gm = math.exp(log_mean.item())
            median = cat_ratios.median().item()
            try:
                x = cat_ratios
                n = int(x.numel())
                def _k(q: float) -> int:
                    return int(max(1, min(n, round(q * (n - 1)) + 1)))
                p10 = float(torch.kthvalue(x, _k(0.10)).values.item())
                p90 = float(torch.kthvalue(x, _k(0.90)).values.item())
            except Exception:
                p10 = float("nan")
                p90 = float("nan")
            return {
                "gm": float(gm),
                "median": float(median),
                "p10": float(p10),
                "p90": float(p90),
                "min": float(cat_ratios.min().item()),
                "max": float(cat_ratios.max().item()),
            }

        any_noise_global = False
        all_ratios = []
        all_dt_ratios = []
        per_group = []

        for gi, group in enumerate(self.param_groups):
            lr = float(group.get('lr', 0.0))
            beta = float(group.get('beta', 0.99))
            eps_g = float(group.get('eps', 1e-5))
            preconditioning = bool(group.get('preconditioning', True))
            preconditioner = str(group.get('preconditioner', 'rmsprop'))
            add_noise = bool(group.get('add_noise', True))
            noise_scale = float(group.get('noise_scale', 1.0))
            temperature = float(group.get('temperature', 1.0))
            # If this group injects any noise, flag it
            if add_noise and noise_scale > 0.0 and temperature > 0.0:
                any_noise_global = True
            # Skip groups with no parameters or undefined lr
            if lr <= 0.0:
                per_group.append({
                    "group_name": str(group.get("group_name", f"group{gi}")),
                    **_summarize(torch.tensor([], device=self.param_groups[0]["params"][0].device) if (self.param_groups and self.param_groups[0].get("params")) else torch.tensor([])),
                })
                continue
            any_noise_group = bool(add_noise and noise_scale > 0.0 and temperature > 0.0)
            group_ratios = []
            group_dt_ratios = []
            for p in group['params']:
                if p is None:
                    continue
                state = self.state[p]
                if 'ema_g' not in state or 'ema_g2' not in state:
                    continue
                ema_g = state['ema_g']
                ema_g2 = state['ema_g2']
                # var of drift-scaled grad (non-negative clamp)
                var_g = (ema_g2 - ema_g * ema_g).clamp_min(0.0)
                # Preconditioner metric (diagonal proxy in update space)
                if preconditioning:
                    precond = str(preconditioner).lower()
                    if precond in {"matrix", "blockdiag_fisher", "matrix_ema"}:
                        M_inv = state.get('matrix_inv', None)
                        if M_inv is not None and p.ndim == 2 and p.shape[1] == 4:
                            # Use diag of M^{-1} as a per-parameter variance proxy
                            G = torch.diagonal(M_inv, dim1=-2, dim2=-1)  # (N,4)
                        else:
                            G = torch.ones_like(ema_g)
                    elif precond in {"monge", "shampoo"}:
                        G = state.get("precond_diag", None)
                        if G is None or not isinstance(G, torch.Tensor):
                            G = torch.ones_like(ema_g)
                    else:
                        v = state.get('exp_avg_sq', None)
                        if v is None:
                            G = torch.ones_like(ema_g)
                        else:
                            if precond == 'rmsprop':
                                G = 1.0 / (eps_g + v.sqrt())
                            else:
                                step = int(state.get('step', 1))
                                v_hat = v / (1.0 - (beta ** max(step, 1)))
                                G = 1.0 / (eps_g + v_hat.sqrt())
                else:
                    G = torch.ones_like(ema_g)
                # Langevin noise variance per parameter in update space
                var_noise = (2.0 * lr * max(temperature, 0.0)) * (noise_scale * noise_scale) * G
                denom = torch.where(var_noise > 0.0, var_noise, var_noise.new_full(var_noise.shape, eps))
                # Gradient noise-induced update variance
                num = (lr * lr) * (G * G) * var_g
                ratio = (num / denom).clamp_min(1e-30)
                # Track dt column (index 3) for hypocenter-like tensors (N,4).
                if ratio.ndim == 2 and int(ratio.shape[1]) == 4:
                    spatial = ratio[:, :3]
                    spatial_mask = torch.isfinite(spatial)
                    if spatial_mask.any():
                        rr = spatial[spatial_mask].flatten()
                        all_ratios.append(rr)
                        group_ratios.append(rr)
                    dt_ratio = ratio[:, 3]
                    dt_mask = torch.isfinite(dt_ratio)
                    if dt_mask.any():
                        rdt = dt_ratio[dt_mask].flatten()
                        all_dt_ratios.append(rdt)
                        group_dt_ratios.append(rdt)
                else:
                    finite_mask = torch.isfinite(ratio)
                    if finite_mask.any():
                        rr = ratio[finite_mask].flatten()
                        all_ratios.append(rr)
                        group_ratios.append(rr)

            # Per-group summary (only meaningful when this group actually injects noise)
            if any_noise_group and group_ratios:
                gcat = torch.cat(group_ratios)
                gstats = _summarize(gcat)
            else:
                gstats = {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            if any_noise_group and group_dt_ratios:
                gdt = torch.cat(group_dt_ratios)
                gdt_stats = _summarize(gdt)
                gstats["dt_gm"] = float(gdt_stats.get("gm", 0.0))
                gstats["dt_median"] = float(gdt_stats.get("median", 0.0))
            else:
                gstats["dt_gm"] = 0.0
                gstats["dt_median"] = 0.0
            gstats["group_name"] = str(group.get("group_name", f"group{gi}"))
            per_group.append(gstats)

        # Global summary
        if (not any_noise_global) or (not all_ratios):
            out = {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0, "dt_gm": 0.0, "dt_median": 0.0}
            out["per_group"] = per_group
            return out

        cat_ratios = torch.cat(all_ratios) if all_ratios else torch.tensor([])
        out = _summarize(cat_ratios)
        if all_dt_ratios:
            dt_cat = torch.cat(all_dt_ratios)
            dt_stats = _summarize(dt_cat)
            out["dt_gm"] = float(dt_stats.get("gm", 0.0))
            out["dt_median"] = float(dt_stats.get("median", 0.0))
        else:
            out["dt_gm"] = 0.0
            out["dt_median"] = 0.0
        out["per_group"] = per_group
        return out

    @torch.no_grad()
    def temperature_stats(self) -> dict:
        """
        Placeholder for temperature statistics. SGLD does not have a stationary
        momentum distribution, so thermal energy diagnostics are less direct
        than in SGHMC.
        """
        return {
            "msq_gm": float("nan"), "msq_median": float("nan"),
            "var_gm": float("nan"), "var_median": float("nan"),
            "msq_median_over_target": float("nan"),
            "var_median_over_target": float("nan")
        }


class AdaptiveDriftSGLDAdam(torch.optim.Optimizer):
    """
    SGLD with adaptive drift (Adam-variant) from:
      Kim, Song, Liang (2020) "Stochastic Gradient Langevin Dynamics Algorithms with Adaptive Drifts".

    Update:
      theta <- theta - lr * (g + a * A) + sqrt(2 * lr * T) * noise
    where A is an Adam-style normalized momentum term:
      m <- beta1 * m + (1 - beta1) * g_adapt
      v <- beta2 * v + (1 - beta2) * g_adapt^2
      A = m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params,
        *,
        n_obs: int,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        drift_scale: float = 1.0,
        add_noise: bool = True,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if not 0.0 < eps_adam or not math.isfinite(float(eps_adam)):
            raise ValueError(f"Invalid eps_adam: {eps_adam}")
        if not 0.0 <= drift_scale or not math.isfinite(float(drift_scale)):
            raise ValueError(f"Invalid drift_scale: {drift_scale}")
        if n_obs <= 0:
            raise ValueError(f"Invalid n_obs: {n_obs}")

        defaults = dict(
            lr=float(lr),
            beta1=float(beta1),
            beta2=float(beta2),
            eps_adam=float(eps_adam),
            drift_scale=float(drift_scale),
            n_obs=int(n_obs),
            add_noise=bool(add_noise),
            temperature=1.0,
            noise_scale=1.0,
            preconditioning=False,
            preconditioner="none",
            freeze_preconditioner=False,
            # Compatibility keys for logging/helpers
            beta=float(beta2),
            eps=float(eps_adam),
            grad_ema_beta=0.99,
        )
        super().__init__(params, defaults)

    def set_lr(self, new_lr: float) -> None:
        for g in self.param_groups:
            g["lr"] = float(new_lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            beta1 = float(group.get("beta1", 0.9))
            beta2 = float(group.get("beta2", 0.999))
            eps_adam = float(group.get("eps_adam", 1e-8))
            drift_scale = float(group.get("drift_scale", 1.0))
            n_obs = int(group.get("n_obs", 1))
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            grad_ema_beta = float(group.get("grad_ema_beta", 0.99))

            if not math.isfinite(noise_scale) or noise_scale < 0.0:
                noise_scale = 0.0

            for p in group["params"]:
                if p is None or p.grad is None:
                    continue

                grad_mean = p.grad.data
                # Optional gauge projection (remove translation mode before adaptation).
                try:
                    gauge_enable = bool(getattr(self, "_gauge_project_enable", False))
                    gauge_param = getattr(self, "_gauge_project_param", None)
                    if gauge_enable and (gauge_param is p):
                        if isinstance(grad_mean, torch.Tensor) and grad_mean.ndim == 2 and int(grad_mean.shape[1]) >= 4:
                            dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                            mode = str(getattr(self, "_gauge_project_mode", "global"))
                            cid = getattr(self, "_gauge_cluster_ids", None)
                            cc = getattr(self, "_gauge_cluster_counts", None)
                            project_event_mean_inplace(grad_mean, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                except Exception:
                    pass

                # Algorithm 1 uses mean gradient g_t (no N scaling).
                drift_grad = grad_mean

                dd_degree = getattr(p, "_dd_degree", None)
                if dd_degree is not None:
                    try:
                        adapt_grad = grad_mean / dd_degree
                    except Exception:
                        adapt_grad = grad_mean
                else:
                    adapt_grad = grad_mean

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["ema_g"] = torch.zeros_like(p)
                    state["ema_g2"] = torch.zeros_like(p)

                state["step"] += 1
                t = int(state["step"])
                m = state.get("m", None)
                v = state.get("v", None)
                if m is None or not isinstance(m, torch.Tensor) or m.shape != p.shape:
                    m = torch.zeros_like(p)
                    state["m"] = m
                if v is None or not isinstance(v, torch.Tensor) or v.shape != p.shape:
                    v = torch.zeros_like(p)
                    state["v"] = v

                m.mul_(beta1).add_(adapt_grad, alpha=(1.0 - beta1))
                v.mul_(beta2).addcmul_(adapt_grad, adapt_grad, value=(1.0 - beta2))

                # Algorithm 1 uses uncorrected moments and lambda inside the sqrt.
                adapt = m / (v.add(eps_adam).sqrt())

                # Update EMA stats (drift-scaled gradients) for diagnostics.
                ema_g = state.get("ema_g", None)
                ema_g2 = state.get("ema_g2", None)
                if ema_g is None:
                    ema_g = torch.zeros_like(p)
                    state["ema_g"] = ema_g
                if ema_g2 is None:
                    ema_g2 = torch.zeros_like(p)
                    state["ema_g2"] = ema_g2
                ema_g.mul_(grad_ema_beta).add_(drift_grad, alpha=(1.0 - grad_ema_beta))
                ema_g2.mul_(grad_ema_beta).addcmul_(drift_grad, drift_grad, value=(1.0 - grad_ema_beta))

                # Total drift = (lr/2) * (g + a*A)
                update = 0.5 * lr * (drift_grad + drift_scale * adapt)

                if add_noise and noise_scale > 0.0:
                    # Algorithm 1: noise ~ sqrt(lr / N)
                    std = math.sqrt(lr / float(max(1, n_obs))) * noise_scale
                    noise = torch.randn_like(p) * std
                    # Optional gauge projection of injected noise.
                    try:
                        if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                            if bool(getattr(self, "_gauge_project_apply_noise", True)):
                                dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                                mode = str(getattr(self, "_gauge_project_mode", "global"))
                                cid = getattr(self, "_gauge_cluster_ids", None)
                                cc = getattr(self, "_gauge_cluster_counts", None)
                                project_event_mean_inplace(noise, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                    except Exception:
                        pass
                    update = update + noise

                # Optional gauge projection of total update.
                try:
                    if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                        dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                        mode = str(getattr(self, "_gauge_project_mode", "global"))
                        cid = getattr(self, "_gauge_cluster_ids", None)
                        cc = getattr(self, "_gauge_cluster_counts", None)
                        project_event_mean_inplace(update, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                except Exception:
                    pass

                p.add_(-update)

        return loss

    @torch.no_grad()
    def grad_vs_noise_stats(self) -> dict:
        """
        Compute statistics (GeoMean, Median, P10, P90, Min, Max) of the ratio:
        (update variance from minibatch gradient noise) / (injected Langevin noise variance).
        """
        eps = 1e-30

        def _summarize(cat_ratios: torch.Tensor) -> dict:
            if cat_ratios is None or (not isinstance(cat_ratios, torch.Tensor)) or cat_ratios.numel() == 0:
                return {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            log_mean = torch.log(cat_ratios.clamp_min(1e-20)).mean()
            gm = math.exp(log_mean.item())
            median = cat_ratios.median().item()
            try:
                x = cat_ratios
                n = int(x.numel())
                def _k(q: float) -> int:
                    return int(max(1, min(n, round(q * (n - 1)) + 1)))
                p10 = float(torch.kthvalue(x, _k(0.10)).values.item())
                p90 = float(torch.kthvalue(x, _k(0.90)).values.item())
            except Exception:
                p10 = float("nan")
                p90 = float("nan")
            return {
                "gm": float(gm),
                "median": float(median),
                "p10": float(p10),
                "p90": float(p90),
                "min": float(cat_ratios.min().item()),
                "max": float(cat_ratios.max().item()),
            }

        any_noise_global = False
        all_ratios = []
        all_dt_ratios = []
        per_group = []
        all_var_g = []
        all_var_noise = []

        for gi, group in enumerate(self.param_groups):
            lr = float(group.get("lr", 0.0))
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            temperature = float(group.get("temperature", 1.0))
            if add_noise and noise_scale > 0.0:
                any_noise_global = True
            if lr <= 0.0:
                per_group.append({
                    "group_name": str(group.get("group_name", f"group{gi}")),
                    **_summarize(torch.tensor([], device=self.param_groups[0]["params"][0].device) if (self.param_groups and self.param_groups[0].get("params")) else torch.tensor([])),
                })
                continue
            any_noise_group = bool(add_noise and noise_scale > 0.0)
            group_ratios = []
            group_dt_ratios = []
            for p in group.get("params", []):
                if p is None:
                    continue
                state = self.state.get(p, {})
                if "ema_g" not in state or "ema_g2" not in state:
                    continue
                ema_g = state["ema_g"]
                ema_g2 = state["ema_g2"]
                var_g = (ema_g2 - ema_g * ema_g).clamp_min(0.0)
                n_obs = int(group.get("n_obs", 1))
                var_noise = (lr / float(max(1, n_obs))) * (noise_scale * noise_scale)
                denom = var_g.new_full(var_g.shape, max(var_noise, eps))
                num = (lr * lr) * var_g
                ratio = (num / denom).clamp_min(1e-30)
                # Track dt column (index 3) for hypocenter-like tensors (N,4).
                if ratio.ndim == 2 and int(ratio.shape[1]) == 4:
                    spatial = ratio[:, :3]
                    spatial_mask = torch.isfinite(spatial)
                    if spatial_mask.any():
                        rr = spatial[spatial_mask].flatten()
                        all_ratios.append(rr)
                        group_ratios.append(rr)
                        all_var_g.append(var_g[:, :3][spatial_mask].flatten())
                        all_var_noise.append(var_noise.new_full(var_g[:, :3][spatial_mask].shape, var_noise).flatten())
                    dt_ratio = ratio[:, 3]
                    dt_mask = torch.isfinite(dt_ratio)
                    if dt_mask.any():
                        rdt = dt_ratio[dt_mask].flatten()
                        all_dt_ratios.append(rdt)
                        group_dt_ratios.append(rdt)
                else:
                    finite_mask = torch.isfinite(ratio)
                    if finite_mask.any():
                        rr = ratio[finite_mask].flatten()
                        all_ratios.append(rr)
                        group_ratios.append(rr)
                        all_var_g.append(var_g[finite_mask].flatten())
                        all_var_noise.append(var_noise.new_full(var_g[finite_mask].shape, var_noise).flatten())

            if any_noise_group and group_ratios:
                gcat = torch.cat(group_ratios)
                gstats = _summarize(gcat)
            else:
                gstats = {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            if any_noise_group and group_dt_ratios:
                gdt = torch.cat(group_dt_ratios)
                gdt_stats = _summarize(gdt)
                gstats["dt_gm"] = float(gdt_stats.get("gm", 0.0))
                gstats["dt_median"] = float(gdt_stats.get("median", 0.0))
            else:
                gstats["dt_gm"] = 0.0
                gstats["dt_median"] = 0.0
            gstats["group_name"] = str(group.get("group_name", f"group{gi}"))
            per_group.append(gstats)

        if (not any_noise_global) or (not all_ratios):
            out = {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0, "dt_gm": 0.0, "dt_median": 0.0}
            out["per_group"] = per_group
            return out

        cat_ratios = torch.cat(all_ratios) if all_ratios else torch.tensor([])
        out = _summarize(cat_ratios)
        if all_dt_ratios:
            dt_cat = torch.cat(all_dt_ratios)
            dt_stats = _summarize(dt_cat)
            out["dt_gm"] = float(dt_stats.get("gm", 0.0))
            out["dt_median"] = float(dt_stats.get("median", 0.0))
        else:
            out["dt_gm"] = 0.0
            out["dt_median"] = 0.0
        try:
            if all_var_g:
                out["var_g_median"] = float(torch.cat(all_var_g).median().item())
            if all_var_noise:
                out["var_noise_median"] = float(torch.cat(all_var_noise).median().item())
        except Exception:
            pass
        out["per_group"] = per_group
        return out


    @torch.no_grad()
    def grad_vs_noise_geomean(self) -> float:
        """
        Legacy wrapper for grad_vs_noise_stats['gm']
        """
        stats = self.grad_vs_noise_stats()
        return stats["gm"]

    @torch.no_grad()
    def preconditioner_stats(self):
        """
        Return summary stats of the preconditioner (dict):
          {min, p25, median, p75, max}
        - rmsprop: stats of diagonal G
        - matrix/blockdiag_fisher: stats of diagonal entries of M^{-1} (from state['matrix_inv'])
        """
        try:
            def _five_num(x: torch.Tensor) -> dict:
                x = x.detach()
                x = x.reshape(-1)
                # Filter non-finite (shouldn't happen, but keeps logs robust)
                if x.numel() == 0:
                    return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
                finite = torch.isfinite(x)
                if torch.any(finite):
                    x = x[finite]
                if x.numel() == 0:
                    return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}

                n = int(x.numel())
                # kthvalue is 1-indexed
                def _k(q: float) -> int:
                    return int(max(1, min(n, round(q * (n - 1)) + 1)))

                # Use selection (kthvalue) to avoid full sort
                p25 = float(torch.kthvalue(x, _k(0.25)).values.item())
                med = float(torch.kthvalue(x, _k(0.50)).values.item())
                p75 = float(torch.kthvalue(x, _k(0.75)).values.item())
                return {
                    "min": float(x.min().item()),
                    "p25": p25,
                    "median": med,
                    "p75": p75,
                    "max": float(x.max().item()),
                }

            if len(self.param_groups) == 0:
                return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
            group = self.param_groups[0]
            preconditioning = bool(group.get('preconditioning', True))
            if not preconditioning:
                return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}

            eps = float(group.get('eps', 1e-5))
            beta = float(group.get('beta', 0.99))
            preconditioner = str(group.get('preconditioner', 'rmsprop')).lower()

            # Find first parameter with state
            p = None
            for q in group.get('params', []):
                if q is not None:
                    p = q
                    break
            if p is None:
                return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}

            state = self.state.get(p, {})

            # {{ edit }} matrix stats
            if preconditioner in {"matrix", "blockdiag_fisher", "matrix_ema"}:
                M_inv = state.get('matrix_inv', None)
                if M_inv is None:
                    return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
                # M_inv: (N,4,4) for dX_src
                diagonals = torch.diagonal(M_inv, dim1=-2, dim2=-1)
                return _five_num(diagonals)

            # existing diagonal stats
            v = state.get('exp_avg_sq', None)
            if v is None:
                G = torch.ones_like(p)
            else:
                if preconditioner == 'rmsprop':
                    G = 1.0 / (eps + v.sqrt())
                else:
                    step = int(state.get('step', 1))
                    v_hat = v / (1.0 - (beta ** max(step, 1)))
                    G = 1.0 / (eps + v_hat.sqrt())
            return _five_num(G)
        except Exception:
            return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}


@torch.no_grad()
def transplant_v_from_adam(adam_opt, sgld_opt):
    # copy Adam's exp_avg_sq into sampler state['exp_avg_sq'] (same shape, same dtype)
    adam_state = adam_opt.state
    sgld_state = sgld_opt.state
    for group in sgld_opt.param_groups:
        for p in group['params']:
            if p is None:
                continue

            # {{ edit }} Ensure state dict exists
            st = sgld_state[p]

            # {{ edit }} Ensure required keys exist for pSGLD.step()
            st.setdefault('step', 0)
            st.setdefault('ema_g', torch.zeros_like(p))
            st.setdefault('ema_g2', torch.zeros_like(p))

            # {{ edit }} Initialize RMSProp/Adam exp_avg_sq if needed
            if p in adam_state and 'exp_avg_sq' in adam_state[p]:
                v_src = adam_state[p]['exp_avg_sq']
                st['exp_avg_sq'] = v_src.detach().clone()
            else:
                st.setdefault('exp_avg_sq', torch.zeros_like(p))

            # {{ edit }} If using blockdiag_fisher (alias: matrix_ema) on (N,4), ensure exp_avg_outer exists
            precond = str(group.get('preconditioner', 'rmsprop')).lower()
            preconditioning = bool(group.get('preconditioning', True))
            if preconditioning and precond in {"blockdiag_fisher", "matrix_ema"} and (p.ndim == 2 and p.shape[1] == 4):
                st.setdefault('exp_avg_outer', torch.zeros((p.shape[0], 4, 4), device=p.device, dtype=p.dtype))
                # If block partitioning is active, initialize blockwise EMA storage too.
                bm = group.get("blockdiag_fisher_block_members", None)
                bs = group.get("blockdiag_fisher_block_sizes", None)
                if bm is not None and bs is not None:
                    try:
                        K = int(bm.shape[0])
                        Smax = int(bm.shape[1])
                        Dmax = int(4 * Smax)
                        st.setdefault("block_exp_avg_outer", torch.zeros((K, Dmax, Dmax), device=p.device, dtype=p.dtype))
                    except Exception:
                        pass

@torch.no_grad()
def heartbeat_poststep_from(prev_params, params, rel_floor_scale=1e-3, abs_floor=1e-8):
    # Deprecated diagnostics; keeping signature for compatibility if imported.
    return
