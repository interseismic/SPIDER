import torch
import math

from .gauge import project_event_mean_inplace

class pSGLD(torch.optim.Optimizer):
    """
    Preconditioned Stochastic Gradient Langevin Dynamics (pSGLD) optimizer.

    This optimizer combines stochastic gradient descent with Langevin dynamics
    for Bayesian sampling. It supports preconditioning using RMSprop-style
    adaptive learning rates.

    Optionally includes the Γ(θ) correction term from the original pSGLD
    paper to account for the drift induced by a state-dependent preconditioner.
    We use a diagonal, low-cost approximation suitable for RMSprop/Adam-style
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
            preconditioner (str): 'rmsprop' (default) or 'adam' to control G(θ)
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

        # Allow blockdiag_fisher (alias: matrix_ema)
        if preconditioner == "matrix_ema":
            preconditioner = "blockdiag_fisher"
        if preconditioner not in {"rmsprop", "adam", "matrix", "blockdiag_fisher"}:
            raise ValueError("preconditioner must be 'rmsprop', 'adam', 'matrix', or 'blockdiag_fisher' (alias: 'matrix_ema')")

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
                        if preconditioner in {"rmsprop", "adam"}:
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

                if preconditioning:
                    v = state['exp_avg_sq']
                    if not freeze_preconditioner:
                        v.mul_(beta).addcmul_(grad_for_precond, grad_for_precond, value=1 - beta)
                    
                    # Preconditioner choice
                    if preconditioner == 'rmsprop':
                        G = 1.0 / (eps + v.sqrt())
                    else:  # 'adam' bias-corrected second moment
                        v_hat = v / (1.0 - (beta ** state['step']))
                        G = 1.0 / (eps + v_hat.sqrt())
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

                # Gamma correction term (approximate, diagonal case)
                # Γ_i ≈ - (1-β) * g_i * sqrt(v_i) / (eps + sqrt(v_i))^2
                if include_gamma and preconditioning:
                    if preconditioner == 'rmsprop':
                        sqrt_v = v.sqrt().clamp_min(0.0)
                        denom = (eps + sqrt_v)
                        gamma = - (1.0 - beta) * raw_grad * (sqrt_v / (denom * denom))
                    else:
                        v_hat = v / (1.0 - (beta ** state['step']))
                        sqrt_v = v_hat.sqrt().clamp_min(0.0)
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
    def grad_vs_noise_stats(self) -> dict:
        """
        Compute statistics (GeoMean, Median, P10, P90, Min, Max) of the ratio:
        (update variance from minibatch gradient noise) / (injected Langevin noise variance).
        """
        eps = 1e-30
        any_noise = False
        all_ratios = []
        for group in self.param_groups:
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
                any_noise = True
            # Skip groups with no parameters or undefined lr
            if lr <= 0.0:
                continue
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
                finite_mask = torch.isfinite(ratio)
                if finite_mask.any():
                    all_ratios.append(ratio[finite_mask].flatten())

        if not any_noise or not all_ratios:
            return {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
        
        cat_ratios = torch.cat(all_ratios)
        if cat_ratios.numel() == 0:
             return {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}

        # Geometric Mean (clamped to avoid log(0))
        log_mean = torch.log(cat_ratios.clamp_min(1e-20)).mean()
        gm = math.exp(log_mean.item())
        
        # Median and other stats
        median = cat_ratios.median().item()
        # Quantiles (selection to avoid full sort)
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
        min_val = cat_ratios.min().item()
        max_val = cat_ratios.max().item()
        
        return {"gm": gm, "median": median, "p10": p10, "p90": p90, "min": min_val, "max": max_val}

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
        - rmsprop/adam: stats of diagonal G
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
