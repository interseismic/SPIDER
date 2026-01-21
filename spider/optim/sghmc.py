import math
from typing import Optional
import torch

from .gauge import project_event_mean_inplace


class SGHMC(torch.optim.Optimizer):
    """
    Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) with diagonal RMSprop
    preconditioning (G). Maintains per-parameter momentum and second-moment
    statistics for G, and injects noise consistent with the friction term.

    Discretization (per-parameter, diagonal G):
      p <- (1 - alpha) * p - lr * (G * g) + sqrt(2 * alpha * lr) * noise_scale * sqrt(T) * sqrt(G) * N(0, I)
      theta <- theta + p

    Notes:
    - Drift always uses n_obs * ḡ (sum-loglik convention); RMSprop stats use minibatch-mean ḡ.
    - G is computed via RMSprop: G = 1 / (eps + sqrt(v_hat)), with v updated by beta.
    - If freeze_preconditioner is True, v is not updated during step().
    """

    def __init__(
        self,
        params,
        n_obs: int,
        lr: float = 1e-3,
        beta: float = 0.99,
        eps: float = 1e-5,
        alpha: float = 0.01,
        preconditioning: bool = True,
        add_noise: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon parameter: {eps}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if n_obs <= 0:
            raise ValueError(f"Invalid n_obs: {n_obs}")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            alpha=alpha,
            n_obs=n_obs,
            preconditioning=preconditioning,
            add_noise=add_noise,
            temperature=1.0,
            noise_scale=0.0,  # set externally (e.g., phase 3 ramp)
            freeze_preconditioner=False,
            grad_ema_beta=0.99,
        )
        super().__init__(params, defaults)

    def set_lr(self, new_lr: float):
        for g in self.param_groups:
            g["lr"] = float(new_lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Be robust to older/foreign optimizer state dicts that may lack SGHMC keys.
            # (We also guard at the checkpoint load site, but this keeps SGHMC from hard-crashing.)
            lr = float(group.get("lr", self.defaults.get("lr", 1e-3)))
            beta = float(group.get("beta", self.defaults.get("beta", 0.99)))
            eps = float(group.get("eps", self.defaults.get("eps", 1e-5)))
            alpha = float(group.get("alpha", self.defaults.get("alpha", 0.01)))
            n_obs = int(group.get("n_obs", self.defaults.get("n_obs", 1)))
            group.setdefault("lr", lr)
            group.setdefault("beta", beta)
            group.setdefault("eps", eps)
            group.setdefault("alpha", alpha)
            group.setdefault("n_obs", n_obs)
            preconditioning = bool(group.get("preconditioning", True))
            # Diagonal preconditioning: "rmsprop" (default).
            # Matrix preconditioning: "matrix" (rare; primarily used in legacy workflows / diagnostics).
            preconditioner_type = str(group.get("preconditioner", "rmsprop")).strip().lower()
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            temperature = float(group.get("temperature", 1.0))
            freeze_preconditioner = bool(group.get("freeze_preconditioner", False))
            grad_ema_beta = float(group.get("grad_ema_beta", 0.99))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # --- Optional gauge projection: remove translation mode before preconditioner stats update ---
                try:
                    gauge_enable = bool(getattr(self, "_gauge_project_enable", False))
                    gauge_param = getattr(self, "_gauge_project_param", None)
                    if gauge_enable and (gauge_param is p):
                        if isinstance(grad, torch.Tensor) and grad.ndim == 2 and int(grad.shape[1]) >= 4:
                            dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                            mode = str(getattr(self, "_gauge_project_mode", "global"))
                            cid = getattr(self, "_gauge_cluster_ids", None)
                            cc = getattr(self, "_gauge_cluster_counts", None)
                            project_event_mean_inplace(grad, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                except Exception:
                    pass
                
                # Drift always uses N * ḡ
                grad_for_drift = grad.mul(n_obs)
                
                dd_degree = getattr(p, "_dd_degree", None)
                if dd_degree is not None:
                    grad_for_precond = grad / dd_degree
                else:
                    grad_for_precond = grad
                
                state = self.state[p]

                # Initialize state
                if "momentum" not in state:
                    state["step"] = 0
                    if preconditioning and preconditioner_type == "rmsprop":
                        state.setdefault("exp_avg_sq", torch.zeros_like(p))
                    state["momentum"] = torch.zeros_like(p)
                    state.setdefault("ema_g", torch.zeros_like(p))
                    state.setdefault("ema_g2", torch.zeros_like(p))

                # Ensure keys
                v = state.get("exp_avg_sq", None)
                m = state.get("momentum", None)
                ema_g = state.get("ema_g")
                ema_g2 = state.get("ema_g2")
                if m is None:
                    m = torch.zeros_like(p)
                    state["momentum"] = m
                if ema_g is None:
                    ema_g = torch.zeros_like(p)
                    state["ema_g"] = ema_g
                if ema_g2 is None:
                    ema_g2 = torch.zeros_like(p)
                    state["ema_g2"] = ema_g2
                state["step"] = int(state.get("step", 0)) + 1
                
                # Prepare Matrix Preconditioner parts
                M_inv = None
                L_fac = None
                if preconditioner_type == "matrix":
                    M_inv = state.get('matrix_inv', None)
                    L_fac = state.get('matrix_L', None)

                # Update diagonal preconditioner stats and compute G
                G = None
                if preconditioner_type == "rmsprop" and preconditioning:
                    if v is None:
                        v = torch.zeros_like(p)
                        state["exp_avg_sq"] = v
                    if not freeze_preconditioner:
                        v.mul_(beta).addcmul_(grad_for_precond, grad_for_precond, value=(1.0 - beta))
                    # Compute diagonal preconditioner G
                    # Use a bias-corrected variant for numerical stability (especially early in warmup).
                    step_i = int(state.get("step", 1))
                    if step_i > 0:
                        v_hat = v / (1.0 - (beta ** step_i))
                    else:
                        v_hat = v
                    G = 1.0 / (eps + v_hat.sqrt())
                elif preconditioner_type != "matrix":
                    G = torch.ones_like(p)

                # Update EMA gradient statistics (using drift-scaled grads)
                # Keep diagonal statistics for diagnostics even if using matrix precond
                ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))

                # Momentum update
                # Frictional decay
                m.mul_(1.0 - alpha)
                
                # Drift term: -lr * Precond * g
                if M_inv is not None:
                    # Matrix Preconditioning
                    # M_inv: (N, 4, 4), grad_for_drift: (N, 4)
                    # We need batch matmul: (N, 4, 4) @ (N, 4, 1) -> (N, 4, 1)
                    drift_force = torch.matmul(M_inv, grad_for_drift.unsqueeze(-1)).squeeze(-1)
                    m.add_(drift_force, alpha=-lr)
                elif G is not None:
                    # Diagonal Preconditioning
                    m.addcmul_(G, grad_for_drift, value=-lr)
                else:
                    # Identity (should be covered by G=ones, but safety fallback)
                    m.add_(grad_for_drift, alpha=-lr)

                # Noise injection
                if add_noise and temperature > 0.0 and noise_scale > 0.0:
                    std = math.sqrt(2.0 * alpha * lr) * math.sqrt(max(temperature, 0.0)) * noise_scale
                    
                    if L_fac is not None:
                        # Matrix Noise: L_fac @ epsilon
                        # L_fac L_fac^T = M_inv
                        epsilon = torch.randn_like(p)
                        noise_vec = torch.matmul(L_fac, epsilon.unsqueeze(-1)).squeeze(-1)
                        m.add_(noise_vec, alpha=std)
                    elif G is not None:
                        # Diagonal Noise: sqrt(G) * epsilon
                        noise = torch.randn_like(p) * std * G.sqrt()
                        # Optional gauge projection of injected noise (prevents mean translation drift).
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
                        m.add_(noise)
                    else:
                         # Identity Noise
                        noise = torch.randn_like(p) * std
                        m.add_(noise)

                # Optional: project momentum mean as well (helps in SGHMC where momentum carries drift).
                try:
                    if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                        if bool(getattr(self, "_gauge_project_apply_momentum", True)):
                            dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                            mode = str(getattr(self, "_gauge_project_mode", "global"))
                            cid = getattr(self, "_gauge_cluster_ids", None)
                            cc = getattr(self, "_gauge_cluster_counts", None)
                            project_event_mean_inplace(m, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                except Exception:
                    pass

                # Parameter update
                p.add_(m)

        return loss

    @torch.no_grad()
    def preconditioner_stats(self):
        """
        Return summary stats for the preconditioner (dict):
          {min, p25, median, p75, max}

        - Diagonal: stats of diagonal G.
        - Matrix (FIM): stats of diagonal elements of M^{-1}.
        """
        try:
            def _nan() -> dict:
                return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}

            def _five_num(x: torch.Tensor) -> dict:
                x = x.detach().reshape(-1)
                if x.numel() == 0:
                    return _nan()
                finite = torch.isfinite(x)
                if torch.any(finite):
                    x = x[finite]
                if x.numel() == 0:
                    return _nan()
                n = int(x.numel())
                # kthvalue is 1-indexed; use selection to avoid full sort
                def _k(q: float) -> int:
                    return int(max(1, min(n, round(q * (n - 1)) + 1)))
                p25 = float(torch.kthvalue(x, _k(0.25)).values.item())
                med = float(torch.kthvalue(x, _k(0.50)).values.item())
                p75 = float(torch.kthvalue(x, _k(0.75)).values.item())
                return {"min": float(x.min().item()), "p25": p25, "median": med, "p75": p75, "max": float(x.max().item())}

            if len(self.param_groups) == 0:
                return _nan()
            group = self.param_groups[0]
            preconditioning = bool(group.get("preconditioning", True))
            preconditioner_type = str(group.get("preconditioner", "rmsprop")).lower()
            if not preconditioning:
                return _nan()
            
            eps = float(group.get("eps", 1e-5))
            beta = float(group.get("beta", 0.99))
            
            # First param with state
            p = None
            for q in group.get("params", []):
                if q is not None:
                    p = q
                    break
            if p is None:
                return _nan()
            
            state = self.state.get(p, {})
            
            if preconditioner_type == "matrix":
                M_inv = state.get("matrix_inv", None)
                if M_inv is None:
                    return _nan()
                diagonals = torch.diagonal(M_inv, dim1=-2, dim2=-1)
                return _five_num(diagonals)

            v = state.get("exp_avg_sq", None)
            if v is None:
                G = torch.ones_like(p)
            else:
                step = int(state.get("step", 1))
                v_hat = v / (1.0 - (beta ** max(step, 1)))
                G = 1.0 / (eps + v_hat.sqrt())
            return _five_num(G)
        except Exception:
            return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}

    @torch.no_grad()
    def grad_vs_noise_stats(self) -> dict:
        """
        Compute statistics (GeoMean, Median, P10, P90, Min, Max) of the ratio:
        (update variance from minibatch gradient noise) / (injected momentum noise variance).
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
        per_group = []

        for gi, group in enumerate(self.param_groups):
            lr = float(group.get("lr", 0.0))
            beta = float(group.get("beta", 0.99))
            eps_g = float(group.get("eps", 1e-5))
            preconditioning = bool(group.get("preconditioning", True))
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            temperature = float(group.get("temperature", 1.0))
            alpha = float(group.get("alpha", 0.01))
            
            if add_noise and noise_scale > 0.0 and temperature > 0.0 and alpha > 0.0:
                any_noise_global = True
            
            if lr <= 0.0:
                per_group.append({
                    "group_name": str(group.get("group_name", f"group{gi}")),
                    "gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0,
                })
                continue
            any_noise_group = bool(add_noise and noise_scale > 0.0 and temperature > 0.0 and alpha > 0.0)
            group_ratios = []

            for p in group["params"]:
                if p is None:
                    continue
                state = self.state[p]
                v = state.get("exp_avg_sq", None)
                ema_g = state.get("ema_g")
                ema_g2 = state.get("ema_g2")
                
                if ema_g is None or ema_g2 is None:
                    continue
                
                var_g = (ema_g2 - ema_g * ema_g).clamp_min(0.0)
                
                if preconditioning:
                    if v is None:
                        G = torch.ones_like(p)
                    else:
                        step = int(state.get("step", 1))
                        v_hat = v / (1.0 - (beta ** max(step, 1)))
                        G = 1.0 / (eps_g + v_hat.sqrt())
                else:
                    G = torch.ones_like(p)
                
                # SGHMC noise variance per parameter in momentum/update space
                var_noise = (2.0 * alpha * lr) * (noise_scale * noise_scale) * max(temperature, 0.0) * G
                denom = torch.where(var_noise > 0.0, var_noise, var_noise.new_full(var_noise.shape, eps))
                
                # Gradient noise-induced update variance via EMA stats
                num = (lr * lr) * (G * G) * var_g
                
                ratio = (num / denom).clamp_min(1e-30)
                finite_mask = torch.isfinite(ratio)
                if finite_mask.any():
                    rr = ratio[finite_mask].flatten()
                    all_ratios.append(rr)
                    group_ratios.append(rr)

            if any_noise_group and group_ratios:
                gcat = torch.cat(group_ratios)
                gstats = _summarize(gcat)
            else:
                gstats = {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            gstats["group_name"] = str(group.get("group_name", f"group{gi}"))
            per_group.append(gstats)

        if (not any_noise_global) or (not all_ratios):
            out = {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            out["per_group"] = per_group
            return out

        cat_ratios = torch.cat(all_ratios) if all_ratios else torch.tensor([])
        out = _summarize(cat_ratios)
        out["per_group"] = per_group
        return out

    @torch.no_grad()
    def temperature_stats(self) -> dict:
        """
        Estimate an "effective temperature" from the stationary momentum variance.

        This is a heuristic diagnostic meant for *sampling* phases when the chain is in a
        roughly stationary regime (gradients not dominated by deterministic drift).

        For our SGHMC discretization (per-parameter, diagonal G):
            m <- (1 - alpha) m + sqrt(2 alpha lr) * noise_scale * sqrt(T) * sqrt(G) * N(0, I) + (drift terms)

        Ignoring drift, m is AR(1) with coefficient (1-alpha). The stationary variance is:
            Var(m) = Var(eta) / (1 - (1-alpha)^2) = (2 alpha lr noise_scale^2 T G) / (2 alpha - alpha^2)
                   = (2 lr noise_scale^2 T G) / (2 - alpha)

        Solving for T gives the per-element estimate:
            T_eff = ((2 - alpha) / 2) * m^2 / (lr * noise_scale^2 * G)

        Important: m can have a nonzero mean component when drift is present, which inflates m^2.
        We therefore compute *two* variants:
          - teff_msq: based on m^2 (can be inflated by drift)
          - teff_var: based on Var(m) = mean((m - mean(m))^2) elementwise (more "thermal")

        We aggregate elementwise T_eff across all parameters and return {gm, median, min, max} for both.
        Also returns T_eff / T_target aggregates, which should be ~1 if temperature scaling is correct.
        """
        eps = 1e-30
        all_msq = []
        all_msq_over = []
        all_var = []
        all_var_over = []
        any = False

        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            beta = float(group.get("beta", 0.99))
            eps_g = float(group.get("eps", 1e-5))
            preconditioning = bool(group.get("preconditioning", True))
            preconditioner_type = group.get("preconditioner", "rmsprop")  # "rmsprop" or "matrix"
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            temperature = float(group.get("temperature", 1.0))
            alpha = float(group.get("alpha", 0.01))

            if lr <= 0.0 or noise_scale <= 0.0 or not add_noise or temperature <= 0.0:
                continue

            # scale factor from stationary variance derivation
            scale = (2.0 - alpha) / 2.0
            if not math.isfinite(scale) or scale <= 0.0:
                continue

            for p in group["params"]:
                if p is None:
                    continue
                state = self.state.get(p, {})
                m = state.get("momentum", None)
                if m is None:
                    continue

                # Compute effective G used for noise injection
                if preconditioning and preconditioner_type == "matrix":
                    M_inv = state.get("matrix_inv", None)
                    if M_inv is not None and M_inv.ndim >= 2:
                        # For event tensors (N,4) with M_inv (N,4,4), use diagonal as per-dim scaling.
                        try:
                            G_eff = torch.diagonal(M_inv, dim1=-2, dim2=-1)
                        except Exception:
                            G_eff = torch.ones_like(m)
                        # Broadcast if needed
                        if G_eff.shape != m.shape:
                            try:
                                G_eff = G_eff.expand_as(m)
                            except Exception:
                                G_eff = torch.ones_like(m)
                    else:
                        G_eff = torch.ones_like(m)
                else:
                    if preconditioning:
                        v = state.get("exp_avg_sq", None)
                        if v is None:
                            G_eff = torch.ones_like(m)
                        else:
                            step = int(state.get("step", 1))
                            v_hat = v / (1.0 - (beta ** max(step, 1)))
                            G_eff = 1.0 / (eps_g + v_hat.sqrt())
                    else:
                        G_eff = torch.ones_like(m)

                denom = (lr * (noise_scale * noise_scale)) * G_eff
                denom = torch.where(denom > 0.0, denom, denom.new_full(denom.shape, eps))
                # m^2-based (may include drift mean)
                teff_msq = (scale * (m * m)) / denom
                teff_msq = teff_msq.clamp_min(1e-30)
                finite = torch.isfinite(teff_msq)
                if finite.any():
                    any = True
                    flat_msq = teff_msq[finite].flatten()
                    all_msq.append(flat_msq)
                    all_msq_over.append((flat_msq / max(temperature, eps)).clamp_min(1e-30))

                # variance-based (remove mean drift component)
                m0 = m - m.mean()
                teff_var = (scale * (m0 * m0)) / denom
                teff_var = teff_var.clamp_min(1e-30)
                finite2 = torch.isfinite(teff_var)
                if finite2.any():
                    any = True
                    flat_var = teff_var[finite2].flatten()
                    all_var.append(flat_var)
                    all_var_over.append((flat_var / max(temperature, eps)).clamp_min(1e-30))

        if (not any) or (not all_msq and not all_var):
            return {
                "msq_gm": float("nan"),
                "msq_median": float("nan"),
                "msq_min": float("nan"),
                "msq_max": float("nan"),
                "msq_gm_over_target": float("nan"),
                "msq_median_over_target": float("nan"),
                "var_gm": float("nan"),
                "var_median": float("nan"),
                "var_min": float("nan"),
                "var_max": float("nan"),
                "var_gm_over_target": float("nan"),
                "var_median_over_target": float("nan"),
            }

        def _agg(x: torch.Tensor) -> tuple[float, float, float, float]:
            log_mean = torch.log(x).mean()
            gm = math.exp(log_mean.item())
            med = x.median().item()
            mn = x.min().item()
            mx = x.max().item()
            return float(gm), float(med), float(mn), float(mx)

        msq = torch.cat(all_msq) if all_msq else None
        var = torch.cat(all_var) if all_var else None

        msq_gm, msq_med, msq_min, msq_max = _agg(msq) if msq is not None else (float("nan"),) * 4
        var_gm, var_med, var_min, var_max = _agg(var) if var is not None else (float("nan"),) * 4

        msq_over = torch.cat(all_msq_over) if all_msq_over else None
        var_over = torch.cat(all_var_over) if all_var_over else None
        msq_gm_over, msq_med_over, _, _ = _agg(msq_over) if msq_over is not None else (float("nan"),) * 4
        var_gm_over, var_med_over, _, _ = _agg(var_over) if var_over is not None else (float("nan"),) * 4

        return {
            "msq_gm": msq_gm,
            "msq_median": msq_med,
            "msq_min": msq_min,
            "msq_max": msq_max,
            "msq_gm_over_target": msq_gm_over,
            "msq_median_over_target": msq_med_over,
            "var_gm": var_gm,
            "var_median": var_med,
            "var_min": var_min,
            "var_max": var_max,
            "var_gm_over_target": var_gm_over,
            "var_median_over_target": var_med_over,
        }

    @torch.no_grad()
    def grad_vs_noise_geomean(self) -> float:
        """
        Legacy wrapper for grad_vs_noise_stats['gm']
        """
        stats = self.grad_vs_noise_stats()
        return stats["gm"]

    # NOTE: Removed drift_vs_noise_per_dim() diagnostic. We no longer log drift_ratio_* metrics
    # to W&B (too noisy/expensive), and keeping this method encourages accidental reintroduction.


