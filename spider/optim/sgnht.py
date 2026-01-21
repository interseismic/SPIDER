import math

import torch

from .gauge import project_event_mean_inplace


class SGNHT(torch.optim.Optimizer):
    """
    Stochastic Gradient Nose-Hoover Thermostat (SGNHT) with optional diagonal RMSprop
    preconditioning (G). Maintains per-parameter momentum and a scalar thermostat.
    Injects diffusion noise controlled by `diffusion` (A).

    Discretization (per-parameter, diagonal G):
      m <- m - lr * (G * g) - lr * xi * m + sqrt(2 A lr) * noise_scale * sqrt(T) * sqrt(G) * N(0, I)
      xi <- xi + (lr / Q) * (mean(m^2 / G) - T)
      theta <- theta + m
    """

    def __init__(
        self,
        params,
        n_obs: int,
        lr: float = 1e-3,
        beta: float = 0.99,
        eps: float = 1e-5,
        diffusion: float = 0.01,
        thermostat_mass: float = 1.0,
        preconditioning: bool = True,
        add_noise: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon parameter: {eps}")
        if not 0.0 <= diffusion:
            raise ValueError(f"Invalid diffusion parameter: {diffusion}")
        if not (thermostat_mass > 0.0):
            raise ValueError(f"Invalid thermostat_mass: {thermostat_mass}")
        if n_obs <= 0:
            raise ValueError(f"Invalid n_obs: {n_obs}")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            diffusion=diffusion,
            thermostat_mass=thermostat_mass,
            n_obs=n_obs,
            preconditioning=preconditioning,
            add_noise=add_noise,
            preconditioner="rmsprop",
            temperature=1.0,
            noise_scale=0.0,
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
            lr = float(group.get("lr", self.defaults.get("lr", 1e-3)))
            beta = float(group.get("beta", self.defaults.get("beta", 0.99)))
            eps = float(group.get("eps", self.defaults.get("eps", 1e-5)))
            diffusion = float(group.get("diffusion", self.defaults.get("diffusion", 0.01)))
            thermostat_mass = float(group.get("thermostat_mass", self.defaults.get("thermostat_mass", 1.0)))
            n_obs = int(group.get("n_obs", self.defaults.get("n_obs", 1)))
            preconditioning = bool(group.get("preconditioning", True))
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

                # Optional gauge projection (remove translation mode before preconditioner stats update).
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

                grad_for_drift = grad.mul(n_obs)
                dd_degree = getattr(p, "_dd_degree", None)
                if dd_degree is not None:
                    grad_for_precond = grad / dd_degree
                else:
                    grad_for_precond = grad

                state = self.state[p]
                if "momentum" not in state:
                    state["step"] = 0
                    if preconditioning and preconditioner_type == "rmsprop":
                        state.setdefault("exp_avg_sq", torch.zeros_like(p))
                    state["momentum"] = torch.zeros_like(p)
                    state["thermostat"] = torch.zeros((), dtype=p.dtype, device=p.device)
                    state.setdefault("ema_g", torch.zeros_like(p))
                    state.setdefault("ema_g2", torch.zeros_like(p))

                v = state.get("exp_avg_sq", None)
                m = state.get("momentum", None)
                xi = state.get("thermostat", None)
                ema_g = state.get("ema_g")
                ema_g2 = state.get("ema_g2")
                if m is None:
                    m = torch.zeros_like(p)
                    state["momentum"] = m
                if xi is None:
                    xi = torch.zeros((), dtype=p.dtype, device=p.device)
                    state["thermostat"] = xi
                if ema_g is None:
                    ema_g = torch.zeros_like(p)
                    state["ema_g"] = ema_g
                if ema_g2 is None:
                    ema_g2 = torch.zeros_like(p)
                    state["ema_g2"] = ema_g2
                state["step"] = int(state.get("step", 0)) + 1

                M_inv = None
                L_fac = None
                if preconditioner_type == "matrix":
                    M_inv = state.get("matrix_inv", None)
                    L_fac = state.get("matrix_L", None)

                # Update diagonal preconditioner stats and compute G.
                G = None
                if preconditioner_type == "rmsprop" and preconditioning:
                    if v is None:
                        v = torch.zeros_like(p)
                        state["exp_avg_sq"] = v
                    if not freeze_preconditioner:
                        v.mul_(beta).addcmul_(grad_for_precond, grad_for_precond, value=(1.0 - beta))
                    step_i = int(state.get("step", 1))
                    if step_i > 0:
                        v_hat = v / (1.0 - (beta ** step_i))
                    else:
                        v_hat = v
                    G = 1.0 / (eps + v_hat.sqrt())
                elif preconditioner_type != "matrix":
                    G = torch.ones_like(p)

                # EMA gradient stats (diagnostic only)
                ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))

                # Drift term
                if M_inv is not None:
                    drift_force = torch.matmul(M_inv, grad_for_drift.unsqueeze(-1)).squeeze(-1)
                    m.add_(drift_force, alpha=-lr)
                    G_eff = torch.diagonal(M_inv, dim1=-2, dim2=-1) if M_inv.ndim >= 2 else torch.ones_like(m)
                elif G is not None:
                    m.addcmul_(G, grad_for_drift, value=-lr)
                    G_eff = G
                else:
                    m.add_(grad_for_drift, alpha=-lr)
                    G_eff = torch.ones_like(m)

                # Thermostat friction
                try:
                    m.add_(m, alpha=-lr * float(xi.item()))
                except Exception:
                    m.add_(m, alpha=-lr * float(xi))

                # Noise injection (diffusion)
                if add_noise and temperature > 0.0 and noise_scale > 0.0 and diffusion > 0.0:
                    std = math.sqrt(2.0 * diffusion * lr) * math.sqrt(max(temperature, 0.0)) * noise_scale
                    if L_fac is not None:
                        epsilon = torch.randn_like(p)
                        noise_vec = torch.matmul(L_fac, epsilon.unsqueeze(-1)).squeeze(-1)
                        m.add_(noise_vec, alpha=std)
                    elif G is not None:
                        noise = torch.randn_like(p) * std * G.sqrt()
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
                        noise = torch.randn_like(p) * std
                        m.add_(noise)

                # Thermostat update (scalar per parameter tensor)
                try:
                    denom = G_eff.clamp_min(1e-12)
                    m2_over_g = (m * m) / denom
                    avg_kin = m2_over_g.mean()
                    xi.add_((avg_kin - float(temperature)), alpha=(lr / thermostat_mass))
                except Exception:
                    pass

                # Optional: project momentum mean as well
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
                if not isinstance(M_inv, torch.Tensor):
                    return _nan()
                if M_inv.ndim >= 2:
                    diag = torch.diagonal(M_inv, dim1=-2, dim2=-1)
                else:
                    diag = M_inv
                return _five_num(diag)

            # RMSprop-style diagonal preconditioner
            v = state.get("exp_avg_sq", None)
            if v is None or not isinstance(v, torch.Tensor):
                return _nan()
            step_i = int(state.get("step", 1))
            if step_i > 0:
                v_hat = v / (1.0 - (beta ** step_i))
            else:
                v_hat = v
            G = 1.0 / (eps + v_hat.sqrt())
            return _five_num(G)
        except Exception:
            return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}

    @torch.no_grad()
    def sgnht_stats(self) -> dict:
        """
        Diagnostics for SGNHT:
          - xi_mean / xi_median (thermostat)
          - kinetic_over_target_{gm,median}: mean(m^2 / G) / T
        """
        eps = 1e-30
        xi_all = []
        kin_all = []
        kin_over_all = []
        any = False

        for group in self.param_groups:
            beta = float(group.get("beta", 0.99))
            eps_g = float(group.get("eps", 1e-5))
            preconditioning = bool(group.get("preconditioning", True))
            preconditioner_type = str(group.get("preconditioner", "rmsprop")).strip().lower()
            temperature = float(group.get("temperature", 1.0))

            for p in group["params"]:
                if p is None:
                    continue
                state = self.state.get(p, {})
                m = state.get("momentum", None)
                if m is None:
                    continue
                xi = state.get("thermostat", None)
                if isinstance(xi, torch.Tensor):
                    xi_all.append(xi.detach().flatten())

                # Effective diagonal scaling for kinetic energy
                if preconditioning and preconditioner_type == "matrix":
                    M_inv = state.get("matrix_inv", None)
                    if M_inv is not None and M_inv.ndim >= 2:
                        try:
                            G_eff = torch.diagonal(M_inv, dim1=-2, dim2=-1)
                        except Exception:
                            G_eff = torch.ones_like(m)
                        if G_eff.shape != m.shape:
                            try:
                                G_eff = G_eff.expand_as(m)
                            except Exception:
                                G_eff = torch.ones_like(m)
                    else:
                        G_eff = torch.ones_like(m)
                elif preconditioning:
                    v = state.get("exp_avg_sq", None)
                    if v is None:
                        G_eff = torch.ones_like(m)
                    else:
                        step = int(state.get("step", 1))
                        v_hat = v / (1.0 - (beta ** max(step, 1)))
                        G_eff = 1.0 / (eps_g + v_hat.sqrt())
                else:
                    G_eff = torch.ones_like(m)

                denom = G_eff.clamp_min(1e-12)
                kin = (m * m) / denom
                kin = kin.clamp_min(1e-30)
                finite = torch.isfinite(kin)
                if finite.any():
                    any = True
                    flat = kin[finite].flatten()
                    kin_all.append(flat)
                    kin_over_all.append((flat / max(temperature, eps)).clamp_min(1e-30))

        if (not any) or (not kin_all):
            return {
                "xi_mean": float("nan"),
                "xi_median": float("nan"),
                "kinetic_gm_over_target": float("nan"),
                "kinetic_median_over_target": float("nan"),
            }

        kin_cat = torch.cat(kin_all) if kin_all else None
        kin_over = torch.cat(kin_over_all) if kin_over_all else None
        if kin_cat is None or kin_over is None:
            return {
                "xi_mean": float("nan"),
                "xi_median": float("nan"),
                "kinetic_gm_over_target": float("nan"),
                "kinetic_median_over_target": float("nan"),
            }

        log_mean = torch.log(kin_over).mean()
        gm = float(math.exp(log_mean.item()))
        med = float(kin_over.median().item())

        if xi_all:
            xi_cat = torch.cat(xi_all)
            xi_mean = float(xi_cat.mean().item())
            xi_med = float(xi_cat.median().item())
        else:
            xi_mean = float("nan")
            xi_med = float("nan")

        return {
            "xi_mean": xi_mean,
            "xi_median": xi_med,
            "kinetic_gm_over_target": gm,
            "kinetic_median_over_target": med,
        }
