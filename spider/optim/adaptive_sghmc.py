import math
from typing import Optional

import torch

from .gauge import project_event_mean_inplace


class AdaptiveSGHMC(torch.optim.Optimizer):
    """
    Adaptive (scale-adapted) SGHMC implementation in the spirit of BOHAMIANN / Springenberg et al. (2016).

    This is a diagonal-preconditioned SGHMC variant with a burn-in adaptation of:
      - tau (moving window / step-size adaptation)
      - g (running mean of gradients)
      - v_hat (running second moment / "variance" proxy of gradients)

    Notes for SPIDER integration:
    - SPIDER's loss returns minibatch *mean* gradients; the sampler drift expects "sum-loglik"
      convention. We therefore scale gradients by `scale_grad` (typically n_obs = N_total).
    - The original BOHAMIANN formulation has an implicit noise term; we expose `add_noise`,
      `noise_scale`, and `temperature` to stay compatible with SPIDER's phase logic.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-2,
        num_burn_in_steps: int = 3000,
        epsilon: float = 1e-16,
        mdecay: float = 0.05,
        scale_grad: float = 1.0,
        add_noise: bool = True,
    ) -> None:
        if lr < 0.0 or not math.isfinite(float(lr)):
            raise ValueError(f"Invalid learning rate: {lr}")
        if num_burn_in_steps < 0:
            raise ValueError(f"Invalid num_burn_in_steps: {num_burn_in_steps}")
        if epsilon < 0.0 or not math.isfinite(float(epsilon)):
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if mdecay < 0.0 or not math.isfinite(float(mdecay)):
            raise ValueError(f"Invalid mdecay: {mdecay}")

        defaults = dict(
            lr=float(lr),
            num_burn_in_steps=int(num_burn_in_steps),
            epsilon=float(epsilon),
            mdecay=float(mdecay),
            scale_grad=float(scale_grad),
            add_noise=bool(add_noise),
            # SPIDER-compatible knobs (set by epoch runner / locate)
            temperature=1.0,
            noise_scale=1.0,
            freeze_preconditioner=False,
            # metadata used in SPIDER logging helpers
            preconditioning=True,
            preconditioner="adaptive_sghmc",
            beta=float("nan"),  # not used; present for compatibility
            eps=float(epsilon),  # alias for compatibility
            n_obs=int(max(1, int(scale_grad))),  # alias for compatibility (overridden by backend factory)
            is_burnin=True,
            # EMA gradient statistics for diagnostics
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
            mdecay = float(group.get("mdecay", 0.05))
            epsilon = float(group.get("epsilon", group.get("eps", 1e-16)))

            # SPIDER convention: scale_grad typically equals n_obs (dataset size)
            scale_grad = float(group.get("scale_grad", group.get("n_obs", 1.0)))
            scale_grad_t: Optional[torch.Tensor] = None

            # SPIDER-compatible noise controls
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            temperature = float(group.get("temperature", 1.0))

            num_burn = int(group.get("num_burn_in_steps", 0))
            freeze_precond = bool(group.get("freeze_preconditioner", False))
            is_burnin_flag = group.get("is_burnin", None)

            # Temperature/noise guards
            if not math.isfinite(temperature) or temperature < 0.0:
                temperature = 0.0
            if not math.isfinite(noise_scale) or noise_scale < 0.0:
                noise_scale = 0.0

            for p in group["params"]:
                if p is None or p.grad is None:
                    continue

                st = self.state[p]
                # Be robust to partially-initialized / checkpoint-restored states:
                # older states may exist without our keys, or may omit 'iteration'.
                if "iteration" not in st:
                    st["iteration"] = 0
                st.setdefault("tau", torch.ones_like(p))
                st.setdefault("g", torch.ones_like(p))
                st.setdefault("v_hat", torch.ones_like(p))
                st.setdefault("momentum", torch.zeros_like(p))
                st.setdefault("ema_g", torch.zeros_like(p))
                st.setdefault("ema_g2", torch.zeros_like(p))

                st["iteration"] = int(st.get("iteration", 0)) + 1
                it = int(st["iteration"])

                # Convert scale_grad to tensor once per param-group/device/dtype
                if scale_grad_t is None or scale_grad_t.device != p.device or scale_grad_t.dtype != p.dtype:
                    scale_grad_t = torch.tensor(scale_grad, device=p.device, dtype=p.dtype)

                tau = st["tau"]
                g = st["g"]
                v_hat = st["v_hat"]
                momentum = st["momentum"]

                # SPIDER convention:
                # - Autograd produces minibatch-mean gradients of the *average* negative log posterior.
                # - Sampler drift wants the "sum-loglik" convention, so we scale by n_obs (scale_grad).
                #
                # For consistency with SGHMC/pSGLD:
                # - Use drift_grad = N * ḡ for the dynamics
                # - Use precond/adaptation stats from the minibatch-mean gradient (optionally degree-normalized)
                #   so the preconditioner is less sensitive to dataset size and heterogeneous event degrees.
                grad_mean = p.grad.data
                # --- Optional gauge projection: remove translation mode before adaptation/preconditioner updates ---
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
                drift_grad = grad_mean * scale_grad_t

                # Optional DD-degree normalization (attached by spider.core.state._attach_dd_preconditioner_metric).
                # This mirrors SGHMC/pSGLD behavior where degree scaling affects preconditioner statistics,
                # not the drift scaling itself.
                dd_degree = getattr(p, "_dd_degree", None)
                if dd_degree is not None:
                    try:
                        precond_grad = grad_mean / dd_degree
                    except Exception:
                        precond_grad = grad_mean
                else:
                    precond_grad = grad_mean

                # tau_inv = 1/(tau+1)
                tau_inv = 1.0 / (tau + 1.0)

                # Burn-in adaptation (optionally disabled or frozen)
                # SPIDER integration note:
                # - BOHAMIANN uses a *step-count* burn-in (num_burn_in_steps).
                # - In SPIDER, we optionally drive burn-in by *epochs* via the param-group flag `is_burnin`,
                #   which is set by the training loop (Phase 3 => burn-in, Phase 4 => sampling).
                # Behavior:
                # - If `is_burnin` is present, it takes precedence.
                # - Otherwise, fall back to the original step-count semantics.
                do_burnin = bool(is_burnin_flag) if is_burnin_flag is not None else (it <= num_burn)
                if do_burnin and (not freeze_precond):
                    # Eq. 9 (Springenberg et al. 2016): update tau, g, v_hat
                    tau.add_(-tau * (g * g / (v_hat + epsilon)) + 1.0)
                    g.add_(-g * tau_inv + tau_inv * precond_grad)
                    v_hat.add_(-v_hat * tau_inv + tau_inv * (precond_grad * precond_grad))

                # Diagonal preconditioner
                minv_t = 1.0 / (torch.sqrt(v_hat) + epsilon)

                # Update EMA gradient statistics (drift-scaled gradients).
                try:
                    grad_ema_beta = float(group.get("grad_ema_beta", 0.99))
                except Exception:
                    grad_ema_beta = 0.99
                ema_g = st.get("ema_g")
                ema_g2 = st.get("ema_g2")
                if ema_g is None:
                    ema_g = torch.zeros_like(p)
                    st["ema_g"] = ema_g
                if ema_g2 is None:
                    ema_g2 = torch.zeros_like(p)
                    st["ema_g2"] = ema_g2
                ema_g.mul_(grad_ema_beta).add_(drift_grad, alpha=(1.0 - grad_ema_beta))
                ema_g2.mul_(grad_ema_beta).addcmul_(drift_grad, drift_grad, value=(1.0 - grad_ema_beta))

                # BOHAMIANN noise variance term
                # epsilon_var = 2 * lr^2 * mdecay * minv_t - lr^4
                eps_var = (2.0 * (lr * lr) * mdecay * minv_t) - (lr ** 4)
                eps_var = torch.clamp(eps_var, min=1e-16)

                if add_noise and noise_scale > 0.0 and temperature > 0.0:
                    # Allow SPIDER to ramp/scale noise and apply a temperature (std scales by sqrt(T)).
                    sigma = torch.sqrt(eps_var) * float(noise_scale) * math.sqrt(float(temperature))
                    noise = torch.normal(mean=torch.zeros_like(drift_grad), std=sigma)
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
                else:
                    noise = torch.zeros_like(drift_grad)

                # Momentum update (Eq. 10 right)
                momentum.add_(-(lr * lr) * minv_t * drift_grad - mdecay * momentum + noise)

                # Optional: project momentum mean as well (helps because momentum carries across steps).
                try:
                    if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                        if bool(getattr(self, "_gauge_project_apply_momentum", True)):
                            dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                            mode = str(getattr(self, "_gauge_project_mode", "global"))
                            cid = getattr(self, "_gauge_cluster_ids", None)
                            cc = getattr(self, "_gauge_cluster_counts", None)
                            project_event_mean_inplace(momentum, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                except Exception:
                    pass

                # Theta update (Eq. 10 left): theta += momentum
                p.data.add_(momentum)

        return loss

    @torch.no_grad()
    def preconditioner_stats(self):
        """
        Return summary stats of the effective diagonal preconditioner minv_t (dict):
          {min, p25, median, p75, max}
        """
        try:
            if len(self.param_groups) == 0:
                return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
            group = self.param_groups[0]
            eps = float(group.get("epsilon", group.get("eps", 1e-16)))
            # find first param with state
            p = None
            for q in group.get("params", []):
                if q is not None:
                    p = q
                    break
            if p is None:
                return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
            st = self.state.get(p, {})
            v_hat = st.get("v_hat", None)
            if v_hat is None:
                return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
            minv_t = 1.0 / (torch.sqrt(v_hat) + eps)
            x = minv_t.detach().reshape(-1)
            finite = torch.isfinite(x)
            if torch.any(finite):
                x = x[finite]
            if x.numel() == 0:
                return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}
            n = int(x.numel())
            def _k(q: float) -> int:
                return int(max(1, min(n, round(q * (n - 1)) + 1)))
            p25 = float(torch.kthvalue(x, _k(0.25)).values.item())
            med = float(torch.kthvalue(x, _k(0.50)).values.item())
            p75 = float(torch.kthvalue(x, _k(0.75)).values.item())
            return {"min": float(x.min().item()), "p25": p25, "median": med, "p75": p75, "max": float(x.max().item())}
        except Exception:
            return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan")}

    @torch.no_grad()
    def grad_vs_noise_stats(self) -> dict:
        """
        Compute stats of the ratio:
          (momentum variance induced by minibatch gradient noise) / (injected momentum noise variance).

        For AdaptiveSGHMC:
          momentum update includes:  -(lr^2) * minv_t * grad
          so gradient-noise-induced variance ≈ (lr^4) * (minv_t^2) * Var(grad)
        Injected noise variance per element:
          Var_noise = eps_var * noise_scale^2 * temperature
        where eps_var = 2 * lr^2 * mdecay * minv_t - lr^4 (clamped).

        We estimate Var(grad) from EMA stats: var_g = ema_g2 - ema_g^2.
        """
        eps_small = 1e-30

        def _summarize(cat: torch.Tensor) -> dict:
            if cat is None or (not isinstance(cat, torch.Tensor)) or cat.numel() == 0:
                return {"gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            log_mean = torch.log(cat.clamp_min(1e-20)).mean()
            gm = math.exp(float(log_mean.item()))
            try:
                x = cat
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
                "median": float(cat.median().item()),
                "p10": float(p10),
                "p90": float(p90),
                "min": float(cat.min().item()),
                "max": float(cat.max().item()),
            }

        any_noise_global = False
        all_ratios = []
        per_group = []

        for gi, group in enumerate(self.param_groups):
            lr = float(group.get("lr", 0.0))
            mdecay = float(group.get("mdecay", 0.05))
            epsilon = float(group.get("epsilon", group.get("eps", 1e-16)))
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            temperature = float(group.get("temperature", 1.0))

            if add_noise and noise_scale > 0.0 and temperature > 0.0:
                any_noise_global = True
            if lr <= 0.0:
                per_group.append({
                    "group_name": str(group.get("group_name", f"group{gi}")),
                    "gm": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0,
                })
                continue
            any_noise_group = bool(add_noise and noise_scale > 0.0 and temperature > 0.0)
            group_ratios = []

            for p in group.get("params", []):
                if p is None:
                    continue
                st = self.state.get(p, {})
                v_hat = st.get("v_hat", None)
                ema_g = st.get("ema_g", None)
                ema_g2 = st.get("ema_g2", None)
                if v_hat is None or ema_g is None or ema_g2 is None:
                    continue

                minv_t = 1.0 / (torch.sqrt(v_hat) + epsilon)
                eps_var = (2.0 * (lr * lr) * mdecay * minv_t) - (lr ** 4)
                eps_var = torch.clamp(eps_var, min=1e-16)

                var_g = (ema_g2 - ema_g * ema_g).clamp_min(0.0)
                var_drift = (lr ** 4) * (minv_t * minv_t) * var_g

                if add_noise and noise_scale > 0.0 and temperature > 0.0:
                    var_noise = eps_var * (noise_scale * noise_scale) * max(temperature, 0.0)
                else:
                    var_noise = torch.zeros_like(var_drift)

                denom = torch.where(var_noise > 0.0, var_noise, var_noise.new_full(var_noise.shape, eps_small))
                ratio = (var_drift / denom).clamp_min(1e-30)
                finite = torch.isfinite(ratio)
                if finite.any():
                    rr = ratio[finite].flatten()
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

        cat = torch.cat(all_ratios) if all_ratios else torch.tensor([])
        out = _summarize(cat)
        out["per_group"] = per_group
        return out

    @torch.no_grad()
    def temperature_stats(self) -> dict:
        """
        Heuristic effective-temperature diagnostic from the momentum distribution.

        Ignoring drift, momentum follows an AR(1):
          m <- (1 - mdecay) m + noise,  noise ~ N(0, eps_var * noise_scale^2 * T)

        Stationary variance (per element) is:
          Var(m) = Var(noise) / (1 - (1 - mdecay)^2) = Var(noise) / (2*mdecay - mdecay^2)

        Solve for T:
          T_eff = (2*mdecay - mdecay^2) * Var(m) / (eps_var * noise_scale^2)

        We report both:
          - msq_*: based on m^2 (can be inflated by drift mean)
          - var_*: based on (m - mean(m))^2 (more "thermal")
        """
        eps_small = 1e-30
        all_msq = []
        all_var = []
        all_msq_over = []
        all_var_over = []

        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            mdecay = float(group.get("mdecay", 0.05))
            epsilon = float(group.get("epsilon", group.get("eps", 1e-16)))
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            temperature = float(group.get("temperature", 1.0))

            if (not add_noise) or lr <= 0.0 or noise_scale <= 0.0 or temperature <= 0.0:
                continue
            scale = (2.0 * mdecay - (mdecay * mdecay))
            if (not math.isfinite(scale)) or scale <= 0.0:
                continue

            for p in group.get("params", []):
                if p is None:
                    continue
                st = self.state.get(p, {})
                v_hat = st.get("v_hat", None)
                m = st.get("momentum", None)
                if v_hat is None or m is None:
                    continue

                minv_t = 1.0 / (torch.sqrt(v_hat) + epsilon)
                eps_var = (2.0 * (lr * lr) * mdecay * minv_t) - (lr ** 4)
                eps_var = torch.clamp(eps_var, min=1e-16)

                denom = eps_var * (noise_scale * noise_scale)
                denom = torch.where(denom > 0.0, denom, denom.new_full(denom.shape, eps_small))

                teff_msq = (scale * (m * m)) / denom
                teff_msq = teff_msq.clamp_min(1e-30)
                fin = torch.isfinite(teff_msq)
                if fin.any():
                    flat = teff_msq[fin].flatten()
                    all_msq.append(flat)
                    all_msq_over.append((flat / max(temperature, eps_small)).clamp_min(1e-30))

                m0 = m - m.mean()
                teff_var = (scale * (m0 * m0)) / denom
                teff_var = teff_var.clamp_min(1e-30)
                fin2 = torch.isfinite(teff_var)
                if fin2.any():
                    flat2 = teff_var[fin2].flatten()
                    all_var.append(flat2)
                    all_var_over.append((flat2 / max(temperature, eps_small)).clamp_min(1e-30))

        if (not all_msq) and (not all_var):
            return {
                "msq_gm": float("nan"),
                "msq_median": float("nan"),
                "var_gm": float("nan"),
                "var_median": float("nan"),
                "msq_median_over_target": float("nan"),
                "var_median_over_target": float("nan"),
            }

        def _gm_med(x: torch.Tensor) -> tuple[float, float]:
            log_mean = torch.log(x.clamp_min(1e-20)).mean()
            gm = math.exp(float(log_mean.item()))
            med = float(x.median().item())
            return float(gm), float(med)

        msq = torch.cat(all_msq) if all_msq else None
        var = torch.cat(all_var) if all_var else None
        msq_over = torch.cat(all_msq_over) if all_msq_over else None
        var_over = torch.cat(all_var_over) if all_var_over else None

        msq_gm, msq_med = _gm_med(msq) if msq is not None else (float("nan"), float("nan"))
        var_gm, var_med = _gm_med(var) if var is not None else (float("nan"), float("nan"))
        _, msq_med_over = _gm_med(msq_over) if msq_over is not None else (float("nan"), float("nan"))
        _, var_med_over = _gm_med(var_over) if var_over is not None else (float("nan"), float("nan"))

        return {
            "msq_gm": msq_gm,
            "msq_median": msq_med,
            "var_gm": var_gm,
            "var_median": var_med,
            "msq_median_over_target": msq_med_over,
            "var_median_over_target": var_med_over,
        }


