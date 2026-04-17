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


def _build_lrd_metric(
    *,
    group: dict,
    state: dict,
    grad_for_precond: torch.Tensor,
    beta: float,
    eps: float,
    freeze_preconditioner: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build/update a low-rank-plus-diagonal preconditioner for a parameter tensor.

    Returns:
      d_flat: (D,) diagonal term (positive)
      U: (D, r) orthonormal basis
      lam: (r,) non-negative low-rank spectrum
    """
    g = grad_for_precond
    D = int(g.numel())
    g_flat = g.reshape(-1)

    rank_cfg = int(group.get("lrd_rank", 16))
    rank = max(0, min(int(rank_cfg), D))
    mode = str(group.get("lrd_mode", "svd")).strip().lower()
    if mode in {"randomized_svd", "stochastic_svd"}:
        mode = "svd"
    if mode not in {"svd", "oja"}:
        mode = "svd"
    update_every = max(1, int(group.get("lrd_update_every", 20)))
    buffer_size = max(max(2, rank + 1), int(group.get("lrd_buffer_size", 64)))
    oja_eta = float(group.get("lrd_oja_eta", 0.02))
    if (not math.isfinite(oja_eta)) or (oja_eta <= 0.0):
        oja_eta = 0.02
    diag_floor = float(group.get("lrd_diag_floor", eps))
    if (not math.isfinite(diag_floor)) or (diag_floor <= 0.0):
        diag_floor = max(float(eps), 1e-12)

    v = state.get("exp_avg_sq", None)
    if (not isinstance(v, torch.Tensor)) or v.shape != g.shape or v.device != g.device or v.dtype != g.dtype:
        v = torch.zeros_like(g)
        state["exp_avg_sq"] = v
    if not freeze_preconditioner:
        v.mul_(beta).addcmul_(g, g, value=(1.0 - beta))
    step_i = int(state.get("step", 1))
    if 0.0 <= float(beta) < 1.0:
        v_hat = v / (1.0 - (float(beta) ** max(step_i, 1)))
    else:
        v_hat = v
    d_flat = (1.0 / (float(eps) + v_hat.clamp_min(0.0).sqrt())).reshape(-1).clamp_min(diag_floor)

    if rank <= 0:
        U = torch.zeros((D, 0), device=g.device, dtype=g.dtype)
        lam = torch.zeros((0,), device=g.device, dtype=g.dtype)
        state["lrd_U"] = U
        state["lrd_lambda"] = lam
        state["precond_diag"] = d_flat.reshape_as(g).detach()
        return d_flat, U, lam

    U = state.get("lrd_U", None)
    lam = state.get("lrd_lambda", None)
    if (not isinstance(U, torch.Tensor)) or U.shape != (D, rank) or U.device != g.device or U.dtype != g.dtype:
        U = torch.zeros((D, rank), device=g.device, dtype=g.dtype)
    if (not isinstance(lam, torch.Tensor)) or lam.shape != (rank,) or lam.device != g.device or lam.dtype != g.dtype:
        lam = torch.zeros((rank,), device=g.device, dtype=g.dtype)

    if not freeze_preconditioner:
        if mode == "oja":
            if not bool(torch.any(torch.isfinite(U))) or float(U.abs().sum().item()) == 0.0:
                gn = float(g_flat.norm().item())
                if gn > 0.0:
                    U[:, 0] = g_flat / max(gn, 1e-12)
                if rank > 1:
                    U[:, 1:] = torch.randn((D, rank - 1), device=g.device, dtype=g.dtype)
                try:
                    U, _ = torch.linalg.qr(U, mode="reduced")
                except Exception:
                    pass
            q = U.mT @ g_flat
            U = U + (oja_eta * torch.outer(g_flat, q))
            try:
                U, _ = torch.linalg.qr(U, mode="reduced")
            except Exception:
                pass
            if U.shape[1] > rank:
                U = U[:, :rank].contiguous()
            if U.shape[1] < rank:
                U_pad = torch.zeros((D, rank), device=g.device, dtype=g.dtype)
                if U.shape[1] > 0:
                    U_pad[:, : U.shape[1]] = U
                U = U_pad
            q = U.mT @ g_flat
            lam.mul_(beta).addcmul_(q, q, value=(1.0 - beta))
        else:
            buf = state.get("lrd_grad_buffer", None)
            if (
                (not isinstance(buf, torch.Tensor))
                or buf.ndim != 2
                or int(buf.shape[1]) != int(D)
                or buf.device != g.device
                or buf.dtype != g.dtype
            ):
                buf = torch.zeros((0, D), device=g.device, dtype=g.dtype)
            buf = torch.cat([buf, g_flat.detach().unsqueeze(0)], dim=0)
            if int(buf.shape[0]) > int(buffer_size):
                buf = buf[-int(buffer_size) :, :]
            state["lrd_grad_buffer"] = buf.detach()
            if (int(state.get("step", 1)) % int(update_every) == 0) and int(buf.shape[0]) >= max(2, rank):
                X = buf - buf.mean(dim=0, keepdim=True)
                try:
                    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
                    r_eff = min(rank, int(Vh.shape[0]), int(S.shape[0]))
                    if r_eff > 0:
                        U_new = Vh[:r_eff, :].mT.contiguous()
                        lam_new = (S[:r_eff] * S[:r_eff]) / float(max(1, int(X.shape[0]) - 1))
                        U.zero_()
                        U[:, :r_eff] = U_new
                        lam.mul_(beta)
                        lam[:r_eff].add_(lam_new.to(dtype=lam.dtype, device=lam.device), alpha=(1.0 - beta))
                except Exception:
                    pass

    lam = lam.clamp_min(0.0)
    diag_low = torch.zeros((D,), device=g.device, dtype=g.dtype)
    if int(U.numel()) > 0 and int(lam.numel()) > 0:
        try:
            diag_low = (U * U).matmul(lam)
        except Exception:
            diag_low = torch.zeros((D,), device=g.device, dtype=g.dtype)
    diag_proxy = (d_flat + diag_low).clamp_min(diag_floor).reshape_as(g)
    state["lrd_U"] = U.detach()
    state["lrd_lambda"] = lam.detach()
    state["precond_diag"] = diag_proxy.detach()
    return d_flat, U, lam

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

    Supported preconditioners in current runtime:
    - `rmsprop` (diagonal)
    - `lrd` (low-rank plus diagonal)
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
        if preconditioner in {"none", "false", ""}:
            if preconditioning:
                raise ValueError("preconditioner cannot be 'none' when preconditioning=True")
            # Keep a valid label even when preconditioning is disabled.
            preconditioner = "rmsprop"
        if preconditioner not in {"rmsprop", "lrd"}:
            raise ValueError(
                f"preconditioner must be 'rmsprop' or 'lrd'; got '{preconditioner}'"
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
                grad_for_precond = raw_grad

                state = self.state[p]

                # Initialize state if not present
                if "ema_g" not in state:
                    state['step'] = 0
                    if preconditioning:
                        if preconditioner == "rmsprop":
                            state.setdefault('exp_avg_sq', torch.zeros_like(p))
                    state['ema_g'] = torch.zeros_like(p)
                    state['ema_g2'] = torch.zeros_like(p)

                # Ensure step exists
                if 'step' not in state:
                    state['step'] = 0
                state['step'] += 1

                if preconditioning and preconditioner == "lrd":
                    # Non-diagonal preconditioner (LRD)
                    ema_g = state['ema_g']
                    ema_g2 = state['ema_g2']
                    ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                    ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))
                    state['ema_g'] = ema_g
                    state['ema_g2'] = ema_g2

                    if preconditioner == "lrd":
                        d_flat, U_lrd, lam_lrd = _build_lrd_metric(
                            group=group,
                            state=state,
                            grad_for_precond=grad_for_precond,
                            beta=float(beta),
                            eps=float(eps),
                            freeze_preconditioner=bool(freeze_preconditioner),
                        )
                        g_flat = grad_for_drift.reshape(-1)
                        pre_flat = d_flat * g_flat
                        if int(U_lrd.numel()) > 0 and int(lam_lrd.numel()) > 0:
                            try:
                                q = U_lrd.mT @ g_flat
                                pre_flat = pre_flat + (U_lrd @ (lam_lrd * q))
                            except Exception:
                                pass
                        update = lr * pre_flat.reshape_as(p)
                        if add_noise:
                            temp = max(0.0, temperature)
                            std = math.sqrt(2.0 * lr * temp) * noise_scale
                            noise_flat = torch.randn_like(g_flat) * d_flat.sqrt()
                            if int(U_lrd.numel()) > 0 and int(lam_lrd.numel()) > 0:
                                try:
                                    z2 = torch.randn((int(lam_lrd.numel()),), device=p.device, dtype=p.dtype)
                                    noise_flat = noise_flat + (U_lrd @ (lam_lrd.clamp_min(0.0).sqrt() * z2))
                                except Exception:
                                    pass
                            noise = std * noise_flat.reshape_as(p)
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
                    if precond in {"lrd"}:
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
        - lrd: stats of diagonal proxy from state['precond_diag']
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

            if preconditioner == "lrd":
                G_diag = state.get("precond_diag", None)
                if isinstance(G_diag, torch.Tensor):
                    return _five_num(G_diag)

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

@torch.no_grad()
def heartbeat_poststep_from(prev_params, params, rel_floor_scale=1e-3, abs_floor=1e-8):
    # Deprecated diagnostics; keeping signature for compatibility if imported.
    return
