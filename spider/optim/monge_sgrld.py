from __future__ import annotations

import math
from typing import Iterable

import torch

from .gauge import project_event_mean_inplace


class MongeSGLD(torch.optim.Optimizer):
    """
    Stochastic Gradient Riemannian Langevin Dynamics with Monge metric.

    Implements the Monge metric from:
    Yu et al. "Scalable Stochastic Gradient Riemannian Langevin Dynamics
    in Non-Diagonal Metrics" (2023), Section 4.

    We follow the paper's Euler-Maruyama discretization and ignore the
    Gamma correction term, consistent with their experiments and bounds.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        n_obs: int,
        lr: float = 1e-3,
        alpha: float = 1.0,
        ema_beta: float = 0.9,
        eps: float = 1e-12,
        add_noise: bool = True,
    ):
        if n_obs <= 0:
            raise ValueError(f"Invalid n_obs: {n_obs}")
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not (0.0 <= ema_beta < 1.0):
            raise ValueError(f"Invalid ema_beta: {ema_beta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(
            lr=float(lr),
            alpha=float(alpha),
            ema_beta=float(ema_beta),
            eps=float(eps),
            n_obs=int(n_obs),
            add_noise=bool(add_noise),
            temperature=1.0,
            noise_scale=1.0,
        )
        super().__init__(params, defaults)

    def set_lr(self, new_lr: float) -> None:
        for group in self.param_groups:
            group["lr"] = float(new_lr)

    def set_temperature(self, new_temperature: float) -> None:
        if new_temperature < 0.0 or not math.isfinite(new_temperature):
            raise ValueError(f"Invalid temperature: {new_temperature}")
        for group in self.param_groups:
            group["temperature"] = float(new_temperature)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group.get("lr", 1e-3))
            alpha = float(group.get("alpha", 1.0))
            ema_beta = float(group.get("ema_beta", 0.9))
            eps = float(group.get("eps", 1e-12))
            n_obs = int(group.get("n_obs", 1))
            add_noise = bool(group.get("add_noise", True))
            temperature = float(group.get("temperature", 1.0))
            noise_scale = float(group.get("noise_scale", 1.0))

            params_with_grad = []
            g_norm_sq = None
            g_dot_grad = None

            # First pass: update EMA and compute global stats.
            for p in group["params"]:
                if p.grad is None:
                    continue
                raw_grad = p.grad

                # Gauge projection (gradient)
                try:
                    if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                        if isinstance(raw_grad, torch.Tensor) and raw_grad.ndim == 2 and int(raw_grad.shape[1]) >= 4:
                            dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                            mode = str(getattr(self, "_gauge_project_mode", "global"))
                            cid = getattr(self, "_gauge_cluster_ids", None)
                            cc = getattr(self, "_gauge_cluster_counts", None)
                            project_event_mean_inplace(raw_grad, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                except Exception:
                    pass

                state = self.state[p]
                ema = state.get("monge_ema", None)
                if ema is None:
                    ema = torch.zeros_like(p)
                    state["monge_ema"] = ema
                ema.mul_(ema_beta).add_(raw_grad, alpha=(1.0 - ema_beta))

                params_with_grad.append((p, raw_grad, ema))

                contrib_norm = torch.sum(ema * ema)
                contrib_dot = torch.sum(ema * raw_grad)
                g_norm_sq = contrib_norm if g_norm_sq is None else (g_norm_sq + contrib_norm)
                g_dot_grad = contrib_dot if g_dot_grad is None else (g_dot_grad + contrib_dot)

            if not params_with_grad:
                continue

            if g_norm_sq is None:
                g_norm_sq = torch.zeros((), dtype=params_with_grad[0][0].dtype, device=params_with_grad[0][0].device)
            if g_dot_grad is None:
                g_dot_grad = torch.zeros_like(g_norm_sq)

            alpha_sq = g_norm_sq.new_tensor(alpha * alpha)
            denom = 1.0 + alpha_sq * g_norm_sq
            if float(g_norm_sq.item()) <= eps:
                f_minus1 = g_norm_sq.new_tensor(0.0)
                f_minus_half = g_norm_sq.new_tensor(0.0)
            else:
                f_minus1 = -alpha_sq / denom
                f_minus_half = (1.0 / g_norm_sq) * (1.0 / torch.sqrt(denom) - 1.0)

            # Drift update using G^{-1}.
            for p, raw_grad, ema in params_with_grad:
                drift_vec = raw_grad + f_minus1 * ema * g_dot_grad
                drift_update = -lr * drift_vec
                p.add_(drift_update)

            # Noise update using G^{-1/2}.
            noise_update = None
            if add_noise and temperature > 0.0 and noise_scale > 0.0:
                std = (
                    math.sqrt(2.0 * lr)
                    * math.sqrt(max(temperature, 0.0))
                    * noise_scale
                    / math.sqrt(max(n_obs, 1))
                )
                noises = []
                g_dot_noise = None
                for p, _, ema in params_with_grad:
                    noise = torch.randn_like(p)
                    noises.append(noise)
                    contrib = torch.sum(ema * noise)
                    g_dot_noise = contrib if g_dot_noise is None else (g_dot_noise + contrib)
                if g_dot_noise is None:
                    g_dot_noise = g_norm_sq.new_tensor(0.0)

                for (p, _, ema), noise in zip(params_with_grad, noises):
                    noise_vec = noise + f_minus_half * ema * g_dot_noise
                    try:
                        if bool(getattr(self, "_gauge_project_enable", False)) and (getattr(self, "_gauge_project_param", None) is p):
                            if bool(getattr(self, "_gauge_project_apply_noise", True)):
                                dims = tuple(getattr(self, "_gauge_project_dims", (0, 1, 2)))
                                mode = str(getattr(self, "_gauge_project_mode", "global"))
                                cid = getattr(self, "_gauge_cluster_ids", None)
                                cc = getattr(self, "_gauge_cluster_counts", None)
                                project_event_mean_inplace(noise_vec, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
                    except Exception:
                        pass
                    noise_update = noise_vec * std
                    p.add_(noise_update)

            # Stats (last param processed)
            try:
                last_grad = params_with_grad[-1][1]
                grad_norm = float(torch.sqrt(torch.mean(last_grad.detach().float() ** 2)).item())
                drift_var = float(torch.mean(((-lr * (last_grad + f_minus1 * params_with_grad[-1][2] * g_dot_grad)).detach().float()) ** 2).item())
                noise_var = float(torch.mean(noise_update.detach().float() ** 2).item()) if isinstance(noise_update, torch.Tensor) else 0.0
                update = (-lr * (last_grad + f_minus1 * params_with_grad[-1][2] * g_dot_grad)) + (noise_update if isinstance(noise_update, torch.Tensor) else 0.0)
                update_norm = float(torch.sqrt(torch.mean(update.detach().float() ** 2)).item())
                ratio = drift_var / noise_var if noise_var > 0.0 else float("nan")
                self._last_stats = {
                    "grad_norm": grad_norm,
                    "update_norm": update_norm,
                    "drift_var": drift_var,
                    "noise_var": noise_var,
                    "drift_noise_var_ratio": ratio,
                    "monge_g_norm": float(torch.sqrt(g_norm_sq.detach().float()).item()),
                }
            except Exception:
                self._last_stats = {}

        return loss
