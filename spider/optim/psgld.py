from __future__ import annotations

import math
import torch

from .gauge import project_event_mean_inplace


class pSGLD(torch.optim.Optimizer):
    """
    Simplified pSGLD for spider.
    - Diagonal RMSprop preconditioning only.
    - Optional gauge projection on gradients/noise.
    """

    def __init__(
        self,
        params,
        *,
        n_obs: int,
        lr: float = 1e-3,
        beta: float = 0.99,
        eps: float = 1e-5,
        preconditioning: bool = True,
        include_gamma: bool = False,
        add_noise: bool = True,
    ):
        if n_obs <= 0:
            raise ValueError(f"Invalid n_obs: {n_obs}")
        defaults = dict(
            lr=float(lr),
            beta=float(beta),
            eps=float(eps),
            n_obs=int(n_obs),
            preconditioning=bool(preconditioning),
            include_gamma=bool(include_gamma),
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
            beta = float(group.get("beta", 0.99))
            eps = float(group.get("eps", 1e-5))
            n_obs = int(group.get("n_obs", 1))
            preconditioning = bool(group.get("preconditioning", True))
            include_gamma = bool(group.get("include_gamma", False))
            add_noise = bool(group.get("add_noise", True))
            temperature = float(group.get("temperature", 1.0))
            noise_scale = float(group.get("noise_scale", 1.0))

            for p in group["params"]:
                if p.grad is None:
                    continue
                raw_grad = p.grad
                grad_norm = float(torch.sqrt(torch.mean(raw_grad.detach().float() ** 2)).item())

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

                # Use minibatch-mean gradient as-is (no n_obs scaling).
                grad_for_drift = raw_grad

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1

                if preconditioning:
                    v = state.get("exp_avg_sq", None)
                    if v is None:
                        v = torch.zeros_like(p)
                        state["exp_avg_sq"] = v
                    v.mul_(beta).addcmul_(raw_grad, raw_grad, value=(1.0 - beta))
                    v_hat = v / (1.0 - beta ** state["step"])
                    G = 1.0 / (eps + v_hat.sqrt())
                else:
                    G = torch.ones_like(p)

                # Optional Gamma correction (disabled by default)
                if include_gamma and preconditioning:
                    # Diagonal approximation to Γ(θ) term
                    gamma = (lr * G) * (grad_for_drift)
                else:
                    gamma = 0.0

                drift_update = -lr * G * grad_for_drift
                if include_gamma and preconditioning:
                    drift_update = drift_update - gamma
                p.add_(drift_update)

                # Noise
                noise_update = None
                if add_noise and temperature > 0.0 and noise_scale > 0.0:
                    std = (
                        math.sqrt(2.0 * lr)
                        * math.sqrt(max(temperature, 0.0))
                        * noise_scale
                        / math.sqrt(max(n_obs, 1))
                    )
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
                    p.add_(noise)
                    noise_update = noise
                else:
                    noise_update = None

                try:
                    drift_var = float(torch.mean(drift_update.detach().float() ** 2).item())
                    noise_var = float(torch.mean(noise_update.detach().float() ** 2).item()) if isinstance(noise_update, torch.Tensor) else 0.0
                    update = drift_update + (noise_update if isinstance(noise_update, torch.Tensor) else 0.0)
                    update_norm = float(torch.sqrt(torch.mean(update.detach().float() ** 2)).item())
                    ratio = drift_var / noise_var if noise_var > 0.0 else float("nan")
                    self._last_stats = {
                        "grad_norm": grad_norm,
                        "update_norm": update_norm,
                        "drift_var": drift_var,
                        "noise_var": noise_var,
                        "drift_noise_var_ratio": ratio,
                    }
                except Exception:
                    self._last_stats = {}

        return loss
