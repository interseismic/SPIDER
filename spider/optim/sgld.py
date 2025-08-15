import torch
import math
from typing import Optional, Callable

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
    """

    def __init__(self, params, n_obs, lr=1e-3, beta=0.99, eps=1e-5,
                 preconditioning=True, add_noise=True,
                 preconditioner: str = "rmsprop",
                 scale_grad_by_n_obs: bool = True,
                 include_gamma: bool = True):
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
            scale_grad_by_n_obs (bool): If True, scale only the drift by n_obs
                when your loss is mean-reduced; preconditioner stats use raw grads.
            include_gamma (bool): If True, add a diagonal approximation to the
                Γ(θ) correction term from pSGLD. This adds a small extra drift
                accounting for state-dependent G. The original paper notes Γ can
                be omitted with negligible bias when beta≈1.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon parameter: {eps}")
        if n_obs <= 0:
            raise ValueError(f"Invalid n_obs: {n_obs}")

        if preconditioner not in {"rmsprop", "adam"}:
            raise ValueError("preconditioner must be 'rmsprop' or 'adam'")

        defaults = dict(lr=lr, beta=beta, eps=eps, n_obs=n_obs,
                        preconditioning=preconditioning, add_noise=add_noise,
                        preconditioner=preconditioner,
                        scale_grad_by_n_obs=scale_grad_by_n_obs,
                        include_gamma=include_gamma)
        super().__init__(params, defaults)

    def set_lr(self, new_lr):
        """Set learning rate for all parameter groups."""
        for group in self.param_groups:
            group['lr'] = new_lr

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
            scale_grad_by_n_obs = group.get('scale_grad_by_n_obs', True)
            preconditioner = group.get('preconditioner', 'rmsprop')
            include_gamma = group.get('include_gamma', True)
            noise_scale = float(group.get('noise_scale', 1.0))

            for p in group['params']:
                if p.grad is None:
                    continue

                raw_grad = p.grad
                # Use raw gradients for preconditioner statistics
                grad_for_precond = raw_grad
                # Optionally scale only the drift by n_obs if loss is mean-reduced
                grad_for_drift = raw_grad.mul(n_obs) if scale_grad_by_n_obs else raw_grad
                state = self.state[p]

                # Initialize state if not present
                if len(state) == 0:
                    state['step'] = 0
                    if preconditioning:
                        state['exp_avg_sq'] = torch.zeros_like(p)

                # Ensure all required state keys exist
                if 'step' not in state:
                    state['step'] = 0
                if preconditioning and 'exp_avg_sq' not in state:
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                # Increment step counter
                state['step'] += 1

                if preconditioning:
                    v = state['exp_avg_sq']
                    v.mul_(beta).addcmul_(grad_for_precond, grad_for_precond, value=1 - beta)
                    # Preconditioner choice
                    if preconditioner == 'rmsprop':
                        G = 1.0 / (eps + v.sqrt())
                    else:  # 'adam' bias-corrected second moment
                        v_hat = v / (1.0 - (beta ** state['step']))
                        G = 1.0 / (eps + v_hat.sqrt())
                else:
                    G = torch.ones_like(p)

                # Compute update step
                update = lr * G * grad_for_drift

                # Gamma correction term (approximate, diagonal case)
                # Γ_i ≈ - (1-β) * g_i * sqrt(v_i) / (eps + sqrt(v_i))^2
                if include_gamma and preconditioning:
                    if preconditioner == 'rmsprop':
                        sqrt_v = v.sqrt().clamp_min(0.0)
                        denom = (eps + sqrt_v)
                        gamma = - (1.0 - beta) * grad_for_precond * (sqrt_v / (denom * denom))
                    else:
                        v_hat = v / (1.0 - (beta ** state['step']))
                        sqrt_v = v_hat.sqrt().clamp_min(0.0)
                        denom = (eps + sqrt_v)
                        gamma = - (1.0 - beta) * grad_for_precond * (sqrt_v / (denom * denom))
                    update = update + lr * gamma

                # Add Langevin noise if requested: std = sqrt(2 * lr) * sqrt(G) * noise_scale
                if add_noise:
                    noise = torch.randn_like(p) * math.sqrt(2.0 * lr) * G.sqrt() * noise_scale
                    update += noise

                # Update parameter
                p.add_(-update)

        return loss


@torch.no_grad()
def transplant_v_from_adam(adam_opt, sgld_opt):
    # copy Adam's exp_avg_sq into SGLD state['exp_avg_sq'] (same shape, same dtype)
    adam_state = adam_opt.state
    sgld_state = sgld_opt.state
    for group in sgld_opt.param_groups:
        for p in group['params']:
            if p is None: continue
            if p in adam_state and 'exp_avg_sq' in adam_state[p]:
                v_src = adam_state[p]['exp_avg_sq']
                sgld_state[p]['exp_avg_sq'] = v_src.detach().clone()
            else:
                sgld_state[p]['exp_avg_sq'] = torch.zeros_like(p)

@torch.no_grad()
def heartbeat_poststep_from(prev_params, params, rel_floor_scale=1e-3, abs_floor=1e-8):
    # Deprecated diagnostics; keeping signature for compatibility if imported.
    return
