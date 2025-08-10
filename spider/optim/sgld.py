import torch
import math
from typing import Iterable, Optional, Callable
from torch.optim.optimizer import Optimizer
import collections

class SGLD(torch.optim.Optimizer):
    """
    Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    This optimizer combines stochastic gradient descent with Langevin dynamics
    for Bayesian sampling. It supports preconditioning using RMSprop-style
    adaptive learning rates.
    """

    def __init__(self, params, n_obs, lr=1e-3, beta=0.99, eps=1e-5,
                 preconditioning=True, add_noise=True):
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
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon parameter: {eps}")
        if n_obs <= 0:
            raise ValueError(f"Invalid n_obs: {n_obs}")

        defaults = dict(lr=lr, beta=beta, eps=eps, n_obs=n_obs,
                       preconditioning=preconditioning, add_noise=add_noise)
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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state if not present
                if len(state) == 0:
                    state['step'] = 0
                    if preconditioning:
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1

                if preconditioning:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                    # Compute preconditioning matrix
                    G = 1.0 / (eps + exp_avg_sq.sqrt())
                else:
                    G = torch.ones_like(p)

                # Compute update step
                update = lr * G * grad

                # Add Langevin noise if requested
                if add_noise:
                    noise_scale = torch.sqrt(2.0 * lr * G / n_obs)
                    noise = torch.randn_like(p) * noise_scale
                    update += noise

                # Update parameter
                p.add_(-update)

        return loss

    def sample_step(self, grad_ll, grad_lp):
        """
        Custom sampling step that takes explicit gradients.
        This maintains backward compatibility with the existing code.

        Args:
            grad_ll: Likelihood gradient
            grad_lp: Prior gradient
        """
        with torch.no_grad():
            # Get the first (and only) parameter group
            group = self.param_groups[0]
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            n_obs = group['n_obs']
            preconditioning = group['preconditioning']

            # Get the parameter (should be ΔX_src)
            p = group['params'][0]
            state = self.state[p]

            # Initialize state if not present
            if len(state) == 0:
                state['step'] = 0
                if preconditioning:
                    state['exp_avg_sq'] = torch.zeros_like(p)

            state['step'] += 1

            # Combine gradients safely
            combined_grad = grad_ll + grad_lp

            if preconditioning:
                exp_avg_sq = state['exp_avg_sq']
                # Use grad_ll for preconditioning to avoid issues with combined gradients
                exp_avg_sq.mul_(beta).addcmul_(grad_ll, grad_ll, value=1 - beta)
                G = 1.0 / (eps + exp_avg_sq.sqrt())
            else:
                G = torch.ones_like(p)

            # Compute update step
            sgd_step = lr * G * combined_grad
            langevin_step = torch.sqrt(2.0 * lr * G / n_obs) * torch.randn_like(p)

            # Update parameter
            p.add_(-sgd_step - langevin_step)

            # Clear intermediate tensors to help with memory management
            del combined_grad, sgd_step, langevin_step

class SGLDAdamPrecond(Optimizer):
    """
    SGLD with Adam-style diagonal preconditioner (no divergence term).

    Key points
    ----------
    • Preconditioner v_t is updated from the UNscaled gradient of your provided loss.
    • Normalized objectives (loss divided by N) supported in two ways:
        mode='scale_grad'  : drift uses (N * grad_norm), base lr as given.  [default]
        mode='scale_lr'    : effective lr = base_lr * N, drift uses grad_norm.
      (Both are correct; try 'scale_grad' first. Use 'scale_lr' if A is twitchy.)
    • Burn-in helpers:
        - adapt_only=True  : update v_t only (no parameter movement, no noise).
        - noise_scale∈[0,1]: scales injected noise; warm it up 0→1 across steps.
    • Stability clamps (optional):
        - max_drift_norm : clamp ||M * g_eff|| per parameter tensor.
        - max_rel_step   : elementwise |Δθ| ≤ max_rel_step * (|θ| + rel_eps).

    Update (per parameter θ)
    ------------------------
        g_raw  = ∂(your loss)/∂θ
        g_eff  = grad_scale * g_raw
        v_t    = β2 v_{t-1} + (1-β2) * g_raw^2
        v_hat  = v_t / (1 - β2^t)   (if bias correction enabled)
        M      = 1 / (sqrt(v_hat) + eps)
        θ_new  = θ + (η_eff/2) * (M * g_eff) + 𝒩(0, η_eff M)

      with:
        η_eff = base lr            if mode='scale_grad' or unnormalized
        η_eff = base lr * N        if mode='scale_lr'
        g_eff = g_raw * N          if mode='scale_grad' and normalized
        g_eff = g_raw              otherwise
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-4,
                 beta2: float = 0.999,
                 eps: float = 1e-6,                   # slightly larger default helps early steps
                 freeze_preconditioner: bool = True,
                 use_bias_correction: bool = True,
                 # normalization / scaling
                 normalized_loss: bool = False,
                 n_data: Optional[int] = None,
                 mode: str = 'scale_grad',            # 'scale_grad' or 'scale_lr'
                 # burn-in helpers
                 adapt_only: bool = False,            # update v_t only; no param update, no noise
                 noise_scale: float = 1.0,            # scales the injected noise (0..1)
                 # clamps
                 max_drift_norm: Optional[float] = None,
                 max_rel_step: Optional[float] = None,
                 rel_eps: float = 1e-8,
                 ):
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if not (0.0 < beta2 < 1.0):
            raise ValueError("beta2 must be in (0,1)")
        if mode not in ('scale_grad', 'scale_lr'):
            raise ValueError("mode must be 'scale_grad' or 'scale_lr'")
        if normalized_loss and n_data is None:
            raise ValueError("Provide n_data when normalized_loss=True.")
        if not (0.0 <= noise_scale):
            raise ValueError("noise_scale must be >= 0")

        defaults = dict(
            lr=lr, beta2=beta2, eps=eps,
            freeze_preconditioner=freeze_preconditioner,
            use_bias_correction=use_bias_correction,
            normalized_loss=normalized_loss,
            n_data=n_data,
            mode=mode,
            adapt_only=adapt_only,
            noise_scale=noise_scale,
            max_drift_norm=max_drift_norm,
            max_rel_step=max_rel_step,
            rel_eps=rel_eps,
        )
        super().__init__(params, defaults)

        self._num_steps = 0
        self._frozen = freeze_preconditioner

    # ---------- controls ----------
    @torch.no_grad()
    def freeze_preconditioner(self): self._frozen = True

    @torch.no_grad()
    def unfreeze_preconditioner(self): self._frozen = False

    @torch.no_grad()
    def set_adapt_only(self, flag: bool):
        for g in self.param_groups: g['adapt_only'] = bool(flag)

    @torch.no_grad()
    def set_noise_scale(self, s: float):
        for g in self.param_groups: g['noise_scale'] = float(s)

    @torch.no_grad()
    def zero_stats(self):
        """Reset v buffers and step counter (fresh burn-in)."""
        for group in self.param_groups:
            for p in group['params']:
                if p is None: continue
                st = self.state[p]
                st['v'] = torch.zeros_like(p)
        self._num_steps = 0

    # ---------- main step ----------
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        self._num_steps += 1

        for group in self.param_groups:
            lr = group['lr']
            beta2 = group['beta2']
            eps = group['eps']
            use_bc = group['use_bias_correction']
            normalized_loss = group['normalized_loss']
            n_data = group['n_data']
            mode = group['mode']
            adapt_only = group['adapt_only']
            noise_scale = group['noise_scale']
            max_drift_norm = group['max_drift_norm']
            max_rel_step = group['max_rel_step']
            rel_eps = group['rel_eps']

            # Effective lr and gradient scaling
            if normalized_loss and mode == 'scale_lr':
                eff_lr = lr * float(n_data)   # scale step size, not grads
                grad_scale = 1.0
            elif normalized_loss and mode == 'scale_grad':
                eff_lr = lr                   # scale grads by N, keep lr
                grad_scale = float(n_data)
            else:
                eff_lr = lr                   # unnormalized objective
                grad_scale = 1.0

            bc = (1.0 - beta2 ** self._num_steps) if use_bc else 1.0

            for p in group['params']:
                if p.grad is None:
                    continue

                g_raw = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st['v'] = torch.zeros_like(p)
                v = st['v']

                # Update v_t with UNscaled gradient (stable for huge N)
                if not self._frozen:
                    v.mul_(beta2).addcmul_(g_raw, g_raw, value=(1.0 - beta2))

                if adapt_only:
                    # Burn-in phase: only accumulate v_t; skip movement/noise
                    continue

                # Build preconditioner
                v_hat = v / bc if use_bc else v
                denom = v_hat.sqrt().add_(eps)
                M = 1.0 / denom

                # Drift using (possibly) scaled gradient
                g_eff = g_raw * grad_scale
                drift_core = M * g_eff

                # Optional clamp on drift (per tensor)
                if max_drift_norm is not None:
                    nrm = drift_core.norm()
                    if torch.isfinite(nrm) and nrm > max_drift_norm:
                        drift_core.mul_(max_drift_norm / (nrm + 1e-12))

                delta = 0.5 * eff_lr * drift_core

                # Optional elementwise relative step clamp
                if max_rel_step is not None:
                    cap = max_rel_step * (p.abs() + rel_eps)
                    delta = torch.clamp(delta, min=-cap, max=cap)

                # Langevin noise: N(0, eff_lr * M), optionally warmed up by noise_scale
                if noise_scale > 0.0:
                    noise_std = math.sqrt(eff_lr) * M.clamp_min(0.0).sqrt()
                    noise = torch.empty_like(p).normal_(0.0, 1.0).mul_(noise_std * noise_scale)
                    p.add_(delta).add_(noise)
                else:
                    p.add_(delta)

        return loss

@torch.no_grad()
def transplant_v_from_adam(adam_opt, sgld_opt):
    # copy Adam's exp_avg_sq into SGLD state['v'] (same shape, same dtype)
    adam_state = adam_opt.state
    sgld_state = sgld_opt.state
    for group in sgld_opt.param_groups:
        for p in group['params']:
            if p is None: continue
            if p in adam_state and 'exp_avg_sq' in adam_state[p]:
                v_src = adam_state[p]['exp_avg_sq']
                sgld_state[p]['v'] = v_src.detach().clone()
            else:
                sgld_state[p]['v'] = torch.zeros_like(p)

# ---------- helper: suggest a conservative base lr from current state ----------
@torch.no_grad()
def suggest_lr_from_state(opt: SGLDAdamPrecond,
                          u_mult: float = 1e-3,
                          sigma_mult: float = 1e-4):
    """
    Suggest a base lr that keeps:
      max |(M*g_eff)| step contribution <= u, and
      max noise std per-dimension <= sigma,
    with u = u_mult * RMS(theta), sigma = sigma_mult * RMS(theta).

    Returns: (eta_base, eta_drift_eff_ceiling, eta_noise_eff_ceiling)
    Note: If mode='scale_lr', the optimizer multiplies base lr by N internally.
    """
    # RMS(θ)
    numel, sq = 0, 0.0
    for g in opt.param_groups:
        for p in g['params']:
            if p is None: continue
            numel += p.numel()
            sq += float((p**2).sum())
    theta_rms = (sq / max(1, numel))**0.5
    u = u_mult * theta_rms
    sigma = sigma_mult * theta_rms

    max_Mg, max_M = 0.0, 0.0
    for g in opt.param_groups:
        beta2 = g['beta2']; eps = g['eps']
        use_bc = g['use_bias_correction']
        normalized_loss = g['normalized_loss']
        n_data = g['n_data']
        mode = g['mode']

        # Mirror eff lr / grad scaling logic for Mg sizing (we only need gs here)
        if normalized_loss and mode == 'scale_lr':
            gs = 1.0
        elif normalized_loss and mode == 'scale_grad':
            gs = float(n_data)
        else:
            gs = 1.0

        bc = (1.0 - beta2 ** max(1, opt._num_steps)) if use_bc else 1.0

        for p in g['params']:
            if p.grad is None: continue
            v = opt.state[p]['v']
            v_hat = v / bc if use_bc else v
            M = (v_hat.sqrt().add_(eps)).reciprocal()

            g_raw = p.grad
            g_eff = g_raw * gs

            max_M = max(max_M, float(M.max()))
            max_Mg = max(max_Mg, float((M * g_eff).abs().max()))

    # Effective ceilings for η_eff
    eta_drift_eff = (2.0 * u) / (max_Mg + 1e-12) if max_Mg > 0 else float('inf')
    eta_noise_eff = (sigma ** 2) / (max_M + 1e-12) if max_M > 0 else float('inf')
    eta_eff = 0.5 * min(eta_drift_eff, eta_noise_eff)

    # Convert to base lr (if mode='scale_lr', base = eff / N)
    base_lrs = []
    for g in opt.param_groups:
        if g['normalized_loss'] and g['mode'] == 'scale_lr':
            base_lrs.append(eta_eff / float(g['n_data']))
        else:
            base_lrs.append(eta_eff)
    eta_base = min(base_lrs) if base_lrs else eta_eff
    return eta_base, eta_drift_eff, eta_noise_eff

@torch.no_grad()
def print_phase2_diagnostics(opt):
    """
    Call after loss.backward(), before opt.step().
    Reports:
      - max(M)
      - max(|M * g_eff|)   (where g_eff matches your mode/normalization)
    """
    mx_M = 0.0
    mx_Mg = 0.0
    mx_abs_g = 0.0

    for group in opt.param_groups:
        beta2 = group['beta2']
        eps = group['eps']
        use_bc = group['use_bias_correction']
        normalized_loss = group['normalized_loss']
        mode = group['mode']
        n_data = group.get('n_data', None)

        # gradient scaling (matches the optimizer’s logic)
        if normalized_loss and mode == 'scale_grad':
            grad_scale = float(n_data)
        else:
            grad_scale = 1.0  # unnormalized or mode='scale_lr'

        # bias-correction factor for v
        bc = (1.0 - beta2 ** max(1, opt._num_steps)) if use_bc else 1.0

        for p in group['params']:
            if p.grad is None:
                continue

            # pull v (if missing, treat as zeros)
            st = opt.state.get(p, {})
            v = st.get('v', torch.zeros_like(p))

            v_hat = v / bc if use_bc else v
            # M = 1 / (sqrt(v_hat) + eps)
            M = (v_hat.sqrt().add_(eps)).reciprocal()

            g_raw = p.grad
            g_eff = g_raw * grad_scale

            # update maxima
            mx_M = max(mx_M, float(M.max()))
            mx_Mg = max(mx_Mg, float((M * g_eff).abs().max()))
            mx_abs_g = max(mx_abs_g, float(g_raw.abs().max()))

    print(f"[Phase 2 diag] max(M) = {mx_M:.3e}   max(|M*g_eff|) = {mx_Mg:.3e}   max|g_raw| = {mx_abs_g:.3e}")

@torch.no_grad()
def phase3_compute_base_lr(opt, u_rel=1e-3, u_abs=1e-6,
                           sigma_rel=3e-4, sigma_abs=1e-4):
    """
    Read current grads/v_t from `opt` *after* you've called loss.backward(),
    and compute a BASE lr (works for mode='scale_lr' or 'scale_grad').
    """
    # RMS(theta)
    numel, sq = 0, 0.0
    for g in opt.param_groups:
        for p in g['params']:
            if p is None: continue
            numel += p.numel()
            sq += float((p**2).sum())
    theta_rms = (sq / max(1, numel))**0.5

    max_M = 0.0
    max_Mg = 0.0
    mode = None
    n_data = 1.0
    noise_scale = 1.0

    for g in opt.param_groups:
        beta2, eps = g['beta2'], g['eps']
        use_bc = g['use_bias_correction']
        mode = g['mode']
        noise_scale = float(g['noise_scale'])
        normalized = g['normalized_loss']
        n_data = float(g.get('n_data', 1.0))

        # grad scaling must mirror your SGLDAdamPrecond
        gs = n_data if (normalized and mode == 'scale_grad') else 1.0
        bc = (1.0 - beta2 ** max(1, opt._num_steps)) if use_bc else 1.0

        for p in g['params']:
            if p.grad is None: continue
            v = opt.state[p]['v']
            v_hat = v / bc if use_bc else v
            M = (v_hat.sqrt().add_(eps)).reciprocal()
            g_eff = p.grad * gs
            max_M  = max(max_M,  float(M.max()))
            max_Mg = max(max_Mg, float((M * g_eff).abs().max()))

    # targets (relative to RMS(theta), with absolute floors)
    u = max(u_rel * theta_rms, u_abs)
    sigma = max(sigma_rel * theta_rms, sigma_abs)

    # effective step ceilings (include current noise_scale)
    eta_drift_eff = (2.0 * u) / (max_Mg + 1e-12) if max_Mg > 0 else float('inf')
    eta_noise_eff = (sigma / (max(1e-12, noise_scale) * math.sqrt(max_M + 1e-12)))**2 if max_M > 0 else float('inf')
    eta_eff = 0.5 * min(eta_drift_eff, eta_noise_eff)

    # convert to BASE lr (class multiplies by N internally in mode='scale_lr')
    base_lr = (eta_eff / n_data) if mode == 'scale_lr' else eta_eff
    return base_lr, dict(max_M=max_M, max_Mg=max_Mg, eta_eff=eta_eff,
                         noise_scale=noise_scale, mode=mode)

class Phase3Stopper:
    """
    Tracks BASE lr over a rolling window and decides when to stop Phase 3.
    Stop when noise_scale==1 and mean relative change ≤ tol.
    """
    def __init__(self, window=100, tol=0.10):
        self.window = int(window)
        self.tol = float(tol)
        self.hist = collections.deque(maxlen=self.window)

    def update(self, base_lr, noise_scale) -> bool:
        self.hist.append(float(base_lr))
        if noise_scale < 1.0 or len(self.hist) < self.window:
            return False
        med = float(torch.median(torch.tensor(list(self.hist))))
        if med == 0.0:
            return True
        rel = [abs(x - med) / (abs(med) + 1e-12) for x in self.hist]
        return (sum(rel) / len(rel)) <= self.tol

@torch.no_grad()
def phase3_step_control(opt, step_idx: int, ramp_len: int,
                        stopper: Phase3Stopper,
                        *, set_lr: bool = True,
                        u_rel=1e-3, u_abs=1e-6,
                        sigma_rel=3e-4, sigma_abs=1e-4,
                        verbose: bool = False):
    """
    Call this AFTER loss.backward() and BEFORE opt.step().

    - Ramps noise_scale based on (step_idx, ramp_len)
    - Computes BASE lr from current grads/preconditioner
    - Optionally sets group['lr'] = base_lr
    - Returns (done, base_lr, info_dict)
    """
    # 1) ramp noise 0→1
    s = float(min(1.0, (step_idx + 1) / max(1, ramp_len)))
    opt.set_noise_scale(s)

    # 2) compute BASE lr from current state
    base_lr, info = phase3_compute_base_lr(opt, u_rel, u_abs, sigma_rel, sigma_abs)

    # 3) optionally set it
    if set_lr:
        for g in opt.param_groups:
            g['lr'] = base_lr

    if verbose and (step_idx % max(1, ramp_len // 10) == 0):
        print(f"[Phase3 ctrl] s={s:.3f} base_lr={base_lr:.3e} "
              f"eta_eff={info['eta_eff']:.3e} max(M)={info['max_M']:.3e} max|Mg|={info['max_Mg']:.3e}")

    # 4) stopping rule
    done = stopper.update(base_lr, s)
    return done, base_lr, info

@torch.no_grad()
def _theta_rms(params):
    n, s = 0, 0.0
    for p in params:
        if p is None: continue
        n += p.numel()
        s += float((p**2).sum())
    return (s / max(1, n))**0.5

@torch.no_grad()
def _max_M_and_Mg(opt):
    """Return (max_M, max_|M*g_eff|, eta_eff, noise_scale)."""
    max_M = 0.0
    max_Mg = 0.0
    eta_eff = None
    noise_scale = None

    for g in opt.param_groups:
        lr = g['lr']
        mode = g['mode']
        n_data = float(g.get('n_data', 1.0))
        eta_eff = lr * n_data if mode == 'scale_lr' else lr

        beta2 = g['beta2']; eps = g['eps']; use_bc = g['use_bias_correction']
        noise_scale = float(g.get('noise_scale', 1.0))
        normalized = g.get('normalized_loss', False)

        # grad scaling must mirror your optimizer’s logic
        grad_scale = n_data if (normalized and mode == 'scale_grad') else 1.0
        bc = (1.0 - beta2 ** max(1, getattr(opt, "_num_steps", 1))) if use_bc else 1.0

        for p in g['params']:
            if p.grad is None: continue
            v = opt.state[p]['v']
            v_hat = v / bc if use_bc else v
            M = (v_hat.sqrt().add_(eps)).reciprocal()
            max_M  = max(max_M,  float(M.max()))
            max_Mg = max(max_Mg, float((M * (p.grad * grad_scale)).abs().max()))
    return max_M, max_Mg, eta_eff, noise_scale

@torch.no_grad()
def snapshot_tensors(params):
    """params: iterable of Tensors/Parameters. Returns detached clones."""
    return [p.detach().clone() for p in params]

@torch.no_grad()
def theta_rms_from(params):
    n, s = 0, 0.0
    for p in params:
        n += p.numel()
        s += float((p**2).sum())
    return (s / max(1, n))**0.5

@torch.no_grad()
def sgld_heartbeat_prestep_params(params, opt, loss_value=None, phase="auto"):
    """
    Call AFTER loss.backward() and BEFORE opt.step().
    `params` is an iterable (e.g., [ΔX_src]).
    """
    theta_r = theta_rms_from(params)

    # scan optimizer state to get max(M) and max|M*g_eff|
    max_M = 0.0
    max_Mg = 0.0
    eta_eff = None
    noise_scale = None

    for g in opt.param_groups:
        lr = g['lr']; mode = g['mode']; n_data = float(g.get('n_data', 1.0))
        eta_eff = lr * n_data if mode == 'scale_lr' else lr

        beta2 = g['beta2']; eps = g['eps']; use_bc = g['use_bias_correction']
        noise_scale = float(g.get('noise_scale', 1.0))
        normalized = g.get('normalized_loss', False)
        grad_scale = n_data if (normalized and mode == 'scale_grad') else 1.0
        bc = (1.0 - beta2 ** max(1, getattr(opt, "_num_steps", 1))) if use_bc else 1.0

        for p in g['params']:
            if p.grad is None: continue
            v = opt.state[p]['v']
            v_hat = v / bc if use_bc else v
            M = (v_hat.sqrt().add_(eps)).reciprocal()
            max_M  = max(max_M,  float(M.max()))
            max_Mg = max(max_Mg, float((M * (p.grad * grad_scale)).abs().max()))

    drift_inf = 0.5 * (eta_eff or 0.0) * max_Mg
    noise_std_max = math.sqrt(max(eta_eff or 0.0, 0.0)) * math.sqrt(max_M) * (noise_scale or 1.0)
    snr = drift_inf / (noise_std_max + 1e-12)

    loss_str = "" if loss_value is None else f" loss={float(loss_value):.3e}"
    frozen = bool(getattr(opt, "_frozen", False))
    phase_lbl = "S4-SAMPLE" if (noise_scale >= 0.999 and frozen) else "OPT"
    print(f"[{phase_lbl}] max(M)={max_M:.2e} max|M*g|={max_Mg:.2e} "
          f"eta_eff={eta_eff:.2e} drift_inf={drift_inf:.2e} "
          f"noise_max={noise_std_max:.2e} SNR={snr:.2f}{loss_str}")

@torch.no_grad()
def heartbeat_poststep_from(prev_params, params, rel_floor_scale=1e-3, abs_floor=1e-8):
    # RMS(theta) for scale
    n, s = 0, 0.0
    for p in params:
        n += p.numel(); s += float((p**2).sum())
    theta_rms = (s / max(1, n))**0.5
    rel_floor = max(rel_floor_scale * theta_rms, abs_floor)

    numel, sq_d, max_rel = 0, 0.0, 0.0
    any_nan = False; any_inf = False
    for p0, p1 in zip(prev_params, params):
        d = p1 - p0
        sq_d += float((d**2).sum()); numel += p1.numel()
        denom = p1.abs().clamp_min(rel_floor)
        rel = (d.abs() / denom).max()
        max_rel = max(max_rel, float(rel))
        any_nan |= torch.isnan(p1).any().item() or torch.isnan(d).any().item()
        any_inf |= torch.isinf(p1).any().item() or torch.isinf(d).any().item()
    delta_rms = (sq_d / max(1, numel))**0.5
    print(f"[MOVE] ΔRMS={delta_rms:.2e} max_rel={max_rel:.2e} (floor={rel_floor:.2e}) NaN={any_nan} Inf=False")
