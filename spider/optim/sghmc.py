import math
from typing import Optional
import torch

from .gauge import project_event_mean_inplace

_COMPONENT_LRD_ALIASES = {"component_lrd", "cc_lrd", "block_lrd", "component-lrd"}


def _canonical_preconditioner_name(preconditioner: str) -> str:
    p = str(preconditioner).strip().lower()
    if p in _COMPONENT_LRD_ALIASES:
        return "component_lrd"
    return p


def _blocked_reparam_scale(group: dict, p: torch.Tensor) -> torch.Tensor | None:
    if not bool(group.get("reparam_blocked_enable", False)):
        return None
    if (not isinstance(p, torch.Tensor)) or p.ndim != 2 or int(p.shape[1]) < 4:
        return None
    s_xyz = float(group.get("reparam_blocked_spatial_scale", 1.0))
    s_dt = float(group.get("reparam_blocked_dt_scale", 1.0))
    if (not math.isfinite(s_xyz)) or s_xyz <= 0.0:
        s_xyz = 1.0
    if (not math.isfinite(s_dt)) or s_dt <= 0.0:
        s_dt = 1.0
    if abs(s_xyz - 1.0) < 1e-12 and abs(s_dt - 1.0) < 1e-12:
        return None
    scale = torch.ones_like(p)
    scale[:, :3] = float(s_xyz)
    scale[:, 3] = float(s_dt)
    return scale


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


def _build_component_event_groups(
    *,
    state: dict,
    component_ids: torch.Tensor,
    n_events: int,
) -> list[torch.Tensor]:
    cache = state.get("lrd_component_groups_cache", None)
    if isinstance(cache, dict):
        cid_cached = cache.get("component_ids", None)
        groups_cached = cache.get("groups", None)
        if (
            isinstance(cid_cached, torch.Tensor)
            and isinstance(groups_cached, list)
            and cid_cached.device == component_ids.device
            and int(cid_cached.numel()) == int(component_ids.numel())
            and int(n_events) == int(cache.get("n_events", -1))
        ):
            try:
                if int(cid_cached.data_ptr()) == int(component_ids.data_ptr()):
                    return groups_cached
            except Exception:
                pass
    if n_events <= 0:
        groups: list[torch.Tensor] = []
    else:
        cid = component_ids.to(dtype=torch.int64)
        order = torch.argsort(cid)
        if int(order.numel()) <= 0:
            groups = []
        else:
            cid_sorted = cid.index_select(0, order)
            if int(cid_sorted.numel()) <= 1:
                groups = [order]
            else:
                split = torch.nonzero(cid_sorted[1:] != cid_sorted[:-1], as_tuple=False).flatten() + 1
                starts = torch.cat([split.new_tensor([0]), split], dim=0)
                ends = torch.cat([split, split.new_tensor([int(order.numel())])], dim=0)
                groups = []
                for s, e in zip(starts.tolist(), ends.tolist()):
                    if int(e) > int(s):
                        groups.append(order[int(s):int(e)])
    state["lrd_component_groups_cache"] = {
        "component_ids": component_ids,
        "groups": groups,
        "n_events": int(n_events),
    }
    return groups


def _build_component_lrd_metric(
    *,
    group: dict,
    state: dict,
    grad_for_precond: torch.Tensor,
    beta: float,
    eps: float,
    freeze_preconditioner: bool,
) -> tuple[torch.Tensor, list[dict]]:
    """
    Component-wise low-rank+diagonal preconditioner.

    This keeps low-rank couplings strictly within DD connected components.
    Returns:
      d_flat: full diagonal term over all entries
      factors: list of per-component factors {event_idx, U, lam}
    """
    g = grad_for_precond
    if g.ndim != 2:
        d_flat, U, lam = _build_lrd_metric(
            group=group,
            state=state,
            grad_for_precond=grad_for_precond,
            beta=beta,
            eps=eps,
            freeze_preconditioner=freeze_preconditioner,
        )
        factors: list[dict] = []
        if int(U.numel()) > 0 and int(lam.numel()) > 0:
            factors.append({"event_idx": None, "U": U, "lam": lam})
        return d_flat, factors

    component_ids = group.get("lrd_component_ids", None)
    if not isinstance(component_ids, torch.Tensor) or component_ids.ndim != 1 or int(component_ids.numel()) != int(g.shape[0]):
        d_flat, U, lam = _build_lrd_metric(
            group=group,
            state=state,
            grad_for_precond=grad_for_precond,
            beta=beta,
            eps=eps,
            freeze_preconditioner=freeze_preconditioner,
        )
        factors = []
        if int(U.numel()) > 0 and int(lam.numel()) > 0:
            factors.append({"event_idx": None, "U": U, "lam": lam})
        return d_flat, factors

    if component_ids.device != g.device:
        component_ids = component_ids.to(device=g.device)

    rank_cfg = int(group.get("lrd_rank", 16))
    mode = str(group.get("lrd_mode", "svd")).strip().lower()
    if mode in {"randomized_svd", "stochastic_svd"}:
        mode = "svd"
    if mode not in {"svd", "oja"}:
        mode = "svd"
    update_every = max(1, int(group.get("lrd_update_every", 20)))
    buffer_size = max(max(2, rank_cfg + 1), int(group.get("lrd_buffer_size", 64)))
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
    d2 = (1.0 / (float(eps) + v_hat.clamp_min(0.0).sqrt())).clamp_min(diag_floor)
    d_flat = d2.reshape(-1)

    comp_states = state.get("lrd_component_states", None)
    if not isinstance(comp_states, dict):
        comp_states = {}
    factors: list[dict] = []
    diag_low = torch.zeros_like(g)
    n_events, d_event = int(g.shape[0]), int(g.shape[1])
    event_groups = _build_component_event_groups(state=state, component_ids=component_ids, n_events=n_events)

    for ev_idx in event_groups:
        m = int(ev_idx.numel())
        if m <= 0:
            continue
        Dk = int(m * d_event)
        rank = max(0, min(int(rank_cfg), Dk))
        if rank <= 0:
            continue
        try:
            comp_key = int(component_ids[int(ev_idx[0].item())].item())
        except Exception:
            comp_key = int(len(factors))
        sub = comp_states.get(comp_key, {})
        if not isinstance(sub, dict):
            sub = {}

        gk = g.index_select(0, ev_idx).reshape(-1)
        U = sub.get("U", None)
        lam = sub.get("lam", None)
        if (not isinstance(U, torch.Tensor)) or U.shape != (Dk, rank) or U.device != g.device or U.dtype != g.dtype:
            U = torch.zeros((Dk, rank), device=g.device, dtype=g.dtype)
        if (not isinstance(lam, torch.Tensor)) or lam.shape != (rank,) or lam.device != g.device or lam.dtype != g.dtype:
            lam = torch.zeros((rank,), device=g.device, dtype=g.dtype)

        if not freeze_preconditioner:
            if mode == "oja":
                if not bool(torch.any(torch.isfinite(U))) or float(U.abs().sum().item()) == 0.0:
                    gn = float(gk.norm().item())
                    if gn > 0.0:
                        U[:, 0] = gk / max(gn, 1e-12)
                    if rank > 1:
                        U[:, 1:] = torch.randn((Dk, rank - 1), device=g.device, dtype=g.dtype)
                    try:
                        U, _ = torch.linalg.qr(U, mode="reduced")
                    except Exception:
                        pass
                q = U.mT @ gk
                U = U + (oja_eta * torch.outer(gk, q))
                try:
                    U, _ = torch.linalg.qr(U, mode="reduced")
                except Exception:
                    pass
                if U.shape[1] > rank:
                    U = U[:, :rank].contiguous()
                if U.shape[1] < rank:
                    U_pad = torch.zeros((Dk, rank), device=g.device, dtype=g.dtype)
                    if U.shape[1] > 0:
                        U_pad[:, : U.shape[1]] = U
                    U = U_pad
                q = U.mT @ gk
                lam.mul_(beta).addcmul_(q, q, value=(1.0 - beta))
            else:
                buf = sub.get("grad_buffer", None)
                if (
                    (not isinstance(buf, torch.Tensor))
                    or buf.ndim != 2
                    or int(buf.shape[1]) != int(Dk)
                    or buf.device != g.device
                    or buf.dtype != g.dtype
                ):
                    buf = torch.zeros((0, Dk), device=g.device, dtype=g.dtype)
                buf = torch.cat([buf, gk.detach().unsqueeze(0)], dim=0)
                if int(buf.shape[0]) > int(buffer_size):
                    buf = buf[-int(buffer_size):, :]
                sub["grad_buffer"] = buf.detach()
                if (int(step_i) % int(update_every) == 0) and int(buf.shape[0]) >= max(2, rank):
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
        sub["U"] = U.detach()
        sub["lam"] = lam.detach()
        comp_states[comp_key] = sub

        if int(U.numel()) > 0 and int(lam.numel()) > 0:
            try:
                diag_low_k = (U * U).matmul(lam).reshape(m, d_event)
                diag_low[ev_idx, :] = diag_low_k
                factors.append({"event_idx": ev_idx, "U": U.detach(), "lam": lam.detach()})
            except Exception:
                pass

    diag_proxy = (d2 + diag_low).clamp_min(diag_floor)
    state["lrd_component_states"] = comp_states
    state["precond_diag"] = diag_proxy.detach()
    return d_flat, factors


def _apply_component_lowrank_drift(
    *,
    grad_for_drift: torch.Tensor,
    pre_flat: torch.Tensor,
    factors: list[dict],
) -> torch.Tensor:
    if not factors:
        return pre_flat
    out_flat = pre_flat
    pre2 = pre_flat.reshape_as(grad_for_drift) if grad_for_drift.ndim == 2 else None
    d_event = int(grad_for_drift.shape[1]) if grad_for_drift.ndim == 2 else 0
    for fac in factors:
        ev_idx = fac.get("event_idx", None)
        U = fac.get("U", None)
        lam = fac.get("lam", None)
        if not isinstance(U, torch.Tensor) or not isinstance(lam, torch.Tensor):
            continue
        if int(U.numel()) <= 0 or int(lam.numel()) <= 0:
            continue
        if ev_idx is None:
            try:
                g_flat = grad_for_drift.reshape(-1)
                q = U.mT @ g_flat
                out_flat = out_flat + (U @ (lam * q))
                if pre2 is not None:
                    pre2 = out_flat.reshape_as(grad_for_drift)
            except Exception:
                continue
            continue
        if pre2 is None or not isinstance(ev_idx, torch.Tensor) or int(ev_idx.numel()) <= 0:
            continue
        try:
            gk = grad_for_drift.index_select(0, ev_idx).reshape(-1)
            q = U.mT @ gk
            add = (U @ (lam * q)).reshape(int(ev_idx.numel()), d_event)
            pre2[ev_idx, :] = pre2.index_select(0, ev_idx) + add
            out_flat = pre2.reshape(-1)
        except Exception:
            continue
    return out_flat


def _apply_component_lowrank_noise(
    *,
    grad_shape_like: torch.Tensor,
    noise_flat: torch.Tensor,
    factors: list[dict],
) -> torch.Tensor:
    if not factors:
        return noise_flat
    out_flat = noise_flat
    noise2 = noise_flat.reshape_as(grad_shape_like) if grad_shape_like.ndim == 2 else None
    d_event = int(grad_shape_like.shape[1]) if grad_shape_like.ndim == 2 else 0
    for fac in factors:
        ev_idx = fac.get("event_idx", None)
        U = fac.get("U", None)
        lam = fac.get("lam", None)
        if not isinstance(U, torch.Tensor) or not isinstance(lam, torch.Tensor):
            continue
        if int(U.numel()) <= 0 or int(lam.numel()) <= 0:
            continue
        if ev_idx is None:
            try:
                z2 = torch.randn((int(lam.numel()),), device=out_flat.device, dtype=out_flat.dtype)
                out_flat = out_flat + (U @ (lam.clamp_min(0.0).sqrt() * z2))
                if noise2 is not None:
                    noise2 = out_flat.reshape_as(grad_shape_like)
            except Exception:
                continue
            continue
        if noise2 is None or not isinstance(ev_idx, torch.Tensor) or int(ev_idx.numel()) <= 0:
            continue
        try:
            z2 = torch.randn((int(lam.numel()),), device=noise2.device, dtype=noise2.dtype)
            add = (U @ (lam.clamp_min(0.0).sqrt() * z2)).reshape(int(ev_idx.numel()), d_event)
            noise2[ev_idx, :] = noise2.index_select(0, ev_idx) + add
            out_flat = noise2.reshape(-1)
        except Exception:
            continue
    return out_flat


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
            # Supported preconditioners in current runtime:
            # - rmsprop (diagonal)
            # - lrd (global low-rank plus diagonal)
            # - component_lrd (component-wise low-rank plus diagonal)
            preconditioner_type = _canonical_preconditioner_name(group.get("preconditioner", "rmsprop"))
            add_noise = bool(group.get("add_noise", True))
            noise_scale = float(group.get("noise_scale", 1.0))
            temperature = float(group.get("temperature", 1.0))
            freeze_preconditioner = bool(group.get("freeze_preconditioner", False))
            grad_ema_beta = float(group.get("grad_ema_beta", 0.99))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                reparam_scale = _blocked_reparam_scale(group, p)

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
                grad_eff = grad if reparam_scale is None else (grad * reparam_scale)
                grad_for_drift = grad_eff.mul(n_obs)
                grad_for_precond = grad_eff
                
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

                if preconditioning and preconditioner_type in {"lrd", "component_lrd"}:
                    if preconditioner_type == "component_lrd":
                        d_flat, comp_factors = _build_component_lrd_metric(
                            group=group,
                            state=state,
                            grad_for_precond=grad_for_precond,
                            beta=float(beta),
                            eps=float(eps),
                            freeze_preconditioner=bool(freeze_preconditioner),
                        )
                        g_flat = grad_for_drift.reshape(-1)
                        pre_flat = d_flat * g_flat
                        pre_flat = _apply_component_lowrank_drift(
                            grad_for_drift=grad_for_drift,
                            pre_flat=pre_flat,
                            factors=comp_factors,
                        )
                    else:
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
                    # Keep diagnostics consistent with other paths.
                    ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                    ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))

                    m.mul_(1.0 - alpha)
                    drift_inc = pre_flat.reshape_as(p)
                    if reparam_scale is not None:
                        drift_inc = drift_inc * reparam_scale
                    m.add_(drift_inc, alpha=-lr)

                    if add_noise and temperature > 0.0 and noise_scale > 0.0:
                        std = math.sqrt(2.0 * alpha * lr) * math.sqrt(max(temperature, 0.0)) * noise_scale
                        noise_flat = torch.randn_like(g_flat) * d_flat.sqrt()
                        if preconditioner_type == "component_lrd":
                            noise_flat = _apply_component_lowrank_noise(
                                grad_shape_like=grad_for_drift,
                                noise_flat=noise_flat,
                                factors=comp_factors,
                            )
                        else:
                            if int(U_lrd.numel()) > 0 and int(lam_lrd.numel()) > 0:
                                try:
                                    z2 = torch.randn((int(lam_lrd.numel()),), device=p.device, dtype=p.dtype)
                                    noise_flat = noise_flat + (U_lrd @ (lam_lrd.clamp_min(0.0).sqrt() * z2))
                                except Exception:
                                    pass
                        noise = std * noise_flat.reshape_as(p)
                        if reparam_scale is not None:
                            noise = noise * reparam_scale
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

                    p.add_(m)
                    continue

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
                else:
                    G = torch.ones_like(p)

                # Update EMA gradient statistics (using drift-scaled grads)
                ema_g.mul_(grad_ema_beta).add_(grad_for_drift, alpha=(1.0 - grad_ema_beta))
                ema_g2.mul_(grad_ema_beta).addcmul_(grad_for_drift, grad_for_drift, value=(1.0 - grad_ema_beta))

                # Momentum update
                # Frictional decay
                m.mul_(1.0 - alpha)
                
                # Drift term: -lr * Precond * g
                if G is not None:
                    # Diagonal Preconditioning
                    drift_inc = G * grad_for_drift
                    if reparam_scale is not None:
                        drift_inc = drift_inc * reparam_scale
                    m.add_(drift_inc, alpha=-lr)
                else:
                    # Identity (should be covered by G=ones, but safety fallback)
                    drift_inc = grad_for_drift
                    if reparam_scale is not None:
                        drift_inc = drift_inc * reparam_scale
                    m.add_(drift_inc, alpha=-lr)

                # Noise injection
                if add_noise and temperature > 0.0 and noise_scale > 0.0:
                    std = math.sqrt(2.0 * alpha * lr) * math.sqrt(max(temperature, 0.0)) * noise_scale
                    
                    if G is not None:
                        # Diagonal Noise: sqrt(G) * epsilon
                        noise = torch.randn_like(p) * std * G.sqrt()
                        if reparam_scale is not None:
                            noise = noise * reparam_scale
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
                        if reparam_scale is not None:
                            noise = noise * reparam_scale
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
            preconditioner_type = _canonical_preconditioner_name(group.get("preconditioner", "rmsprop"))
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
            
            if preconditioner_type in {"lrd", "component_lrd"}:
                G_diag = state.get("precond_diag", None)
                if isinstance(G_diag, torch.Tensor):
                    return _five_num(G_diag)

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
            preconditioner_type = _canonical_preconditioner_name(group.get("preconditioner", "rmsprop"))
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
                    if preconditioner_type in {"lrd", "component_lrd"}:
                        G = state.get("precond_diag", None)
                        if (not isinstance(G, torch.Tensor)) or G.shape != p.shape:
                            G = torch.ones_like(p)
                    elif v is None:
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
            preconditioner_type = _canonical_preconditioner_name(group.get("preconditioner", "rmsprop"))  # "rmsprop" | "lrd" | "component_lrd"
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
                if preconditioning and preconditioner_type in {"lrd", "component_lrd"}:
                    G_eff = state.get("precond_diag", None)
                    if (not isinstance(G_eff, torch.Tensor)) or G_eff.shape != m.shape:
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


