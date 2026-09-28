"""
Block Gauss-Newton metric for the SPIDER samplers (`preconditioning.type = "block_gn"`).

Motivation
----------
The event-shift posterior is extremely stiff: an event with thousands of differential times
has a conditional posterior a few metres wide, an event with a handful of rows is bounded only
by the event prior (hundreds of metres to kilometres). A diagonal RMSprop metric built from
minibatch gradient statistics resolves only a small part of that range (its entries scale like
the posterior *width*, not the variance, and the `eps` floor compresses them further), so weakly
observed events random-walk across their marginals thousands of times more slowly than well
observed ones. The same stiffness appears between the spatial and origin-time coordinates.

Method
------
For every event `i` the Gauss-Newton (Fisher) block of the data term is

    F_i = sum_{rows r touching i} w_r J_{r,i}^T J_{r,i},      w_r = 1 / sigma_phase(r)^2,

where `J_{r,i}` is the 1x4 Jacobian of the predicted differential time with respect to the
event's (x, y, z, t) shift (`[-grad T, -1]` for the first endpoint, `[+grad T, +1]` for the
second). Adding the event-prior precision `P_i` (hierarchical per-cluster matrix when present,
otherwise the fixed Gaussian prior, otherwise an explicit fallback) gives the Laplace
approximation of the event's *conditional* precision given its neighbours. Its inverse

    M_i = (F_i + P_i)^{-1},          L_i L_i^T = M_i   (symmetric square root)

is the per-event metric.

The samplers take the preconditioned Langevin step

    dX_i <- dX_i - s * M_i * g_i + sqrt(2 s T) * L_i * z_i,       z_i ~ N(0, I_4),

with `g_i` the sum-likelihood gradient (`N * minibatch mean`) and `s` a *dimensionless* step:
for a locally quadratic posterior, `s` is the fraction of the Newton step taken per iteration,
independent of how well each event is observed and of the units of the coordinates. Every
event therefore mixes on the same time scale in every coordinate, and the blocked
reparameterization (`dt_scale`) is unnecessary (it is ignored under `block_gn`).

Gauge projection is required. Differential-time data are (nearly) invariant under a rigid
translation of a connected component, so the component mean is held only by priors; the
centroid prior `0.5 n_k m_k^T P_c m_k` (a 10 m std is typical) has a curvature that exceeds any
single event's block by orders of magnitude while being spread over all events of the
component. A block-diagonal Newton step is unstable along that direction, so `block_gn` is only
allowed together with `inference.runtime.gauge_projection.enabled = true`: the component mean
(in the projected dims) is a hard constraint, exactly as with the other preconditioners under
gauge projection.

On the constrained manifold the step is the *exact* Newton step of the block-diagonal model,
not the projection of the unconstrained one (that would let the kilometre-scale steps of weak
events leak into the metre-scale events through the mean and diverge). With the constraint
`sum_{i in k} E dX_i = 0` (E selects the projected dims) and multiplier `lambda_k`:

    y_i   = M_i (g_i + E^T lambda_k),      lambda_k = -S_k^{-1} sum_{i in k} E M_i g_i,
    S_k   = sum_{i in k} E M_i E^T                                        (one dxd matrix per component)
    xi_i  = z_i - M_i E^T S_k^{-1} sum_{j in k} E z_j,   z_j ~ N(0, M_j)   (constrained noise)

`y` has zero component mean by construction and `xi ~ N(0, (P H P)^+)` is the Gaussian
conditioned on the constraint. On the constrained subspace the data Hessian is bounded by twice
the block diagonal (normalized-Laplacian bound), so the drift-only iteration is a contraction for
every `s < 1` regardless of how heterogeneous the blocks are.

Because the metric is held fixed between explicit refreshes, no pSGLD Gamma correction is
needed and the sampling kernel is exactly time-homogeneous within a phase.

Step-size semantics
-------------------
The samplers receive `lr = lr_user / N` and multiply minibatch-mean gradients by `N`. Under
`block_gn` the natural step is `s = lr * N = lr_user`, i.e. **`inference.sampler.lr[1..3]` is the
dimensionless step directly** (typical values 0.05-0.5; values >= 2 diverge on a quadratic).

Refresh policy
--------------
The metric is (re)computed lazily on the first sampler step after it is requested. Requests are
issued at the start of every sampler phase, every `refresh_every_epochs` epochs, and after each
hierarchical-prior update (`refresh_on_prior_update`); the last two are skipped while the
preconditioner is frozen for the phase (`freeze_preconditioner_sampling`), so a frozen Phase 4
runs one fixed metric.

Limits
------
* The metric is block-diagonal: it equalizes *per-event* stiffness. Collective slow modes
  (weakly anchored sub-groups moving together) are not accelerated.
* The data weights are the independent-Gaussian `phase_unc` terms; the shared-event
  random-effect correlation is ignored in the metric (it only makes the metric slightly
  conservative, which is safe).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from spider.utils.console import info, warn

BLOCK_GN_NAME = "block_gn"

# Config defaults (mirrored in docs/sampler-options.md and configuration-reference.md).
BLOCK_GN_DEFAULTS: Dict[str, Any] = {
    "fallback_std": None,          # [km, km, km, s] prior std used only when no event prior is active
    "min_std": None,               # [km, km, km, s] floor on the implied per-coordinate std (adds variance)
    "refresh_every_epochs": None,  # int | None
    "refresh_on_prior_update": True,
    "batch_size": 262144,          # rows per Jacobian batch
    "jitter": 1e-6,                # relative eigenvalue floor for (F + P)
}

_EVENT_DIM = 4
_TINY = torch.finfo(torch.float64).tiny


def read_block_gn_config(params: dict) -> Dict[str, Any]:
    """Resolve the block_gn options from materialized runtime params (legacy-bridge keys)."""
    cfg = dict(BLOCK_GN_DEFAULTS)
    for key in BLOCK_GN_DEFAULTS:
        pk = f"sampler_preconditioning_block_gn_{key}"
        if pk in params and params[pk] is not None:
            cfg[key] = params[pk]
    return cfg


def is_block_gn_group(group: dict) -> bool:
    return bool(group.get("preconditioning", False)) and (
        str(group.get("preconditioner", "")).strip().lower() == BLOCK_GN_NAME
    )


def sampler_uses_block_gn(sampler) -> bool:
    if sampler is None or not hasattr(sampler, "param_groups"):
        return False
    return any(is_block_gn_group(g) for g in sampler.param_groups)


# --------------------------------------------------------------------------------------
# Metric construction
# --------------------------------------------------------------------------------------
def compute_event_fisher_blocks(
    *,
    II: torch.Tensor,
    YY: torch.Tensor,
    X_src: torch.Tensor,
    dX: torch.Tensor,
    model: torch.nn.Module,
    sigma_ps: Sequence[float],
    batch_size: int = 262144,
) -> torch.Tensor:
    """Accumulate the per-event 4x4 Gauss-Newton blocks `sum_r w_r J^T J` over all rows.

    Returns a float64 tensor of shape (n_events, 4, 4) on `dX.device`.
    """
    device = dX.device
    n_ev = int(dX.shape[0])
    N = int(II.shape[0])
    F = torch.zeros((n_ev, _EVENT_DIM, _EVENT_DIM), dtype=torch.float64, device=device)
    if N == 0:
        return F
    sp, ss = float(sigma_ps[0]), float(sigma_ps[1])
    inv_var_p, inv_var_s = 1.0 / (sp * sp), 1.0 / (ss * ss)
    bs = max(1, int(batch_size))
    pos_all = (X_src + dX).detach()
    was_training = bool(getattr(model, "training", False))
    for i0 in range(0, N, bs):
        i1 = min(N, i0 + bs)
        idx = II[i0:i1]
        y = YY[i0:i1]
        e1 = idx[:, 0].to(torch.int64)
        e2 = idx[:, 1].to(torch.int64)
        X_rec = y[:, 1:4]
        phase = y[:, 4:5]
        w = torch.where(y[:, 4] > 0.5,
                        torch.full_like(y[:, 0], inv_var_s),
                        torch.full_like(y[:, 0], inv_var_p)).to(torch.float64)
        B = int(e1.numel())
        with torch.enable_grad():
            leaf = torch.cat([pos_all[e1, :3], pos_all[e2, :3]], dim=0).detach().requires_grad_(True)
            inp = torch.cat([leaf, torch.cat([X_rec, X_rec], 0), torch.cat([phase, phase], 0)], dim=1)
            T = model(inp).reshape(-1)
            grad = torch.autograd.grad(T.sum(), leaf, create_graph=False, retain_graph=False)[0]
        g1 = grad[:B]
        g2 = grad[B:]
        ones = torch.ones((B, 1), device=device, dtype=grad.dtype)
        J1 = torch.cat([-g1, -ones], dim=1).to(torch.float64)   # (B,4)
        J2 = torch.cat([g2, ones], dim=1).to(torch.float64)
        ok = torch.isfinite(J1).all(dim=1) & torch.isfinite(J2).all(dim=1) & torch.isfinite(w)
        if not bool(ok.all()):
            J1, J2, w, e1, e2 = J1[ok], J2[ok], w[ok], e1[ok], e2[ok]
        wb = w.view(-1, 1, 1)
        F.index_add_(0, e1, torch.bmm(J1.unsqueeze(2), J1.unsqueeze(1)) * wb)
        F.index_add_(0, e2, torch.bmm(J2.unsqueeze(2), J2.unsqueeze(1)) * wb)
    if was_training:
        model.train()
    return F


def event_prior_precision_blocks(
    *,
    n_events: int,
    prior_precision: Optional[torch.Tensor],
    cluster_ids: Optional[torch.Tensor],
    fallback_std: Optional[Sequence[float]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Per-event (n_events,4,4) float64 prior precision, or None when nothing is available."""
    if isinstance(prior_precision, torch.Tensor):
        P = prior_precision.detach().to(device=device, dtype=torch.float64)
        if P.ndim == 2:
            return P.unsqueeze(0).expand(n_events, _EVENT_DIM, _EVENT_DIM)
        if P.ndim == 3:
            if int(P.shape[0]) == 1 or cluster_ids is None:
                return P[:1].expand(n_events, _EVENT_DIM, _EVENT_DIM)
            cid = cluster_ids.to(device=device, dtype=torch.int64).reshape(-1)
            return P.index_select(0, cid)
        raise ValueError(f"prior_precision must be (4,4) or (K,4,4); got {tuple(P.shape)}")
    if fallback_std is not None:
        std = torch.as_tensor([float(s) for s in fallback_std], dtype=torch.float64, device=device)
        if std.numel() != _EVENT_DIM or bool((std <= 0).any()):
            raise ValueError("block_gn.fallback_std must be 4 positive numbers [km, km, km, s]")
        return torch.diag(1.0 / (std * std)).unsqueeze(0).expand(n_events, _EVENT_DIM, _EVENT_DIM)
    return None


def _floored_inverse_and_sqrt(H: torch.Tensor, jitter: float, extra_diag_var: Optional[torch.Tensor] = None):
    """Batch (n,4,4) SPD-ish -> (M, L, std, n_clamped) with a relative eigenvalue floor."""
    H = 0.5 * (H + H.transpose(1, 2))
    evals, evecs = torch.linalg.eigh(H)                          # ascending
    lam_max = evals[:, -1:].clamp_min(0.0)
    # Relative floor per block; blocks with no information at all borrow the typical scale of
    # the informative blocks so the metric stays finite.
    informative = lam_max > 0
    lam_ref = lam_max[informative].median() if bool(informative.any()) else torch.tensor(1.0, dtype=H.dtype, device=H.device)
    lam_scale = torch.where(informative, lam_max, lam_ref.expand_as(lam_max))
    floor = (lam_scale * float(max(jitter, 0.0))).clamp_min(_TINY)
    clamped = evals < floor
    evals_f = torch.maximum(evals, floor)
    M = torch.matmul(evecs * (1.0 / evals_f).unsqueeze(1), evecs.transpose(1, 2))
    if extra_diag_var is not None:
        M = M + torch.diag(extra_diag_var).unsqueeze(0)
    M = 0.5 * (M + M.transpose(1, 2))
    # Symmetric square root (L L^T = M): unlike Cholesky it cannot fail on blocks that are SPD
    # only up to roundoff.
    mu, W = torch.linalg.eigh(M)
    L = torch.matmul(W * torch.sqrt(mu.clamp_min(_TINY)).unsqueeze(1), W.transpose(1, 2))
    std = torch.sqrt(torch.diagonal(M, dim1=1, dim2=2).clamp_min(0.0))
    return M, L, std, int(clamped.any(dim=1).sum().item())


def assemble_block_metric(
    F: torch.Tensor,
    P: Optional[torch.Tensor],
    *,
    jitter: float = 1e-6,
    min_std: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Invert `H = F + P` block-wise (eigenvalue-floored) and take its symmetric square root.

    Returns dict(M=(n,4,4) float32, L=(n,4,4), std=(n,4) implied per-coordinate std,
    n_clamped, n_unobserved).
    """
    H = F.to(torch.float64)
    if P is not None:
        H = H + P.to(torch.float64)
    extra = None
    if min_std is not None:
        ms = torch.as_tensor([float(s) for s in min_std], dtype=torch.float64, device=H.device)
        if ms.numel() != _EVENT_DIM or bool((ms < 0).any()):
            raise ValueError("block_gn.min_std must be 4 non-negative numbers [km, km, km, s]")
        extra = ms * ms
    M, L, std, n_clamped = _floored_inverse_and_sqrt(H, jitter, extra)
    out: Dict[str, Any] = {
        "M": M.to(torch.float32).contiguous(),
        "L": L.to(torch.float32).contiguous(),
        "std": std.to(torch.float32).contiguous(),
        "n_clamped": n_clamped,
        "n_unobserved": int((F.abs().sum(dim=(1, 2)) == 0).sum().item()),
    }
    return out


def _maybe_broadcast(t: torch.Tensor) -> torch.Tensor:
    """Make the metric bit-identical across DDP ranks (each rank sums the full dataset in a
    different floating-point order; ranks must apply identical updates)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(t, src=0)
    except Exception:
        pass
    return t


def compute_block_gn_metric_from_state(state, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the metric from a LocateState (data, model, current shifts, active priors)."""
    params = getattr(state, "params", {}) or {}
    dX = state.dX_src.detach()
    n_ev = int(dX.shape[0])
    device = dX.device

    # Data weights: the active likelihood's phase uncertainties.
    sigma_ps = None
    try:
        from spider.core.state import _current_noise_scales
        sp, ss = _current_noise_scales(state)
        sigma_ps = (float(sp), float(ss))
    except Exception:
        pu = params.get("phase_unc", None)
        if isinstance(pu, (list, tuple)) and len(pu) >= 2:
            sigma_ps = (float(pu[0]), float(pu[1]))
    if sigma_ps is None:
        raise RuntimeError("block_gn: cannot resolve phase uncertainties (state.scale_theta / params['phase_unc'])")

    cluster_ids = getattr(state, "cluster_ids", None)
    F = compute_event_fisher_blocks(
        II=state.II, YY=state.YY, X_src=state.X_src, dX=dX, model=state.model,
        sigma_ps=sigma_ps, batch_size=int(cfg.get("batch_size", BLOCK_GN_DEFAULTS["batch_size"])),
    )
    F = _maybe_broadcast(F)

    # Prior precision: hierarchical matrix -> fixed Gaussian prior -> explicit fallback -> none.
    prior_enabled = bool(params.get("prior_event_enable", True)) and bool(params.get("_prior_event_runtime_enable", True))
    P_src = "none"
    prior_precision = getattr(state, "event_precision_matrix", None) if prior_enabled else None
    fallback_std = cfg.get("fallback_std", None)
    if isinstance(prior_precision, torch.Tensor):
        P_src = "hierarchical" if prior_precision.ndim == 3 else "event_prior"
    elif prior_enabled and isinstance(params.get("prior_event_std", None), (list, tuple)) and len(params["prior_event_std"]) == _EVENT_DIM:
        fallback_std = params["prior_event_std"]
        P_src = "event_prior_std"
    elif fallback_std is not None:
        P_src = "fallback_std"
    P = event_prior_precision_blocks(
        n_events=n_ev, prior_precision=prior_precision, cluster_ids=cluster_ids,
        fallback_std=fallback_std, device=device,
    )

    out = assemble_block_metric(F, P, jitter=float(cfg.get("jitter", 1e-6)), min_std=cfg.get("min_std", None))
    out["prior_source"] = P_src
    out["sigma_ps"] = sigma_ps

    # Summary line: implied conditional std percentiles per coordinate.
    try:
        std = out["std"]
        q = torch.tensor([0.05, 0.5, 0.95], device=std.device)
        pct = torch.quantile(std.to(torch.float32), q, dim=0)  # (3,4)
        def _fmt(j: int, scale: float, unit: str) -> str:
            return f"{pct[0, j] * scale:.0f}/{pct[1, j] * scale:.0f}/{pct[2, j] * scale:.0f} {unit}"
        info(
            "Block Gauss-Newton metric built: implied conditional std p5/p50/p95 "
            f"x {_fmt(0, 1e3, 'm')} | y {_fmt(1, 1e3, 'm')} | z {_fmt(2, 1e3, 'm')} | t {_fmt(3, 1e3, 'ms')} "
            f"| prior={P_src} | unobserved events={out['n_unobserved']} | eigen-floored={out['n_clamped']}",
            section="PRECOND",
        )
        if P is None:
            warn(
                "block_gn: no event prior is active and block_gn.fallback_std is unset; weakly observed "
                "events are regularized only by the relative eigenvalue floor (jitter). Set "
                "inference.sampler.preconditioning.block_gn.fallback_std to bound their step size.",
                section="PRECOND",
            )
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------------------
# Optimizer-side plumbing
# --------------------------------------------------------------------------------------
GAUGE_REQUIRED_MESSAGE = (
    "preconditioning.type='block_gn' requires inference.runtime.gauge_projection.enabled=true: the "
    "component-mean (translation) mode is held only by the centroid prior, whose curvature is far above "
    "any single event's block, and a block-diagonal metric is unstable along it unless the mean is "
    "constrained."
)
GAUGE_DIMS_MESSAGE = (
    "preconditioning.type='block_gn' with the centroid prior enabled requires "
    "inference.runtime.gauge_projection.dims to cover all four coordinates [0, 1, 2, 3]: the centroid "
    "prior acts on every coordinate of the component mean."
)


def check_block_gn_gauge_requirement(*, gauge_enabled: bool, gauge_dims=None, centroid_prior_enabled: bool = False) -> None:
    if not bool(gauge_enabled):
        raise ValueError(GAUGE_REQUIRED_MESSAGE)
    if centroid_prior_enabled and gauge_dims is not None:
        try:
            dims = sorted(int(d) for d in gauge_dims)
        except Exception:
            dims = []
        if dims != [0, 1, 2, 3]:
            raise ValueError(GAUGE_DIMS_MESSAGE)


def block_gn_gauge_spec(opt: torch.optim.Optimizer, p: torch.Tensor):
    """(cid (n,) int64, K, dims tuple) describing the gauge constraint attached to the optimizer
    for parameter `p`, or None when gauge projection is not active for it."""
    if not bool(getattr(opt, "_gauge_project_enable", False)) or (getattr(opt, "_gauge_project_param", None) is not p):
        return None
    dims = tuple(int(d) for d in getattr(opt, "_gauge_project_dims", (0, 1, 2)))
    mode = str(getattr(opt, "_gauge_project_mode", "global"))
    cid = getattr(opt, "_gauge_cluster_ids", None)
    if mode == "cluster" and isinstance(cid, torch.Tensor) and int(cid.numel()) == int(p.shape[0]):
        cid = cid.to(device=p.device, dtype=torch.int64).reshape(-1)
        cc = getattr(opt, "_gauge_cluster_counts", None)
        K = int(cc.numel()) if isinstance(cc, torch.Tensor) else int(cid.max().item()) + 1
    else:
        cid = torch.zeros(int(p.shape[0]), dtype=torch.int64, device=p.device)
        K = 1
    return cid, K, dims


def attach_block_gn_provider(opt: torch.optim.Optimizer, *, params: dict, state) -> None:
    """Give the optimizer a closure that rebuilds the metric from the live LocateState, and
    request an initial build. Called by the backend factory; safe to call for other types."""
    if not sampler_uses_block_gn(opt):
        return
    check_block_gn_gauge_requirement(
        gauge_enabled=bool(params.get("gauge_project_enable", False)),
        gauge_dims=params.get("gauge_project_dims", None),
        centroid_prior_enabled=bool(params.get("prior_centroid_enable", False)),
    )
    if not bool(getattr(opt, "_gauge_project_enable", False)):
        raise RuntimeError("block_gn: gauge projection is enabled in config but not attached to the sampler")
    cfg = read_block_gn_config(params)
    setattr(opt, "_block_gn_cfg", cfg)
    setattr(opt, "_block_gn_provider", lambda: compute_block_gn_metric_from_state(state, cfg))
    request_block_gn_refresh(opt)


def request_block_gn_refresh(opt: torch.optim.Optimizer) -> None:
    for g in getattr(opt, "param_groups", []):
        if is_block_gn_group(g):
            g["block_gn_refresh_pending"] = True


def ensure_block_gn_metric(opt: torch.optim.Optimizer, group: dict, st: dict, p: torch.Tensor):
    """Return (M, L) for parameter `p`, rebuilding via the provider when requested/missing."""
    need = bool(group.get("block_gn_refresh_pending", False)) or ("block_gn_M" not in st) or ("block_gn_L" not in st)
    if need:
        provider = getattr(opt, "_block_gn_provider", None)
        if provider is None:
            if "block_gn_M" in st and "block_gn_L" in st:
                group["block_gn_refresh_pending"] = False
            else:
                raise RuntimeError(
                    "block_gn preconditioner has no metric: attach a provider "
                    "(spider.optim.block_gn.attach_block_gn_provider) or set state['block_gn_M'/'block_gn_L']."
                )
        else:
            out = provider()
            st["block_gn_M"] = out["M"].to(device=p.device)
            st["block_gn_L"] = out["L"].to(device=p.device)
            st["block_gn_std"] = out["std"].to(device=p.device)
            st["block_gn_builds"] = int(st.get("block_gn_builds", 0)) + 1
            group["block_gn_refresh_pending"] = False
    M, L = st["block_gn_M"], st["block_gn_L"]
    if p.ndim != 2 or int(p.shape[1]) != _EVENT_DIM or tuple(M.shape) != (int(p.shape[0]), _EVENT_DIM, _EVENT_DIM):
        raise RuntimeError(
            f"block_gn metric shape {tuple(M.shape)} does not match parameter {tuple(p.shape)}; "
            "the block_gn preconditioner supports the (n_events, 4) event-shift tensor only."
        )
    if M.dtype != p.dtype:
        M = M.to(p.dtype); L = L.to(p.dtype)
        st["block_gn_M"], st["block_gn_L"] = M, L
    return M, L


def block_gn_drift_and_noise(
    *,
    M: torch.Tensor,
    L: torch.Tensor,
    g: torch.Tensor,
    noise_std: float,
    gauge=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Metric-preconditioned gradient (n,4) and, when `noise_std > 0`, a matching noise draw.

    Without `gauge`: drift = M g, noise = noise_std * L z ~ N(0, noise_std^2 M).
    With `gauge = (cid, K, dims)`: the exact constrained Newton step and the constrained
    Gaussian draw under `sum_{i in k} g_dims(dX_i) = 0` (see module docstring); both have zero
    component mean in `dims` by construction.
    """
    drift = torch.bmm(M, g.unsqueeze(-1)).squeeze(-1)
    noise = None
    if noise_std > 0.0:
        noise = noise_std * torch.bmm(L, torch.randn_like(g).unsqueeze(-1)).squeeze(-1)
    if gauge is None:
        return drift, noise
    cid, K, dims = gauge
    if len(dims) == 0:
        return drift, noise
    d_idx = torch.as_tensor(list(dims), dtype=torch.int64, device=M.device)
    Md = M.index_select(2, d_idx)                                  # (n,4,d)  = M_i E^T
    Mdd = Md.index_select(1, d_idx)                                # (n,d,d)  = E M_i E^T
    S = torch.zeros((K, len(dims), len(dims)), dtype=torch.float64, device=M.device)
    S.index_add_(0, cid, Mdd.to(torch.float64))
    S = 0.5 * (S + S.transpose(1, 2))
    # components with no events (K from counts) would be singular: guard with an identity
    empty = torch.diagonal(S, dim1=1, dim2=2).sum(1) <= 0
    if bool(empty.any()):
        S[empty] = torch.eye(len(dims), dtype=torch.float64, device=M.device)
    Sinv = torch.linalg.inv(S)                                    # (K,d,d)

    def _constrain(v: torch.Tensor) -> torch.Tensor:
        # v (n,4) -> v - M E^T S^{-1} sum_k E v   (zero component mean in dims)
        sum_k = torch.zeros((K, len(dims)), dtype=torch.float64, device=M.device)
        sum_k.index_add_(0, cid, v.index_select(1, d_idx).to(torch.float64))
        lam = torch.bmm(Sinv, sum_k.unsqueeze(-1)).squeeze(-1).to(v.dtype)   # (K,d)
        return v - torch.bmm(Md, lam.index_select(0, cid).unsqueeze(-1)).squeeze(-1)

    drift = _constrain(drift)
    if noise is not None:
        noise = _constrain(noise)
    return drift, noise


def block_gn_natural_step(group: dict) -> float:
    """`lr` arrives as lr_user / N; the dimensionless step is lr_user = lr * N."""
    return float(group.get("lr", 0.0)) * float(group.get("n_obs", 1))


def block_gn_diag_metric(st: dict, like: torch.Tensor) -> torch.Tensor:
    """Diagonal proxy of the metric (for gradient-noise diagnostics)."""
    M = st.get("block_gn_M", None)
    if isinstance(M, torch.Tensor) and M.ndim == 3 and int(M.shape[0]) == int(like.shape[0]):
        return torch.diagonal(M, dim1=1, dim2=2).to(like.dtype)
    return torch.ones_like(like)


# --------------------------------------------------------------------------------------
# Runner-side hooks (called from the phase loops / prior update)
# --------------------------------------------------------------------------------------
def block_gn_epoch_hook(state, sampler, *, epoch: int, phase_start: bool, log: bool = True) -> None:
    """Request a metric rebuild at phase start and at the configured cadence (unless frozen)."""
    if not sampler_uses_block_gn(sampler):
        return
    g0 = sampler.param_groups[0]
    if phase_start:
        request_block_gn_refresh(sampler)
        if log:
            info(f"block_gn: metric rebuild scheduled at phase start (epoch {int(epoch)}); "
                 f"natural step s={block_gn_natural_step(g0):g}", section="PRECOND")
        return
    if bool(g0.get("freeze_preconditioner", False)):
        return
    cfg = getattr(sampler, "_block_gn_cfg", None) or read_block_gn_config(getattr(state, "params", {}) or {})
    every = cfg.get("refresh_every_epochs", None)
    if every is not None and int(every) > 0 and int(epoch) > 0 and (int(epoch) % int(every)) == 0:
        request_block_gn_refresh(sampler)


def block_gn_prior_updated_hook(state, sampler) -> None:
    """Called after a hierarchical-prior update; rebuild unless frozen or disabled by config."""
    if not sampler_uses_block_gn(sampler):
        return
    g0 = sampler.param_groups[0]
    if bool(g0.get("freeze_preconditioner", False)):
        return
    cfg = getattr(sampler, "_block_gn_cfg", None) or read_block_gn_config(getattr(state, "params", {}) or {})
    if bool(cfg.get("refresh_on_prior_update", True)):
        request_block_gn_refresh(sampler)
