from typing import Dict, List
import time
import torch
import math
import numpy as np
import torch.distributed as dist
from spider.core.hierarchy import update_precision_hyperparameter
from spider.utils.console import info, warn

from spider.core.state import LocateState, _current_noise_scales, _clamp_dX_inplace
from spider.core.batching import _ensure_owner_buckets, _iter_event_batches
from spider.core.modeling import (
    posterior_loss,
    compute_likelihood_loss,
    compute_prior_loss,
    compute_residuals,
    write_output,
)
from spider.utils.wandb_gates import want_wandb_group as _want_wandb_group
from spider.optim.gauge import project_event_mean_inplace


# Standardized stdout helper
def _log(*parts, section: str = "RUN", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)


def _maybe_apply_gauge_projection_for_optimizer(state: LocateState, optimizer: torch.optim.Optimizer) -> None:
    """
    Apply gauge projection (remove translation mode) for optimizers that do NOT implement it internally.

    Sampler backends in `spider.optim.*` already apply gauge projection inside their `step()` method
    (to gradients and optionally to injected noise/momentum). Phase-1 MAP uses torch.optim.Adam,
    which does not. This hook makes `runtime.gauge_projection` apply to locate-map as well.
    """
    try:
        if not bool(state.params.get("gauge_project_enable", False)):
            return
    except Exception:
        return

    # Avoid double-projecting for SPIDER samplers which already do this in optimizer.step().
    try:
        mod = str(getattr(optimizer.__class__, "__module__", "") or "")
        if mod.startswith("spider.optim"):
            return
    except Exception:
        pass

    try:
        dims_v = state.params.get("gauge_project_dims", [0, 1, 2])
        if not isinstance(dims_v, list) or len(dims_v) == 0:
            dims_v = [0, 1, 2]
        dims = tuple(int(d) for d in dims_v)
    except Exception:
        dims = (0, 1, 2)
    try:
        mode = str(state.params.get("gauge_project_mode", "global")).strip().lower()
    except Exception:
        mode = "global"
    if mode not in {"global", "cluster"}:
        mode = "global"

    cid = state.cluster_ids if mode == "cluster" else None
    cc = state.cluster_counts if mode == "cluster" else None

    # Project the gradient of dX_src in-place.
    try:
        g = getattr(state.dX_src, "grad", None)
        if isinstance(g, torch.Tensor) and g.ndim == 2 and int(g.shape[0]) == int(state.dX_src.shape[0]):
            project_event_mean_inplace(g, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
    except Exception:
        pass


def _maybe_apply_gauge_projection_to_momentum_buffers(state: LocateState, optimizer: torch.optim.Optimizer) -> None:
    """
    Optionally project optimizer momentum buffers for non-sampler optimizers (e.g., Adam).

    This prevents the gauge (translation) mode from accumulating in exp_avg / momentum buffers.
    """
    try:
        if not bool(state.params.get("gauge_project_enable", False)):
            return
        if not bool(state.params.get("gauge_project_apply_momentum", True)):
            return
    except Exception:
        return
    # SPIDER samplers handle momentum/noise internally.
    try:
        mod = str(getattr(optimizer.__class__, "__module__", "") or "")
        if mod.startswith("spider.optim"):
            return
    except Exception:
        pass
    try:
        dims_v = state.params.get("gauge_project_dims", [0, 1, 2])
        if not isinstance(dims_v, list) or len(dims_v) == 0:
            dims_v = [0, 1, 2]
        dims = tuple(int(d) for d in dims_v)
    except Exception:
        dims = (0, 1, 2)
    try:
        mode = str(state.params.get("gauge_project_mode", "global")).strip().lower()
    except Exception:
        mode = "global"
    if mode not in {"global", "cluster"}:
        mode = "global"
    cid = state.cluster_ids if mode == "cluster" else None
    cc = state.cluster_counts if mode == "cluster" else None

    p = getattr(state, "dX_src", None)
    if p is None:
        return
    st = getattr(optimizer, "state", {}).get(p, None)  # type: ignore[call-arg]
    if not isinstance(st, dict):
        return
    # Common momentum buffer keys:
    # - Adam/AdamW: 'exp_avg' (first moment)
    # - SGD: 'momentum_buffer'
    for k in ("exp_avg", "momentum_buffer"):
        try:
            buf = st.get(k, None)
            if isinstance(buf, torch.Tensor) and buf.ndim == 2 and int(buf.shape[0]) == int(p.shape[0]):
                project_event_mean_inplace(buf, dims=dims, mode=mode, cluster_ids=cid, cluster_counts=cc)
        except Exception:
            pass


def _ddp_info(params: dict) -> tuple[bool, int, int, bool]:
    """
    Returns (enabled, rank, world_size, is_main) for torchrun/DDP mode.
    We treat DDP as enabled only when torch.distributed is initialized and world_size>1.
    """
    try:
        ws = int(params.get("_ddp_world_size", 1) or 1)
        rk = int(params.get("_ddp_rank", 0) or 0)
    except Exception:
        ws, rk = 1, 0
    enabled = bool(ws > 1) and bool(dist.is_available()) and bool(dist.is_initialized())
    if enabled:
        try:
            # Trust the runtime communicator if available.
            ws = int(dist.get_world_size())
            rk = int(dist.get_rank())
        except Exception:
            pass
    is_main = (int(rk) == 0)
    return enabled, int(rk), int(ws), bool(is_main)


def _ddp_allreduce_grads(optimizer: torch.optim.Optimizer) -> None:
    """All-reduce gradients (SUM) across ranks. Assumes dist is initialized."""
    grads = []
    # Control tensor device for tiny collectives (e.g., all_gather on totals).
    # IMPORTANT: even if a rank has no grads this step, it must still participate
    # in the same collectives as other ranks, otherwise NCCL will deadlock.
    ctrl_dev = None
    for g in optimizer.param_groups:  # type: ignore[attr-defined]
        for p in g.get("params", []):
            if p is None:
                continue
            if ctrl_dev is None and isinstance(p, torch.Tensor):
                ctrl_dev = p.device
            gg = getattr(p, "grad", None)
            if gg is None:
                continue
            if not isinstance(gg, torch.Tensor) or gg.numel() <= 0:
                continue
            grads.append(gg)

    if ctrl_dev is None:
        # Extremely defensive: if optimizer has no tensor params, fall back to CPU.
        ctrl_dev = torch.device("cpu")

    # Coalesce into a single buffer to reduce per-parameter allreduce overhead.
    # This matters a lot on systems without fast GPU interconnect, where many small allreduces
    # can dominate the step time.
    dev0 = grads[0].device if grads else ctrl_dev
    dt0 = grads[0].dtype if grads else torch.float32
    same = True
    total = 0
    for gg in grads:
        total += int(gg.numel())
        if gg.device != dev0 or gg.dtype != dt0:
            same = False
            break

    # Fail fast if different ranks have different parameter sets / grad sizes.
    # IMPORTANT: avoid ReduceOp.MIN/MAX on int64 here — some NCCL stacks return garbage for those.
    # Instead, all_gather the scalar totals and compare exactly.
    total_t = torch.tensor([int(total)], device=dev0, dtype=torch.int64)
    try:
        ws = int(dist.get_world_size())
    except Exception:
        ws = 0
    if ws > 1:
        totals = [torch.empty_like(total_t) for _ in range(ws)]
        dist.all_gather(totals, total_t)
        vals = [int(t.item()) for t in totals]
        if any(v != vals[0] for v in vals[1:]):
            raise RuntimeError(
                "DDP grad buffer size mismatch across ranks: "
                + ", ".join(f"rank{i}={v}" for i, v in enumerate(vals))
                + ". This usually means some rank is missing a Parameter (e.g., optional latent disabled) or "
                "a rank hit an exception and skipped initializing part of the model."
            )

    # If *all* ranks have total==0, there's nothing to reduce this step.
    # (But we still did the all_gather above to keep the collective schedule aligned.)
    if (not grads) or total <= 0:
        return

    if (not same):
        # Fallback: allreduce each grad separately (after the mismatch check above).
        for gg in grads:
            dist.all_reduce(gg, op=dist.ReduceOp.SUM)
        return

    # Reuse a persistent buffer attached to the optimizer to avoid allocating every step.
    buf = getattr(optimizer, "_ddp_grad_buffer", None)
    try:
        if not isinstance(buf, torch.Tensor) or int(buf.numel()) != int(total) or buf.device != dev0 or buf.dtype != dt0:
            buf = torch.empty((int(total),), device=dev0, dtype=dt0)
            setattr(optimizer, "_ddp_grad_buffer", buf)
    except Exception:
        buf = torch.empty((int(total),), device=dev0, dtype=dt0)
        try:
            setattr(optimizer, "_ddp_grad_buffer", buf)
        except Exception:
            pass

    # Pack
    off = 0
    for gg in grads:
        n = int(gg.numel())
        buf[off : off + n].copy_(gg.reshape(-1))
        off += n

    # Allreduce once
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    # Unpack
    off = 0
    for gg in grads:
        n = int(gg.numel())
        gg.copy_(buf[off : off + n].view_as(gg))
        off += n


def _ddp_set_step_seed(params: dict, step: int, *, device: torch.device) -> None:
    """
    Make sampler noise deterministic across ranks by resetting RNG state per step.
    This is critical for SGHMC/pSGLD because the optimizer step injects random noise.
    """
    try:
        base = int(params.get("runtime_seed", 0))
    except Exception:
        base = 0
    s = int(base + 10000019 * int(step))
    try:
        torch.manual_seed(s)
    except Exception:
        pass
    try:
        if device.type == "cuda":
            # IMPORTANT: do NOT call manual_seed_all() here.
            # That can initialize CUDA contexts on *all* visible GPUs, including devices not used by this rank,
            # which can cause major slowdowns and can violate user-intended device selection.
            torch.cuda.manual_seed(s)
    except Exception:
        pass


def _maybe_write_map_csv(state: LocateState, epoch: int) -> None:
    """
    Periodically write a MAP CSV snapshot during Phase 1.

    NOTE: This is intentionally a lightweight side-effect used only when `write_map_csv=True`
    is passed to `_run_epoch` (Phase 1).
    """
    # In torchrun/DDP mode, only rank0 should emit files.
    try:
        ddp_enabled, _rk, _ws, ddp_is_main = _ddp_info(state.params)
        if ddp_enabled and not ddp_is_main:
            return
    except Exception:
        pass
    if epoch % 100 != 0:
        return
    try:
        X_src1 = (state.X_src + state.dX_src).detach().cpu().numpy()
        origins = write_output(
            state.origins0,
            X_src1,
            state.X_src.detach().cpu().numpy(),
            state.projector,
        )
        origins.write_csv(f"{state.params['catalog_outfile']}_MAP.csv")
    except Exception:
        # Best-effort only; never fail training for CSV output.
        return


def _get_diagnostics_cfg(params: dict) -> dict:
    """Return the diagnostics config dict (new schema only).

    Diagnostics are configured under `inference.diagnostics`.
    Legacy top-level `diagnostics` is no longer supported.
    """
    inf = params.get("inference", None)
    if not isinstance(inf, dict):
        raise KeyError("Missing required config block: inference")
    d = inf.get("diagnostics", None)
    if not isinstance(d, dict):
        raise KeyError("Missing required config block: inference.diagnostics")
    return d

def _update_svrg_snapshot(state: LocateState, batch_size: int, optimizer: torch.optim.Optimizer) -> None:
    """
    Compute full gradient at current parameters and store as snapshot.
    This is an expensive O(N) operation.
    """
    _log("SVRG: Updating full gradient snapshot...")
    
    # Store snapshot of parameters
    state.svrg_dX_snapshot = state.dX_src.detach().clone()
    
    # Compute full gradient as an *average-gradient* consistent with our minibatch convention.
    # IMPORTANT:
    # - Do NOT call posterior_loss in a per-batch loop: it includes the prior term scaled by 1/N,
    #   which would get counted multiple times.
    # - Instead: add the prior ONCE, and add likelihood contributions as a weighted sum of batch means.
    from spider.core.modeling import compute_likelihood_loss, compute_prior_loss
    
    # Clear any existing grads
    optimizer.zero_grad(set_to_none=True)
    if state.dX_src.grad is not None:
        state.dX_src.grad.zero_()
    
    N = int(state.N)
    bs = max(int(batch_size), 10000)  # use a large batch for efficiency if possible
    
    # Current noise scales (constant during this snapshot computation)
    σp, σs = _current_noise_scales(state)
    
    # 1. Prior
    l_prior = compute_prior_loss(
        ΔX_src=state.dX_src,
        prior_event=state.prior_event,
        prior_centroid=state.prior_centroid,
        σ_p=σp,
        σ_s=σs,
        N_total=state.N,
        params=state.params,
        cluster_ids=state.cluster_ids,
        cluster_counts=state.cluster_counts,
        event_precision_matrix=state.event_precision_matrix,
    )
    l_prior.backward()
    
    # 2. Likelihood
    for i in range(0, N, bs):
        i_end = min(i + bs, N)
        bs_curr = i_end - i
        
        II_b = state.II[i:i_end]
        YY_b = state.YY[i:i_end]
             
        # Compute mean NLL for this batch
        nll_batch = compute_likelihood_loss(
            idx=II_b,
            y=YY_b,
            X_src=state.X_src,
            ΔX_src=state.dX_src,
            model=state.model,
            σ_p=σp,
            σ_s=σs,
            params=state.params,
            nuisance_delta=None
        )
        
        # Scale by fraction of total data to contribute to global mean
        loss_chunk = nll_batch * (bs_curr / N)
        loss_chunk.backward()
        
    # Store result
    state.svrg_grad_full = state.dX_src.grad.detach().clone()
    
    # Zero out again to leave clean state
    state.dX_src.grad.zero_()
    
    # Scale adjustment if user wants Total Gradient logic
    # (Check if optimizer scales by N_obs)
    # The optimizer wrapper usually handles scaling.
    # However, if we feed this into the SVRG correction formula:
    # v = g_batch - g_snap + g_full
    # All terms must be consistently scaled.
    # Our batches in _run_epoch produce gradients from posterior_loss().
    # posterior_loss() returns Average Posterior NLL.
    # So g_batch is Average Gradient.
    # Our g_full computation above produces Average Gradient.
    # So they match! 
    # (The optimizer might multiply by N later, but that applies to the sum v, which is fine).
    
    _log(f"SVRG: Snapshot updated. Grad norm: {state.svrg_grad_full.norm().item():.3e}")

def _set_backend_noise(optimizer: torch.optim.Optimizer, *, enabled: bool, scale: float) -> None:
    """Set noise flags consistently for any sampler backend."""
    if not hasattr(optimizer, "param_groups"):
        return
    for g in optimizer.param_groups:  # type: ignore[attr-defined]
        g["add_noise"] = bool(enabled and (scale > 0.0))
        g["noise_scale"] = float(scale if enabled else 0.0)

def _run_epoch(
    state: LocateState,
    epoch_index: int,
    optimizer: torch.optim.Optimizer,
    *,
    is_sampling: bool = False,
    noise_scale_factor: float = 0.0,
    grad_clip_norm: float = 0.0,
    write_map_csv: bool = False,
) -> Dict[str, float]:
    """
    Unified epoch execution for all phases.
    
    Args:
        state: The LocateState object
        epoch_index: Current epoch index (for logging/seeding)
        optimizer: The optimizer to step (Adam or SGLD)
        is_sampling: If True, collect samples (Phase 4)
        noise_scale_factor: Factor to scale Langevin noise (0.0-1.0).
                           Handled generically for any sampler backend.
        grad_clip_norm: If > 0, clip gradients to this norm.
        write_map_csv: If True, check/write MAP CSV (Phase 1 behavior)
    """
    if bool(is_sampling):
        phase_id = 4
    elif float(noise_scale_factor) > 0.0:
        phase_id = 3
    elif isinstance(optimizer, torch.optim.Adam):
        phase_id = 1
    else:
        phase_id = 2

    # --- Per-prior runtime enables (materialized by validate_and_materialize_priors) ---
    # Hard break: priors are active in all phases when enabled (no per-phase scheduling).
    state.params["_prior_event_runtime_enable"] = bool(state.params.get("prior_event_enable", True))
    state.params["_prior_centroid_runtime_enable"] = bool(state.params.get("prior_centroid_enable", False))
    # Noise prior removed (fixed phase_unc only).

    ddp_enabled, ddp_rank, ddp_world_size, ddp_is_main = _ddp_info(state.params)

    use_event_batches = bool(state.params.get("event_batch_enable", False))
    if ddp_enabled and use_event_batches:
        raise ValueError(
            "torchrun/DDP mode for `spider sample` currently supports only standard batching "
            "(inference.batching.standard.*). Disable inference.batching.event_batches.enabled."
        )
    permute_time_s = 0.0
    if not use_event_batches:
        # Allow a per-run seed offset (e.g. for multi-GPU independent chains).
        seed0 = int(state.params.get("runtime_seed", 0))
        shuffle = bool(state.params.get("batch_shuffle", True))
        # Optional: timing + one-line confirmation (helps diagnose big-N performance).
        t_perm0 = time.time()
        state.begin_epoch_rr(seed=int(seed0 + epoch_index), shuffle=shuffle)
        permute_time_s = float(time.time() - t_perm0)
        # Expose whether this epoch is shuffled to lower-level code (e.g. caching in likelihoods).
        state.params["_runtime_batch_shuffle"] = bool(shuffle)

    # Expose epoch index to lower-level code paths (e.g., likelihood caching / refresh schedules).
    # This is an internal implementation detail, not a user-facing config key.
    state.params["_runtime_epoch_index"] = int(epoch_index)
        
    epoch_start_time = time.time()
    total_loss_vals: List[float] = []
    # Weighted loss aggregation (by number of rows/edges in each batch) so epoch "loss"
    # is comparable across batch sizes / event-batch partitioning.
    total_loss_weighted_sum: float = 0.0
    total_loss_weighted_denom: int = 0

    # Optional: uncollapsed shared-event latent diagnostics (accumulated across minibatches; one sync at end).
    b_delta_sum = None
    b_delta_sumsq = None
    b_delta_maxabs = None
    b_delta_count = None
    # Also track RMS of *event-level* latent endpoints b(e1), b(e2) used by DD rows.
    # This is more interpretable than RMS of the underlying parameter tensor in inducing_gp mode
    b_end_sumsq_p = None
    b_end_sumsq_s = None
    b_end_count_p = None
    b_end_count_s = None
    # slowness_inducing_gp: track RMS of the *event-level u vectors* at endpoints (more interpretable than dual coeffs).
    u_end_sumsq_p = None
    u_end_sumsq_s = None
    u_end_count_p = None
    u_end_count_s = None
    u_end_maxnorm_p = None
    u_end_maxnorm_s = None
    corr_dc_sumsq_p = None
    corr_dc_sumsq_s = None
    corr_dc_count_p = None
    corr_dc_count_s = None
    corr_dc_maxabs_p = None
    corr_dc_maxabs_s = None
    # Online ESS metrics are computed only when enough *saved samples* exist and the cadence triggers.
    # To avoid gaps in W&B time series (epochs where ESS isn't recomputed), we cache the last metrics
    # on the state object and re-log them each epoch.
    ess_online_updated_this_epoch = False
    if not hasattr(state, "_ess_online_last_metrics"):
        setattr(state, "_ess_online_last_metrics", None)
    if not hasattr(state, "_ess_online_error_count"):
        setattr(state, "_ess_online_error_count", 0)

    # Batching parameters
    if use_event_batches:
        seed0 = int(state.params.get("runtime_seed", 0))
        epoch_seed = int(seed0 + epoch_index)
        # Use sgld params by default, fallback to warmup if key missing (Phase 1 uses warmup key)
        # But to simplify, we can look at the phase. Phase 1 sets its own max_edges logic.
        # For unified logic, we'll try sgld key first, then warmup key.
        # Actually, looking at original code, Phase 1 used 'event_batch_max_edges_warmup', others 'sgld'.
        # We will rely on the caller to set these in params or just pick one logic.
        # Let's stick to the logic: if optim is Adam (Phase 1), use warmup key?
        # Or just use 'event_batch_max_edges' generic key if specific one is missing.
        # To be safe and simple:
        max_edges_key = "event_batch_max_edges_sgld"
        if isinstance(optimizer, torch.optim.Adam):
             max_edges_key = "event_batch_max_edges_warmup"
             
        max_edges = int(state.params.get(max_edges_key, state.params.get("event_batch_max_edges", 0)))
        events_per_batch = int(state.params.get("event_batch_size", 256))
        reorder_all = bool(state.params.get("event_bucket_reorder_all", False))
        reuse_epochs = int(state.params.get("event_bucket_reuse_epochs", 1))
        
        _ensure_owner_buckets(
            state,
            epoch_index=epoch_seed,
            events_per_batch=events_per_batch,
            max_edges_per_batch=max_edges,
            reorder_all=reorder_all,
            reuse_epochs=reuse_epochs,
        )
        
        # Determine iterator
        if state._bucket_offsets is None or (reorder_all and (state._bucket_II is None or state._bucket_YY is None)):
            batch_iter = _iter_event_batches(state, epoch_seed=epoch_seed, max_edges_per_batch=max_edges, events_per_batch=events_per_batch)
            use_buckets = False
        else:
            batch_iter = range(int(state._bucket_offsets.numel()) - 1)
            use_buckets = True
            
    else:
        # Standard batching
        bs_key = "batch_size_sgld"
        if isinstance(optimizer, torch.optim.Adam):
            bs_key = "batch_size_warmup"
        batch_size = int(state.params.get(bs_key, 10000))
        # IMPORTANT: avoid an extra empty final batch when N is exactly divisible by batch_size.
        # `range(0, N//bs + 1)` yields a last iteration with i_start==i_end==N (no data),
        # which can create rank-divergent control flow and pointless DDP collectives.
        if batch_size <= 0:
            batch_iter = range(0)
        else:
            n_batches = int((int(state.N) + int(batch_size) - 1) // int(batch_size))
            batch_iter = range(int(n_batches))
        use_buckets = False
        # Expose standard batching parameters for lower-level caching.
        state.params["_runtime_batch_size"] = int(batch_size)
        state.params["_runtime_batching_mode"] = "standard"

    # Noise setup (generic sampler backend)
    # If noise_scale_factor > 0, enable noise; else disable (e.g., Phase 2). Phase 3 ramps it.
    _set_backend_noise(optimizer, enabled=(noise_scale_factor > 0.0), scale=float(noise_scale_factor))
    # SVRG Snapshot Update (only if SVRG enabled and we are sampling)
    svrg_enabled = state.svrg_enable and is_sampling
    if ddp_enabled and svrg_enabled:
        raise ValueError("SVRG is not supported in torchrun/DDP mode (it requires extra full-gradient bookkeeping). Disable inference.diagnostics.svrg.enabled.")
    if svrg_enabled:
        # Check if we need to update snapshot (e.g. every epoch)
        # For simplicity, update at start of every epoch for now if enabled
        _update_svrg_snapshot(state, batch_size=batch_size, optimizer=optimizer)

    try:
        diag0 = _get_diagnostics_cfg(state.params)
        # Optional profiling: collapsed shared_event_re likelihood (PCG) cost + workload stats.
        # Expected config location: inference.diagnostics.profile_shared_event_re (bool).
        if isinstance(diag0, dict) and bool(diag0.get("profile_shared_event_re", False)):
            state.params["_se_re_time_ms_sum"] = 0.0
            state.params["_se_re_time_ms_count"] = 0
            state.params["_se_re_groups_sum"] = 0
            state.params["_se_re_groups_pcg_sum"] = 0
            state.params["_se_re_groups_fallback_sum"] = 0
            state.params["_se_re_max_rows_max"] = 0
            state.params["_se_re_max_nodes_max"] = 0
            # Runtime gate read by modeling.py
            try:
                if isinstance(K_full, torch.Tensor) and K_full.ndim == 2:
                    M_static = int(K_full.shape[0])
                else:
                    if isinstance(K_blocks, list) and K_blocks:
                        ms = []
                        for K in K_blocks:
                            if isinstance(K, torch.Tensor) and K.ndim == 2:
                                ms.append(int(K.shape[0]))
                        if ms:
                            M_static = int(max(ms))
            except Exception:
                M_static = 0
            state.params["_sl_re_max_M_static"] = int(M_static)
            state.params["_sl_re_max_M_max"] = int(M_static)
    except Exception:
        pass

    # Best-effort: total batches (works for standard batching / ranges)
    total_batches = None
    try:
        if isinstance(batch_iter, range):
            total_batches = int(len(batch_iter))
    except Exception:
        total_batches = None

    # Inner Loop
    for bi, batch_item in enumerate(batch_iter):
        optimizer.zero_grad(set_to_none=True)
        
        # Prepare batch data
        if use_event_batches:
            if use_buckets:
                # batch_item is index in offsets
                i0 = int(state._bucket_offsets[batch_item].item())
                i1 = int(state._bucket_offsets[batch_item + 1].item())
                if i1 <= i0: continue
                
                if reorder_all and state._bucket_II is not None:
                    II_b = state._bucket_II[i0:i1, :]
                    YY_b = state._bucket_YY[i0:i1, :]
                    rows = None
                else:
                    rows = state._bucket_rows_order[i0:i1]
                    # Fail fast with a clear error if bucket row indices are invalid.
                    try:
                        N_rt = int(state.II.shape[0])
                        if isinstance(rows, torch.Tensor) and rows.numel() > 0:
                            mn = int(rows.min().item())
                            mx = int(rows.max().item())
                            if mn < 0 or mx >= N_rt:
                                raise ValueError(
                                    f"owner_buckets produced out-of-range row indices: min={mn} max={mx} N={N_rt} "
                                    f"(bucket={int(batch_item)} slice=[{i0}:{i1}])"
                                )
                    except Exception:
                        raise
                    II_b = state.II.index_select(0, rows)
                    YY_b = state.YY.index_select(0, rows)
                # Expose stable bucket id to lower-level code (e.g., correlated likelihood caching).
                # This is an internal implementation detail, not a user-facing config key.
                state.params["_runtime_bucket_id"] = int(batch_item)
                # Also expose a bucket "generation" token that changes whenever buckets are rebuilt.
                # Without this, caches keyed only by bucket_id will collide across rebuilds because
                # bucket ids are reused from 0..num_buckets-1 each rebuild.
                try:
                    state.params["_runtime_bucket_gen"] = int(getattr(state, "_bucket_last_epoch", -1))
                except Exception:
                    state.params["_runtime_bucket_gen"] = -1
                try:
                    if reorder_all and getattr(state, "_bucket_p_counts", None) is not None:
                        state.params["_runtime_bucket_p_count"] = int(state._bucket_p_counts[batch_item].item())
                    else:
                        state.params["_runtime_bucket_p_count"] = -1
                except Exception:
                    state.params["_runtime_bucket_p_count"] = -1

                # If available, provide precomputed per-bucket phase graphs (nodes/u/v) to the likelihood.
                # This avoids per-batch torch.unique/remapping when grouping='phase'.
                try:
                    if reorder_all and getattr(state, "_bucket_nodes_p", None) is not None:
                        bi = int(batch_item)
                        state.params["_runtime_bucket_nodes_p"] = state._bucket_nodes_p[bi]
                        state.params["_runtime_bucket_u_p"] = state._bucket_u_p[bi]
                        state.params["_runtime_bucket_v_p"] = state._bucket_v_p[bi]
                        state.params["_runtime_bucket_nodes_s"] = state._bucket_nodes_s[bi]
                        state.params["_runtime_bucket_u_s"] = state._bucket_u_s[bi]
                        state.params["_runtime_bucket_v_s"] = state._bucket_v_s[bi]
                    else:
                        state.params["_runtime_bucket_nodes_p"] = None
                        state.params["_runtime_bucket_u_p"] = None
                        state.params["_runtime_bucket_v_p"] = None
                        state.params["_runtime_bucket_nodes_s"] = None
                        state.params["_runtime_bucket_u_s"] = None
                        state.params["_runtime_bucket_v_s"] = None
                except Exception:
                    state.params["_runtime_bucket_nodes_p"] = None
                    state.params["_runtime_bucket_u_p"] = None
                    state.params["_runtime_bucket_v_p"] = None
                    state.params["_runtime_bucket_nodes_s"] = None
                    state.params["_runtime_bucket_u_s"] = None
                    state.params["_runtime_bucket_v_s"] = None

                # Preferred: pre-chunked phase blocks (each chunk <= max_nodes), fully deterministic.
                try:
                    if reorder_all and getattr(state, "_bucket_chunks_p", None) is not None:
                        bi = int(batch_item)
                        state.params["_runtime_bucket_chunks_p"] = state._bucket_chunks_p[bi]
                        state.params["_runtime_bucket_chunks_s"] = state._bucket_chunks_s[bi]
                    else:
                        state.params["_runtime_bucket_chunks_p"] = None
                        state.params["_runtime_bucket_chunks_s"] = None
                except Exception:
                    state.params["_runtime_bucket_chunks_p"] = None
                    state.params["_runtime_bucket_chunks_s"] = None

                # Provide per-row station index (stable int id for (network,station)) when available.
                # on float receiver coordinates.
                try:
                    sta_b = None
                    if reorder_all and getattr(state, "_bucket_station_index", None) is not None:
                        sta_b = state._bucket_station_index[i0:i1]
                    elif rows is not None and getattr(state, "row_station_index", None) is not None:
                        sta_b = state.row_station_index.index_select(0, rows)
                    state.params["_runtime_bucket_station_index"] = sta_b
                except Exception:
                    state.params["_runtime_bucket_station_index"] = None
                # Provide per-row component ids aligned to this bucket slice when available.
                try:
                    comp_b = None
                    if reorder_all and getattr(state, "_bucket_comp_index", None) is not None:
                        comp_b = state._bucket_comp_index[i0:i1]
                    state.params["_runtime_bucket_comp_index"] = comp_b
                except Exception:
                    state.params["_runtime_bucket_comp_index"] = None
                # Not a standard batch; clear standard ids to avoid accidental cache hits.
                state.params["_runtime_batch_id"] = -1
                state.params["_runtime_batch_i0"] = -1
                state.params["_runtime_batch_i1"] = -1
            else:
                # batch_item is batch_idx tensor
                # Defensive: ensure batch row indices are in-range before index_select.
                try:
                    N_rt = int(state.II.shape[0])
                    if isinstance(batch_item, torch.Tensor) and batch_item.numel() > 0:
                        mn = int(batch_item.min().item())
                        mx = int(batch_item.max().item())
                        if mn < 0 or mx >= N_rt:
                            raise ValueError(
                                f"event_batches produced out-of-range row indices: min={mn} max={mx} N={N_rt}"
                            )
                except Exception:
                    raise
                II_b = state.II.index_select(0, batch_item)
                YY_b = state.YY.index_select(0, batch_item)
                rows = batch_item # for SSST index select if needed logic? 
                # Actually SSST logic uses batch_idx directly in fallback loop
                state.params["_runtime_bucket_id"] = -1
                state.params["_runtime_bucket_gen"] = -1
                state.params["_runtime_bucket_p_count"] = -1
                state.params["_runtime_bucket_nodes_p"] = None
                state.params["_runtime_bucket_u_p"] = None
                state.params["_runtime_bucket_v_p"] = None
                state.params["_runtime_bucket_nodes_s"] = None
                state.params["_runtime_bucket_u_s"] = None
                state.params["_runtime_bucket_v_s"] = None
                state.params["_runtime_bucket_chunks_p"] = None
                state.params["_runtime_bucket_chunks_s"] = None
                try:
                    if getattr(state, "row_station_index", None) is not None:
                        state.params["_runtime_bucket_station_index"] = state.row_station_index.index_select(0, batch_item)
                    else:
                        state.params["_runtime_bucket_station_index"] = None
                except Exception:
                    state.params["_runtime_bucket_station_index"] = None
                state.params["_runtime_bucket_comp_index"] = None
                # Not a standard batch; clear standard ids to avoid accidental cache hits.
                state.params["_runtime_batch_id"] = -1
                state.params["_runtime_batch_i0"] = -1
                state.params["_runtime_batch_i1"] = -1
        else:
            # batch_item is j (index of batch)
            i_start = batch_item * batch_size
            i_end = min(i_start + batch_size, state.N)
            II_b = state.II_epoch[i_start:i_end, :]
            YY_b = state.YY_epoch[i_start:i_end]
            global_bsz = int(i_end - i_start)
            # indices for SSST
            # If using standard batching, we assume contiguous indices in epoch permutation
            rows = None 
            state.params["_runtime_bucket_id"] = -1
            state.params["_runtime_bucket_gen"] = -1
            state.params["_runtime_bucket_p_count"] = -1
            state.params["_runtime_bucket_nodes_p"] = None
            state.params["_runtime_bucket_u_p"] = None
            state.params["_runtime_bucket_v_p"] = None
            state.params["_runtime_bucket_nodes_s"] = None
            state.params["_runtime_bucket_u_s"] = None
            state.params["_runtime_bucket_v_s"] = None
            state.params["_runtime_bucket_chunks_p"] = None
            state.params["_runtime_bucket_chunks_s"] = None
            # Stable standard batch id (only meaningful when _runtime_batch_shuffle is false).
            state.params["_runtime_batch_id"] = int(batch_item)
            state.params["_runtime_batch_i0"] = int(i_start)
            state.params["_runtime_batch_i1"] = int(i_end)
            try:
                if getattr(state, "row_station_index_epoch", None) is not None:
                    state.params["_runtime_bucket_station_index"] = state.row_station_index_epoch[i_start:i_end]
                elif getattr(state, "row_station_index", None) is not None:
                    state.params["_runtime_bucket_station_index"] = state.row_station_index[i_start:i_end]
                else:
                    state.params["_runtime_bucket_station_index"] = None
            except Exception:
                state.params["_runtime_bucket_station_index"] = None
            state.params["_runtime_bucket_comp_index"] = None

        # --- DDP shard: split *this batch* across ranks (global batch size is the configured batch size) ---
        if ddp_enabled:
            try:
                # We only support standard batching here (event-batch path returns earlier).
                B = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
                if B != int(global_bsz):
                    global_bsz = int(B)
                # IMPORTANT:
                # Use a *strided* shard (interleaved rows) rather than a contiguous slice.
                #
                # Rationale: if rows are partially sorted (e.g., by phase/component/station),
                # contiguous sharding can give different ranks different data *types* within the same
                # global batch. That can lead to rank-specific unused Parameters (grad=None) and
                # NCCL deadlocks in our manual gradient allreduce.
                #
                # Strided sharding is deterministic and tends to preserve mixture across ranks.
                if global_bsz > 0:
                    sel = torch.arange(int(ddp_rank), int(global_bsz), int(ddp_world_size), device=II_b.device)
                    II_b = II_b.index_select(0, sel)
                    YY_b = YY_b.index_select(0, sel)
                sta_rt = state.params.get("_runtime_bucket_station_index", None)
                if isinstance(sta_rt, torch.Tensor) and int(sta_rt.shape[0]) == int(global_bsz):
                    if global_bsz > 0:
                        state.params["_runtime_bucket_station_index"] = sta_rt.index_select(0, sel)
            except Exception:
                # Leave batch unsharded if something goes wrong; better than crashing mid-run.
                pass

        # Noise scales for loss
        σp, σs = _current_noise_scales(state)

        # ---- Fail-fast index validation (prevents opaque CUDA IndexKernel asserts) ----
        # Validate event indices in II_b are within [0, n_events).
        # This MUST run outside any try/except that might swallow the error.
        try:
            Ne_rt = int(state.X_src.shape[0])
            if Ne_rt > 0 and isinstance(II_b, torch.Tensor) and II_b.numel() > 0:
                e1_rt = II_b[:, 0].to(torch.int64)
                e2_rt = II_b[:, 1].to(torch.int64)
                mn = int(torch.minimum(e1_rt.min(), e2_rt.min()).item())
                mx = int(torch.maximum(e1_rt.max(), e2_rt.max()).item())
                if mn < 0 or mx >= Ne_rt:
                    raise ValueError(f"Batch II has out-of-range event indices: min={mn} max={mx} n_events={Ne_rt}")
        except Exception:
            raise

        # Validate station indices (if present) are within [0, n_stations).
        try:
            sta_rt = state.params.get("_runtime_bucket_station_index", None)
            n_stations_rt = int(getattr(state, "n_stations", 0))
            if isinstance(sta_rt, torch.Tensor) and sta_rt.numel() > 0 and n_stations_rt > 0:
                mn = int(sta_rt.min().item())
                mx = int(sta_rt.max().item())
                if mn < 0 or mx >= n_stations_rt:
                    raise ValueError(f"Batch station index out of range: min={mn} max={mx} n_stations={n_stations_rt}")
        except Exception:
            raise
        
        nuisance_delta = None
        prof_se_lat = False
        se_lat_t0 = None
        try:
            diag = _get_diagnostics_cfg(state.params)
        except Exception:
            prof_se_lat = False

        # Optional: uncollapsed shared-event latent random effects b[s,event,phase] (sampled).
        # Adds nuisance term: (b_{s,e2,phase} - b_{s,e1,phase}) to predicted differential time.
        try:
            if prof_se_lat:
                import time as _time
                if state.device.type == "cuda":
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                se_lat_t0 = _time.perf_counter()
                sta_b = state.params.get("_runtime_bucket_station_index", None)
                if isinstance(sta_b, torch.Tensor) and int(sta_b.shape[0]) == int(II_b.shape[0]) and isinstance(b_lat, torch.Tensor):
                    e1 = II_b[:, 0].to(torch.int64)
                    e2 = II_b[:, 1].to(torch.int64)
                    sta_bi = sta_b.to(torch.int64)
                    ph = YY_b[:, 4]
                    is_s = (ph >= 0.5)

                    # Fail fast if event indices are out of range for any event-indexed tensors.
                    # This prevents CUDA device-side asserts inside index_select on per-event arrays.
                    try:
                        Ne_rt = int(state.X_src.shape[0])
                        if Ne_rt > 0 and e1.numel() > 0:
                            mn = int(torch.minimum(e1.min(), e2.min()).item())
                            mx = int(torch.maximum(e1.max(), e2.max()).item())
                            if mn < 0 or mx >= Ne_rt:
                                raise ValueError(f"II contains out-of-range event indices in batch: min={mn} max={mx} n_events={Ne_rt}")
                    except Exception:
                        raise

                    if mode == "inducing_gp":
                        # Inducing-point GP (predictive-process mean):
                        # b(e) ≈ Σ_j k(e,u_j) * c_j, where c are inducing coefficients and k is an RBF kernel.
                        # We store per-event neighbor inducing indices + kernel values from Stage 3.
                        # Optional Stage 5 (FITC): inflate per-observation noise by the diagonal residual variance
                        #   Var[ε_e] = (1 - Q_ee) * τ^2, so for a differential (e1,e2): Var[ε_e2 - ε_e1] = (r2 + r1) * τ^2.
                        try:
                                if isinstance(resid, torch.Tensor) and resid.ndim == 1:
                                    r1 = resid.index_select(0, e1).to(torch.float32)
                                    r2 = resid.index_select(0, e2).to(torch.float32)
                                    rsum = (r1 + r2).clamp_min(0.0)
                                    tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                                    extra = torch.where(is_s, rsum * (tau_s * tau_s), rsum * (tau_p * tau_p)).to(torch.float32)
                        except Exception:
                            pass
                        if isinstance(nei_idx, torch.Tensor) and isinstance(nei_k, torch.Tensor) and b_lat.ndim == 3 and int(b_lat.shape[2]) == 2:
                            # Two modes:
                            # - per-station coefficients (b_lat shape (n_stations, M, 2))
                            # - station-basis coefficients (b_lat shape (R, M, 2) + W_sta shape (n_stations, R))
                            # Robust mode selection to avoid OOB indexing:
                            # - If b_lat[0] matches n_stations -> per-station
                            # - Else if W exists and b_lat[0] matches W.shape[1] -> basis
                            # - Else -> disable this contribution (better than crashing CUDA)
                            n_stations_rt = int(getattr(state, "n_stations", 0))
                            is_per_station = (n_stations_rt > 0) and (int(b_lat.shape[0]) == int(n_stations_rt))
                            is_basis = (
                                isinstance(W_sta, torch.Tensor)
                                and W_sta.ndim == 2
                                and int(b_lat.shape[0]) == int(W_sta.shape[1])
                                and int(W_sta.shape[0]) == int(n_stations_rt)
                            )

                            # Defensive: ensure station indices are in range before any index_select.
                            # Avoid CUDA device-side asserts; if violated, skip this nuisance term.
                            ok_sta = True
                            try:
                                if n_stations_rt <= 0:
                                    ok_sta = False
                                elif sta_bi.numel() > 0:
                                    mn = int(sta_bi.min().item())
                                    mx = int(sta_bi.max().item())
                                    if mn < 0 or mx >= n_stations_rt:
                                        ok_sta = False
                            except Exception:
                                ok_sta = False
                            if not ok_sta:
                                delta_b = None
                            else:
                                # Also defensively check neighbor inducing indices against M_total for this parameterization.
                                # This catches mismatches between interpolation NPZ and coefficient tensor shape without crashing CUDA.
                                ok_nei = True
                                try:
                                    M_total_rt = int(b_lat.shape[1])
                                    if M_total_rt <= 0:
                                        ok_nei = False
                                    else:
                                        # sample check on the two endpoint sets (cheap relative to failing later)
                                        idx1 = nei_idx.index_select(0, e1)
                                        idx2 = nei_idx.index_select(0, e2)
                                        mx1 = int(idx1.max().item()) if idx1.numel() > 0 else -1
                                        mx2 = int(idx2.max().item()) if idx2.numel() > 0 else -1
                                        mn1 = int(idx1.min().item()) if idx1.numel() > 0 else 0
                                        mn2 = int(idx2.min().item()) if idx2.numel() > 0 else 0
                                        mx_all = max(mx1, mx2)
                                        mn_all = min(mn1, mn2)
                                        # -1 is allowed padding; anything >= M_total is invalid
                                        if mx_all >= M_total_rt or mn_all < -1:
                                            ok_nei = False
                                except Exception:
                                    ok_nei = False
                                if not ok_nei:
                                    delta_b = None
                                elif is_basis:
                                    # Basis mode: b(s,e,phase) = Σ_r W[s,r] * a_r(e,phase)
                                    #
                                    # Performance note: the naive implementation gathers per-rank coefficients and then
                                    # weights by W, which costs ~O(R) extra work. Since n_stations is small (e.g. ~52),
                                    # it is faster to first compute per-station inducing coefficients via matmul:
                                    #   C_sta[:, j, phase] = W[:, :] @ A[:, j, phase]   (U x M)
                                    # and then do the same gather+weighted-sum as the per-station path.
                                    A = b_lat.to(torch.float32)  # (R,M,2)
                                    R = int(A.shape[0])
                                    M = int(A.shape[1])
                                    # Compress stations in this minibatch
                                    sta_u, sta_inv = torch.unique(sta_bi, sorted=False, return_inverse=True)  # (U,), (B,)
                                    W_u = W_sta.index_select(0, sta_u).to(torch.float32)  # (U,R)
                                    # (U,M) per phase
                                    C_u_P = torch.matmul(W_u, A[:, :, 0])  # (U,M)
                                    C_u_S = torch.matmul(W_u, A[:, :, 1])  # (U,M)

                                    def _recon_endpoint_scalar(C_u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
                                        # C_u: (U,M), returns (B,)
                                        idx = nei_idx.index_select(0, e)  # (B,m)
                                        kk = nei_k.index_select(0, e).to(torch.float32)  # (B,m)
                                        msk = (idx >= 0)
                                        idxc = idx.clamp_min(0)
                                        w = kk * msk.to(torch.float32)  # (B,m)
                                        # Gather per-row station-specific coefficients at neighbor indices:
                                        # (B,m) = C_u[sta_inv[:,None], idxc]
                                        vals = C_u[sta_inv.unsqueeze(1), idxc]
                                        return (vals * w).sum(dim=1)  # (B,)

                                    b1P = _recon_endpoint_scalar(C_u_P, e1)
                                    b2P = _recon_endpoint_scalar(C_u_P, e2)
                                    b1S = _recon_endpoint_scalar(C_u_S, e1)
                                    b2S = _recon_endpoint_scalar(C_u_S, e2)
                                    # Keep endpoint tensors for diagnostics below
                                    b1 = torch.stack([b1P, b1S], dim=1)
                                    b2 = torch.stack([b2P, b2S], dim=1)
                                    dP = (b2P - b1P).to(torch.float32)
                                    dS = (b2S - b1S).to(torch.float32)
                                    delta_b = torch.where(is_s, dS, dP)
                                elif is_per_station:
                                    # Per-station coefficients: compute both channels for all rows (single fused gather).
                                    ev12 = torch.cat([e1, e2], dim=0)
                                    idx12 = nei_idx.index_select(0, ev12)  # (2B,m)
                                    k12 = nei_k.index_select(0, ev12)      # (2B,m)
                                    idx12c = idx12.clamp_min(0)
                                    w12 = k12.to(torch.float32) * (idx12 >= 0).to(torch.float32)

                                    b_lat_sta = b_lat.index_select(0, sta_bi).to(torch.float32)  # (B,M,2)
                                    m = int(idx12c.shape[1])
                                    idx12c3 = idx12c.view(2, -1, m).unsqueeze(-1).expand(-1, -1, -1, 2)  # (2,B,m,2)
                                    g12 = torch.gather(b_lat_sta.unsqueeze(0).expand(2, -1, -1, -1), 2, idx12c3)  # (2,B,m,2)
                                    b12 = (g12 * w12.view(2, -1, m).unsqueeze(-1)).sum(dim=2)  # (2,B,2)
                                    b1 = b12[0]
                                    b2 = b12[1]

                                    d = (b2 - b1)
                                    delta_b = torch.where(is_s, d[:, 1], d[:, 0])
                                else:
                                    delta_b = None
                        else:
                            delta_b = None
                    else:
                        # Explicit event-latent parameterization (full / graph_gmrf):
                        # - per-station: b_lat shape (n_stations, n_events, 2)
                        # - station-basis: b_lat shape (R, n_events, 2) with W_sta (n_stations, R)
                        if b_lat.ndim == 3 and int(b_lat.shape[2]) == 2:
                            n_stations_rt = int(getattr(state, "n_stations", 0))
                            is_per_station = (n_stations_rt > 0) and (int(b_lat.shape[0]) == int(n_stations_rt))
                            is_basis = (
                                isinstance(W_sta, torch.Tensor)
                                and W_sta.ndim == 2
                                and int(W_sta.shape[0]) == int(n_stations_rt)
                                and int(b_lat.shape[0]) == int(W_sta.shape[1])
                            )

                            if is_basis:
                                # b(s,e,phase) = Σ_r W[s,r] * a_r(e,phase)
                                A = b_lat.to(torch.float32)  # (R,Ne,2)
                                W = W_sta.index_select(0, sta_bi).to(torch.float32)  # (B,R)
                                # Gather event endpoints in basis space: (R,B,2) -> (B,R,2)
                                A1 = A.index_select(1, e1).transpose(0, 1).contiguous()
                                A2 = A.index_select(1, e2).transpose(0, 1).contiguous()
                                b1 = (W.unsqueeze(-1) * A1).sum(dim=1)  # (B,2)
                                b2 = (W.unsqueeze(-1) * A2).sum(dim=1)  # (B,2)
                                d = (b2 - b1).to(torch.float32)
                                delta_b = torch.where(is_s, d[:, 1], d[:, 0])
                            elif is_per_station:
                                # Direct per-station coefficients
                                b1 = b_lat[sta_bi, e1]  # (B,2)
                                b2 = b_lat[sta_bi, e2]  # (B,2)
                                d = (b2 - b1).to(torch.float32)
                                delta_b = torch.where(is_s, d[:, 1], d[:, 0])
                            else:
                                delta_b = None
                        else:
                            delta_b = None

                    if isinstance(delta_b, torch.Tensor):
                        nuisance_delta = delta_b if nuisance_delta is None else (nuisance_delta + delta_b)
                        if want_lat_diag:
                            if b_delta_sum is None:
                                b_delta_sum = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_delta_sumsq = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_delta_maxabs = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_delta_count = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_end_sumsq_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_end_sumsq_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_end_count_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                b_end_count_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                            b_delta_sum = b_delta_sum + delta_b.sum()
                            b_delta_sumsq = b_delta_sumsq + (delta_b * delta_b).sum()
                            b_delta_maxabs = torch.maximum(b_delta_maxabs, delta_b.abs().max())
                            b_delta_count = b_delta_count + float(delta_b.numel())
                            # Endpoint RMS diagnostics:
                            # For each DD row, we can also track the RMS of the *event-level* latent values at both endpoints.
                            # This remains meaningful for both parameterizations:
                            #   - full: bP1/bP2/bS1/bS2 are direct event-level values
                            #   - inducing_gp: b1P/b2P/b1S/b2S are reconstructed event-level values via interpolation
                            try:
                                _bP1 = b1[:, 0].to(torch.float32)
                                _bP2 = b2[:, 0].to(torch.float32)
                                _bS1 = b1[:, 1].to(torch.float32)
                                _bS2 = b2[:, 1].to(torch.float32)
                                is_p_f = (~is_s).to(torch.float32)
                                is_s_f = is_s.to(torch.float32)
                                b_end_sumsq_p = b_end_sumsq_p + ((_bP1 * _bP1 + _bP2 * _bP2) * is_p_f).sum()
                                b_end_sumsq_s = b_end_sumsq_s + ((_bS1 * _bS1 + _bS2 * _bS2) * is_s_f).sum()
                                # Each row contributes two endpoints
                                b_end_count_p = b_end_count_p + (2.0 * is_p_f.sum())
                                b_end_count_s = b_end_count_s + (2.0 * is_s_f.sum())
                            except Exception:
                                pass
        except Exception:
            pass
        finally:
            if prof_se_lat and (se_lat_t0 is not None):
                try:
                    import time as _time
                    if state.device.type == "cuda":
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    dt_ms = 1000.0 * float(_time.perf_counter() - se_lat_t0)
                    # Accumulate per-epoch; emit once at end like other per-epoch metrics.
                    if "_se_lat_time_ms_sum" not in state.params:
                        state.params["_se_lat_time_ms_sum"] = 0.0
                        state.params["_se_lat_time_ms_count"] = 0
                    state.params["_se_lat_time_ms_sum"] = float(state.params["_se_lat_time_ms_sum"]) + float(dt_ms)
                    state.params["_se_lat_time_ms_count"] = int(state.params["_se_lat_time_ms_count"]) + 1
                except Exception:
                    pass

        # sigma_inflation removed (start fresh).

        if ddp_enabled:
            # DDP-safe loss construction:
            # - Likelihood term is the *global batch mean* across all ranks: (1/B) Σ_i NLL_i.
            #   Each rank computes a local mean and scales by (B_r / B).
            # - Prior term is already scaled by 1/N_total in compute_prior_loss(); we want it included once,
            #   so we add it as (1/world_size) per rank.
            try:
                local_bsz = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
            except Exception:
                local_bsz = 0
            try:
                B_global = int(global_bsz) if "global_bsz" in locals() else int(local_bsz)
            except Exception:
                B_global = int(local_bsz)
            B_global = max(1, int(B_global))

            if local_bsz > 0:
                # Optional profiling: time the shared_event_re likelihood computation.
                try:
                    diag = _get_diagnostics_cfg(state.params)
                    prof_se_re = bool(diag.get("profile_shared_event_re", False)) if isinstance(diag, dict) else False
                except Exception:
                    prof_se_re = False
                se_re_t0 = None
                sl_re_t0 = None
                try:
                    diag = _get_diagnostics_cfg(state.params)
                except Exception:
                    prof_sl_re = False
                if prof_se_re and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        se_re_t0 = _time.perf_counter()
                    except Exception:
                        se_re_t0 = None
                    try:
                        import time as _time
                        sl_re_t0 = _time.perf_counter()
                    except Exception:
                        sl_re_t0 = None
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_prelog", False)):
                    state.params["_shared_event_re_prelog"] = True
                    try:
                        n_rows = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
                        solver = str(state.params.get("_shared_event_re_solver", ""))
                        grouping = str(state.params.get("_shared_event_re_grouping", ""))
                        # Sentinels to verify shared_event_re block executed.
                        state.params["_shared_event_re_runtime_last_groups"] = -1
                        state.params["_shared_event_re_runtime_last_groups_pcg"] = -1
                        state.params["_shared_event_re_runtime_last_groups_fallback_diag"] = -1
                        info(
                            f"shared_event_re prelog rows={n_rows} solver={solver} grouping={grouping}",
                            section="LIKELIHOOD",
                        )
                    except Exception:
                        pass
                loss_like = compute_likelihood_loss(
                    idx=II_b,
                    y=YY_b,
                    X_src=state.X_src,
                    ΔX_src=state.dX_src,
                    model=state.model,
                    σ_p=σp,
                    σ_s=σs,
                    params=state.params,
                    nuisance_delta=nuisance_delta,
                )
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_postlog", False)):
                    state.params["_shared_event_re_postlog"] = True
                    g = int(state.params.get("_shared_event_re_runtime_last_groups", -1))
                    g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", -1))
                    g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", -1))
                    info(
                        f"shared_event_re postlog groups={g} pcg={g_pcg} fallback={g_fb}",
                        section="LIKELIHOOD",
                    )
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_batch_logged", False)):
                    state.params["_shared_event_re_batch_logged"] = True
                    try:
                        n_rows = int(II_b.shape[0])
                        info(f"shared_event_re batch rows={n_rows}", section="LIKELIHOOD")
                        if n_rows > 0:
                            ph_id = torch.where(
                                YY_b[:, 4] < 0.5,
                                torch.zeros_like(YY_b[:, 4], dtype=torch.int64),
                                torch.ones_like(YY_b[:, 4], dtype=torch.int64),
                            )
                            grouping = str(state.params.get("_shared_event_re_grouping", "phase")).strip().lower()
                            if grouping in {"stationphase", "station-phase"}:
                                grouping = "station_phase"
                            if grouping == "station_phase":
                                sta_idx = state.params.get("_runtime_bucket_station_index", None)
                                if isinstance(sta_idx, torch.Tensor) and int(sta_idx.numel()) == int(ph_id.numel()):
                                    keys = (sta_idx.to(dtype=torch.int64) * 2) + ph_id
                                else:
                                    keys = ph_id
                                    grouping = "phase"
                            else:
                                keys = ph_id
                            keys_sorted, _ = torch.sort(keys)
                            if keys_sorted.numel() > 0:
                                is_new = torch.ones_like(keys_sorted, dtype=torch.bool)
                                is_new[1:] = keys_sorted[1:] != keys_sorted[:-1]
                                n_groups = int(torch.nonzero(is_new, as_tuple=False).shape[0])
                            else:
                                n_groups = 0
                            info(f"shared_event_re batch grouping={grouping} groups={n_groups}", section="LIKELIHOOD")
                    except Exception as e:
                        pass
                        pass
                        pass
                        pass
                        info(f"shared_event_re batch log failed: {e}", section="LIKELIHOOD")
                if not bool(state.params.get("_shared_event_re_flag_logged", False)):
                    state.params["_shared_event_re_flag_logged"] = True
                    info(
                        f"shared_event_re flag={bool(state.params.get('_shared_event_re_enabled', False))}",
                        section="LIKELIHOOD",
                    )
                # One-time shared_event_re runtime report (from modeling.py stats).
                if bool(state.params.get("_shared_event_re_enabled", False)):
                    g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                    g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                    g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                    g_rows = int(state.params.get("_shared_event_re_runtime_last_groups_rows_cap", 0) or 0)
                    g_nodes = int(state.params.get("_shared_event_re_runtime_last_groups_nodes_cap", 0) or 0)
                    g_tau0 = int(state.params.get("_shared_event_re_runtime_last_groups_tau_zero", 0) or 0)
                    mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                    mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                    mr_all = int(state.params.get("_shared_event_re_runtime_last_max_rows_all", 0) or 0)
                    mn_all = int(state.params.get("_shared_event_re_runtime_last_max_nodes_all", 0) or 0)
                    tg = state.params.get("_shared_event_re_tau_s", [0.0, 0.0])
                    log_every = int(state.params.get("_shared_event_re_stats_log_every_epochs", 0) or 0)
                    if log_every > 0 and (epoch_index % log_every == 0):
                        grouping = str(state.params.get("_shared_event_re_runtime_last_grouping", "phase"))
                        info(
                            f"shared_event_re stats epoch={epoch_index} grouping={grouping} "
                            f"groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"rows_cap={g_rows} nodes_cap={g_nodes} tau0={g_tau0} "
                            f"max_rows={mr} max_nodes={mn} max_rows_all={mr_all} max_nodes_all={mn_all}",
                            section="LIKELIHOOD",
                        )
                    if not bool(state.params.get("_shared_event_re_reported", False)):
                        state.params["_shared_event_re_reported"] = True
                        info(
                            f"shared_event_re stats groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"max_rows={mr} max_nodes={mn} tau_s={tg}",
                            section="LIKELIHOOD",
                        )
                    if g_fb > 0:
                        info(
                            f"shared_event_re fallback reasons rows_cap={g_rows} nodes_cap={g_nodes} tau_zero={g_tau0}",
                            section="LIKELIHOOD",
                        )
                        if (not ddp_enabled) or ddp_is_main:
                            if bool(state.params.get("_shared_event_re_auto_tune_nodes_cap", False)) and (g_nodes > 0):
                                cur = int(state.params.get("_shared_event_re_max_nodes_per_group", 0) or 0)
                                cap = int(state.params.get("_shared_event_re_auto_tune_nodes_max", 0) or 0)
                                target = int(min(max(mn_all, cur), cap)) if cap > 0 else int(max(mn_all, cur))
                                if target > cur and mn_all > 0:
                                    state.params["_shared_event_re_max_nodes_per_group"] = target
                                    info(
                                        f"shared_event_re auto-tune: max_nodes_per_group -> {target}",
                                        section="LIKELIHOOD",
                                    )
                            if bool(state.params.get("_shared_event_re_auto_tune_rows_cap", False)) and (g_rows > 0):
                                cur = int(state.params.get("_shared_event_re_max_rows_per_group", 0) or 0)
                                cap = int(state.params.get("_shared_event_re_auto_tune_rows_max", 0) or 0)
                                target = int(min(max(mr_all, cur), cap)) if cap > 0 else int(max(mr_all, cur))
                                if target > cur and mr_all > 0:
                                    state.params["_shared_event_re_max_rows_per_group"] = target
                                    info(
                                        f"shared_event_re auto-tune: max_rows_per_group -> {target}",
                                        section="LIKELIHOOD",
                                    )
                    if g > 0 and g_pcg == 0 and g_fb >= g:
                        info(
                            "shared_event_re warning: all groups fell back to diagonal; "
                            "increase shared_event_re.max_nodes_per_group or max_rows_per_group",
                            section="LIKELIHOOD",
                        )
                if bool(state.params.get("_shared_event_re_whitening_enabled", False)) and not bool(state.params.get("_shared_event_re_whitening_reported", False)):
                    state.params["_shared_event_re_whitening_reported"] = True
                    wg = int(state.params.get("_shared_event_re_whitening_last_groups", 0) or 0)
                    wchol = int(state.params.get("_shared_event_re_whitening_last_groups_chol", 0) or 0)
                    wfb = int(state.params.get("_shared_event_re_whitening_last_groups_fallback_diag", 0) or 0)
                    wrows = int(state.params.get("_shared_event_re_whitening_last_groups_rows_cap", 0) or 0)
                    wnodes = int(state.params.get("_shared_event_re_whitening_last_groups_nodes_cap", 0) or 0)
                    wtau0 = int(state.params.get("_shared_event_re_whitening_last_groups_tau_zero", 0) or 0)
                    wmr = int(state.params.get("_shared_event_re_whitening_last_max_rows", 0) or 0)
                    wmn = int(state.params.get("_shared_event_re_whitening_last_max_nodes", 0) or 0)
                    info(
                        f"shared_event_re whitening groups={wg} chol={wchol} fallback={wfb} max_rows={wmr} max_nodes={wmn}",
                        section="LIKELIHOOD",
                    )
                    if wfb > 0:
                        info(
                            f"shared_event_re whitening fallback reasons rows_cap={wrows} nodes_cap={wnodes} tau_zero={wtau0}",
                            section="LIKELIHOOD",
                        )
                # One-time debug: compare shared_event_re vs baseline loss on this batch.
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_debug_compare_logged", False)):
                    state.params["_shared_event_re_debug_compare_logged"] = True
                    try:
                        with torch.no_grad():
                            params_dbg = dict(state.params)
                            params_dbg["_shared_event_re_enabled"] = False
                            loss_base = compute_likelihood_loss(
                                idx=II_b,
                                y=YY_b,
                                X_src=state.X_src,
                                ΔX_src=state.dX_src,
                                model=state.model,
                                σ_p=σp,
                                σ_s=σs,
                                params=params_dbg,
                                nuisance_delta=nuisance_delta,
                            )
                        delta = float((loss_like - loss_base).detach().item())
                        _log(f"[shared_event_re] loss delta vs baseline = {delta:.6e}", flush=True)
                    except Exception as e:
                        _log(f"[shared_event_re] debug compare failed: {e}", flush=True)
                    try:
                        with torch.no_grad():
                            params_dbg = dict(state.params)
                            loss_base = compute_likelihood_loss(
                                idx=II_b,
                                y=YY_b,
                                X_src=state.X_src,
                                ΔX_src=state.dX_src,
                                model=state.model,
                                σ_p=σp,
                                σ_s=σs,
                                params=params_dbg,
                                nuisance_delta=nuisance_delta,
                            )
                        delta = float((loss_like - loss_base).detach().item())
                    except Exception as e:
                        pass
                # One-time shared_event_re runtime summary after the first loss call.
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_runtime_logged", False)):
                    state.params["_shared_event_re_runtime_logged"] = True
                    try:
                        g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                        g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                        g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                        _log(
                            f"[shared_event_re] runtime groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"max_rows={mr} max_nodes={mn}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    try:
                        _log(
                            f"max_rows={mr} max_nodes={mn}",
                            flush=True,
                        )
                    except Exception:
                        pass
                if prof_se_re and (se_re_t0 is not None) and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - se_re_t0)
                        state.params["_se_re_time_ms_sum"] = float(state.params.get("_se_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        state.params["_se_re_time_ms_count"] = int(state.params.get("_se_re_time_ms_count", 0) or 0) + 1
                        # Workload stats from modeling.py (set per-call)
                        g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                        g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                        g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                        state.params["_se_re_groups_sum"] = int(state.params.get("_se_re_groups_sum", 0) or 0) + g
                        state.params["_se_re_groups_pcg_sum"] = int(state.params.get("_se_re_groups_pcg_sum", 0) or 0) + g_pcg
                        state.params["_se_re_groups_fallback_sum"] = int(state.params.get("_se_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_se_re_max_rows_max"] = max(int(state.params.get("_se_re_max_rows_max", 0) or 0), mr)
                        state.params["_se_re_max_nodes_max"] = max(int(state.params.get("_se_re_max_nodes_max", 0) or 0), mn)
                    except Exception:
                        pass
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - sl_re_t0)
                        # Workload stats from modeling.py (set per-call)
                        state.params["_sl_re_groups_sum"] = int(state.params.get("_sl_re_groups_sum", 0) or 0) + g
                        state.params["_sl_re_groups_woodbury_sum"] = int(state.params.get("_sl_re_groups_woodbury_sum", 0) or 0) + g_w
                        state.params["_sl_re_groups_fallback_sum"] = int(state.params.get("_sl_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_sl_re_max_rows_max"] = max(int(state.params.get("_sl_re_max_rows_max", 0) or 0), mr)
                        state.params["_sl_re_max_nodes_max"] = max(int(state.params.get("_sl_re_max_nodes_max", 0) or 0), mn)
                        state.params["_sl_re_max_M_max"] = max(int(state.params.get("_sl_re_max_M_max", 0) or 0), int(state.params.get("_sl_re_max_M_static", 0) or 0))
                    except Exception:
                        pass
            else:
                loss_like = torch.tensor(0.0, device=state.device, dtype=torch.float32)

            loss_prior = compute_prior_loss(
                ΔX_src=state.dX_src,
                prior_event=state.prior_event,
                prior_centroid=state.prior_centroid,
                σ_p=σp,
                σ_s=σs,
                N_total=state.N,
                params=state.params,
                cluster_ids=state.cluster_ids,
                cluster_counts=state.cluster_counts,
                event_precision_matrix=state.event_precision_matrix,
            )

            loss = loss_like * (float(local_bsz) / float(B_global)) + (loss_prior / float(ddp_world_size))
        else:
            # Non-DDP path: compute likelihood + prior explicitly (same math as posterior_loss),
            # so we can reuse shared_event_re profiling/timing.
            try:
                local_bsz = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
            except Exception:
                local_bsz = 0

            if local_bsz > 0:
                # Optional profiling: time the shared_event_re likelihood computation.
                try:
                    diag = _get_diagnostics_cfg(state.params)
                    prof_se_re = bool(diag.get("profile_shared_event_re", False)) if isinstance(diag, dict) else False
                except Exception:
                    prof_se_re = False
                    prof_sl_re = False
                se_re_t0 = None
                sl_re_t0 = None
                if prof_se_re and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        se_re_t0 = _time.perf_counter()
                    except Exception:
                        se_re_t0 = None
                    try:
                        import time as _time
                        sl_re_t0 = _time.perf_counter()
                    except Exception:
                        sl_re_t0 = None
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_prelog", False)):
                    state.params["_shared_event_re_prelog"] = True
                    try:
                        n_rows = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
                        solver = str(state.params.get("_shared_event_re_solver", ""))
                        grouping = str(state.params.get("_shared_event_re_grouping", ""))
                        # Sentinels to verify shared_event_re block executed.
                        state.params["_shared_event_re_runtime_last_groups"] = -1
                        state.params["_shared_event_re_runtime_last_groups_pcg"] = -1
                        state.params["_shared_event_re_runtime_last_groups_fallback_diag"] = -1
                        info(
                            f"shared_event_re prelog rows={n_rows} solver={solver} grouping={grouping}",
                            section="LIKELIHOOD",
                        )
                    except Exception:
                        pass

                loss_like = compute_likelihood_loss(
                    idx=II_b,
                    y=YY_b,
                    X_src=state.X_src,
                    ΔX_src=state.dX_src,
                    model=state.model,
                    σ_p=σp,
                    σ_s=σs,
                    params=state.params,
                    nuisance_delta=nuisance_delta,
                )
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_postlog", False)):
                    state.params["_shared_event_re_postlog"] = True
                    g = int(state.params.get("_shared_event_re_runtime_last_groups", -1))
                    g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", -1))
                    g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", -1))
                    info(
                        f"shared_event_re postlog groups={g} pcg={g_pcg} fallback={g_fb}",
                        section="LIKELIHOOD",
                    )
                if bool(state.params.get("_shared_event_re_enabled", False)):
                    g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                    g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                    g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                    g_rows = int(state.params.get("_shared_event_re_runtime_last_groups_rows_cap", 0) or 0)
                    g_nodes = int(state.params.get("_shared_event_re_runtime_last_groups_nodes_cap", 0) or 0)
                    g_tau0 = int(state.params.get("_shared_event_re_runtime_last_groups_tau_zero", 0) or 0)
                    mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                    mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                    tg = state.params.get("_shared_event_re_tau_s", [0.0, 0.0])
                    if not bool(state.params.get("_shared_event_re_reported", False)):
                        state.params["_shared_event_re_reported"] = True
                        info(
                            f"shared_event_re stats groups={g} pcg={g_pcg} fallback={g_fb} "
                            f"max_rows={mr} max_nodes={mn} tau_s={tg}",
                            section="LIKELIHOOD",
                        )
                    if g_fb > 0:
                        info(
                            f"shared_event_re fallback reasons rows_cap={g_rows} nodes_cap={g_nodes} tau_zero={g_tau0}",
                            section="LIKELIHOOD",
                        )
                        if (not ddp_enabled) or ddp_is_main:
                            if bool(state.params.get("_shared_event_re_auto_tune_nodes_cap", False)) and (g_nodes > 0):
                                cur = int(state.params.get("_shared_event_re_max_nodes_per_group", 0) or 0)
                                cap = int(state.params.get("_shared_event_re_auto_tune_nodes_max", 0) or 0)
                                target = int(min(max(mn_all, cur), cap)) if cap > 0 else int(max(mn_all, cur))
                                if target > cur and mn_all > 0:
                                    state.params["_shared_event_re_max_nodes_per_group"] = target
                                    info(
                                        f"shared_event_re auto-tune: max_nodes_per_group -> {target}",
                                        section="LIKELIHOOD",
                                    )
                            if bool(state.params.get("_shared_event_re_auto_tune_rows_cap", False)) and (g_rows > 0):
                                cur = int(state.params.get("_shared_event_re_max_rows_per_group", 0) or 0)
                                cap = int(state.params.get("_shared_event_re_auto_tune_rows_max", 0) or 0)
                                target = int(min(max(mr_all, cur), cap)) if cap > 0 else int(max(mr_all, cur))
                                if target > cur and mr_all > 0:
                                    state.params["_shared_event_re_max_rows_per_group"] = target
                                    info(
                                        f"shared_event_re auto-tune: max_rows_per_group -> {target}",
                                        section="LIKELIHOOD",
                                    )
                    if g > 0 and g_pcg == 0 and g_fb >= g:
                        info(
                            "shared_event_re warning: all groups fell back to diagonal; "
                            "increase shared_event_re.max_nodes_per_group or max_rows_per_group",
                            section="LIKELIHOOD",
                        )
                if bool(state.params.get("_shared_event_re_enabled", False)) and not bool(state.params.get("_shared_event_re_debug_compare_logged", False)):
                    state.params["_shared_event_re_debug_compare_logged"] = True
                    try:
                        with torch.no_grad():
                            params_dbg = dict(state.params)
                            params_dbg["_shared_event_re_enabled"] = False
                            loss_base = compute_likelihood_loss(
                                idx=II_b,
                                y=YY_b,
                                X_src=state.X_src,
                                ΔX_src=state.dX_src,
                                model=state.model,
                                σ_p=σp,
                                σ_s=σs,
                                params=params_dbg,
                                nuisance_delta=nuisance_delta,
                            )
                        delta = float((loss_like - loss_base).detach().item())
                        info(f"shared_event_re loss delta vs baseline = {delta:.6e}", section="LIKELIHOOD")
                    except Exception as e:
                        info(f"shared_event_re debug compare failed: {e}", section="LIKELIHOOD")

                if prof_se_re and (se_re_t0 is not None) and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - se_re_t0)
                        state.params["_se_re_time_ms_sum"] = float(state.params.get("_se_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        state.params["_se_re_time_ms_count"] = int(state.params.get("_se_re_time_ms_count", 0) or 0) + 1
                        # Workload stats from modeling.py (set per-call)
                        g = int(state.params.get("_shared_event_re_runtime_last_groups", 0) or 0)
                        g_pcg = int(state.params.get("_shared_event_re_runtime_last_groups_pcg", 0) or 0)
                        g_fb = int(state.params.get("_shared_event_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_shared_event_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_shared_event_re_runtime_last_max_nodes", 0) or 0)
                        state.params["_se_re_groups_sum"] = int(state.params.get("_se_re_groups_sum", 0) or 0) + g
                        state.params["_se_re_groups_pcg_sum"] = int(state.params.get("_se_re_groups_pcg_sum", 0) or 0) + g_pcg
                        state.params["_se_re_groups_fallback_sum"] = int(state.params.get("_se_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_se_re_max_rows_max"] = max(int(state.params.get("_se_re_max_rows_max", 0) or 0), mr)
                        state.params["_se_re_max_nodes_max"] = max(int(state.params.get("_se_re_max_nodes_max", 0) or 0), mn)
                    except Exception:
                        pass
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - sl_re_t0)
                        state.params["_sl_re_groups_sum"] = int(state.params.get("_sl_re_groups_sum", 0) or 0) + g
                        state.params["_sl_re_groups_woodbury_sum"] = int(state.params.get("_sl_re_groups_woodbury_sum", 0) or 0) + g_w
                        state.params["_sl_re_groups_fallback_sum"] = int(state.params.get("_sl_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_sl_re_max_rows_max"] = max(int(state.params.get("_sl_re_max_rows_max", 0) or 0), mr)
                        state.params["_sl_re_max_nodes_max"] = max(int(state.params.get("_sl_re_max_nodes_max", 0) or 0), mn)
                        state.params["_sl_re_max_M_max"] = max(int(state.params.get("_sl_re_max_M_max", 0) or 0), int(state.params.get("_sl_re_max_M_static", 0) or 0))
                    except Exception:
                        pass
            else:
                loss_like = torch.tensor(0.0, device=state.device, dtype=torch.float32)

            loss_prior = compute_prior_loss(
                ΔX_src=state.dX_src,
                prior_event=state.prior_event,
                prior_centroid=state.prior_centroid,
                σ_p=σp,
                σ_s=σs,
                N_total=state.N,
                params=state.params,
                cluster_ids=state.cluster_ids,
                cluster_counts=state.cluster_counts,
                event_precision_matrix=state.event_precision_matrix,
            )
            loss = loss_like + loss_prior

        # L2 Reg (Phase 1/General)
        if state.nuisance_enable and state.nuisance_alpha is not None:
             lam = float(state.params.get("nuisance_alpha_l2", 0.0))
             if lam > 0.0:
                 # In DDP mode, treat this like a prior/regularizer (include once).
                 reg = lam * (state.nuisance_alpha.square().mean())
                 loss = loss + (reg / float(ddp_world_size) if ddp_enabled else reg)

        # Minibatch graph Laplacian penalty (suppresses internal floppy deformation modes)
        # Uses the same observed event pairs (II_b) that define the DD graph.
        # if lap_w > 0.0 and II_b is not None and II_b.numel() > 0:
        #     e1 = II_b[:, 0]
        #     e2 = II_b[:, 1]
        #     dx1 = state.dX_src.index_select(0, e1)
        #     dx2 = state.dX_src.index_select(0, e2)
        #     diff = (dx1 - dx2) * lap_dims
        #     lap_term = (diff * diff).sum(dim=1).mean()
        #     loss = loss + lap_w * lap_term

        # IMPORTANT DDP invariant:
        # All ranks must execute the same sequence of collectives.
        #
        # Previously, a rank could hit a non-finite loss and `continue` here, while other ranks
        # proceeded into gradient allreduce, causing NCCL to deadlock until watchdog timeout.
        #
        # Fix: in DDP mode, detect non-finite loss, synchronize a `bad_loss` flag, and if any
        # rank is bad, run a *dummy* backward that touches all parameters (zero gradients) so
        # the collective schedule stays aligned. Then skip the optimizer step on all ranks.
        bad_loss = 0
        try:
            bad_loss = 0 if bool(torch.isfinite(loss).item()) else 1
        except Exception:
            bad_loss = 1
        if ddp_enabled:
            try:
                bad_t0 = torch.tensor([bad_loss], device=state.device, dtype=torch.int32)
                dist.all_reduce(bad_t0, op=dist.ReduceOp.MAX)
                bad_loss = int(bad_t0.item())
            except Exception:
                bad_loss = 1
        if (not ddp_enabled) and bad_loss:
            # Non-DDP: keep historical behavior (skip this batch).
            continue

        if ddp_enabled and bad_loss:
            # Dummy loss: depends on parameters but yields zero gradients.
            # This ensures p.grad tensors exist and `_ddp_allreduce_grads` can run safely.
            loss0 = torch.zeros((), device=state.device, dtype=torch.float32)
            try:
                for g in optimizer.param_groups:  # type: ignore[attr-defined]
                    for p in g.get("params", []):
                        if isinstance(p, torch.Tensor) and bool(getattr(p, "requires_grad", False)):
                            loss0 = loss0 + (p.sum() * 0.0)
            except Exception:
                pass
            loss = loss0

        loss.backward()

        # DDP: all-reduce gradients (SUM) so all ranks take identical optimizer steps.
        if ddp_enabled:
            _ddp_allreduce_grads(optimizer)
            # Abort step if any rank produced non-finite gradients
            bad = 0
            try:
                for g in optimizer.param_groups:  # type: ignore[attr-defined]
                    for p in g.get("params", []):
                        if p is None or getattr(p, "grad", None) is None:
                            continue
                        if not torch.isfinite(p.grad).all():
                            bad = 1
                            break
                    if bad:
                        break
            except Exception:
                bad = 1
            try:
                bad_t = torch.tensor([bad], device=state.device, dtype=torch.int32)
                dist.all_reduce(bad_t, op=dist.ReduceOp.MAX)
                bad = int(bad_t.item())
            except Exception:
                bad = 1
            if bad:
                optimizer.zero_grad(set_to_none=True)
                continue
            # Also skip the step if any rank had a non-finite loss (handled via dummy backward above).
            if bad_loss:
                optimizer.zero_grad(set_to_none=True)
                continue

            try:
                    def _rms(t: torch.Tensor | None) -> float:
                        if not isinstance(t, torch.Tensor) or t.numel() == 0:
                            return float("nan")
                        return float(t.detach().square().mean().sqrt().item())
                    def _grms(t: torch.Tensor | None) -> float:
                        if not isinstance(t, torch.Tensor):
                            return float("nan")
                        g = getattr(t, "grad", None)
                        if not isinstance(g, torch.Tensor) or g.numel() == 0:
                            return float("nan")
                        return float(g.detach().square().mean().sqrt().item())
                    _log(
                        f"comp_p={_rms(s_cp):.3g}/{_grms(s_cp):.3g} "
                        f"comp_s={_rms(s_cs):.3g}/{_grms(s_cs):.3g} "
                        f"sta_p={_rms(s_sp):.3g}/{_grms(s_sp):.3g} "
                        f"sta_s={_rms(s_ss):.3g}/{_grms(s_ss):.3g}",
                        flush=True,
                    )
            except Exception:
                pass

        # SVRG Correction
        if svrg_enabled and state.svrg_grad_full is not None and state.svrg_dX_snapshot is not None:
            # 1. Compute grad at snapshot location for THIS batch
            # We need to temporarily swap dX_src to snapshot
            # Detach current grad first
            grad_batch_current = state.dX_src.grad.clone()
            
            # Swap params
            dX_current_data = state.dX_src.data.clone()
            state.dX_src.data.copy_(state.svrg_dX_snapshot)
            if state.dX_src.grad is not None:
                state.dX_src.grad.zero_()
                
            # Compute loss at snapshot
            loss_snap = posterior_loss(
                idx=II_b,
                y=YY_b,
                X_src=state.X_src,
                ΔX_src=state.dX_src, # Now holding snapshot
                model=state.model,
                prior_event=state.prior_event,
                prior_centroid=state.prior_centroid,
                σ_p=σp,
                σ_s=σs,
                N_total=state.N,
                params=state.params,
                nuisance_delta=nuisance_delta,
                cluster_ids=state.cluster_ids,
                cluster_counts=state.cluster_counts,
                event_precision_matrix=state.event_precision_matrix,
            )
            loss_snap.backward()
            grad_batch_snapshot = state.dX_src.grad
            
            # Restore current params
            state.dX_src.data.copy_(dX_current_data)
            
            # Apply Correction: g = g_curr - g_snap + g_full
            # Note: We must handle potential None grads or shape mismatches carefully
            if grad_batch_snapshot is not None:
                # Corrected gradient
                corrected_grad = grad_batch_current - grad_batch_snapshot + state.svrg_grad_full
                state.dX_src.grad.copy_(corrected_grad)
            else:
                # Fallback: restore original
                state.dX_src.grad.copy_(grad_batch_current)

        # Grad Checks
        if state.dX_src.grad is not None and not torch.isfinite(state.dX_src.grad).all():
            optimizer.zero_grad(set_to_none=True)
            continue

        # Optional: per-dimension learning-rate multiplier for ΔT (origin time correction).
        # This is implemented as a gradient scaler so it works for Adam and our SGLD/SGHMC backends.
        try:
            dt_lr_mult = float(state.params.get("dt_lr_mult", 1.0))
            if math.isfinite(dt_lr_mult) and dt_lr_mult > 0.0 and (dt_lr_mult != 1.0):
                g = state.dX_src.grad
                if isinstance(g, torch.Tensor) and g.ndim == 2 and int(g.shape[1]) >= 4:
                    g[:, 3].mul_(dt_lr_mult)
        except Exception:
            pass

        # Gauge projection (translation-mode removal): apply to MAP/Adam as well (sampler backends handle internally).
        _maybe_apply_gauge_projection_for_optimizer(state, optimizer)
            
        # Clipping
        if grad_clip_norm > 0.0:
            params_to_clip = [state.dX_src]
            if state.nuisance_enable and state.nuisance_alpha is not None and state.nuisance_alpha.grad is not None:
                params_to_clip.append(state.nuisance_alpha)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=grad_clip_norm)
            
        # Ensure sampler noise is identical across ranks (SGHMC/pSGLD inject noise in optimizer.step()).
        if ddp_enabled:
            _ddp_set_step_seed(state.params, int(state.global_step_count), device=state.device)
        optimizer.step()
        _maybe_apply_gauge_projection_to_momentum_buffers(state, optimizer)
        
        # Safety Check
        if not torch.isfinite(state.dX_src).all():
             with torch.no_grad():
                 state.dX_src.data = torch.nan_to_num(state.dX_src.data, nan=0.0, posinf=0.0, neginf=0.0)
        
        _clamp_dX_inplace(state)
        
        # Sampling (Phase 4)
        if is_sampling:
            with torch.no_grad():
                # Save every N steps (to in-memory buffer; flushing to disk is handled elsewhere).
                write_samples = bool(state.params.get("write_samples", True))
                save_every_n = int(state.params.get("save_every_n", 1))
                if write_samples and save_every_n > 0 and ddp_is_main and (state.global_step_count % save_every_n == 0):
                    # Initialize sampling wall-clock start time on first saved sample.
                    # This is used for online ESS/sec diagnostics (independent of GPU timings).
                    try:
                        if not hasattr(state, "_sampling_wall_t0") or getattr(state, "_sampling_wall_t0") is None:
                            setattr(state, "_sampling_wall_t0", float(time.time()))
                    except Exception:
                        pass

                    state.samples.append(state.dX_src.detach().cpu().clone())
                    # Noise learning removed (fixed phase_unc only).

                    # Optional online ESS/IACT diagnostic using the in-memory samples buffer.
                    try:
                        if bool(state.params.get("ess_online_enabled", False)) and _want_wandb_group(state.params, "ess_online"):
                            every = int(state.params.get("ess_online_every_n_samples", 0))
                            window = int(state.params.get("ess_online_window", 0))
                            if window <= 0:
                                window = 512
                            min_needed = max(8, int(window))
                            if every > 0 and len(state.samples) >= min_needed and (len(state.samples) % every) == 0:
                                n_events_total = int(state.dX_src.shape[0])
                                k = int(state.params.get("ess_online_n_events", 0))
                                k = min(k, n_events_total)
                                if k > 0:
                                    seed = int(state.params.get("ess_online_seed", 0))
                                    rng = np.random.default_rng(seed)
                                    # Cache the chosen subset so it's stable.
                                    if "_ess_online_event_idx" not in state.params:
                                        idx = rng.choice(n_events_total, size=k, replace=False).astype(np.int64)
                                        state.params["_ess_online_event_idx"] = idx.tolist()
                                    idx_np = np.asarray(state.params.get("_ess_online_event_idx", []), dtype=np.int64)
                                    if idx_np.size > 0:
                                        dims = state.params.get("ess_online_dims", [0, 1, 2])
                                        if not isinstance(dims, list) or len(dims) == 0:
                                            dims = [0, 1, 2]
                                        max_lag_v = int(state.params.get("ess_online_max_lag", 0))
                                        from spider.diagnostics.online_ess import compute_online_ess
                                        res = compute_online_ess(
                                            samples=state.samples,
                                            event_idx=torch.as_tensor(idx_np, dtype=torch.int64),
                                            dims=[int(d) for d in dims],
                                            window=int(window),
                                            max_lag=(max_lag_v if max_lag_v > 0 else None),
                                        )
                                        ess_online_last = res.to_metrics(prefix="ess_online")

                                        # Derive ESS/sec style metrics so users can tune batch size empirically.
                                        # Use sampling wall-clock time since the first saved sample and also the
                                        # delta rate since the previous ESS update.
                                        try:
                                            now_t = float(time.time())
                                            t0 = float(getattr(state, "_sampling_wall_t0", now_t))
                                            elapsed_s = max(1e-9, now_t - t0)
                                            ess_med = float(ess_online_last.get("ess_online/ess_median", float("nan")))
                                            n_samp = float(ess_online_last.get("ess_online/n_samples", float("nan")))
                                            ess_online_last["ess_online/elapsed_s"] = float(elapsed_s)
                                            if np.isfinite(ess_med):
                                                ess_online_last["ess_online/ess_per_s_median"] = float(ess_med / elapsed_s)
                                            if np.isfinite(n_samp):
                                                ess_online_last["ess_online/samples_per_s"] = float(n_samp / elapsed_s)

                                            prev_t = getattr(state, "_ess_online_last_wall_time", None)
                                            prev_ess = getattr(state, "_ess_online_last_ess_median", None)
                                            if prev_t is not None and prev_ess is not None:
                                                dt = max(1e-9, float(now_t) - float(prev_t))
                                                dess = float(ess_med) - float(prev_ess) if np.isfinite(ess_med) else float("nan")
                                                if np.isfinite(dess):
                                                    ess_online_last["ess_online/ess_per_s_delta_median"] = float(dess / dt)
                                                ess_online_last["ess_online/delta_s"] = float(dt)

                                            setattr(state, "_ess_online_last_wall_time", float(now_t))
                                            setattr(state, "_ess_online_last_ess_median", float(ess_med))
                                        except Exception:
                                            pass

                                        setattr(state, "_ess_online_last_metrics", ess_online_last)
                                        setattr(state, "_ess_online_last_n_samples", int(ess_online_last.get("ess_online/n_samples", 0)))
                                        ess_online_updated_this_epoch = True
                                        # Terminal output intentionally suppressed; metrics are still logged.
                    except Exception:
                        # Don't crash sampling for diagnostics; count failures so users can see if it's flaky.
                        try:
                            setattr(state, "_ess_online_error_count", int(getattr(state, "_ess_online_error_count", 0)) + 1)
                        except Exception:
                            pass
                        
                # Periodic cache clear
                ec_every = int(state.params.get("cuda_empty_cache_every", 0))
                if ec_every > 0 and (state.global_step_count % ec_every == 0):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Stats
        # In DDP mode, the per-rank loss is constructed so that SUM across ranks equals the true global loss.
        # Only rank0 logs/aggregates to avoid duplicated histories.
        if ddp_enabled:
            try:
                loss_sum = loss.detach().clone()
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                loss_f = float(loss_sum.item())
            except Exception:
                loss_f = float("nan")
        else:
            loss_f = float(loss.item())
        if (not ddp_enabled) or ddp_is_main:
            total_loss_vals.append(loss_f)
        try:
            # Weight by number of rows/edges in this batch.
            # NOTE: must use the current batch tensor (II_b); other locals named `idx` exist in this function
            # for ESS diagnostics and are unrelated to batch size.
            if ddp_enabled:
                bsz = int(global_bsz) if "global_bsz" in locals() else int(II_b.shape[0])
            else:
                bsz = int(II_b.shape[0]) if isinstance(II_b, torch.Tensor) else 0
        except Exception:
            bsz = 0
        if bsz > 0 and ((not ddp_enabled) or ddp_is_main):
            total_loss_weighted_sum += loss_f * float(bsz)
            total_loss_weighted_denom += int(bsz)
        state.global_step_count += 1

    # End of Epoch Stats & Logging
    with torch.no_grad():
        # Optional: Gibbs update for hierarchical event prior precision (Wishart hyperprior).
        # This is allowed in any phase; updates are controlled only by `every_epochs` cadence.
        try:
            hier_enable = bool(getattr(state, "hierarchical_prior_enable", False)) and bool(state.params.get("hierarchical_event_prior", False))
        except Exception:
            hier_enable = False
        if hier_enable:
            # Hard break: per-phase scheduling removed; hyper-updates are always allowed when enabled.
            state.params["_hierarchical_hyper_runtime_enable"] = True
            try:
                if "hierarchical_prior_dof" not in state.params or "_hierarchical_scale_std" not in state.params:
                    raise KeyError("Hierarchical event hyperprior requires hierarchical_prior_dof and _hierarchical_scale_std (from priors.event.hyper.params)")
                if "_hierarchical_update_every_epochs" not in state.params:
                    raise KeyError("Hierarchical event hyperprior requires priors.event.hyper.update.every_epochs")
                update_every = int(state.params["_hierarchical_update_every_epochs"])
                update_every = max(1, update_every)
                do_hier_update = (epoch_index % update_every) == 0
            except Exception:
                do_hier_update = False
            if do_hier_update:
                try:
                    nu = float(state.params["hierarchical_prior_dof"])
                    p_std = torch.tensor(state.params["_hierarchical_scale_std"], device=state.device, dtype=torch.float32)
                    V_inv_common = nu * torch.diag(p_std ** 2)

                    if state.cluster_ids is not None and state.cluster_counts is not None:
                        K = int(state.cluster_counts.shape[0])
                        new_P0s = []
                        for k in range(K):
                            mask_k = (state.cluster_ids == k)
                            dX_k = state.dX_src[mask_k]
                            if dX_k.shape[0] == 0:
                                new_P0s.append(V_inv_common * (1.0 / max(nu, 1e-12)))
                                continue
                            P0_k = update_precision_hyperparameter(dX_k.detach(), nu, V_inv_common, mode="sample")
                            new_P0s.append(P0_k)
                        state.event_precision_matrix = torch.stack(new_P0s, dim=0)
                    else:
                        new_P0 = update_precision_hyperparameter(state.dX_src.detach(), nu, V_inv_common, mode="sample")
                        state.event_precision_matrix = new_P0
                    # Lightweight visibility: report implied stds periodically (or every update if update_every is large).
                    try:
                        # Prefer the existing diagnostics cadence knob if present.
                        # Default to something visible (10) so users can confirm updates are happening.
                        log_every = int(state.params.get("display_precond_every", state.params.get("hierarchical_log_every_epochs", 10)))
                        log_every = max(1, log_every)
                        if (epoch_index % log_every) == 0:
                            P0_all = state.event_precision_matrix
                            P0_mean = P0_all.mean(dim=0) if isinstance(P0_all, torch.Tensor) and P0_all.ndim == 3 else P0_all
                            Cov = torch.linalg.inv(P0_mean)
                            stds = torch.sqrt(torch.diag(Cov)).clamp_min(torch.tensor(0.0, device=Cov.device, dtype=Cov.dtype))
                            corr_zt = float(Cov[2, 3].item()) / (float(stds[2].item()) * float(stds[3].item()) + 1e-12)
                            state.params["_hierarchical_event_std_est"] = [float(stds[0].item()), float(stds[1].item()), float(stds[2].item()), float(stds[3].item())]
                            info(
                                f"Hierarchical event prior updated: std≈[{stds[0]:.3g},{stds[1]:.3g},{stds[2]:.3g},{stds[3]:.3g}] corr_zt≈{corr_zt:.3g}",
                                section="PRIORS",
                            )
                    except Exception:
                        pass
                except Exception:
                    # Avoid silent failures; warn once per epoch.
                    warn("Hierarchical event prior update failed; leaving P0 unchanged for this epoch.", section="PRIORS")

        # Prefer weighted mean over edges; fallback to unweighted mean if something went wrong.
        if total_loss_weighted_denom > 0:
            total_loss_mean = total_loss_weighted_sum / float(total_loss_weighted_denom)
        else:
            total_loss_mean = (sum(total_loss_vals) / len(total_loss_vals)) if total_loss_vals else 0.0
        epoch_time = time.time() - epoch_start_time
        
        # Phase 1 MAP CSV
        if write_map_csv:
            _maybe_write_map_csv(state, epoch_index)
            
        # Compute MADs occasionally
        # Logic differs slightly per phase in original code, but can be standardized
        # Phase 1: % 10
        # Phase 2/3/4: customized interval
        # We can pass a flag or check params here.
        # Simplifying: check standardized keys or default
        phase_name = "phase1" if isinstance(optimizer, torch.optim.Adam) else "phase4" # simplified
        # Actually rely on param keys directly
        # ... (omitted for brevity, logic handled in return values)

        # Update stats tensor
        st = state.stats_tensor
        st[0] = state.dX_src[:, 0].mean()
        st[1] = state.dX_src[:, 1].mean()
        st[2] = state.dX_src[:, 2].mean()
        st[3] = torch.abs(state.dX_src[:, 0]).median()
        st[4] = torch.abs(state.dX_src[:, 1]).median()
        st[5] = torch.abs(state.dX_src[:, 2]).median()
        st[6] = torch.sqrt(state.dX_src[:, 0]**2 + state.dX_src[:, 1]**2 + state.dX_src[:, 2]**2).max()
        st[7] = torch.quantile(torch.sqrt(state.dX_src[:, 0]**2 + state.dX_src[:, 1]**2 + state.dX_src[:, 2]**2), 0.90).item()
        
        stats_cpu = st.detach().cpu().numpy()

        # Time component (ΔT / origin-time correction) stats are not stored in stats_tensor
        # to preserve checkpoint/backward compatibility. Compute separately for logging.
        #
        # IMPORTANT interpretability note:
        # In a pure differential-time likelihood, adding a constant to *all* origin times does not
        # change dt_pred (only differences matter). That can make raw dt_* metrics drift slowly.
        # To diagnose meaningful relative-time mixing, also compute "centered" stats where we
        # subtract the per-component (cluster) mean if clusters exist, else the global mean.
        try:
            dT = state.dX_src[:, 3].to(torch.float32)
            dt_mean = float(dT.mean().detach().cpu().item())
            dt_med_abs = float(torch.abs(dT).median().detach().cpu().item())
            dt_max_abs = float(torch.abs(dT).max().detach().cpu().item())
            dt_p90_abs = float(torch.quantile(torch.abs(dT), 0.90).detach().cpu().item())
            # Centered ΔT (remove per-connected-component mean when available)
            try:
                dT0 = None
                cid = getattr(state, "cluster_ids", None)
                cc = getattr(state, "cluster_counts", None)
                if isinstance(cid, torch.Tensor) and isinstance(cc, torch.Tensor) and cid.numel() == dT.numel():
                    K = int(cc.numel())
                    if K > 0:
                        sums = torch.zeros((K,), device=dT.device, dtype=dT.dtype)
                        sums.index_add_(0, cid.to(torch.int64), dT)
                        denom = cc.to(device=dT.device, dtype=dT.dtype).view(-1).clamp_min(1.0)
                        means = sums / denom
                        dT0 = dT - means.index_select(0, cid.to(torch.int64))
                if dT0 is None:
                    dT0 = dT - dT.mean()
                dt_centered_std = float(dT0.std(unbiased=False).detach().cpu().item())
                dt_centered_med_abs = float(torch.abs(dT0).median().detach().cpu().item())
                dt_centered_p90_abs = float(torch.quantile(torch.abs(dT0), 0.90).detach().cpu().item())
            except Exception:
                dt_centered_std = float("nan")
                dt_centered_med_abs = float("nan")
                dt_centered_p90_abs = float("nan")
        except Exception:
            dt_mean = float("nan")
            dt_med_abs = float("nan")
            dt_max_abs = float("nan")
            dt_p90_abs = float("nan")
            dt_centered_std = float("nan")
            dt_centered_med_abs = float("nan")
            dt_centered_p90_abs = float("nan")
        
        metrics = {
            "loss": total_loss_mean,
            "epoch_time": epoch_time,
            "permute_time": float(permute_time_s),
            "dx_mean": stats_cpu[0],
            "dy_mean": stats_cpu[1],
            "dz_mean": stats_cpu[2],
            "dx_med_abs": stats_cpu[3],
            "dy_med_abs": stats_cpu[4],
            "dz_med_abs": stats_cpu[5],
            "dr_max": stats_cpu[6],
            "dr_90": stats_cpu[7],
            # ΔT (seconds)
            "dt_mean": dt_mean,
            "dt_med_abs": dt_med_abs,
            "dt_max_abs": dt_max_abs,
            "dt_p90_abs": dt_p90_abs,
            "dt_centered_std": dt_centered_std,
            "dt_centered_med_abs": dt_centered_med_abs,
            "dt_centered_p90_abs": dt_centered_p90_abs,
        }
        # Optional: timing + workload summary for collapsed shared_event_re likelihood (per epoch).
        try:
            diag = _get_diagnostics_cfg(state.params)
            if isinstance(diag, dict) and bool(diag.get("profile_shared_event_re", False)) and bool(state.params.get("_shared_event_re_enabled", False)):
                c = int(state.params.get("_se_re_time_ms_count", 0) or 0)
                s = float(state.params.get("_se_re_time_ms_sum", 0.0) or 0.0)
                if c > 0:
                    metrics["shared_event_re/time_ms_mean"] = float(s / float(c))
                    metrics["shared_event_re/time_ms_sum"] = float(s)
                    metrics["shared_event_re/time_batches"] = float(c)
                    # Workload (means over batches + max over epoch)
                    metrics["shared_event_re/groups_mean"] = float(int(state.params.get("_se_re_groups_sum", 0) or 0) / float(c))
                    metrics["shared_event_re/groups_pcg_mean"] = float(int(state.params.get("_se_re_groups_pcg_sum", 0) or 0) / float(c))
                    metrics["shared_event_re/groups_fallback_diag_mean"] = float(int(state.params.get("_se_re_groups_fallback_sum", 0) or 0) / float(c))
                    metrics["shared_event_re/max_rows_max"] = float(int(state.params.get("_se_re_max_rows_max", 0) or 0))
                    metrics["shared_event_re/max_nodes_max"] = float(int(state.params.get("_se_re_max_nodes_max", 0) or 0))
                if bool(state.params.get("_shared_event_re_station_phase_enabled", False)):
                    metrics["shared_event_re/station_phase_groups"] = float(int(state.params.get("_shared_event_re_station_phase_last_groups", 0) or 0))
                    metrics["shared_event_re/station_phase_quad"] = float(state.params.get("_shared_event_re_station_phase_last_quad", 0.0) or 0.0)
        except Exception:
            pass
        return metrics

