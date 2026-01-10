from typing import Dict, List
import time
import torch
import math
import numpy as np
import torch.distributed as dist
from spider.core.hierarchy import update_precision_hyperparameter
from spider.utils.console import info, warn
from spider.core.state import LocateState, _current_noise_scales, _clamp_dX_inplace, _apply_shared_event_latent_constraints_inplace
from spider.core.batching import _ensure_owner_buckets, _iter_event_batches
from spider.core.modeling import (
    posterior_loss,
    compute_likelihood_loss,
    compute_prior_loss,
    compute_residuals,
    write_output,
)
from spider.utils.wandb_gates import want_wandb_group as _want_wandb_group


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
    for g in optimizer.param_groups:  # type: ignore[attr-defined]
        for p in g.get("params", []):
            if p is None:
                continue
            gg = getattr(p, "grad", None)
            if gg is None:
                continue
            if not isinstance(gg, torch.Tensor) or gg.numel() <= 0:
                continue
            grads.append(gg)
    if not grads:
        return

    # Coalesce into a single buffer to reduce per-parameter allreduce overhead.
    # This matters a lot on systems without fast GPU interconnect, where many small allreduces
    # can dominate the step time.
    dev0 = grads[0].device
    dt0 = grads[0].dtype
    same = True
    total = 0
    for gg in grads:
        total += int(gg.numel())
        if gg.device != dev0 or gg.dtype != dt0:
            same = False
            break
    if (not same) or total <= 0:
        # Fallback: allreduce each grad separately.
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
            torch.cuda.manual_seed_all(s)
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
    print("SVRG: Updating full gradient snapshot...")
    
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
    
    print(f"SVRG: Snapshot updated. Grad norm: {state.svrg_grad_full.norm().item():.3e}")

def _set_backend_noise(optimizer: torch.optim.Optimizer, *, enabled: bool, scale: float) -> None:
    """Set noise flags consistently for any sampler backend."""
    if not hasattr(optimizer, "param_groups"):
        return
    # AdaptiveSGHMC uses a BOHAMIANN-style update which *can* inject noise, but in SPIDER we
    # still want consistent Phase semantics:
    # - Phase 2: noise_scale_factor=0 => effectively deterministic drift (no injected noise)
    # - Phase 3: ramp noise_scale_factor up
    # - Phase 4: full sampling noise
    #
    # We detect AdaptiveSGHMC via the param-group preconditioner tag set by the backend factory.
    force_on = False
    try:
        if len(optimizer.param_groups) > 0:  # type: ignore[attr-defined]
            p0 = optimizer.param_groups[0]  # type: ignore[index]
            if str(p0.get("preconditioner", "")).strip().lower() == "adaptive_sghmc":
                force_on = True
    except Exception:
        force_on = False

    for g in optimizer.param_groups:  # type: ignore[attr-defined]
        if force_on:
            g["add_noise"] = True
            # For AdaptiveSGHMC, keep `add_noise=True` but allow `noise_scale` to be 0.0 to
            # match Phase 2 (deterministic) behavior.
            try:
                s = float(scale)
                if not math.isfinite(s) or s < 0.0:
                    s = 0.0
            except Exception:
                s = 0.0
            g["noise_scale"] = s
        else:
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
    state.params["_prior_centroid_runtime_enable"] = bool(state.params.get("prior_centroid_enable", True))
    state.params["_prior_noise_runtime_enable"] = bool(state.params.get("prior_noise_enable", True))

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
        if epoch_index == 0:
            try:
                if (not ddp_enabled) or ddp_is_main:
                    print(
                        f"[spider][INFO][BATCH] standard.shuffle={int(shuffle)} "
                        f"permute_s={permute_time_s:.3f} "
                        f"batch_size={int(state.params.get('batch_size_sgld', 0) or 0)} "
                        f"N={int(getattr(state, 'N', 0) or 0)}",
                        flush=True,
                    )
            except Exception:
                pass
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
    want_lat_diag = bool(state.params.get("_shared_event_latent_enabled", False)) and _want_wandb_group(state.params, "shared_event_latent")
    b_delta_sum = None
    b_delta_sumsq = None
    b_delta_maxabs = None
    b_delta_count = None
    # Also track RMS of *event-level* latent endpoints b(e1), b(e2) used by DD rows.
    # This is more interpretable than RMS of the underlying parameter tensor in inducing_gp mode
    # (where `shared_event_latent_b` stores inducing coefficients rather than event values).
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
        batch_iter = range(0, state.N // batch_size + 1)
        use_buckets = False
        # Expose standard batching parameters for lower-level caching.
        state.params["_runtime_batch_size"] = int(batch_size)
        state.params["_runtime_batching_mode"] = "standard"

    # Noise setup (generic sampler backend)
    # If noise_scale_factor > 0, enable noise; else disable (e.g., Phase 2). Phase 3 ramps it.
    _set_backend_noise(optimizer, enabled=(noise_scale_factor > 0.0), scale=float(noise_scale_factor))

    # For AdaptiveSGHMC we want burn-in driven by *epochs* (Phase 3) rather than optimizer step counts.
    # Expose this via `param_group['is_burnin']`, which the optimizer reads.
    try:
        if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
            p0 = optimizer.param_groups[0]
            if str(p0.get("preconditioner", "")).strip().lower() == "adaptive_sghmc":
                for g in optimizer.param_groups:
                    g["is_burnin"] = bool(int(phase_id) == 3)
    except Exception:
        pass

    # SVRG Snapshot Update (only if SVRG enabled and we are sampling)
    svrg_enabled = state.svrg_enable and is_sampling
    if ddp_enabled and svrg_enabled:
        raise ValueError("SVRG is not supported in torchrun/DDP mode (it requires extra full-gradient bookkeeping). Disable inference.diagnostics.svrg.enabled.")
    if svrg_enabled:
        # Check if we need to update snapshot (e.g. every epoch)
        # For simplicity, update at start of every epoch for now if enabled
        _update_svrg_snapshot(state, batch_size=batch_size, optimizer=optimizer)

    # Optional profiling accumulators (only active when inference.diagnostics.profile_shared_event_latent=true)
    try:
        diag0 = _get_diagnostics_cfg(state.params)
        if isinstance(diag0, dict) and bool(diag0.get("profile_shared_event_latent", False)):
            state.params["_se_lat_time_ms_sum"] = 0.0
            state.params["_se_lat_time_ms_count"] = 0
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
        # Optional profiling: collapsed slowness_re likelihood (inducing GP) cost + breakdown.
        # Expected config location: inference.diagnostics.profile_slowness_re (bool).
        if isinstance(diag0, dict) and bool(diag0.get("profile_slowness_re", False)):
            # Runtime gate read by modeling.py
            state.params["_profile_slowness_re"] = True
            # Profiling knobs (optional)
            try:
                state.params["_profile_slowness_re_max_groups"] = int(diag0.get("profile_slowness_re_max_groups", 2))
            except Exception:
                state.params["_profile_slowness_re_max_groups"] = 2
            try:
                state.params["_profile_slowness_re_use_cuda_events"] = bool(diag0.get("profile_slowness_re_use_cuda_events", True))
            except Exception:
                state.params["_profile_slowness_re_use_cuda_events"] = True
            # Aggregates (per-epoch)
            state.params["_sl_re_time_ms_sum"] = 0.0
            state.params["_sl_re_time_ms_count"] = 0
            state.params["_sl_re_grouping_ms_sum"] = 0.0
            state.params["_sl_re_kernel_ms_sum"] = 0.0
            state.params["_sl_re_assemble_ms_sum"] = 0.0
            state.params["_sl_re_solve_ms_sum"] = 0.0
            state.params["_sl_re_profiled_groups_sum"] = 0
            state.params["_sl_re_pcg_iters_sum"] = 0
            # Workload (per batch)
            state.params["_sl_re_groups_sum"] = 0
            state.params["_sl_re_groups_woodbury_sum"] = 0
            state.params["_sl_re_groups_fallback_sum"] = 0
            state.params["_sl_re_max_rows_max"] = 0
            state.params["_sl_re_max_nodes_max"] = 0
            # Also compute a static upper bound for M from stored inducing artifacts (cheap, one-time).
            M_static = 0
            try:
                K_full = state.params.get("_slowness_re_inducing_K_full", None)
                if isinstance(K_full, torch.Tensor) and K_full.ndim == 2:
                    M_static = int(K_full.shape[0])
                else:
                    K_blocks = state.params.get("_slowness_re_inducing_K_blocks", None)
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

    # Optional: batch-level progress heartbeat (especially helpful in torchrun/DDP where only rank0 prints).
    # This prevents "looks hung" confusion on very large datasets.
    batch_progress_every = 0
    try:
        if bool(state.params.get("verbose", False)):
            diag = _get_diagnostics_cfg(state.params)
            if isinstance(diag, dict):
                # Default: in DDP, print every 10 batches when verbose=true.
                if ddp_enabled:
                    batch_progress_every = int(diag.get("ddp_batch_progress_every", 10))
                else:
                    batch_progress_every = int(diag.get("batch_progress_every", 0))
    except Exception:
        batch_progress_every = 0
    if batch_progress_every < 0:
        batch_progress_every = 0
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
                # This enables station-dependent nuisance models (e.g. shared_event_latent) without relying
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
                # Deterministic contiguous shard per rank.
                s = int((global_bsz * ddp_rank) // ddp_world_size)
                e = int((global_bsz * (ddp_rank + 1)) // ddp_world_size)
                II_b = II_b[s:e, :]
                YY_b = YY_b[s:e]
                sta_rt = state.params.get("_runtime_bucket_station_index", None)
                if isinstance(sta_rt, torch.Tensor) and int(sta_rt.shape[0]) == int(global_bsz):
                    state.params["_runtime_bucket_station_index"] = sta_rt[s:e]
            except Exception:
                # Leave batch unsharded if something goes wrong; better than crashing mid-run.
                pass

        # Optional progress heartbeat at the start of the batch (rank0 only under torchrun).
        try:
            if batch_progress_every > 0 and ((bi % batch_progress_every) == 0) and ((not ddp_enabled) or ddp_is_main):
                done = int(bi)
                tot = int(total_batches) if total_batches is not None else -1
                elapsed = float(time.time() - epoch_start_time)
                if tot > 0:
                    rate = float(done) / max(1e-9, elapsed)
                    eta = float(tot - done) / max(1e-9, rate)
                    print(f"[spider][INFO][BATCH] epoch={epoch_index} batch={done}/{tot} elapsed_s={elapsed:.1f} eta_s={eta:.1f}", flush=True)
                else:
                    print(f"[spider][INFO][BATCH] epoch={epoch_index} batch={done} elapsed_s={elapsed:.1f}", flush=True)
        except Exception:
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
        sigma_extra_var = None
        prof_se_lat = False
        se_lat_t0 = None
        try:
            # Optional runtime profiling: time the shared_event_latent nuisance term reconstruction.
            # Expected config location: inference.diagnostics.profile_shared_event_latent (bool).
            diag = _get_diagnostics_cfg(state.params)
            prof_se_lat = bool(diag.get("profile_shared_event_latent", False)) if isinstance(diag, dict) else False
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
            if bool(state.params.get("_shared_event_latent_enabled", False)):
                mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                b_lat = getattr(state, "shared_event_latent_b", None)
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
                            if bool(state.params.get("_shared_event_latent_inducing_fitc_enable", False)):
                                resid = getattr(state, "shared_event_latent_inducing_fitc_resid", None)
                                if isinstance(resid, torch.Tensor) and resid.ndim == 1:
                                    r1 = resid.index_select(0, e1).to(torch.float32)
                                    r2 = resid.index_select(0, e2).to(torch.float32)
                                    rsum = (r1 + r2).clamp_min(0.0)
                                    tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                                    tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                                    extra = torch.where(is_s, rsum * (tau_s * tau_s), rsum * (tau_p * tau_p)).to(torch.float32)
                                    sigma_extra_var = extra if sigma_extra_var is None else (sigma_extra_var + extra)
                        except Exception:
                            pass
                        nei_idx = getattr(state, "shared_event_latent_inducing_neighbor_idx", None)
                        nei_k = getattr(state, "shared_event_latent_inducing_neighbor_k", None)
                        if isinstance(nei_idx, torch.Tensor) and isinstance(nei_k, torch.Tensor) and b_lat.ndim == 3 and int(b_lat.shape[2]) == 2:
                            # Two modes:
                            # - per-station coefficients (b_lat shape (n_stations, M, 2))
                            # - station-basis coefficients (b_lat shape (R, M, 2) + W_sta shape (n_stations, R))
                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
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
                    elif mode == "slowness_inducing_gp":
                        # Station×phase slowness-vector inducing GP (pair-shared approximation):
                        #
                        # For each station×phase, we model a 3D slowness perturbation vector field u(x) (units s/km)
                        # with a Matérn(3/2) kernel in XYZ (km). We store inducing coefficients c such that:
                        #   u(e) ≈ Σ_j k(||x_e - x_u||) * c_u_j
                        # where c has prior c ~ N(0, K_UU^{-1}) and K_UU is the Matérn kernel matrix.
                        #
                        # Under the "pair-shared" approximation for DD pairs separated by <=~1 km, the DD correction is:
                        #   δt_ij ≈ 0.5*(u(e1)+u(e2)) · (x1 - x2)
                        #
                        # IMPORTANT: we use current locations (X_src + ΔX_src) so gradients flow through geometry.
                        try:
                            nei_idx = getattr(state, "shared_event_latent_inducing_neighbor_idx", None)
                            ind_ev = getattr(state, "shared_event_latent_inducing_event_idx", None)
                            U_xyz = getattr(state, "shared_event_latent_inducing_xyz_km", None)
                            fixed_xyz = bool(state.params.get("_shared_event_latent_inducing_fixed_xyz", False))
                            if not isinstance(nei_idx, torch.Tensor):
                                raise RuntimeError("missing inducing_gp neighbor tensors")
                            if fixed_xyz:
                                if not (isinstance(U_xyz, torch.Tensor) and U_xyz.ndim == 2 and int(U_xyz.shape[1]) >= 3):
                                    raise RuntimeError("slowness_inducing_gp fixed_xyz requires shared_event_latent_inducing_xyz_km")
                            else:
                                if not isinstance(ind_ev, torch.Tensor):
                                    raise RuntimeError("missing inducing_gp inducing_event_idx tensor")
                            if not (isinstance(b_lat, torch.Tensor) and b_lat.ndim == 4 and int(b_lat.shape[2]) == 2 and int(b_lat.shape[3]) == 3):
                                raise RuntimeError("invalid slowness_inducing_gp shared_event_latent_b shape")

                            # Current event coords (km) for endpoints and inducing points (subset of events).
                            Xcur = (state.X_src + state.dX_src)[:, :3].to(torch.float32)
                            x1 = Xcur.index_select(0, e1)  # (B,3)
                            x2 = Xcur.index_select(0, e2)  # (B,3)
                            dx = (x1 - x2).to(torch.float32)  # (B,3)
                            dx_norm2 = (dx * dx).sum(dim=1).to(torch.float32)  # (B,)

                            # Neighbor lists per endpoint
                            idx1 = nei_idx.index_select(0, e1)  # (B,m)
                            idx2 = nei_idx.index_select(0, e2)  # (B,m)
                            m1 = (idx1 >= 0)
                            m2 = (idx2 >= 0)
                            idx1c = idx1.clamp_min(0)
                            idx2c = idx2.clamp_min(0)

                            # Kernel weights: Matérn ν=3/2 in XYZ at current positions.
                            ell = float(state.params.get("_shared_event_latent_ell_km", 0.0))
                            if not (ell > 0.0):
                                raise RuntimeError("slowness_inducing_gp requires ell_km > 0")
                            a = float(np.sqrt(3.0) / float(ell))

                            def _weights(endpoint_x: torch.Tensor, idxc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                                # endpoint_x: (B,3), idxc: (B,m) global inducing idx, mask: (B,m)
                                if fixed_xyz and isinstance(U_xyz, torch.Tensor):
                                    u_xyz = U_xyz.index_select(0, idxc.reshape(-1)).reshape(idxc.shape[0], idxc.shape[1], 3)  # (B,m,3)
                                else:
                                    u_ev = ind_ev.index_select(0, idxc.reshape(-1)).reshape(idxc.shape)  # (B,m) event ids
                                    u_xyz = Xcur.index_select(0, u_ev.reshape(-1)).reshape(idxc.shape[0], idxc.shape[1], 3)  # (B,m,3)
                                d = torch.linalg.norm(endpoint_x.unsqueeze(1) - u_xyz, dim=2).to(torch.float32)  # (B,m)
                                x = (a * d).to(torch.float32)
                                w = (1.0 + x) * torch.exp(-x)
                                return w * mask.to(torch.float32)

                            w1 = _weights(x1, idx1c, m1)  # (B,m)
                            w2 = _weights(x2, idx2c, m2)  # (B,m)

                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                            n_stations_rt = int(getattr(state, "n_stations", 0))
                            is_per_station = (n_stations_rt > 0) and (int(b_lat.shape[0]) == int(n_stations_rt))
                            is_basis = (
                                isinstance(W_sta, torch.Tensor)
                                and W_sta.ndim == 2
                                and int(W_sta.shape[0]) == int(n_stations_rt)
                                and int(b_lat.shape[0]) == int(W_sta.shape[1])
                            )
                            if (not is_per_station) and (not is_basis):
                                raise RuntimeError("slowness_inducing_gp: expected per-station or station-basis coefficients")

                            # Gather coefficients at neighbor indices, returning (B,m,2,3) for each endpoint.
                            if is_per_station:
                                # IMPORTANT: avoid advanced indexing b_lat[sta, idx] here.
                                # On large batches this can be extremely slow due to non-coalesced gather kernels.
                                # Instead, flatten (station, inducing_idx) into a single axis and use index_select.
                                try:
                                    S = int(b_lat.shape[0])
                                    M_total = int(b_lat.shape[1])
                                    flat = b_lat.reshape(S * M_total, 2, 3).to(torch.float32)  # (S*M,2,3)
                                    sta0 = sta_bi.to(torch.int64).clamp_min(0).clamp_max(max(S - 1, 0))
                                    lin1 = (sta0.unsqueeze(1) * M_total + idx1c.to(torch.int64)).reshape(-1)
                                    lin2 = (sta0.unsqueeze(1) * M_total + idx2c.to(torch.int64)).reshape(-1)
                                    C1 = flat.index_select(0, lin1).reshape(idx1c.shape[0], idx1c.shape[1], 2, 3)  # (B,m,2,3)
                                    C2 = flat.index_select(0, lin2).reshape(idx2c.shape[0], idx2c.shape[1], 2, 3)  # (B,m,2,3)
                                except Exception:
                                    # Fallback (should be rare): keep old behavior.
                                    C1 = b_lat[sta_bi.unsqueeze(1), idx1c].to(torch.float32)  # (B,m,2,3)
                                    C2 = b_lat[sta_bi.unsqueeze(1), idx2c].to(torch.float32)  # (B,m,2,3)
                            else:
                                # Basis: reconstruct station-specific coefficients by weighting basis ranks.
                                # A6: (R,M,6) where last dim packs phase×xyz.
                                A = b_lat.to(torch.float32)
                                R = int(A.shape[0]); M = int(A.shape[1])
                                A6 = A.reshape(R, M, 6)
                                Wb = W_sta.index_select(0, sta_bi).to(torch.float32)  # (B,R)
                                Wr = Wb.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # (R,B,1,1)

                                def _gather_station_coeffs(idxc: torch.Tensor) -> torch.Tensor:
                                    # idxc: (B,m) -> (B,m,2,3)
                                    B0, m0 = int(idxc.shape[0]), int(idxc.shape[1])
                                    Aexp = A6.unsqueeze(1).expand(R, B0, M, 6)  # (R,B,M,6)
                                    idxe = idxc.unsqueeze(0).unsqueeze(-1).expand(R, B0, m0, 6)  # (R,B,m,6)
                                    g = torch.gather(Aexp, 2, idxe)  # (R,B,m,6)
                                    g = g.reshape(R, B0, m0, 2, 3)
                                    return (g * Wr.unsqueeze(-1)).sum(dim=0)  # (B,m,2,3)

                                C1 = _gather_station_coeffs(idx1c)
                                C2 = _gather_station_coeffs(idx2c)

                            # Interpolate endpoint u vectors (B,2,3)
                            u1 = (C1 * w1.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (B,2,3)
                            u2 = (C2 * w2.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (B,2,3)
                            u_avg = 0.5 * (u1 + u2)  # (B,2,3)

                            # Convert to time using configured units for tau:
                            # - abs: u_avg is interpreted as slowness vector (s/km) -> seconds = u·dx
                            # - vel_frac: u_avg is interpreted as dimensionless ε (≈ δv/v) -> seconds ≈ (ε·dx) / v(z)
                            units = str(state.params.get("_shared_event_latent_slowness_tau_units", "abs")).strip().lower()
                            if units == "vel_frac":
                                # Lookup vP(z), vS(z) via *GPU* linear interpolation over depth centers.
                                #
                                # IMPORTANT: avoid GPU->CPU numpy round-trips here; with large batches this
                                # can dominate runtime and cause DDP/NCCL timeouts (one rank finishes much later).
                                #
                                # We cache torch tensors in `state.params` so this setup happens once per process.
                                zavg = (0.5 * (x1[:, 2] + x2[:, 2])).to(torch.float32)  # (B,)

                                def _get_v1d_tensors() -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
                                    # Cached tensors
                                    zc_t = state.params.get("_runtime_eikonet_v1d_z_cent_km_t", None)
                                    vp_t = state.params.get("_runtime_eikonet_v1d_vp_km_s_t", None)
                                    vs_t = state.params.get("_runtime_eikonet_v1d_vs_km_s_t", None)
                                    dev0 = dx.device
                                    try:
                                        if (
                                            isinstance(zc_t, torch.Tensor)
                                            and isinstance(vp_t, torch.Tensor)
                                            and isinstance(vs_t, torch.Tensor)
                                            and zc_t.ndim == 1
                                            and vp_t.ndim == 1
                                            and vs_t.ndim == 1
                                            and int(zc_t.numel()) == int(vp_t.numel()) == int(vs_t.numel())
                                            and int(zc_t.numel()) >= 2
                                            and zc_t.device == dev0
                                            and vp_t.device == dev0
                                            and vs_t.device == dev0
                                        ):
                                            return zc_t, vp_t, vs_t
                                    except Exception:
                                        pass

                                    # Build from config arrays (usually stored by EikoNet loading)
                                    try:
                                        z_cent = np.asarray(state.params.get("_eikonet_v1d_depth_centers_km", []), dtype=np.float32).reshape(-1)
                                        vp = np.asarray(state.params.get("_eikonet_v1d_vp_km_s", []), dtype=np.float32).reshape(-1)
                                        vs = np.asarray(state.params.get("_eikonet_v1d_vs_km_s", []), dtype=np.float32).reshape(-1)
                                    except Exception:
                                        z_cent = np.zeros((0,), dtype=np.float32)
                                        vp = np.zeros((0,), dtype=np.float32)
                                        vs = np.zeros((0,), dtype=np.float32)
                                    if z_cent.size < 2 or vp.size != z_cent.size or vs.size != z_cent.size:
                                        return None, None, None

                                    zc = torch.from_numpy(z_cent).to(device=dev0, dtype=torch.float32)
                                    vp0 = torch.from_numpy(vp).to(device=dev0, dtype=torch.float32)
                                    vs0 = torch.from_numpy(vs).to(device=dev0, dtype=torch.float32)
                                    # Enforce sorted z-centers for bucketize/searchsorted
                                    try:
                                        if not bool(torch.all(zc[1:] >= zc[:-1]).item()):
                                            perm = torch.argsort(zc)
                                            zc = zc.index_select(0, perm)
                                            vp0 = vp0.index_select(0, perm)
                                            vs0 = vs0.index_select(0, perm)
                                    except Exception:
                                        pass
                                    # Replace any invalid values with reasonable fallbacks (keeps interpolation stable)
                                    vp0 = torch.where(torch.isfinite(vp0) & (vp0 > 0.0), vp0, torch.full_like(vp0, 6.0))
                                    vs0 = torch.where(torch.isfinite(vs0) & (vs0 > 0.0), vs0, torch.full_like(vs0, 3.5))

                                    # Cache for future batches
                                    state.params["_runtime_eikonet_v1d_z_cent_km_t"] = zc
                                    state.params["_runtime_eikonet_v1d_vp_km_s_t"] = vp0
                                    state.params["_runtime_eikonet_v1d_vs_km_s_t"] = vs0
                                    return zc, vp0, vs0

                                def _interp_torch(zq: torch.Tensor, zc: torch.Tensor, vv: torch.Tensor, v_fallback: float) -> torch.Tensor:
                                    # zq: (B,), zc/vv: (K,) sorted ascending. Clamp to endpoints outside range.
                                    K = int(zc.numel())
                                    if K < 2:
                                        return torch.full_like(zq, float(v_fallback), dtype=torch.float32)
                                    idx = torch.bucketize(zq, zc)  # 0..K
                                    idx0 = (idx - 1).clamp(min=0, max=K - 2)
                                    idx1 = idx0 + 1
                                    z0 = zc.index_select(0, idx0)
                                    z1 = zc.index_select(0, idx1)
                                    v0 = vv.index_select(0, idx0)
                                    v1 = vv.index_select(0, idx1)
                                    denom = (z1 - z0).clamp_min(1e-12)
                                    w = ((zq - z0) / denom).clamp(0.0, 1.0)
                                    out = v0 + w * (v1 - v0)
                                    out = torch.where(torch.isfinite(out) & (out > 0.0), out, torch.full_like(out, float(v_fallback)))
                                    return out.to(torch.float32)

                                zc, vp0, vs0 = _get_v1d_tensors()
                                if zc is None or vp0 is None or vs0 is None:
                                    vP = torch.full_like(zavg, 6.0, dtype=torch.float32)
                                    vS = torch.full_like(zavg, 3.5, dtype=torch.float32)
                                else:
                                    vP = _interp_torch(zavg, zc, vp0, 6.0)
                                    vS = _interp_torch(zavg, zc, vs0, 3.5)
                                vP = vP.clamp_min(1e-6)
                                vS = vS.clamp_min(1e-6)

                                # Optional Stage-5 FITC diagonal correction (variance inflation).
                                # Var(0.5(u1+u2)·dx / v)^approx ≈ 0.25*(r1+r2) * ||dx||^2 * tau^2 / v^2
                                try:
                                    if bool(state.params.get("_shared_event_latent_inducing_fitc_enable", False)):
                                        resid = getattr(state, "shared_event_latent_inducing_fitc_resid", None)
                                        if isinstance(resid, torch.Tensor) and resid.ndim == 1:
                                            r1 = resid.index_select(0, e1).to(torch.float32)
                                            r2 = resid.index_select(0, e2).to(torch.float32)
                                            rsum = (0.25 * (r1 + r2)).clamp_min(0.0)
                                            tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                                            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                                            extraP = rsum * dx_norm2 * (tau_p * tau_p) / (vP * vP)
                                            extraS = rsum * dx_norm2 * (tau_s * tau_s) / (vS * vS)
                                            extra = torch.where(is_s, extraS, extraP).to(torch.float32)
                                            sigma_extra_var = extra if sigma_extra_var is None else (sigma_extra_var + extra)
                                except Exception:
                                    pass

                                dP = (u_avg[:, 0, :] * dx).sum(dim=1) / vP
                                dS = (u_avg[:, 1, :] * dx).sum(dim=1) / vS
                                delta_b = torch.where(is_s, dS, dP)
                            else:
                                # Dot with event separation vector: seconds = (s/km)·(km)
                                # Optional Stage-5 FITC diagonal correction (variance inflation).
                                # Var(0.5(u1+u2)·dx)^approx ≈ 0.25*(r1+r2) * ||dx||^2 * tau^2
                                try:
                                    if bool(state.params.get("_shared_event_latent_inducing_fitc_enable", False)):
                                        resid = getattr(state, "shared_event_latent_inducing_fitc_resid", None)
                                        if isinstance(resid, torch.Tensor) and resid.ndim == 1:
                                            r1 = resid.index_select(0, e1).to(torch.float32)
                                            r2 = resid.index_select(0, e2).to(torch.float32)
                                            rsum = (0.25 * (r1 + r2)).clamp_min(0.0)
                                            tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                                            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                                            extraP = rsum * dx_norm2 * (tau_p * tau_p)
                                            extraS = rsum * dx_norm2 * (tau_s * tau_s)
                                            extra = torch.where(is_s, extraS, extraP).to(torch.float32)
                                            sigma_extra_var = extra if sigma_extra_var is None else (sigma_extra_var + extra)
                                except Exception:
                                    pass
                                dP = (u_avg[:, 0, :] * dx).sum(dim=1)
                                dS = (u_avg[:, 1, :] * dx).sum(dim=1)
                                delta_b = torch.where(is_s, dS, dP)

                            # Diagnostics: log physical u-field magnitudes at endpoints (per-phase).
                            # This is much more interpretable than raw inducing coefficients, especially in the dual/K^{-1} parameterization.
                            try:
                                if want_lat_diag:
                                    u1P = u1[:, 0, :]
                                    u2P = u2[:, 0, :]
                                    u1S = u1[:, 1, :]
                                    u2S = u2[:, 1, :]
                                    # per-row endpoint norms
                                    n1P = torch.linalg.norm(u1P, dim=1).to(torch.float32)
                                    n2P = torch.linalg.norm(u2P, dim=1).to(torch.float32)
                                    n1S = torch.linalg.norm(u1S, dim=1).to(torch.float32)
                                    n2S = torch.linalg.norm(u2S, dim=1).to(torch.float32)
                                    is_p_f = (~is_s).to(torch.float32)
                                    is_s_f = is_s.to(torch.float32)

                                    if u_end_sumsq_p is None:
                                        u_end_sumsq_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_sumsq_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_count_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_count_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_maxnorm_p = torch.zeros((), device=delta_b.device, dtype=torch.float32)
                                        u_end_maxnorm_s = torch.zeros((), device=delta_b.device, dtype=torch.float32)

                                    u_end_sumsq_p = u_end_sumsq_p + ((n1P * n1P + n2P * n2P) * is_p_f).sum()
                                    u_end_sumsq_s = u_end_sumsq_s + ((n1S * n1S + n2S * n2S) * is_s_f).sum()
                                    u_end_count_p = u_end_count_p + (2.0 * is_p_f.sum())
                                    u_end_count_s = u_end_count_s + (2.0 * is_s_f.sum())
                                    # Max endpoint norm (helps detect a small number of pathological events)
                                    u_end_maxnorm_p = torch.maximum(u_end_maxnorm_p, torch.maximum(n1P, n2P).max())
                                    u_end_maxnorm_s = torch.maximum(u_end_maxnorm_s, torch.maximum(n1S, n2S).max())
                            except Exception:
                                pass
                        except Exception:
                            delta_b = None
                    else:
                        # Explicit event-latent parameterization (full / graph_gmrf):
                        # - per-station: b_lat shape (n_stations, n_events, 2)
                        # - station-basis: b_lat shape (R, n_events, 2) with W_sta (n_stations, R)
                        if b_lat.ndim == 3 and int(b_lat.shape[2]) == 2:
                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
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

        # Optional: heteroscedastic likelihood inflation (no latent term).
        # Adds extra per-observation variance in quadrature with the base phase noise:
        #   sigma_eff^2 = sigma_phase^2 + sigma_extra_var
        #
        # IMPORTANT: we detach the distance computation from gradients w.r.t. event locations to avoid
        # a perverse incentive to increase inter-event distances to reduce likelihood weight.
        try:
            if bool(state.params.get("_likelihood_sigma_inflation_enabled", False)) and isinstance(II_b, torch.Tensor) and II_b.numel() > 0:
                mode = str(state.params.get("_likelihood_sigma_inflation_mode", "vel_frac_linear_dd")).strip().lower()
                if mode == "vel_frac_linear_dd":
                    vel_frac = state.params.get("_likelihood_sigma_inflation_vel_frac", [0.0, 0.0])
                    v_km_s = state.params.get("_likelihood_sigma_inflation_v_km_s", [6.0, 3.5])
                    max_d_km = state.params.get("_likelihood_sigma_inflation_max_d_km", None)
                    use_3d = bool(state.params.get("_likelihood_sigma_inflation_use_3d", True))
                    try:
                        f_p = float(vel_frac[0]); f_s = float(vel_frac[1])
                    except Exception:
                        f_p = float(vel_frac); f_s = float(vel_frac)
                    try:
                        v_p = float(v_km_s[0]); v_s = float(v_km_s[1])
                    except Exception:
                        v_p = float(v_km_s); v_s = float(v_km_s)
                    if (f_p > 0.0 or f_s > 0.0) and (v_p > 0.0 and v_s > 0.0):
                        e1 = II_b[:, 0].to(torch.int64)
                        e2 = II_b[:, 1].to(torch.int64)
                        # Current event coordinates (km); detach to avoid gradients through sigma.
                        Xcur = (state.X_src[:, :3] + state.dX_src[:, :3].detach()).to(torch.float32)
                        x1 = Xcur.index_select(0, e1)
                        x2 = Xcur.index_select(0, e2)
                        dxyz = (x2 - x1)
                        if not use_3d:
                            dxyz = dxyz[:, :2]
                        d_km = torch.linalg.norm(dxyz, dim=1).clamp_min(0.0)
                        if max_d_km is not None:
                            try:
                                md = float(max_d_km)
                                if md > 0.0 and math.isfinite(md):
                                    d_km = d_km.clamp_max(md)
                            except Exception:
                                pass
                        ph = YY_b[:, 4]
                        is_s = (ph >= 0.5)
                        # sigma_struct(d) = (f / v) * d  [seconds]
                        slope_p = float(f_p) / float(v_p)
                        slope_s = float(f_s) / float(v_s)
                        sigma_struct = torch.where(is_s, d_km * float(slope_s), d_km * float(slope_p)).to(torch.float32)
                        extra_var = sigma_struct.square().clamp_min(0.0)
                        sigma_extra_var = extra_var if sigma_extra_var is None else (sigma_extra_var + extra_var)

                        # Accumulate per-epoch summary stats (cheap; means only).
                        try:
                            if "_sigma_infl_vel_sum_ms" not in state.params:
                                state.params["_sigma_infl_vel_sum_ms"] = 0.0
                                state.params["_sigma_infl_vel_count"] = 0
                                state.params["_sigma_infl_vel_sum_ms_P"] = 0.0
                                state.params["_sigma_infl_vel_count_P"] = 0
                                state.params["_sigma_infl_vel_sum_ms_S"] = 0.0
                                state.params["_sigma_infl_vel_count_S"] = 0
                                state.params["_sigma_infl_vel_d_km_sum"] = 0.0
                                state.params["_sigma_infl_vel_d_km_count"] = 0
                            ms = (1000.0 * sigma_struct.detach()).to(torch.float32)
                            state.params["_sigma_infl_vel_sum_ms"] = float(state.params.get("_sigma_infl_vel_sum_ms", 0.0) or 0.0) + float(ms.sum().item())
                            state.params["_sigma_infl_vel_count"] = int(state.params.get("_sigma_infl_vel_count", 0) or 0) + int(ms.numel())
                            msP = ms[~is_s]
                            msS = ms[is_s]
                            if int(msP.numel()) > 0:
                                state.params["_sigma_infl_vel_sum_ms_P"] = float(state.params.get("_sigma_infl_vel_sum_ms_P", 0.0) or 0.0) + float(msP.sum().item())
                                state.params["_sigma_infl_vel_count_P"] = int(state.params.get("_sigma_infl_vel_count_P", 0) or 0) + int(msP.numel())
                            if int(msS.numel()) > 0:
                                state.params["_sigma_infl_vel_sum_ms_S"] = float(state.params.get("_sigma_infl_vel_sum_ms_S", 0.0) or 0.0) + float(msS.sum().item())
                                state.params["_sigma_infl_vel_count_S"] = int(state.params.get("_sigma_infl_vel_count_S", 0) or 0) + int(msS.numel())
                            dk = d_km.detach()
                            state.params["_sigma_infl_vel_d_km_sum"] = float(state.params.get("_sigma_infl_vel_d_km_sum", 0.0) or 0.0) + float(dk.sum().item())
                            state.params["_sigma_infl_vel_d_km_count"] = int(state.params.get("_sigma_infl_vel_d_km_count", 0) or 0) + int(dk.numel())
                        except Exception:
                            pass
        except Exception:
            pass

        b_lat_for_prior = (getattr(state, "shared_event_latent_b", None) if bool(state.params.get("_shared_event_latent_enabled", False)) else None)
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
                    prof_sl_re = bool(diag.get("profile_slowness_re", False)) if isinstance(diag, dict) else False
                except Exception:
                    prof_sl_re = False
                if prof_se_re and bool(state.params.get("_shared_event_re_enabled", False)):
                    try:
                        import time as _time
                        se_re_t0 = _time.perf_counter()
                    except Exception:
                        se_re_t0 = None
                if prof_sl_re and bool(state.params.get("_slowness_re_enabled", False)):
                    try:
                        import time as _time
                        sl_re_t0 = _time.perf_counter()
                    except Exception:
                        sl_re_t0 = None
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
                    sigma_extra_var=sigma_extra_var,
                )
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
                if prof_sl_re and (sl_re_t0 is not None) and bool(state.params.get("_slowness_re_enabled", False)):
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - sl_re_t0)
                        state.params["_slowness_re_time_ms_sum"] = float(state.params.get("_slowness_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        state.params["_slowness_re_time_ms_count"] = int(state.params.get("_slowness_re_time_ms_count", 0) or 0) + 1
                        # Workload stats from modeling.py (set per-call)
                        g = int(state.params.get("_slowness_re_runtime_last_groups", 0) or 0)
                        g_w = int(state.params.get("_slowness_re_runtime_last_groups_woodbury", 0) or 0)
                        g_fb = int(state.params.get("_slowness_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_slowness_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_slowness_re_runtime_last_max_nodes", 0) or 0)
                        state.params["_slowness_re_groups_sum"] = int(state.params.get("_slowness_re_groups_sum", 0) or 0) + g
                        state.params["_slowness_re_groups_woodbury_sum"] = int(state.params.get("_slowness_re_groups_woodbury_sum", 0) or 0) + g_w
                        state.params["_slowness_re_groups_fallback_sum"] = int(state.params.get("_slowness_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_slowness_re_max_rows_max"] = max(int(state.params.get("_slowness_re_max_rows_max", 0) or 0), mr)
                        state.params["_slowness_re_max_nodes_max"] = max(int(state.params.get("_slowness_re_max_nodes_max", 0) or 0), mn)
                        # Also update new slowness_re profiler workload counters so W&B plots are consistent.
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
                shared_event_latent_b=b_lat_for_prior,
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
                    prof_sl_re = bool(diag.get("profile_slowness_re", False)) if isinstance(diag, dict) else False
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
                if prof_sl_re and bool(state.params.get("_slowness_re_enabled", False)):
                    try:
                        import time as _time
                        sl_re_t0 = _time.perf_counter()
                    except Exception:
                        sl_re_t0 = None

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
                    sigma_extra_var=sigma_extra_var,
                )

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
                if prof_sl_re and (sl_re_t0 is not None) and bool(state.params.get("_slowness_re_enabled", False)):
                    try:
                        import time as _time
                        if state.device.type == "cuda":
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                        dt_ms = 1000.0 * float(_time.perf_counter() - sl_re_t0)
                        state.params["_slowness_re_time_ms_sum"] = float(state.params.get("_slowness_re_time_ms_sum", 0.0) or 0.0) + float(dt_ms)
                        state.params["_slowness_re_time_ms_count"] = int(state.params.get("_slowness_re_time_ms_count", 0) or 0) + 1
                        g = int(state.params.get("_slowness_re_runtime_last_groups", 0) or 0)
                        g_w = int(state.params.get("_slowness_re_runtime_last_groups_woodbury", 0) or 0)
                        g_fb = int(state.params.get("_slowness_re_runtime_last_groups_fallback_diag", 0) or 0)
                        mr = int(state.params.get("_slowness_re_runtime_last_max_rows", 0) or 0)
                        mn = int(state.params.get("_slowness_re_runtime_last_max_nodes", 0) or 0)
                        state.params["_slowness_re_groups_sum"] = int(state.params.get("_slowness_re_groups_sum", 0) or 0) + g
                        state.params["_slowness_re_groups_woodbury_sum"] = int(state.params.get("_slowness_re_groups_woodbury_sum", 0) or 0) + g_w
                        state.params["_slowness_re_groups_fallback_sum"] = int(state.params.get("_slowness_re_groups_fallback_sum", 0) or 0) + g_fb
                        state.params["_slowness_re_max_rows_max"] = max(int(state.params.get("_slowness_re_max_rows_max", 0) or 0), mr)
                        state.params["_slowness_re_max_nodes_max"] = max(int(state.params.get("_slowness_re_max_nodes_max", 0) or 0), mn)
                        # Also update new slowness_re profiler workload counters so W&B plots are consistent.
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
                shared_event_latent_b=b_lat_for_prior,
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

        if not torch.isfinite(loss):
            # Warning/Skip
            continue
            
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
                sigma_extra_var=sigma_extra_var,
                cluster_ids=state.cluster_ids,
                cluster_counts=state.cluster_counts,
                event_precision_matrix=state.event_precision_matrix,
                shared_event_latent_b=(getattr(state, "shared_event_latent_b", None) if bool(state.params.get("_shared_event_latent_enabled", False)) else None),
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
        
        # Safety Check
        if not torch.isfinite(state.dX_src).all():
             with torch.no_grad():
                 state.dX_src.data = torch.nan_to_num(state.dX_src.data, nan=0.0, posinf=0.0, neginf=0.0)
        
        _clamp_dX_inplace(state)
        _apply_shared_event_latent_constraints_inplace(state)
        
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
                    σp_now, σs_now = _current_noise_scales(state)
                    logσ = torch.stack([torch.log(σp_now).detach().cpu(), torch.log(σs_now).detach().cpu()], dim=0)
                    if state.noise_log_scales is not None:
                        state.noise_log_scales.append(logσ)

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
        # Optional: per-epoch summary for likelihood sigma_inflation (distance/%vel dependent).
        try:
            if bool(state.params.get("_likelihood_sigma_inflation_enabled", False)):
                c = int(state.params.get("_sigma_infl_vel_count", 0) or 0)
                if c > 0:
                    metrics["likelihood/sigma_inflation_struct_mean_ms"] = float(state.params.get("_sigma_infl_vel_sum_ms", 0.0) or 0.0) / float(c)
                cP = int(state.params.get("_sigma_infl_vel_count_P", 0) or 0)
                if cP > 0:
                    metrics["likelihood/sigma_inflation_struct_P_mean_ms"] = float(state.params.get("_sigma_infl_vel_sum_ms_P", 0.0) or 0.0) / float(cP)
                cS = int(state.params.get("_sigma_infl_vel_count_S", 0) or 0)
                if cS > 0:
                    metrics["likelihood/sigma_inflation_struct_S_mean_ms"] = float(state.params.get("_sigma_infl_vel_sum_ms_S", 0.0) or 0.0) / float(cS)
                cd = int(state.params.get("_sigma_infl_vel_d_km_count", 0) or 0)
                if cd > 0:
                    metrics["likelihood/sigma_inflation_d_km_mean"] = float(state.params.get("_sigma_infl_vel_d_km_sum", 0.0) or 0.0) / float(cd)
        except Exception:
            pass
        # Optional: timing summary for shared_event_latent nuisance reconstruction (per epoch).
        try:
            diag = _get_diagnostics_cfg(state.params)
            if isinstance(diag, dict) and bool(diag.get("profile_shared_event_latent", False)):
                c = int(state.params.get("_se_lat_time_ms_count", 0) or 0)
                s = float(state.params.get("_se_lat_time_ms_sum", 0.0) or 0.0)
                if c > 0:
                    metrics["shared_event_latent/time_ms_mean"] = float(s / float(c))
                    metrics["shared_event_latent/time_ms_sum"] = float(s)
                    metrics["shared_event_latent/time_batches"] = float(c)
        except Exception:
            pass
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
        except Exception:
            pass
        # Optional: timing + breakdown summary for collapsed slowness_re likelihood (per epoch).
        try:
            diag = _get_diagnostics_cfg(state.params)
            if isinstance(diag, dict) and bool(diag.get("profile_slowness_re", False)) and bool(state.params.get("_slowness_re_enabled", False)):
                c = int(state.params.get("_sl_re_time_ms_count", 0) or 0)
                s = float(state.params.get("_sl_re_time_ms_sum", 0.0) or 0.0)
                if c > 0:
                    metrics["slowness_re/time_ms_mean"] = float(s / float(c))
                    metrics["slowness_re/time_ms_sum"] = float(s)
                    metrics["slowness_re/time_batches"] = float(c)
                # Grouping is measured once per batch; average over batches.
                if c > 0:
                    metrics["slowness_re/grouping_ms_mean"] = float(float(state.params.get("_sl_re_grouping_ms_sum", 0.0) or 0.0) / float(c))
                # Kernel/assemble/solve are measured per *profiled group*; average over profiled groups.
                gc = int(state.params.get("_sl_re_profiled_groups_sum", 0) or 0)
                if gc > 0:
                    metrics["slowness_re/kernel_ms_mean"] = float(float(state.params.get("_sl_re_kernel_ms_sum", 0.0) or 0.0) / float(gc))
                    metrics["slowness_re/assemble_ms_mean"] = float(float(state.params.get("_sl_re_assemble_ms_sum", 0.0) or 0.0) / float(gc))
                    metrics["slowness_re/solve_ms_mean"] = float(float(state.params.get("_sl_re_solve_ms_sum", 0.0) or 0.0) / float(gc))
                    metrics["slowness_re/profiled_groups"] = float(gc)
                    metrics["slowness_re/pcg_iters_mean"] = float(int(state.params.get("_sl_re_pcg_iters_sum", 0) or 0) / float(gc))
                # Workload (means over batches + max over epoch)
                if c > 0:
                    metrics["slowness_re/groups_mean"] = float(int(state.params.get("_sl_re_groups_sum", 0) or 0) / float(c))
                    metrics["slowness_re/groups_woodbury_mean"] = float(int(state.params.get("_sl_re_groups_woodbury_sum", 0) or 0) / float(c))
                    metrics["slowness_re/groups_fallback_diag_mean"] = float(int(state.params.get("_sl_re_groups_fallback_sum", 0) or 0) / float(c))
                    metrics["slowness_re/max_rows_max"] = float(int(state.params.get("_sl_re_max_rows_max", 0) or 0))
                    metrics["slowness_re/max_nodes_max"] = float(int(state.params.get("_sl_re_max_nodes_max", 0) or 0))
                    metrics["slowness_re/max_M_max"] = float(int(state.params.get("_sl_re_max_M_max", 0) or 0))
        except Exception:
            pass
        # Optional uncollapsed shared-event latent diagnostics.
        # Note: some of these are expensive (e.g., inducing prior energy), so we allow amortization.
        try:
            if bool(state.params.get("_shared_event_latent_enabled", False)) and _want_wandb_group(state.params, "shared_event_latent"):
                # Allow amortizing heavy shared_event_latent diagnostics:
                # inference.diagnostics.shared_event_latent_every_epochs (int, default=1)
                se_every = 1
                try:
                    diag = _get_diagnostics_cfg(state.params)
                    if isinstance(diag, dict) and "shared_event_latent_every_epochs" in diag:
                        se_every = int(diag.get("shared_event_latent_every_epochs", 1))
                except Exception:
                    se_every = 1
                if se_every < 1:
                    se_every = 1
                if (int(epoch_index) % int(se_every)) != 0:
                    raise RuntimeError("skip shared_event_latent diagnostics this epoch")

                b = getattr(state, "shared_event_latent_b", None)
                # Coefficient diagnostics:
                # - full / graph_gmrf / inducing_gp: b is scalar per phase (..,*,2)
                # - slowness_inducing_gp: b is vector per phase (..,*,2,3)
                if isinstance(b, torch.Tensor) and b.ndim == 3 and int(b.shape[2]) == 2:
                    bP = b[:, :, 0]
                    bS = b[:, :, 1]
                    metrics["shared_event_latent/bP_rms"] = float(torch.sqrt((bP * bP).mean()).item())
                    metrics["shared_event_latent/bS_rms"] = float(torch.sqrt((bS * bS).mean()).item())
                    # Global mean drift (should be ~0 under Q with q_diag>0)
                    metrics["shared_event_latent/bP_mean"] = float(bP.mean().item())
                    metrics["shared_event_latent/bS_mean"] = float(bS.mean().item())
                elif isinstance(b, torch.Tensor) and b.ndim == 4 and int(b.shape[2]) == 2 and int(b.shape[3]) == 3:
                    bP = b[:, :, 0, :]  # (S_or_R, M, 3)
                    bS = b[:, :, 1, :]
                    metrics["shared_event_latent/bP_rms"] = float(torch.sqrt((bP * bP).mean()).item())
                    metrics["shared_event_latent/bS_rms"] = float(torch.sqrt((bS * bS).mean()).item())
                    metrics["shared_event_latent/bP_mean"] = float(bP.mean().item())
                    metrics["shared_event_latent/bS_mean"] = float(bS.mean().item())
                    # Direct check of the *station-common mode* constraint:
                    # We care about mean over stations *per event* (K,2), not just global mean.
                    try:
                        n_stations = int(getattr(state, "n_stations", 0) or 0)
                        if n_stations > 0:
                            if int(b.shape[0]) == int(n_stations):
                                mu = b.to(torch.float32).mean(dim=0)  # scalar: (K,2) ; vector: (M,2,3)
                            else:
                                W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                                if (
                                    isinstance(W_sta, torch.Tensor)
                                    and W_sta.ndim == 2
                                    and int(W_sta.shape[0]) == int(n_stations)
                                    and int(b.shape[0]) == int(W_sta.shape[1])
                                ):
                                    W = W_sta.to(device=b.device, dtype=torch.float32)
                                    ones = torch.ones((n_stations,), device=b.device, dtype=torch.float32)
                                    r = (W.transpose(0, 1).matmul(ones)) / float(max(1, n_stations))  # (R,)
                                    mu = torch.tensordot(r, b.to(torch.float32), dims=([0], [0]))  # scalar: (K,2) ; vector: (M,2,3)
                                else:
                                    mu = None
                            if isinstance(mu, torch.Tensor) and mu.numel() > 0:
                                if mu.ndim == 2 and int(mu.shape[1]) == 2:
                                    metrics["shared_event_latent/station_common_P_rms"] = float(torch.sqrt((mu[:, 0] * mu[:, 0]).mean()).item())
                                    metrics["shared_event_latent/station_common_S_rms"] = float(torch.sqrt((mu[:, 1] * mu[:, 1]).mean()).item())
                                elif mu.ndim == 3 and int(mu.shape[1]) == 2 and int(mu.shape[2]) == 3:
                                    metrics["shared_event_latent/station_common_P_rms"] = float(torch.sqrt((mu[:, 0, :] * mu[:, 0, :]).mean()).item())
                                    metrics["shared_event_latent/station_common_S_rms"] = float(torch.sqrt((mu[:, 1, :] * mu[:, 1, :]).mean()).item())
                    except Exception:
                        pass
                    # If station_basis is enabled, b is in basis space (R,M,2). The physically relevant
                    # station-field inducing coefficients are C_sta = W_sta @ A, shape (n_stations,M,2).
                    # Log their scale as well so we can tell if the station field is actually “big” even
                    # when basis coefficients look small (or vice versa).
                    try:
                        mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                        use_sta_basis = bool(state.params.get("_shared_event_latent_station_basis_enabled", False))
                        if mode == "inducing_gp" and use_sta_basis:
                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                            # Only do this when shapes line up; otherwise skip quietly.
                            if (
                                isinstance(W_sta, torch.Tensor)
                                and W_sta.ndim == 2
                                and int(b.shape[0]) == int(W_sta.shape[1])
                                and int(W_sta.shape[0]) > 0
                                and int(b.shape[1]) > 0
                            ):
                                A = b.to(torch.float32)  # (R,M,2)
                                W = W_sta.to(device=A.device, dtype=torch.float32)  # (S,R)
                                C_P = torch.matmul(W, A[:, :, 0])  # (S,M)
                                C_S = torch.matmul(W, A[:, :, 1])  # (S,M)
                                metrics["shared_event_latent/station_coeff_P_rms"] = float(torch.sqrt((C_P * C_P).mean()).item())
                                metrics["shared_event_latent/station_coeff_S_rms"] = float(torch.sqrt((C_S * C_S).mean()).item())
                                metrics["shared_event_latent/station_coeff_P_mean"] = float(C_P.mean().item())
                                metrics["shared_event_latent/station_coeff_S_mean"] = float(C_S.mean().item())
                    except Exception:
                        pass
                    # Prior energy under the shared_event_latent prior.
                    # This is a useful amplitude diagnostic: too small -> b not used; too large -> b drifting / overpowering data fit.
                    mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                    if mode not in {"full", "inducing_gp", "slowness_inducing_gp", "graph_gmrf"}:
                        mode = "full"
                    if mode in {"inducing_gp", "slowness_inducing_gp"}:
                        # Inducing-point GP coefficients prior:
                        #   c ~ N(0, K_UU^{-1})  ⇔  log p(c) ∝ -0.5 * c^T K_UU c
                        offs = state.params.get("_shared_event_latent_inducing_offsets", None)
                        K_blocks = state.params.get("_shared_event_latent_inducing_K_blocks", None)
                        if isinstance(offs, torch.Tensor) and isinstance(K_blocks, list) and K_blocks:
                            tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                            rho = float(state.params.get("_shared_event_latent_rho_ps", 0.0))
                            if (tau_p > 0.0) and (tau_s > 0.0) and (abs(rho) < 1.0):
                                det = (tau_p * tau_p) * (tau_s * tau_s) * (1.0 - rho * rho)
                                inv00 = (tau_s * tau_s) / det
                                inv11 = (tau_p * tau_p) / det
                                inv01 = (-rho * tau_p * tau_s) / det

                                e00 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                                e11 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                                e01 = torch.tensor(0.0, device=b.device, dtype=b.dtype)
                                nb = int(max(0, int(offs.numel()) - 1))
                                for bi in range(nb):
                                    i0 = int(offs[bi].item())
                                    i1 = int(offs[bi + 1].item())
                                    if i1 <= i0:
                                        continue
                                    try:
                                        K = K_blocks[bi]
                                    except Exception:
                                        continue
                                    if not isinstance(K, torch.Tensor) or K.numel() == 0:
                                        continue
                                    Kt = K.to(device=b.device, dtype=b.dtype)
                                    if isinstance(b, torch.Tensor) and b.ndim == 3 and int(b.shape[2]) == 2:
                                        cP = bP[:, i0:i1]
                                        cS = bS[:, i0:i1]
                                        KP = torch.matmul(cP, Kt)
                                        KS = torch.matmul(cS, Kt)
                                        e00 = e00 + (cP * KP).sum()
                                        e11 = e11 + (cS * KS).sum()
                                        e01 = e01 + (cP * KS).sum()
                                    elif isinstance(b, torch.Tensor) and b.ndim == 4 and int(b.shape[2]) == 2 and int(b.shape[3]) == 3:
                                        # Vector slowness coefficients: sum energies across xyz components.
                                        for d in range(3):
                                            cP = b[:, i0:i1, 0, d]
                                            cS = b[:, i0:i1, 1, d]
                                            KP = torch.matmul(cP, Kt)
                                            KS = torch.matmul(cS, Kt)
                                            e00 = e00 + (cP * KP).sum()
                                            e11 = e11 + (cS * KS).sum()
                                            e01 = e01 + (cP * KS).sum()
                                energy = 0.5 * (float(inv00) * e00 + float(inv11) * e11 + 2.0 * float(inv01) * e01)
                                # Degrees of freedom for normalization (scalar or vector)
                                dof = float(max(1, int(b.numel())))
                                metrics["shared_event_latent/prior_energy"] = float(energy.item())
                                metrics["shared_event_latent/prior_energy_per_dof"] = float((energy / dof).item())
                    else:
                        # Full parameterization: Laplacian-GMRF prior over events via fixed graph Q (kNN Laplacian + q_diag I).
                        u = state.params.get("_shared_event_latent_u", None)
                        v = state.params.get("_shared_event_latent_v", None)
                        w = state.params.get("_shared_event_latent_w", None)
                        q_diag = float(state.params.get("_shared_event_latent_q_diag_runtime", state.params.get("_shared_event_latent_q_diag", 0.0)))
                        if isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor) and isinstance(w, torch.Tensor):
                            tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                            tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                            rho = float(state.params.get("_shared_event_latent_rho_ps", 0.0))
                            if (tau_p > 0.0) and (tau_s > 0.0) and (abs(rho) < 1.0):
                                det = (tau_p * tau_p) * (tau_s * tau_s) * (1.0 - rho * rho)
                                inv00 = (tau_s * tau_s) / det
                                inv11 = (tau_p * tau_p) / det
                                inv01 = (-rho * tau_p * tau_s) / det

                                u_i = u.to(torch.int64)
                                v_i = v.to(torch.int64)
                                w_f = w.to(dtype=b.dtype)

                                def _apply_Q(xSN: torch.Tensor) -> torch.Tensor:
                                    y = xSN * float(max(0.0, q_diag))
                                    if int(u_i.numel()) > 0:
                                        xu = xSN.index_select(1, u_i)
                                        xv = xSN.index_select(1, v_i)
                                        diff = xu - xv  # [S,E]
                                        dw = diff * w_f.unsqueeze(0)
                                        y.index_add_(1, u_i, dw)
                                        y.index_add_(1, v_i, -dw)
                                    return y

                                qP = _apply_Q(bP)
                                qS = _apply_Q(bS)
                                e00 = (bP * qP).sum()
                                e11 = (bS * qS).sum()
                                e01 = (bP * qS).sum()
                                energy = 0.5 * (float(inv00) * e00 + float(inv11) * e11 + 2.0 * float(inv01) * e01)
                                dof = float(max(1, int(bP.numel() + bS.numel())))
                                metrics["shared_event_latent/prior_energy"] = float(energy.item())
                                metrics["shared_event_latent/prior_energy_per_dof"] = float((energy / dof).item())
        except Exception:
            pass

        # Decompose total loss into avg likelihood term + (1/N_total) prior term (prior is constant across minibatches).
        try:
            if bool(state.params.get("_shared_event_latent_enabled", False)) and _want_wandb_group(state.params, "shared_event_latent"):
                σp_now, σs_now = _current_noise_scales(state)
                l_prior = compute_prior_loss(
                    state.dX_src,
                    state.prior_event,
                    state.prior_centroid,
                    σp_now,
                    σs_now,
                    int(state.N),
                    state.params,
                    cluster_ids=state.cluster_ids,
                    cluster_counts=state.cluster_counts,
                    event_precision_matrix=state.event_precision_matrix,
                    shared_event_latent_b=getattr(state, "shared_event_latent_b", None),
                )
                l_prior_f = float(l_prior.detach().cpu().item())
                metrics["shared_event_latent/loss_total"] = float(total_loss_mean)
                metrics["shared_event_latent/loss_prior"] = float(l_prior_f)
                metrics["shared_event_latent/loss_like_avg"] = float(total_loss_mean - l_prior_f)
        except Exception:
            pass

        # Per-epoch stats of delta_b nuisance term (accumulated without per-batch sync).
        try:
            if want_lat_diag and (b_delta_count is not None):
                cnt = float(b_delta_count.detach().cpu().item())
                if cnt > 0.0:
                    mean = (b_delta_sum / b_delta_count)
                    rms = torch.sqrt(b_delta_sumsq / b_delta_count)
                    metrics["shared_event_latent/delta_b_mean"] = float(mean.detach().cpu().item())
                    metrics["shared_event_latent/delta_b_rms"] = float(rms.detach().cpu().item())
                    metrics["shared_event_latent/delta_b_maxabs"] = float(b_delta_maxabs.detach().cpu().item())
                # Endpoint RMS (event-level b at e1/e2), split by phase
                if (b_end_count_p is not None) and (b_end_sumsq_p is not None):
                    cntp = float(b_end_count_p.detach().cpu().item())
                    if cntp > 0.0:
                        brms_p = torch.sqrt(b_end_sumsq_p / b_end_count_p)
                        metrics["shared_event_latent/bP_endpoint_rms"] = float(brms_p.detach().cpu().item())
                if (b_end_count_s is not None) and (b_end_sumsq_s is not None):
                    cnts = float(b_end_count_s.detach().cpu().item())
                    if cnts > 0.0:
                        brms_s = torch.sqrt(b_end_sumsq_s / b_end_count_s)
                        metrics["shared_event_latent/bS_endpoint_rms"] = float(brms_s.detach().cpu().item())
                # slowness_inducing_gp: endpoint u-field RMS (per-phase)
                if (u_end_count_p is not None) and (u_end_sumsq_p is not None):
                    cntp = float(u_end_count_p.detach().cpu().item())
                    if cntp > 0.0:
                        urms_p = torch.sqrt(u_end_sumsq_p / u_end_count_p)
                        metrics["shared_event_latent/uP_endpoint_rms"] = float(urms_p.detach().cpu().item())
                if (u_end_count_s is not None) and (u_end_sumsq_s is not None):
                    cnts = float(u_end_count_s.detach().cpu().item())
                    if cnts > 0.0:
                        urms_s = torch.sqrt(u_end_sumsq_s / u_end_count_s)
                        metrics["shared_event_latent/uS_endpoint_rms"] = float(urms_s.detach().cpu().item())
                if u_end_maxnorm_p is not None:
                    metrics["shared_event_latent/uP_endpoint_maxnorm"] = float(u_end_maxnorm_p.detach().cpu().item())
                if u_end_maxnorm_s is not None:
                    metrics["shared_event_latent/uS_endpoint_maxnorm"] = float(u_end_maxnorm_s.detach().cpu().item())
        except Exception:
            pass

        # For inducing-point shared_event_latent modes, also log coefficient magnitudes.
        # This is the highest-signal debug aid when users report "latents blowing up".
        try:
            if want_lat_diag and bool(state.params.get("_shared_event_latent_enabled", False)) and _want_wandb_group(state.params, "shared_event_latent"):
                mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                tau_units = str(state.params.get("_shared_event_latent_slowness_tau_units", "abs")).strip().lower()
                tau_ps = state.params.get("_shared_event_latent_tau_s", [0.0, 0.0])
                try:
                    tau_p = float(tau_ps[0]); tau_s = float(tau_ps[1])
                except Exception:
                    tau_p = float(tau_ps) if tau_ps is not None else 0.0
                    tau_s = float(tau_ps) if tau_ps is not None else 0.0
                metrics["shared_event_latent/tau_units"] = float(1.0 if tau_units == "vel_frac" else 0.0)
                metrics["shared_event_latent/tau_p_config"] = float(tau_p)
                metrics["shared_event_latent/tau_s_config"] = float(tau_s)
                # Flag whether EikoNet v(z) arrays are available (vel_frac scaling depends on this).
                try:
                    vp_arr = np.asarray(state.params.get("_eikonet_v1d_vp_km_s", []), dtype=np.float64).reshape(-1)
                    vs_arr = np.asarray(state.params.get("_eikonet_v1d_vs_km_s", []), dtype=np.float64).reshape(-1)
                    ok = bool(vp_arr.size >= 2 and vs_arr.size == vp_arr.size and np.isfinite(np.nanmedian(vp_arr)) and np.isfinite(np.nanmedian(vs_arr)))
                    metrics["shared_event_latent/eikonet_v1d_available"] = float(1.0 if ok else 0.0)
                except Exception:
                    metrics["shared_event_latent/eikonet_v1d_available"] = float(0.0)

                b = getattr(state, "shared_event_latent_b", None)
                if mode in {"inducing_gp", "slowness_inducing_gp"} and isinstance(b, torch.Tensor):
                    bb = b.detach()
                    # Scalar inducing_gp: (S_or_R, M_total, 2)
                    if bb.ndim == 3 and int(bb.shape[2]) == 2:
                        rms_p = torch.sqrt((bb[:, :, 0] * bb[:, :, 0]).mean().to(torch.float32))
                        rms_s = torch.sqrt((bb[:, :, 1] * bb[:, :, 1]).mean().to(torch.float32))
                        maxabs = bb.abs().max().to(torch.float32)
                        metrics["shared_event_latent/b_coeff_rms_p"] = float(rms_p.cpu().item())
                        metrics["shared_event_latent/b_coeff_rms_s"] = float(rms_s.cpu().item())
                        metrics["shared_event_latent/b_coeff_maxabs"] = float(maxabs.cpu().item())
                    # Vector slowness_inducing_gp: (S_or_R, M_total, 2, 3)
                    elif bb.ndim == 4 and int(bb.shape[2]) == 2 and int(bb.shape[3]) == 3:
                        # RMS across stations/ranks, inducing points, and xyz components
                        rms_p = torch.sqrt((bb[:, :, 0, :] * bb[:, :, 0, :]).mean().to(torch.float32))
                        rms_s = torch.sqrt((bb[:, :, 1, :] * bb[:, :, 1, :]).mean().to(torch.float32))
                        maxabs = bb.abs().max().to(torch.float32)
                        metrics["shared_event_latent/b_coeff_rms_p"] = float(rms_p.cpu().item())
                        metrics["shared_event_latent/b_coeff_rms_s"] = float(rms_s.cpu().item())
                        metrics["shared_event_latent/b_coeff_maxabs"] = float(maxabs.cpu().item())

                    # If we're in vel_frac mode, also log the implied slowness amplitude tau/v (s/km).
                    if tau_units == "vel_frac":
                        try:
                            vp = np.asarray(state.params.get("_eikonet_v1d_vp_km_s", []), dtype=np.float64).reshape(-1)
                            vs = np.asarray(state.params.get("_eikonet_v1d_vs_km_s", []), dtype=np.float64).reshape(-1)
                            vp_med = float(np.nanmedian(vp)) if vp.size > 0 else float("nan")
                            vs_med = float(np.nanmedian(vs)) if vs.size > 0 else float("nan")
                            if np.isfinite(vp_med) and vp_med > 0:
                                metrics["shared_event_latent/vp_v1d_med_km_s"] = float(vp_med)
                                metrics["shared_event_latent/tau_p_over_v_med_s_per_km"] = float(tau_p / vp_med)
                            if np.isfinite(vs_med) and vs_med > 0:
                                metrics["shared_event_latent/vs_v1d_med_km_s"] = float(vs_med)
                                metrics["shared_event_latent/tau_s_over_v_med_s_per_km"] = float(tau_s / vs_med)
                        except Exception:
                            pass
        except Exception:
            pass
        # Online ESS: re-log last-known values to avoid W&B gaps (only if we're actually logging ess_online).
        if bool(state.params.get("ess_online_enabled", False)) and _want_wandb_group(state.params, "ess_online"):
            last = getattr(state, "_ess_online_last_metrics", None)
            if isinstance(last, dict) and len(last) > 0:
                metrics.update(last)
            # Debug/health indicators (safe scalars)
            try:
                metrics["ess_online/updated"] = float(1.0 if ess_online_updated_this_epoch else 0.0)
                metrics["ess_online/error_count"] = float(int(getattr(state, "_ess_online_error_count", 0)))
            except Exception:
                pass

        # Optional diagnostic: evaluate loss on a FIXED subset of rows (same indices every time).
        # This helps distinguish "true drift" from minibatch sampling noise and from changing data permutations.
        try:
            if not _want_wandb_group(state.params, "fixed_eval"):
                raise RuntimeError("skip fixed_eval")
            diag = _get_diagnostics_cfg(state.params)
            fe = diag.get("fixed_eval", {}) if isinstance(diag, dict) else {}
            fe_enabled = bool(fe.get("enabled", False))
            if fe_enabled:
                every = int(fe.get("every_epochs", 25))
                if every < 1:
                    every = 25
                if (epoch_index % every) == 0:
                    n_rows = int(fe.get("n_rows", 200_000))
                    n_rows = max(1, min(n_rows, int(state.N)))
                    seed = int(fe.get("seed", 0))
                    bs_eval = int(fe.get("batch_size", 50_000))
                    bs_eval = max(1, bs_eval)

                    # Cache chosen subset so it stays constant across epochs.
                    if "_fixed_eval_rows" not in state.params:
                        rng = np.random.default_rng(seed)
                        idx_np = rng.choice(int(state.N), size=n_rows, replace=False).astype(np.int64)
                        state.params["_fixed_eval_rows"] = idx_np.tolist()
                    idx_np = np.asarray(state.params.get("_fixed_eval_rows", []), dtype=np.int64)
                    if idx_np.size > 0:
                        # Compute weighted mean loss over the subset in chunks
                        total = 0.0
                        denom = 0
                        σp_eval, σs_eval = _current_noise_scales(state)
                        for i0 in range(0, int(idx_np.size), bs_eval):
                            i1 = min(i0 + bs_eval, int(idx_np.size))
                            rows_t = torch.as_tensor(idx_np[i0:i1], device=state.device, dtype=torch.int64)
                            II_b = state.II.index_select(0, rows_t)
                            YY_b = state.YY.index_select(0, rows_t)
                            nuisance_delta = None
                            l = posterior_loss(
                                idx=II_b,
                                y=YY_b,
                                X_src=state.X_src,
                                ΔX_src=state.dX_src,
                                model=state.model,
                                prior_event=state.prior_event,
                                prior_centroid=state.prior_centroid,
                                σ_p=σp_eval,
                                σ_s=σs_eval,
                                N_total=state.N,
                                params=state.params,
                                nuisance_delta=nuisance_delta,
                                cluster_ids=state.cluster_ids,
                                cluster_counts=state.cluster_counts,
                                event_precision_matrix=state.event_precision_matrix,
                                shared_event_latent_b=(getattr(state, "shared_event_latent_b", None) if bool(state.params.get("_shared_event_latent_enabled", False)) else None),
                            )
                            # l is mean over this chunk
                            w = int(i1 - i0)
                            total += float(l.item()) * float(w)
                            denom += w
                        if denom > 0:
                            metrics["loss_fixed_eval"] = total / float(denom)
        except Exception:
            pass

        # Optional diagnostic: RMS of residuals (base and/or corrected) on a fixed subset of rows.
        # This is the most direct sanity check for posterior calibration in synthetic/noise-free tests.
        try:
            if not _want_wandb_group(state.params, "resid_rms"):
                raise RuntimeError("skip resid_rms")
            diag = _get_diagnostics_cfg(state.params)
            rr_cfg = diag.get("resid_rms", {}) if isinstance(diag, dict) else {}
            rr_enabled = bool(rr_cfg.get("enabled", False))
            if rr_enabled:
                every = int(rr_cfg.get("every_epochs", 10))
                if every < 1:
                    every = 10
                if (epoch_index % every) == 0:
                    n_rows = int(rr_cfg.get("n_rows", 200_000))
                    n_rows = max(1, min(n_rows, int(state.N)))
                    seed = int(rr_cfg.get("seed", 0))
                    bs_eval = int(rr_cfg.get("batch_size", 50_000))
                    bs_eval = max(1, bs_eval)
                    log_base = bool(rr_cfg.get("log_base", True))
                    log_corr = bool(rr_cfg.get("log_corrected", True))

                    # Cache chosen subset so it stays constant across epochs.
                    key_name = "_resid_rms_rows"
                    if key_name not in state.params:
                        rng = np.random.default_rng(seed)
                        idx_np = rng.choice(int(state.N), size=n_rows, replace=False).astype(np.int64)
                        state.params[key_name] = idx_np.tolist()
                    idx_np = np.asarray(state.params.get(key_name, []), dtype=np.int64)
                    if idx_np.size > 0:
                        sumsq_base_p = 0.0
                        sumsq_base_s = 0.0
                        sumsq_base_all = 0.0
                        sumsq_corr_p = 0.0
                        sumsq_corr_s = 0.0
                        sumsq_corr_all = 0.0
                        n_p = 0
                        n_s = 0
                        n_all = 0

                        # Mean correction: shared_event_latent (b) if enabled and available.
                        use_b = bool(state.params.get("_shared_event_latent_enabled", False))
                        if use_b:
                            try:
                                if getattr(state, "shared_event_latent_b", None) is None or getattr(state, "row_station_index", None) is None:
                                    use_b = False
                            except Exception:
                                use_b = False

                        σp_eval, σs_eval = _current_noise_scales(state)
                        sigma_p_base = float(σp_eval.item())
                        sigma_s_base = float(σs_eval.item())

                        for i0 in range(0, int(idx_np.size), bs_eval):
                            i1 = min(i0 + bs_eval, int(idx_np.size))
                            rows_t = torch.as_tensor(idx_np[i0:i1], device=state.device, dtype=torch.int64)
                            II_b = state.II.index_select(0, rows_t)
                            YY_b = state.YY.index_select(0, rows_t)

                            # Base residual: dt_obs - dt_pred (NO nuisance corrections)
                            rb = compute_residuals(II_b, YY_b, state.X_src, state.dX_src, state.model).detach().to(torch.float32)
                            ph = YY_b[:, 4].detach()
                            is_p = (ph < 0.5)

                            if log_base:
                                r2 = (rb * rb)
                                sumsq_base_all += float(r2.sum().item())
                                n_all += int(r2.numel())
                                if bool(is_p.any().item()):
                                    sumsq_base_p += float(r2[is_p].sum().item())
                                    n_p += int(is_p.sum().item())
                                if bool((~is_p).any().item()):
                                    sumsq_base_s += float(r2[~is_p].sum().item())
                                    n_s += int((~is_p).sum().item())

                            if log_corr:
                                # Build nuisance_delta (mean correction) for this batch.
                                nuisance = None
                                if use_b:
                                    try:
                                        b_lat = getattr(state, "shared_event_latent_b", None)
                                        sta = getattr(state, "row_station_index", None)
                                        if not isinstance(b_lat, torch.Tensor):
                                            raise RuntimeError("missing shared_event_latent_b")
                                        if not isinstance(sta, torch.Tensor):
                                            raise RuntimeError("missing row_station_index")
                                        if b_lat.ndim != 3 or int(b_lat.shape[2]) != 2:
                                            raise RuntimeError("invalid shared_event_latent_b shape")
                                        sta_b = sta.index_select(0, rows_t).to(torch.int64)
                                        e1 = II_b[:, 0].to(torch.int64)
                                        e2 = II_b[:, 1].to(torch.int64)
                                        ph2 = YY_b[:, 4]
                                        is_s2 = (ph2 >= 0.5)
                                        mode = str(state.params.get("_shared_event_latent_parameterization", "full")).strip().lower()
                                        if mode not in {"full", "inducing_gp", "graph_gmrf"}:
                                            mode = "full"

                                        if mode == "inducing_gp":
                                            nei_idx = getattr(state, "shared_event_latent_inducing_neighbor_idx", None)
                                            nei_k = getattr(state, "shared_event_latent_inducing_neighbor_k", None)
                                            if not isinstance(nei_idx, torch.Tensor) or not isinstance(nei_k, torch.Tensor):
                                                raise RuntimeError("missing inducing_gp neighbor tensors")
                                            # (B,m) neighbor lists per endpoint event
                                            idx1 = nei_idx.index_select(0, e1)
                                            idx2 = nei_idx.index_select(0, e2)
                                            k1 = nei_k.index_select(0, e1).to(torch.float32)
                                            k2 = nei_k.index_select(0, e2).to(torch.float32)
                                            m1 = (idx1 >= 0)
                                            m2 = (idx2 >= 0)
                                            idx1c = idx1.clamp_min(0)
                                            idx2c = idx2.clamp_min(0)
                                            # Support both parameterizations for inducing_gp:
                                            # - per-station coefficients: b_lat shape (n_stations, M, 2)
                                            # - station-basis coefficients: b_lat shape (R, M, 2) with W_sta shape (n_stations, R)
                                            W_sta = getattr(state, "shared_event_latent_station_basis_W", None)
                                            n_stations_rt = int(getattr(state, "n_stations", 0))
                                            is_basis = (
                                                isinstance(W_sta, torch.Tensor)
                                                and W_sta.ndim == 2
                                                and int(W_sta.shape[0]) == int(n_stations_rt)
                                                and int(b_lat.shape[0]) == int(W_sta.shape[1])
                                            )
                                            is_per_station = (n_stations_rt > 0) and (int(b_lat.shape[0]) == int(n_stations_rt))
                                            if is_basis:
                                                # Basis mode: b(s,·) = Σ_r W[s,r] * a_r(·)
                                                Wb = W_sta.index_select(0, sta_b).to(torch.float32)  # (B,R)
                                                A = b_lat.to(torch.float32)  # (R,M,2)
                                                R = int(A.shape[0])
                                                M = int(A.shape[1])

                                                def _recon_endpoint(idxc: torch.Tensor, kk: torch.Tensor, mm: torch.Tensor) -> torch.Tensor:
                                                    # idxc: (B,m) clamped >=0; kk: (B,m); mm: (B,m) bool
                                                    w = kk * mm.to(torch.float32)  # (B,m)
                                                    m = int(idxc.shape[1])
                                                    idx4 = idxc.unsqueeze(0).unsqueeze(-1).expand(R, -1, -1, 2)  # (R,B,m,2)
                                                    Aexp = A.unsqueeze(1).expand(R, int(idxc.shape[0]), M, 2)
                                                    g = torch.gather(Aexp, 2, idx4)  # (R,B,m,2)
                                                    Wr = Wb.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # (R,B,1,1)
                                                    bm2 = (g * Wr).sum(dim=0)  # (B,m,2)
                                                    return (bm2 * w.unsqueeze(-1)).sum(dim=1)  # (B,2)

                                                b1 = _recon_endpoint(idx1c, k1, m1)
                                                b2 = _recon_endpoint(idx2c, k2, m2)
                                                d = (b2 - b1).to(torch.float32)
                                                delta_b = torch.where(is_s2, d[:, 1], d[:, 0]).to(torch.float32)
                                            elif is_per_station:
                                                # Per-station coefficients
                                                cP = b_lat[:, :, 0]
                                                cS = b_lat[:, :, 1]
                                                v1P = cP[sta_b[:, None], idx1c] * k1 * m1
                                                v2P = cP[sta_b[:, None], idx2c] * k2 * m2
                                                v1S = cS[sta_b[:, None], idx1c] * k1 * m1
                                                v2S = cS[sta_b[:, None], idx2c] * k2 * m2
                                                dP = (v2P.sum(dim=1) - v1P.sum(dim=1)).to(torch.float32)
                                                dS = (v2S.sum(dim=1) - v1S.sum(dim=1)).to(torch.float32)
                                                delta_b = torch.where(is_s2, dS, dP).to(torch.float32)
                                            else:
                                                raise RuntimeError("inducing_gp shared_event_latent_b shape mismatch (neither per-station nor station-basis)")
                                        else:
                                            # Explicit event-latent (full / graph_gmrf):
                                            # - per-station: b_lat shape (n_stations, n_events, 2)
                                            # - station-basis: b_lat shape (R, n_events, 2) with W_sta shape (n_stations, R)
                                            W_sta2 = getattr(state, "shared_event_latent_station_basis_W", None)
                                            n_stations_rt2 = int(getattr(state, "n_stations", 0))
                                            is_per_station2 = (n_stations_rt2 > 0) and (int(b_lat.shape[0]) == int(n_stations_rt2))
                                            is_basis2 = (
                                                isinstance(W_sta2, torch.Tensor)
                                                and W_sta2.ndim == 2
                                                and int(W_sta2.shape[0]) == int(n_stations_rt2)
                                                and int(b_lat.shape[0]) == int(W_sta2.shape[1])
                                            )
                                            if is_basis2:
                                                A2 = b_lat.to(torch.float32)  # (R,Ne,2)
                                                W2 = W_sta2.index_select(0, sta_b).to(torch.float32)  # (B,R)
                                                A1 = A2.index_select(1, e1).transpose(0, 1).contiguous()  # (B,R,2)
                                                A2e = A2.index_select(1, e2).transpose(0, 1).contiguous()  # (B,R,2)
                                                b1 = (W2.unsqueeze(-1) * A1).sum(dim=1)  # (B,2)
                                                b2 = (W2.unsqueeze(-1) * A2e).sum(dim=1)  # (B,2)
                                                d = (b2 - b1).to(torch.float32)
                                                delta_b = torch.where(is_s2, d[:, 1], d[:, 0]).to(torch.float32)
                                            elif is_per_station2:
                                                bP1 = b_lat[sta_b, e1, 0]
                                                bP2 = b_lat[sta_b, e2, 0]
                                                bS1 = b_lat[sta_b, e1, 1]
                                                bS2 = b_lat[sta_b, e2, 1]
                                                delta_b = torch.where(is_s2, (bS2 - bS1), (bP2 - bP1)).to(torch.float32)
                                            else:
                                                raise RuntimeError("explicit shared_event_latent_b shape mismatch (neither per-station nor station-basis)")
                                        nuisance = delta_b if nuisance is None else (nuisance + delta_b)
                                    except Exception:
                                        pass
                                if nuisance is None:
                                    rc = rb
                                else:
                                    rc = rb - nuisance
                                r2c = (rc * rc)
                                sumsq_corr_all += float(r2c.sum().item())
                                if bool(is_p.any().item()):
                                    sumsq_corr_p += float(r2c[is_p].sum().item())
                                if bool((~is_p).any().item()):
                                    sumsq_corr_s += float(r2c[~is_p].sum().item())

                        # Final RMS
                        metrics["resid_rms/n_rows"] = float(int(idx_np.size))
                        metrics["resid_rms/sigma_base_p"] = float(sigma_p_base)
                        metrics["resid_rms/sigma_base_s"] = float(sigma_s_base)
                        if log_base:
                            metrics["resid_rms/base_all"] = float(np.sqrt(sumsq_base_all / max(1, n_all)))
                            metrics["resid_rms/base_p"] = float(np.sqrt(sumsq_base_p / max(1, n_p))) if n_p > 0 else float("nan")
                            metrics["resid_rms/base_s"] = float(np.sqrt(sumsq_base_s / max(1, n_s))) if n_s > 0 else float("nan")
                        if log_corr:
                            metrics["resid_rms/corr_all"] = float(np.sqrt(sumsq_corr_all / max(1, n_all)))
                            metrics["resid_rms/corr_p"] = float(np.sqrt(sumsq_corr_p / max(1, n_p))) if n_p > 0 else float("nan")
                            metrics["resid_rms/corr_s"] = float(np.sqrt(sumsq_corr_s / max(1, n_s))) if n_s > 0 else float("nan")
        except Exception:
            pass

        # Report hierarchical event prior implied stds if enabled (for W&B parity across phases).
        # Keys match existing W&B dashboards: hier_std_x/y/z/t and hier_corr_zt.
        # Also report spatial correlations for quick health checks of the learned covariance.
        try:
            if not _want_wandb_group(state.params, "priors"):
                raise RuntimeError("skip priors metrics")
            if bool(getattr(state, "hierarchical_prior_enable", False)) and state.event_precision_matrix is not None:
                P0_all = state.event_precision_matrix
                P0_mean = P0_all.mean(dim=0) if isinstance(P0_all, torch.Tensor) and P0_all.ndim == 3 else P0_all
                Cov = torch.linalg.inv(P0_mean)
                stds = torch.sqrt(torch.diag(Cov)).clamp_min(torch.tensor(0.0, device=Cov.device, dtype=Cov.dtype))
                s0 = float(stds[0].item())
                s1 = float(stds[1].item())
                s2 = float(stds[2].item())
                s3 = float(stds[3].item())
                metrics.update({
                    "hier_std_x": s0,
                    "hier_std_y": s1,
                    "hier_std_z": s2,
                    "hier_std_t": s3,
                    "hier_corr_xy": float(Cov[0, 1].item()) / (s0 * s1 + 1e-12),
                    "hier_corr_xz": float(Cov[0, 2].item()) / (s0 * s2 + 1e-12),
                    "hier_corr_yz": float(Cov[1, 2].item()) / (s1 * s2 + 1e-12),
                    "hier_corr_zt": float(Cov[2, 3].item()) / (s2 * s3 + 1e-12),
                })
        except Exception:
            pass

        # (Laplacian prior diagnostics removed.)
        # Optional: display preconditioner G stats periodically
        try:
            disp_every = int(state.params.get("display_precond_every", 0))
        except Exception:
            disp_every = 0
        if disp_every > 0 and (epoch_index % disp_every == 0) and _want_wandb_group(state.params, "precond"):
            try:
                if hasattr(optimizer, "preconditioner_stats"):
                    stats = optimizer.preconditioner_stats()  # type: ignore[attr-defined]
                    if isinstance(stats, dict):
                        g_min = float(stats.get("min", float("nan")))
                        g_p25 = float(stats.get("p25", float("nan")))
                        g_med = float(stats.get("median", float("nan")))
                        g_p75 = float(stats.get("p75", float("nan")))
                        g_max = float(stats.get("max", float("nan")))
                        if all([x == x for x in (g_p25, g_med, g_p75)]):  # not NaN
                            if (not ddp_enabled) or ddp_is_main:
                                print(f"Preconditioner G stats: p25={g_p25:.3e}, median={g_med:.3e}, p75={g_p75:.3e}")
                        if g_p25 == g_p25:
                            metrics["precond_g_p25"] = g_p25
                        if g_med == g_med:
                            metrics["precond_g_med"] = g_med
                        if g_p75 == g_p75:
                            metrics["precond_g_p75"] = g_p75
                        if g_min == g_min:
                            metrics["precond_g_min"] = g_min
                        if g_max == g_max:
                            metrics["precond_g_max"] = g_max
                    else:
                        # Backwards compat: (min, median, max)
                        try:
                            g_min, g_med, g_max = stats  # type: ignore[misc]
                            if all([x == x for x in (g_min, g_med, g_max)]):  # not NaN
                                if (not ddp_enabled) or ddp_is_main:
                                    print(f"Preconditioner G stats: min={g_min:.3e}, median={g_med:.3e}, max={g_max:.3e}")
                                metrics["precond_g_min"] = float(g_min)
                                metrics["precond_g_med"] = float(g_med)
                                metrics["precond_g_max"] = float(g_max)
                        except Exception:
                            pass
                else:
                    pg0 = optimizer.param_groups[0] if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0 else {}
                    if (not ddp_enabled) or ddp_is_main:
                        print(
                            "Preconditioner G stats: unavailable (preconditioning disabled or not initialized). "
                            f"preconditioning={pg0.get('preconditioning', None)} preconditioner={pg0.get('preconditioner', None)}"
                        )
            except Exception:
                pass

        return metrics

