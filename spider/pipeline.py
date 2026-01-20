from __future__ import annotations

import os
import math
import time
from typing import Optional

import numpy as np
import torch

from .model import load_eikonet_state_dict
from .modeling import compute_likelihood_loss, compute_prior_loss, compute_residuals
from .data import prepare_input_dfs
from .state import build_state, LocateState
from .optim import pSGLD, MongeSGLD
from .io import save_phase2_bundle, load_phase2_bundle, save_samples_periodic, save_map_locations, get_next_sample_count, clear_samples_file, save_checkpoint
from .utils.console import info, warn
from .utils.wandb_logger import init_wandb, WandbLogger


def _device_from_id(device_id: int) -> torch.device:
    try:
        device_id = int(device_id)
    except Exception:
        device_id = -1
    if device_id < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    warn("CUDA is not available; running on CPU.", section="RUN")
    return torch.device("cpu")


def _apply_clamp(state: LocateState) -> None:
    clamp = state.clamp_abs_dX
    if clamp is None:
        return
    with torch.no_grad():
        for dim in range(4):
            c = float(clamp[dim].item())
            if not math.isfinite(c) or c <= 0.0:
                continue
            state.dX_src[:, dim].clamp_(-c, c)


def _sample_indices(n: int, max_samples: int, seed: int, device: torch.device) -> torch.Tensor:
    if n <= 0:
        return torch.zeros((0,), device=device, dtype=torch.int64)
    max_samples = max(1, int(max_samples))
    if n <= max_samples:
        return torch.arange(n, device=device, dtype=torch.int64)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randperm(n, generator=gen, device="cpu")[:max_samples]
    return idx.to(device=device)


def _quantiles(x: np.ndarray, qs: list[float]) -> list[float]:
    try:
        return [float(np.quantile(x, q)) for q in qs]
    except Exception:
        return [float("nan") for _ in qs]


def _residual_stats(state: LocateState, max_samples: int, seed: int) -> dict:
    out: dict[str, float] = {}
    N = int(state.N)
    if N <= 0:
        return out
    idx = _sample_indices(N, max_samples, seed, state.device)
    if idx.numel() == 0:
        return out
    II = state.II.index_select(0, idx)
    YY = state.YY.index_select(0, idx)
    with torch.no_grad():
        resid = compute_residuals(II, YY, state.X_src, state.dX_src, state.model).detach().cpu().numpy().astype(np.float64)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return out
    out["resid/mean"] = float(np.mean(resid))
    out["resid/std"] = float(np.std(resid))
    out["resid/median"] = float(np.median(resid))
    out["resid/mad"] = float(np.median(np.abs(resid - np.median(resid))))
    q90, q99 = _quantiles(resid, [0.90, 0.99])
    out["resid/p90"] = q90
    out["resid/p99"] = q99
    # P/S RMS (on the same sample)
    ph = YY[:, 4].detach().cpu().numpy()
    is_p = ph < 0.5
    if np.any(is_p):
        out["resid/rms_p"] = float(np.sqrt(np.mean(resid[is_p] ** 2)))
    if np.any(~is_p):
        out["resid/rms_s"] = float(np.sqrt(np.mean(resid[~is_p] ** 2)))
    return out


def _precond_stats(opt: torch.optim.Optimizer, param: torch.Tensor, max_samples: int, seed: int) -> dict:
    out: dict[str, float] = {}
    if not isinstance(opt, pSGLD):
        return out
    st = opt.state.get(param, {})
    v = st.get("exp_avg_sq", None)
    if not isinstance(v, torch.Tensor):
        return out
    eps = float(opt.param_groups[0].get("eps", 1e-5))
    try:
        g = 1.0 / (eps + v.sqrt())
    except Exception:
        return out
    flat = g.detach().flatten()
    n = int(flat.numel())
    if n <= 0:
        return out
    idx = _sample_indices(n, max_samples, seed, device=flat.device)
    samp = flat.index_select(0, idx).cpu().numpy().astype(np.float64)
    if samp.size == 0:
        return out
    q25, q50, q75 = _quantiles(samp, [0.25, 0.50, 0.75])
    out["opt/precond_min"] = float(np.min(samp))
    out["opt/precond_p25"] = q25
    out["opt/precond_median"] = q50
    out["opt/precond_p75"] = q75
    out["opt/precond_max"] = float(np.max(samp))
    return out


def _gauge_stats(state: LocateState) -> dict:
    out: dict[str, float] = {}
    try:
        dims = list(state.params.get("gauge_projection_dims", [0, 1, 2]))
        mu = state.dX_src.detach().mean(dim=0)
        v = mu[dims]
        out["gauge/mean_shift_norm"] = float(torch.linalg.norm(v).item())
        if isinstance(state.cluster_counts, torch.Tensor):
            out["gauge/cluster_count"] = int(state.cluster_counts.numel())
    except Exception:
        return out
    return out


def _clamp_stats(state: LocateState) -> dict:
    out: dict[str, float] = {}
    clamp = state.clamp_abs_dX
    if clamp is None:
        return out
    try:
        abs_dx = torch.abs(state.dX_src.detach())
        mask = abs_dx >= (clamp.view(1, -1) - 1e-12)
        out["clamp/num_clamped"] = int(mask.any(dim=1).sum().item())
    except Exception:
        return out
    return out


def _dx_stats(state: LocateState, max_samples: int, seed: int) -> dict:
    out: dict[str, float] = {}
    n = int(state.dX_src.shape[0])
    if n <= 0:
        return out
    idx = _sample_indices(n, max_samples, seed, state.device)
    if idx.numel() == 0:
        return out
    dx = state.dX_src.detach().index_select(0, idx).cpu().numpy().astype(np.float64)
    if dx.size == 0:
        return out
    abs_dx = np.abs(dx)
    out["dx/med_abs_x"] = float(np.median(abs_dx[:, 0]))
    out["dx/med_abs_y"] = float(np.median(abs_dx[:, 1]))
    out["dx/med_abs_z"] = float(np.median(abs_dx[:, 2]))
    out["dx/med_abs_t"] = float(np.median(abs_dx[:, 3]))
    out["dx/mean_abs_x"] = float(np.mean(abs_dx[:, 0]))
    out["dx/mean_abs_y"] = float(np.mean(abs_dx[:, 1]))
    out["dx/mean_abs_z"] = float(np.mean(abs_dx[:, 2]))
    out["dx/mean_abs_t"] = float(np.mean(abs_dx[:, 3]))
    return out


def _setup_sampler(state: LocateState, lr: float) -> torch.optim.Optimizer:
    backend = str(state.params.get("sampler_backend", "psgld")).lower()
    if backend == "monge":
        opt = MongeSGLD(
            params=[state.dX_src],
            n_obs=int(state.N),
            lr=float(lr),
            alpha=float(state.params.get("sampler_monge_alpha", 1.0)),
            ema_beta=float(state.params.get("sampler_monge_ema_beta", 0.9)),
            eps=float(state.params.get("sampler_monge_eps", 1e-12)),
            add_noise=False,
        )
    else:
        opt = pSGLD(
            params=[state.dX_src],
            n_obs=int(state.N),
            lr=float(lr),
            beta=float(state.params.get("sampler_beta", 0.99)),
            eps=float(state.params.get("sampler_eps", 1e-5)),
            preconditioning=bool(state.params.get("sampler_preconditioning", True)),
            include_gamma=bool(state.params.get("sampler_preconditioning_include_gamma", False)),
            add_noise=False,
        )
    # Attach gauge projection config to optimizer
    if bool(state.params.get("gauge_projection_enabled", False)):
        opt._gauge_project_enable = True
        opt._gauge_project_param = state.dX_src
        opt._gauge_project_dims = tuple(state.params.get("gauge_projection_dims", [0, 1, 2]))
        opt._gauge_project_mode = str(state.params.get("gauge_projection_mode", "global"))
        opt._gauge_cluster_ids = state.cluster_ids
        opt._gauge_cluster_counts = state.cluster_counts
        opt._gauge_project_apply_noise = bool(state.params.get("gauge_projection_apply_noise", True))
    return opt


def _run_epoch(
    *,
    state: LocateState,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    add_noise: bool,
    noise_scale: float,
    grad_clip_norm: float = 0.0,
    shuffle: bool = True,
    phase: str = "phase1",
    epoch: int = 0,
    wandb_logger: Optional[WandbLogger] = None,
) -> dict:
    N = int(state.N)
    if N <= 0:
        return float("nan")
    if batch_size <= 0 or batch_size >= N:
        batch_size = N

    if isinstance(optimizer, (pSGLD, MongeSGLD)):
        for g in optimizer.param_groups:
            g["add_noise"] = bool(add_noise)
            g["noise_scale"] = float(noise_scale)
            g["temperature"] = float(state.params.get("sampler_temperature", 1.0))

    if shuffle:
        perm = torch.randperm(N, device=state.device)
    else:
        perm = torch.arange(N, device=state.device)

    total_loss = 0.0
    total_data = 0.0
    total_prior = 0.0
    total_grad_norm = 0.0
    total_update_norm = 0.0
    total_drift_var = 0.0
    total_noise_var = 0.0
    total_ratio = 0.0
    ratio_count = 0
    total_monge_g_norm = 0.0
    monge_count = 0
    sum_shared_quad = 0.0
    sum_groups_total = 0
    sum_groups_fallback = 0
    sum_pcg_fail = 0
    sum_pcg_iters_mean = 0.0
    sum_edge_weight_mean = 0.0
    max_rows = 0
    max_nodes = 0
    max_pcg_iters = 0
    max_edge_weight = float("nan")
    n_batches = 0
    t0 = time.time()
    log_every_batches = int(state.params.get("wandb_log_every_batches", 0) or 0)
    if "_global_step" not in state.params:
        state.params["_global_step"] = 0
    for i0 in range(0, N, batch_size):
        i1 = min(i0 + batch_size, N)
        idx = perm[i0:i1]
        II = state.II.index_select(0, idx)
        YY = state.YY.index_select(0, idx)
        sta_idx = None
        if isinstance(state.row_station_index, torch.Tensor):
            sta_idx = state.row_station_index.index_select(0, idx)

        optimizer.zero_grad(set_to_none=True)
        batch_metrics: dict = {}
        data_loss = compute_likelihood_loss(
            idx=II,
            y=YY,
            X_src=state.X_src,
            dX_src=state.dX_src,
            model=state.model,
            sigma_p=torch.tensor(state.params["phase_unc"][0], device=state.device),
            sigma_s=torch.tensor(state.params["phase_unc"][1], device=state.device),
            params=state.params,
            row_station_index=sta_idx,
            out_metrics=batch_metrics,
        )
        prior_loss = compute_prior_loss(state.dX_src, prior_std=state.params["prior_event_std"])
        batch_metrics["loss/prior_event"] = float(prior_loss.detach().item())
        loss = data_loss + prior_loss
        loss.backward()

        # Optional per-dimension LR scaling (ΔT)
        dt_lr_mult = float(state.params.get("sampler_dt_lr_mult", 1.0))
        if dt_lr_mult != 1.0:
            try:
                state.dX_src.grad[:, 3].mul_(dt_lr_mult)
            except Exception:
                pass

        if grad_clip_norm and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([state.dX_src], grad_clip_norm)

        grad_norm = float(torch.sqrt(torch.mean(state.dX_src.grad.detach().float() ** 2)).item())
        optimizer.step()
        _apply_clamp(state)

        total_loss += float(loss.detach().item())
        total_data += float(batch_metrics.get("loss/data_nll", float("nan")))
        total_prior += float(batch_metrics.get("loss/prior_event", float("nan")))
        total_grad_norm += float(grad_norm)

        if isinstance(optimizer, (pSGLD, MongeSGLD)):
            st = getattr(optimizer, "_last_stats", {})
            if isinstance(st, dict):
                total_update_norm += float(st.get("update_norm", 0.0))
                total_drift_var += float(st.get("drift_var", 0.0))
                total_noise_var += float(st.get("noise_var", 0.0))
                rr = float(st.get("drift_noise_var_ratio", float("nan")))
                if math.isfinite(rr):
                    total_ratio += rr
                    ratio_count += 1
                mg = float(st.get("monge_g_norm", float("nan")))
                if math.isfinite(mg):
                    total_monge_g_norm += mg
                    monge_count += 1

        if "shared_event_re/quad" in batch_metrics:
            sum_shared_quad += float(batch_metrics.get("shared_event_re/quad", 0.0))
            sum_groups_total += int(batch_metrics.get("shared_event_re/groups_total", 0))
            sum_groups_fallback += int(batch_metrics.get("shared_event_re/groups_fallback", 0))
            sum_pcg_fail += int(batch_metrics.get("shared_event_re/pcg_fail", 0))
            sum_pcg_iters_mean += float(batch_metrics.get("shared_event_re/pcg_iters_mean", 0.0))
            sum_edge_weight_mean += float(batch_metrics.get("shared_event_re/edge_weight_mean", 0.0))
            max_rows = max(max_rows, int(batch_metrics.get("shared_event_re/max_rows", 0)))
            max_nodes = max(max_nodes, int(batch_metrics.get("shared_event_re/max_nodes", 0)))
            max_pcg_iters = max(max_pcg_iters, int(batch_metrics.get("shared_event_re/pcg_iters_max", 0)))
            ew_max = float(batch_metrics.get("shared_event_re/edge_weight_max", float("nan")))
            if math.isfinite(ew_max):
                max_edge_weight = ew_max if not math.isfinite(max_edge_weight) else max(max_edge_weight, ew_max)

        if isinstance(wandb_logger, WandbLogger) and log_every_batches > 0:
            if (n_batches % log_every_batches) == 0:
                step = int(state.params["_global_step"])
                wandb_logger.log(
                    {
                        "loss/total": float(loss.detach().item()),
                        "loss/data_nll": float(batch_metrics.get("loss/data_nll", float("nan"))),
                        "loss/prior_event": float(batch_metrics.get("loss/prior_event", float("nan"))),
                        "opt/grad_norm": float(grad_norm),
                        "opt/update_norm": float(getattr(optimizer, "_last_stats", {}).get("update_norm", float("nan"))),
                        "opt/drift_noise_var_ratio": float(getattr(optimizer, "_last_stats", {}).get("drift_noise_var_ratio", float("nan"))),
                        "opt/monge_g_norm": float(getattr(optimizer, "_last_stats", {}).get("monge_g_norm", float("nan"))),
                        "phase/name": str(phase),
                        "phase/epoch": int(epoch),
                    },
                    step=step,
                )
        state.params["_global_step"] = int(state.params["_global_step"]) + 1
        n_batches += 1

    epoch_time = time.time() - t0
    mean_loss = total_loss / max(n_batches, 1)
    metrics = {
        "loss/total": float(mean_loss),
        "loss/data_nll": float(total_data / max(n_batches, 1)),
        "loss/prior_event": float(total_prior / max(n_batches, 1)),
        "opt/grad_norm": float(total_grad_norm / max(n_batches, 1)),
        "opt/update_norm": float(total_update_norm / max(n_batches, 1)),
        "opt/drift_var": float(total_drift_var / max(n_batches, 1)),
        "opt/noise_var": float(total_noise_var / max(n_batches, 1)),
        "opt/drift_noise_var_ratio": float(total_ratio / max(ratio_count, 1)) if ratio_count > 0 else float("nan"),
        "opt/monge_g_norm": float(total_monge_g_norm / max(monge_count, 1)) if monge_count > 0 else float("nan"),
        "phase/epoch": int(epoch),
        "phase/name": str(phase),
        "phase/steps_per_epoch": int(n_batches),
        "sys/epoch_time_s": float(epoch_time),
        "sys/batch_time_ms": float(epoch_time * 1000.0 / max(n_batches, 1)),
    }

    if sum_shared_quad != 0.0 or sum_groups_total > 0:
        metrics.update(
            {
                "shared_event_re/quad": float(sum_shared_quad / max(n_batches, 1)),
                "shared_event_re/groups_total": int(sum_groups_total),
                "shared_event_re/groups_fallback": int(sum_groups_fallback),
                "shared_event_re/pcg_fail": int(sum_pcg_fail),
                "shared_event_re/max_rows": int(max_rows),
                "shared_event_re/max_nodes": int(max_nodes),
                "shared_event_re/pcg_iters_mean": float(sum_pcg_iters_mean / max(n_batches, 1)),
                "shared_event_re/pcg_iters_max": int(max_pcg_iters),
                "shared_event_re/edge_weight_mean": float(sum_edge_weight_mean / max(n_batches, 1)),
                "shared_event_re/edge_weight_max": float(max_edge_weight),
            }
        )

    if isinstance(optimizer, pSGLD):
        g0 = optimizer.param_groups[0]
        lr = float(g0.get("lr", float("nan")))
        metrics["opt/lr"] = lr
        metrics["opt/precond_enabled"] = int(bool(g0.get("preconditioning", False)))
        metrics["opt/include_gamma"] = int(bool(g0.get("include_gamma", False)))
        metrics["noise/temperature"] = float(g0.get("temperature", float("nan")))
        metrics["noise/scale"] = float(g0.get("noise_scale", float("nan")))
        try:
            n_obs = int(g0.get("n_obs", max(1, int(state.N))))
            metrics["noise/std_effective"] = float(
                math.sqrt(2.0 * lr)
                * math.sqrt(max(float(g0.get("temperature", 1.0)), 0.0))
                * float(g0.get("noise_scale", 1.0))
                / math.sqrt(max(n_obs, 1))
            )
        except Exception:
            metrics["noise/std_effective"] = float("nan")

    return metrics


def _phase1_map(state: LocateState, *, wandb_logger: Optional[WandbLogger] = None) -> None:
    state.optimizer = torch.optim.Adam([state.dX_src], lr=float(state.params.get("lr_warmup", 1e-3)))
    info("Phase 1: MAP (Adam)", section="RUN")
    for epoch in range(int(state.params.get("phase1_epochs", 0))):
        metrics = _run_epoch(
            state=state,
            optimizer=state.optimizer,
            batch_size=int(state.batch_size_warmup),
            add_noise=False,
            noise_scale=0.0,
            grad_clip_norm=0.0,
            shuffle=bool(state.params.get("batching_shuffle", True)),
            phase="phase1",
            epoch=epoch,
            wandb_logger=wandb_logger,
        )
        if epoch % max(int(state.params.get("checkpoint_interval", 50)), 1) == 0 and epoch > 0:
            save_checkpoint(params=state.params, optimizer=state.optimizer, epoch=epoch, N=state.N, dX_src=state.dX_src, phase="phase1")
            if isinstance(wandb_logger, WandbLogger):
                wandb_logger.log({"io/last_checkpoint_epoch": int(epoch)}, step=int(state.params.get("_global_step", 0)))
        if bool(state.params.get("runtime_verbose", True)):
            info(f"phase1 epoch={epoch+1} loss={metrics.get('loss/total', float('nan')):.4e}", section="RUN")
        if isinstance(wandb_logger, WandbLogger):
            max_med = int(state.params.get("wandb_max_median_samples", 200000) or 200000)
            res_samp = int(state.params.get("wandb_residual_sample_size", 200000) or 200000)
            metrics.update(_dx_stats(state, max_med, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_residual_stats(state, res_samp, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_gauge_stats(state))
            metrics.update(_clamp_stats(state))
            wandb_logger.log(metrics, step=int(state.params.get("_global_step", 0)))
    save_map_locations(state.params, state.origins0, state.X_src, state.dX_src, state.projector)


def _phase2_precond(state: LocateState, *, wandb_logger: Optional[WandbLogger] = None) -> None:
    if state.sampler is None:
        state.sampler = _setup_sampler(state, lr=float(state.params.get("sampler_lr_per_phase", [0, 0, 0, 1e-3])[1]))
    else:
        try:
            state.sampler.set_lr(float(state.params.get("sampler_lr_per_phase", [0, 0, 0, 1e-3])[1]))
        except Exception:
            pass
    info("Phase 2: preconditioner warmup (noise off)", section="RUN")
    for epoch in range(int(state.params.get("phase2_epochs", 0))):
        metrics = _run_epoch(
            state=state,
            optimizer=state.sampler,
            batch_size=int(state.batch_size_sgld),
            add_noise=False,
            noise_scale=0.0,
            grad_clip_norm=0.0,
            shuffle=bool(state.params.get("batching_shuffle", True)),
            phase="phase2",
            epoch=epoch,
            wandb_logger=wandb_logger,
        )
        if epoch % max(int(state.params.get("checkpoint_interval", 50)), 1) == 0 and epoch > 0:
            save_checkpoint(params=state.params, optimizer=state.sampler, epoch=epoch, N=state.N, dX_src=state.dX_src, phase="phase2")
            if isinstance(wandb_logger, WandbLogger):
                wandb_logger.log({"io/last_checkpoint_epoch": int(epoch)}, step=int(state.params.get("_global_step", 0)))
        if bool(state.params.get("runtime_verbose", True)):
            info(f"phase2 epoch={epoch+1} loss={metrics.get('loss/total', float('nan')):.4e}", section="RUN")
        if isinstance(wandb_logger, WandbLogger):
            max_med = int(state.params.get("wandb_max_median_samples", 200000) or 200000)
            res_samp = int(state.params.get("wandb_residual_sample_size", 200000) or 200000)
            pre_samp = int(state.params.get("wandb_precond_sample_size", 200000) or 200000)
            metrics.update(_dx_stats(state, max_med, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_residual_stats(state, res_samp, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_precond_stats(state.sampler, state.dX_src, pre_samp, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_gauge_stats(state))
            metrics.update(_clamp_stats(state))
            wandb_logger.log(metrics, step=int(state.params.get("_global_step", 0)))


def _phase3_noise_ramp(state: LocateState, *, wandb_logger: Optional[WandbLogger] = None) -> None:
    if state.sampler is None:
        state.sampler = _setup_sampler(state, lr=float(state.params.get("sampler_lr_per_phase", [0, 0, 0, 1e-3])[2]))
    else:
        try:
            state.sampler.set_lr(float(state.params.get("sampler_lr_per_phase", [0, 0, 0, 1e-3])[2]))
        except Exception:
            pass
    n_epochs = int(state.params.get("phase3_epochs", 0))
    if n_epochs <= 0:
        return
    info("Phase 3: noise ramp", section="RUN")
    for epoch in range(n_epochs):
        noise_scale = float(epoch + 1) / float(max(n_epochs, 1))
        metrics = _run_epoch(
            state=state,
            optimizer=state.sampler,
            batch_size=int(state.batch_size_sgld),
            add_noise=True,
            noise_scale=noise_scale,
            grad_clip_norm=0.0,
            shuffle=bool(state.params.get("batching_shuffle", True)),
            phase="phase3",
            epoch=epoch,
            wandb_logger=wandb_logger,
        )
        if bool(state.params.get("runtime_verbose", True)):
            info(f"phase3 epoch={epoch+1} loss={metrics.get('loss/total', float('nan')):.4e} noise_scale={noise_scale:.3f}", section="RUN")
        if isinstance(wandb_logger, WandbLogger):
            max_med = int(state.params.get("wandb_max_median_samples", 200000) or 200000)
            res_samp = int(state.params.get("wandb_residual_sample_size", 200000) or 200000)
            pre_samp = int(state.params.get("wandb_precond_sample_size", 200000) or 200000)
            metrics.update(_dx_stats(state, max_med, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_residual_stats(state, res_samp, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_precond_stats(state.sampler, state.dX_src, pre_samp, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_gauge_stats(state))
            metrics.update(_clamp_stats(state))
            wandb_logger.log(metrics, step=int(state.params.get("_global_step", 0)))


def _phase4_sampling(state: LocateState, *, wandb_logger: Optional[WandbLogger] = None) -> None:
    if state.sampler is None:
        state.sampler = _setup_sampler(state, lr=float(state.params.get("sampler_lr_per_phase", [0, 0, 0, 1e-3])[3]))
    else:
        try:
            state.sampler.set_lr(float(state.params.get("sampler_lr_per_phase", [0, 0, 0, 1e-3])[3]))
        except Exception:
            pass
    n_epochs = int(state.params.get("phase4_epochs", 0))
    if n_epochs <= 0:
        info("Phase 4: sampling skipped (phase4_epochs=0)", section="RUN")
        return
    info("Phase 4: sampling", section="RUN")
    sample_count = get_next_sample_count(state.params)
    if bool(state.params.get("clear_samples_on_reset", False)) and sample_count == 0:
        clear_samples_file(state.params)
    for epoch in range(n_epochs):
        metrics = _run_epoch(
            state=state,
            optimizer=state.sampler,
            batch_size=int(state.batch_size_sgld),
            add_noise=True,
            noise_scale=1.0,
            grad_clip_norm=0.0,
            shuffle=bool(state.params.get("batching_shuffle", True)),
            phase="phase4",
            epoch=epoch,
            wandb_logger=wandb_logger,
        )
        if bool(state.params.get("runtime_verbose", True)) and (epoch % 10 == 0 or epoch == n_epochs - 1):
            info(f"phase4 epoch={epoch+1} loss={metrics.get('loss/total', float('nan')):.4e}", section="RUN")

        if bool(state.params.get("write_samples", True)):
            save_every_n = int(state.params.get("save_every_n", 1))
            sample_write_interval = int(state.params.get("sample_write_interval", save_every_n))
            if save_every_n > 0 and (epoch % save_every_n == 0):
                if sample_write_interval <= 0 or (epoch % sample_write_interval == 0):
                    save_samples_periodic(
                        params=state.params,
                        origins0=state.origins0,
                        X_src=state.X_src,
                        dX_src=state.dX_src,
                        projector=state.projector,
                        sample_count=sample_count,
                    )
                    if isinstance(wandb_logger, WandbLogger):
                        try:
                            sz = os.path.getsize(state.params.get("samples_outfile", "")) / (1024.0 * 1024.0)
                        except Exception:
                            sz = float("nan")
                        wandb_logger.log(
                            {
                                "phase/samples_written": int(sample_count + 1),
                                "io/last_samples_batch": int(sample_count),
                                "io/samples_file_size_mb": float(sz),
                            },
                            step=int(state.params.get("_global_step", 0)),
                        )
                    sample_count += 1

        if epoch % max(int(state.params.get("checkpoint_interval", 50)), 1) == 0 and epoch > 0:
            save_checkpoint(params=state.params, optimizer=state.sampler, epoch=epoch, N=state.N, dX_src=state.dX_src, phase="phase4")
            if isinstance(wandb_logger, WandbLogger):
                wandb_logger.log({"io/last_checkpoint_epoch": int(epoch)}, step=int(state.params.get("_global_step", 0)))
        if isinstance(wandb_logger, WandbLogger):
            max_med = int(state.params.get("wandb_max_median_samples", 200000) or 200000)
            res_samp = int(state.params.get("wandb_residual_sample_size", 200000) or 200000)
            pre_samp = int(state.params.get("wandb_precond_sample_size", 200000) or 200000)
            metrics.update(_dx_stats(state, max_med, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_residual_stats(state, res_samp, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_precond_stats(state.sampler, state.dX_src, pre_samp, seed=state.params.get("runtime_seed", 0) + epoch))
            metrics.update(_gauge_stats(state))
            metrics.update(_clamp_stats(state))
            wandb_logger.log(metrics, step=int(state.params.get("_global_step", 0)))


def locate_map(*, params: dict, device_id: int, bundle_out: Optional[str] = None) -> None:
    device = _device_from_id(device_id)
    wandb_logger = init_wandb(params)
    if isinstance(wandb_logger, WandbLogger):
        wandb_logger.log({"sys/device": str(device)}, step=0)
    model = load_eikonet_state_dict(
        params["model_file"],
        device=device,
        default_scale=float(params.get("scale", 1.0)),
    )
    stations, dtimes, origins = prepare_input_dfs(params, model=model, device=device)
    state = build_state(params=params, origins0=origins, dtimes=dtimes, model=model, device=device)
    if isinstance(wandb_logger, WandbLogger) and isinstance(params.get("_filter_stats", None), dict):
        if not bool(params.get("_filter_stats_logged", False)):
            wandb_logger.log(params["_filter_stats"], step=int(params.get("_global_step", 0)))
            params["_filter_stats_logged"] = True
    _phase1_map(state, wandb_logger=wandb_logger)
    if not bundle_out:
        ckpt_dir = str(params.get("checkpoint_dir", "")) or "."
        os.makedirs(ckpt_dir, exist_ok=True)
        bundle_out = os.path.join(ckpt_dir, "phase2_bundle.pth")
    save_phase2_bundle(path=bundle_out, params=None, origins0=state.origins0, dtimes=state.dtimes, dX_src=state.dX_src)
    info(f"Wrote Phase-2 bundle {bundle_out}", section="RUN")
    if isinstance(wandb_logger, WandbLogger):
        wandb_logger.finish()


def sample_from_bundle(*, params: dict, device_id: int, bundle_path: str) -> None:
    device = _device_from_id(device_id)
    wandb_logger = init_wandb(params)
    if isinstance(wandb_logger, WandbLogger):
        wandb_logger.log({"sys/device": str(device)}, step=0)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Phase-2 bundle not found: {bundle_path}")
    bundle = load_phase2_bundle(path=bundle_path)
    model = load_eikonet_state_dict(
        params["model_file"],
        device=device,
        default_scale=float(params.get("scale", 1.0)),
    )
    state = build_state(
        params=params,
        origins0=bundle.origins0,
        dtimes=bundle.dtimes,
        model=model,
        device=device,
        dX_src_init=bundle.dX_src,
    )
    state.sampler = _setup_sampler(state, lr=float(params.get("sampler_lr_per_phase", [0, 0, 0, 1e-3])[1]))
    _phase2_precond(state, wandb_logger=wandb_logger)
    _phase3_noise_ramp(state, wandb_logger=wandb_logger)
    _phase4_sampling(state, wandb_logger=wandb_logger)
    if isinstance(wandb_logger, WandbLogger):
        wandb_logger.finish()
