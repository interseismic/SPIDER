"""Location pipeline orchestration (warmup + SGLD) for SPIDER.

This module contains the high-level `locate_all` entrypoint and its helper
functions, organized into clear phases. It uses the lower-level modeling
primitives from `spider.core.modeling`.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Generator
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pyproj import Proj
import collections, math

from spider.optim.sgld import pSGLD
from spider.io.checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint_files
from spider.io.samples import save_samples_periodic, get_next_sample_count, clear_samples_file
from spider.utils import extract_metrics_from_stats_tensor

# Pull modeling primitives we need
from .modeling import (
    posterior_loss,
    compute_residuals_full,
    write_output,
    med_abs_dev_torch,
)

@dataclass
class LocateState:
    """Aggregated runtime state for earthquake location and sampling."""

    params: dict
    device: torch.device
    projector: Proj
    origins0: pl.DataFrame
    dtimes: pl.DataFrame

    # Tensors
    X_src: torch.Tensor
    dX_src: torch.nn.Parameter  # ΔX_src
    II: torch.Tensor
    YY: torch.Tensor

    # Model and priors
    model: nn.Module
    prior_event: torch.distributions.Distribution
    prior_centroid: torch.distributions.Distribution

    # Optim/SGD
    optimizer: torch.optim.Optimizer
    sampler: Optional[torch.optim.Optimizer] = None

    # Config
    N: int = 0
    batch_size_warmup: int = 0
    batch_size_sgld: int = 0
    scale_theta: Optional[torch.Tensor] = None  # [σ_p, σ_s]

    # Stats/samples
    stats_tensor: Optional[torch.Tensor] = None
    samples: Optional[List[torch.Tensor]] = None
    sample_count: int = 0
    global_step_count: int = 0

    II_epoch: Optional[torch.Tensor] = None
    YY_epoch: Optional[torch.Tensor] = None
    _perm_epoch: Optional[torch.Tensor] = None

    def begin_epoch_rr(self, *, seed: Optional[int] = None, use_full_N: bool = True) -> None:
        """
        Prepare random-reshuffled (without replacement) contiguous tensors for this epoch.
        After calling, use `state.II_epoch` and `state.YY_epoch` in place of `state.II`, `state.YY`
        for the duration of the epoch.

        Args:
            seed: Optional integer for reproducible reshuffles (e.g., pass `epoch`).
            use_full_N: If True, use `self.N` rows; otherwise infer from `self.II.shape[0]`.
        """
        N = int(self.N) if use_full_N and self.N else int(self.II.shape[0])

        # Build permutation once per epoch (CPU RNG for determinism across devices)
        gen = torch.Generator(device=self.II.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        perm = torch.randperm(N, generator=gen, device=self.II.device)

        # Create contiguous, permuted views so your [i_start:i_end] slicing stays valid
        self.II_epoch = self.II.index_select(0, perm).contiguous()
        self.YY_epoch = self.YY.index_select(0, perm).contiguous()

        # (optional) stash for debugging/repro
        self._perm_epoch = perm

def _build_initial_state(
    params: dict,
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
) -> LocateState:
    """Prepare tensors, priors, optimizer, and wrap model for multi-GPU."""

    evid_to_row = {row["evid"]: idx for idx, row in enumerate(origins0.iter_rows(named=True))}

    evid1_idx = [evid_to_row[x] for x in dtimes["evid1"]]
    evid2_idx = [evid_to_row[x] for x in dtimes["evid2"]]
    dtimes = dtimes.with_columns(
        [
            pl.Series(evid1_idx).alias("evid1_idx"),
            pl.Series(evid2_idx).alias("evid2_idx"),
            pl.Series(np.arange(dtimes.shape[0])).alias("arid"),
        ]
    )

    projector = Proj(
        proj="laea",
        lat_0=params["lat_min"],
        lon_0=params["lon_min"],
        datum="WGS84",
        units="km",
    )
    XX, YY = projector(
        origins0["longitude"].to_numpy(), origins0["latitude"].to_numpy()
    )

    X_src = torch.zeros(origins0.shape[0], 4, dtype=torch.float32, device=device)
    X_src[:, 0] = torch.tensor(XX, dtype=torch.float32, device=device)
    X_src[:, 1] = torch.tensor(YY, dtype=torch.float32, device=device)
    X_src[:, 2] = torch.tensor(
        origins0["depth"].to_numpy(), dtype=torch.float32, device=device
    )

    dX_src = torch.zeros_like(X_src)
    dX_src.requires_grad_()
    dX_src = torch.nn.Parameter(dX_src)

    II = torch.tensor(
        dtimes[["evid1_idx", "evid2_idx"]].to_numpy(),
        dtype=torch.int64,
        device=device,
    )
    YY = torch.tensor(
        dtimes[["dt", "X", "Y", "Z", "phase"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    model = nn.DataParallel(model, device_ids=params["devices"])  # type: ignore[arg-type]

    batch_size_warmup = params["batch_size_warmup"]
    batch_size_sgld = params["batch_size_sgld"]
    N = dtimes.shape[0]

    prior_event = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(
            len(params["prior_event_std"]), device=device, dtype=torch.float32
        ),
        covariance_matrix=torch.diag(
            torch.tensor(
                params["prior_event_std"], device=device, dtype=torch.float32
            )
            ** 2
        ),
    )
    prior_centroid = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.zeros(
            len(params["prior_centroid_std"]), device=device, dtype=torch.float32
        ),
        covariance_matrix=torch.diag(
            torch.tensor(
                params["prior_centroid_std"], device=device, dtype=torch.float32
            )
            ** 2
        ),
    )

    optimizer = torch.optim.Adam([dX_src], lr=params["lr_warmup"])
    scale_theta = torch.tensor(params["phase_unc"], device=device, dtype=torch.float32)

    state = LocateState(
        params=params,
        device=device,
        projector=projector,
        origins0=origins0,
        dtimes=dtimes,
        X_src=X_src,
        dX_src=dX_src,
        II=II,
        YY=YY,
        model=model,
        prior_event=prior_event,
        prior_centroid=prior_centroid,
        optimizer=optimizer,
        N=N,
        batch_size_warmup=batch_size_warmup,
        batch_size_sgld=batch_size_sgld,
        scale_theta=scale_theta,
        stats_tensor=torch.zeros(8, device=device),
        samples=[],
        sample_count=0,
        global_step_count=0,
    )
    return state


def _maybe_write_map_csv(state: LocateState, epoch: int) -> None:
    if epoch % 100 == 0:
        X_src1 = (state.X_src + state.dX_src).detach().cpu().numpy()
        origins = write_output(
            state.origins0,
            X_src1,
            state.X_src.detach().cpu().numpy(),
            state.projector,
        )
        origins.write_csv(f"{state.params['catalog_outfile']}_MAP.csv")


def _compute_phase_mads(
    state: LocateState, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    residuals = compute_residuals_full(
        state.II, state.YY, state.X_src, state.dX_src, state.model, batch_size, state.N
    )
    idx_p = torch.nonzero(state.YY[:, 4] < 0.5)
    idx_s = torch.nonzero(state.YY[:, 4] > 0.5)
    mad_p = med_abs_dev_torch(residuals[idx_p])
    mad_s = med_abs_dev_torch(residuals[idx_s])
    return mad_p, mad_s


def _phase1_map_warmup(state: LocateState, start_epoch: int = 0, wandb_logger=None) -> None:
    """Noise-free MAP warmup using Adam on ΔX_src."""
    print("Phase 1: MAP estimation with Adam optimizer (noise free)")
    checkpoint_interval = state.params.get("checkpoint_interval", 50)
    for epoch in range(start_epoch, state.params["phase1_epochs"]):
        state.begin_epoch_rr(seed=epoch)
        epoch_start_time = time.time()
        total_loss_vals: List[float] = []

        for j in range(0, state.N // state.batch_size_warmup + 1):
            i_start = j * state.batch_size_warmup
            i_end = min(i_start + state.batch_size_warmup, state.N)

            state.optimizer.zero_grad()
            loss = posterior_loss(
                state.II_epoch[i_start:i_end, :],
                state.YY_epoch[i_start:i_end],
                state.X_src,
                state.dX_src,
                state.model,
                state.prior_event,
                state.prior_centroid,
                state.scale_theta[0],
                state.scale_theta[1],
                state.N,
                normalize=True,
            )
            loss.backward()
            state.optimizer.step()
            with torch.no_grad():
                total_loss_vals.append(float(loss.item()))
                state.global_step_count += 1

        with torch.no_grad():
            if epoch % 10 == 0 or epoch == state.params["phase1_epochs"] - 1:
                mad_p, mad_s = _compute_phase_mads(state, state.batch_size_warmup)
            else:
                mad_p = torch.tensor(0.0, device=state.device)
                mad_s = torch.tensor(0.0, device=state.device)

        _maybe_write_map_csv(state, epoch)

        epoch_time = time.time() - epoch_start_time
        st = state.stats_tensor
        st[0] = state.dX_src[:, 0].mean()
        st[1] = state.dX_src[:, 1].mean()
        st[2] = state.dX_src[:, 2].mean()
        st[3] = torch.abs(state.dX_src[:, 0]).median()
        st[4] = torch.abs(state.dX_src[:, 1]).median()
        st[5] = torch.abs(state.dX_src[:, 2]).median()
        st[6] = torch.sqrt(
            state.dX_src[:, 0] ** 2
            + state.dX_src[:, 1] ** 2
            + state.dX_src[:, 2] ** 2
        ).max()
        st[7] = torch.quantile(
            torch.sqrt(
                state.dX_src[:, 0] ** 2
                + state.dX_src[:, 1] ** 2
                + state.dX_src[:, 2] ** 2
            ),
            0.90,
        ).item()

        total_loss_mean = (sum(total_loss_vals) / len(total_loss_vals)) if total_loss_vals else 0.0
        stats_cpu = st.detach().cpu().numpy()
        
        # Log metrics to wandb if enabled
        if wandb_logger:
            metrics = extract_metrics_from_stats_tensor(st)
            metrics.update({
                "loss": total_loss_mean,
                "mad_p": mad_p.item(),
                "mad_s": mad_s.item(),
                "epoch_time": epoch_time,
                "learning_rate": state.optimizer.param_groups[0]['lr']
            })
            wandb_logger.log_phase1_metrics(epoch, metrics)
        
        print(
            f"it {epoch}",
            f"L: {total_loss_mean:.4e}",
            f"|ΔX|: {stats_cpu[3]:.3e}",
            f"|ΔY|: {stats_cpu[4]:.3e}",
            f"|ΔZ|: {stats_cpu[5]:.3e}",
            f"|ΔR|_max: {stats_cpu[6]:.3e}",
            f"|ΔR|_90: {stats_cpu[7]:.3e}",
            f"MAD_p: {mad_p.item():.3f}",
            f"MAD_s: {mad_s.item():.3f}",
            f"time: {epoch_time:.1f}s",
        )

        # periodic checkpoint for MAP phase
        if checkpoint_interval > 0 and epoch > 0 and (epoch % checkpoint_interval == 0):
            save_checkpoint(
                state.params,
                state.optimizer,
                epoch,
                state.N,
                state.dX_src,
                [],
                state.stats_tensor,
                phase="phase1",
                global_step_count=state.global_step_count,
            )

    if state.params["phase1_epochs"] > 0:
        X_src1 = (state.X_src + state.dX_src).detach().cpu().numpy()
        origins = write_output(
            state.origins0, X_src1, state.X_src.detach().cpu().numpy(), state.projector
        )
        origins.write_csv(f"{state.params['catalog_outfile']}_MAP.csv")
        # final checkpoint for MAP phase
        save_checkpoint(
            state.params,
            state.optimizer,
            state.params["phase1_epochs"] - 1,
            state.N,
            state.dX_src,
            [],
            state.stats_tensor,
            phase="phase1",
            global_step_count=state.global_step_count,
        )


def _setup_sampler(state: LocateState) -> torch.optim.Optimizer:
    # Choose sampler based on parameter setting
    sgld_alg = "psgld"

    if sgld_alg.lower() == "psgld":
        # Use pSGLD (preconditioned SGLD)
        from spider.optim.sgld import pSGLD
        sampler = pSGLD(
            params=[state.dX_src],
            n_obs=state.N,
            lr=float(state.params.get("lr_sgld", 1e-3)),
            beta=float(state.params.get("sgld_beta", 0.99)),
            eps=float(state.params.get("sgld_eps", 1e-5)),
            preconditioning=bool(state.params.get("sgld_preconditioning", True)),
            preconditioner=state.params.get("sgld_preconditioner", "rmsprop"),
            add_noise=False,  # no noise in phases 2; enabled in phase 3
        )
        print("Using pSGLD sampler")
    else:
        raise ValueError(f"Unknown SGLD algorithm: {sgld_alg}. Supported values: 'psgld'")
    
    state.sampler = sampler
    return sampler


def _resume_or_initialize(state: LocateState):
    """Handle checkpoint reset/resume.

    Returns a tuple: (phase, start_epoch, skip_saving_first_epoch, ckpt_data)
    where phase is one of {"phase1","phase2","phase3","phase4"}.
    start_epoch applies to phase1 and phase4 epochs; others ignore it.
    ckpt_data is the raw checkpoint dict if resuming, else None.
    """
    reset_batch_numbers = state.params.get("reset_batch_numbers", False)
    clear_samples_on_reset = state.params.get("clear_samples_on_reset", False)
    
    if reset_batch_numbers:
        print("Resetting batch numbers to 0 (reset_batch_numbers=True)")
        clear_checkpoint_files(state.params)
        if clear_samples_on_reset:
            clear_samples_file(state.params)
        state.sample_count = 0
        return "phase1", 0, False, None

    ckpt = load_checkpoint(state.params, state.device)
    
    # Check if we should clear samples even when resuming from checkpoint
    if clear_samples_on_reset and not reset_batch_numbers:
        print("Clearing samples file while resuming from checkpoint (clear_samples_on_reset=True)")
        clear_samples_file(state.params)
        
    if ckpt is None:
        # fresh run
        state.sample_count = get_next_sample_count(state.params)
        if state.sample_count > 0:
            print(f"Continuing from batch number {state.sample_count} (existing samples file found)")
        else:
            print("Starting with batch number 0 (no existing samples file)")
        return "phase1", 0, False, None

    # adopt tensors and stats from checkpoint
    state.dX_src = ckpt["ΔX_src"]  # type: ignore[assignment]
    # Ensure the optimizer (used in phase1) points at the resumed tensor
    try:
        state.optimizer.param_groups[0]['params'][0] = state.dX_src  # type: ignore[index]
    except Exception as e:
        print(f"Warning: could not reset optimizer parameter reference: {e}")
    state.stats_tensor = ckpt.get("stats_tensor", state.stats_tensor)
    state.samples = []
    state.global_step_count = ckpt.get("global_step_count", 0)
    state.sample_count = get_next_sample_count(state.params)

    # If clearing samples but not resetting batch numbers, start from phase 2
    if clear_samples_on_reset and not reset_batch_numbers:
        print("Starting fresh from phase 2 (as if phase 1 just finished)")
        phase = "phase2"
        start_epoch = 0
    else:
        phase = ckpt.get("phase", "phase4")
        start_epoch = ckpt.get("epoch", 0)

    # Backward compatibility / normalization: handle legacy and spaced names
    phase_str = str(phase).strip().lower().replace(" ", "")
    if phase_str == "map":
        phase = "phase1"
    elif phase_str in {"phase1", "phase2", "phase3", "phase4"}:
        phase = phase_str
    else:
        # Unknown label → default conservatively to phase4 sampling
        print(f"Warning: unknown checkpoint phase label '{phase}', defaulting to 'phase4'")
        phase = "phase4"

    # For phase1, we need to load optimizer state here
    if phase == "phase1":
        try:
            state.optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
            # One-time fix: override LR with current JSON so restarts pick up changes
            if "lr_warmup" in state.params:
                for g in state.optimizer.param_groups:
                    g['lr'] = float(state.params.get("lr_warmup", g.get('lr', 0.0)))
        except Exception as e:
            print(f"Warning: could not load optimizer state: {e}")

    # For SGLD phases we will create sampler later and load then
    skip_saving_first_epoch = True
    print(f"Resuming from phase '{phase}', epoch {start_epoch}")
    return phase, start_epoch, skip_saving_first_epoch, ckpt


def _phase2_preconditioner(state: LocateState, wandb_logger=None) -> None:
    assert state.sampler is not None
    sampler = state.sampler
    # Ensure deterministic updates (no noise)
    for g in sampler.param_groups:
        if 'add_noise' in g:
            g['add_noise'] = False
        if 'noise_scale' in g:
            g['noise_scale'] = 0.0

    print("Phase 2: deterministic preconditioned drift (no noise)")
    for epoch in range(state.params["phase2_epochs"]):
        state.begin_epoch_rr(seed=epoch)
        epoch_start_time = time.time()
        total_loss_vals: List[float] = []
        for j in range(0, state.N // state.batch_size_sgld + 1):
            i_start = j * state.batch_size_sgld
            i_end = min(i_start + state.batch_size_sgld, state.N)

            sampler.zero_grad(set_to_none=True)
            loss = posterior_loss(
                state.II_epoch[i_start:i_end, :],
                state.YY_epoch[i_start:i_end],
                state.X_src,
                state.dX_src,
                state.model,
                state.prior_event,
                state.prior_centroid,
                state.scale_theta[0],
                state.scale_theta[1],
                state.N,
                normalize=True,
            )
            loss.backward()
            sampler.step()
            total_loss_vals.append(float(loss.item()))
            state.global_step_count += 1

        # --- per-epoch summary (like phase 3 style) ---
        with torch.no_grad():
            if epoch % 10 == 0 or epoch == state.params["phase2_epochs"] - 1:
                mad_p, mad_s = _compute_phase_mads(state, state.batch_size_sgld)
            else:
                mad_p = torch.tensor(0.0, device=state.device)
                mad_s = torch.tensor(0.0, device=state.device)

            total_loss_mean = (sum(total_loss_vals) / len(total_loss_vals)) if total_loss_vals else 0.0

        # update stats tensor
        st = state.stats_tensor
        st[0] = state.dX_src[:, 0].mean()
        st[1] = state.dX_src[:, 1].mean()
        st[2] = state.dX_src[:, 2].mean()
        st[3] = torch.abs(state.dX_src[:, 0]).median()
        st[4] = torch.abs(state.dX_src[:, 1]).median()
        st[5] = torch.abs(state.dX_src[:, 2]).median()
        st[6] = torch.sqrt(
            state.dX_src[:, 0] ** 2 +
            state.dX_src[:, 1] ** 2 +
            state.dX_src[:, 2] ** 2
        ).max()
        st[7] = torch.quantile(
            torch.sqrt(
                state.dX_src[:, 0] ** 2 +
                state.dX_src[:, 1] ** 2 +
                state.dX_src[:, 2] ** 2
            ), 0.90,
        ).item()

        stats_cpu = st.detach().cpu().numpy()
        epoch_time = time.time() - epoch_start_time

        # Log metrics to wandb if enabled
        if wandb_logger:
            metrics = extract_metrics_from_stats_tensor(st)
            metrics.update({
                "loss": total_loss_mean,
                "mad_p": mad_p.item(),
                "mad_s": mad_s.item(),
                "epoch_time": epoch_time,
                "learning_rate": sampler.param_groups[0]['lr'],
                "noise_enabled": int(bool(sampler.param_groups[0].get('add_noise', sampler.param_groups[0].get('noise_scale', 0.0) > 0.0)))
            })
            wandb_logger.log_phase2_metrics(epoch, metrics)

        print(
            f"phase2 {epoch+1}/{state.params['phase2_epochs']} "
            f"L: {total_loss_mean:.4e} |ΔX|: {stats_cpu[3]:.3e} |ΔY|: {stats_cpu[4]:.3e} |ΔZ|: {stats_cpu[5]:.3e} "
            f"|ΔR|_max: {stats_cpu[6]:.3e} |ΔR|_90: {stats_cpu[7]:.3e} "
            f"MAD_p: {mad_p.item():.3f} MAD_s: {mad_s.item():.3f} time: {epoch_time:.1f}s "
        )

    # Save checkpoint at end of phase 2
    save_checkpoint(
        state.params,
        state.sampler,  # type: ignore[arg-type]
        epoch=0,
        N=state.N,
        ΔX_src=state.dX_src,
        samples=[],
        stats_tensor=state.stats_tensor,
        phase="phase2",
        global_step_count=state.global_step_count,
    )


def _set_lr_from_state(sampler: torch.optim.Optimizer) -> None:
    # No-op for simplified SGLD; lr is managed directly.
    return


def _phase3_noise_ramp(state: LocateState, wandb_logger=None) -> None:
    assert state.sampler is not None
    sampler = state.sampler
    print("Phase 3: warm up noise (lr constant)")
    ramp_len = int(state.params.get("phase3_epochs", 500))

    # Set constant learning rate
    base_lr = float(state.params.get("lr_sgld", 1e-3))
    sampler.set_lr(base_lr)

    for t in range(ramp_len):
        state.begin_epoch_rr(seed=t)
        epoch_start_time = time.time()
        total_loss_vals: List[float] = []

        for j in range(0, state.N // state.batch_size_sgld + 1):
            # linear ramp of noise scale over phase-3 iterations (not cumulative steps)
            progress = float(min(1.0, (t + 1) / float(ramp_len)))
            for g in sampler.param_groups:
                g['add_noise'] = True
                g['noise_scale'] = progress

            i_start = j * state.batch_size_sgld
            i_end = min(i_start + state.batch_size_sgld, state.N)

            sampler.zero_grad(set_to_none=True)
            loss = posterior_loss(
                state.II_epoch[i_start:i_end, :],
                state.YY_epoch[i_start:i_end],
                state.X_src,
                state.dX_src,
                state.model,
                state.prior_event,
                state.prior_centroid,
                state.scale_theta[0],
                state.scale_theta[1],
                state.N,
                normalize=True,
            )
            loss.backward()
            sampler.step()
            total_loss_vals.append(float(loss.item()))
            state.global_step_count += 1

        # --- per-iteration (ramp step) summary ---
        with torch.no_grad():
            if t % 10 == 0 or t == ramp_len - 1:
                mad_p, mad_s = _compute_phase_mads(state, state.batch_size_sgld)
            else:
                mad_p = torch.tensor(0.0, device=state.device)
                mad_s = torch.tensor(0.0, device=state.device)

            total_loss_mean = (sum(total_loss_vals) / len(total_loss_vals)) if total_loss_vals else 0.0

        # update stats tensor
        st = state.stats_tensor
        st[0] = state.dX_src[:, 0].mean()
        st[1] = state.dX_src[:, 1].mean()
        st[2] = state.dX_src[:, 2].mean()
        st[3] = torch.abs(state.dX_src[:, 0]).median()
        st[4] = torch.abs(state.dX_src[:, 1]).median()
        st[5] = torch.abs(state.dX_src[:, 2]).median()
        st[6] = torch.sqrt(
            state.dX_src[:, 0] ** 2 +
            state.dX_src[:, 1] ** 2 +
            state.dX_src[:, 2] ** 2
        ).max()
        st[7] = torch.quantile(
            torch.sqrt(
                state.dX_src[:, 0] ** 2 +
                state.dX_src[:, 1] ** 2 +
                state.dX_src[:, 2] ** 2
            ), 0.90,
        ).item()

        stats_cpu = st.detach().cpu().numpy()
        epoch_time = time.time() - epoch_start_time

        # Log metrics to wandb if enabled
        if wandb_logger:
            metrics = extract_metrics_from_stats_tensor(st)
            metrics.update({
                "loss": total_loss_mean,
                "mad_p": mad_p.item(),
                "mad_s": mad_s.item(),
                "epoch_time": epoch_time,
                "learning_rate": sampler.param_groups[0]['lr'],
                "noise_enabled": int(bool(sampler.param_groups[0].get('add_noise', sampler.param_groups[0].get('noise_scale', 1.0) > 0.0))),
                "ramp_progress": (t + 1) / ramp_len
            })
            wandb_logger.log_phase3_metrics(t, metrics)

        print(
            f"phase3 {t+1}/{ramp_len} "
            f"L: {total_loss_mean:.4e} |ΔX|: {stats_cpu[3]:.3e} |ΔY|: {stats_cpu[4]:.3e} |ΔZ|: {stats_cpu[5]:.3e} "
            f"|ΔR|_max: {stats_cpu[6]:.3e} |ΔR|_90: {stats_cpu[7]:.3e} "
            f"MAD_p: {mad_p.item():.3f} MAD_s: {mad_s.item():.3f} time: {epoch_time:.1f}s "
            f"lr: {sampler.param_groups[0]['lr']:.2e} noise_scale: {sampler.param_groups[0].get('noise_scale', 0.0):.2f}"
        )

    # Save checkpoint at end of phase 3
    save_checkpoint(
        state.params,
        state.sampler,  # type: ignore[arg-type]
        epoch=0,
        N=state.N,
        ΔX_src=state.dX_src,
        samples=[],
        stats_tensor=state.stats_tensor,
        phase="phase3",
        global_step_count=state.global_step_count,
    )


def _phase4_sampling(
    state: LocateState, start_epoch: int, skip_saving_first_epoch: bool, wandb_logger=None
) -> None:
    assert state.sampler is not None
    sampler = state.sampler
    print("Phase 4: full SGLD sampling with noise injected")
    # Track relative parameter changes over the last N and N2 epochs
    rel_window = int(state.params.get("rel_change_window", 10))
    rel_window2 = int(state.params.get("rel_change_window2", 50))
    param_snapshots = collections.deque(maxlen=max(rel_window, rel_window2) + 1)

    for epoch in range(start_epoch, state.params["phase4_epochs"]):
        state.begin_epoch_rr(seed=epoch)
        epoch_start_time = time.time()
        total_loss_vals: List[float] = []

        for j in range(0, state.N // state.batch_size_sgld + 1):
            i_start = j * state.batch_size_sgld
            i_end = min(i_start + state.batch_size_sgld, state.N)

            sampler.zero_grad(set_to_none=True)
            loss = posterior_loss(
                state.II_epoch[i_start:i_end, :],
                state.YY_epoch[i_start:i_end],
                state.X_src,
                state.dX_src,
                state.model,
                state.prior_event,
                state.prior_centroid,
                state.scale_theta[0],
                state.scale_theta[1],
                state.N,
                normalize=True,
            )
            loss.backward()

            sampler.step()

            with torch.no_grad():
                total_loss_vals.append(float(loss.item()))
                if state.global_step_count % state.params["save_every_n"] == 0:
                    state.samples.append(state.dX_src.detach().cpu().clone())
            state.global_step_count += 1

            if state.global_step_count % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- per-epoch summary ---
        with torch.no_grad():
            if epoch % 10 == 0 or epoch == state.params["phase4_epochs"] - 1:
                mad_p, mad_s = _compute_phase_mads(state, state.batch_size_sgld)
            else:
                mad_p = torch.tensor(0.0, device=state.device)
                mad_s = torch.tensor(0.0, device=state.device)

            total_loss_mean = (sum(total_loss_vals) / len(total_loss_vals)) if total_loss_vals else 0.0

        # update stats tensor
        st = state.stats_tensor
        st[0] = state.dX_src[:, 0].mean()
        st[1] = state.dX_src[:, 1].mean()
        st[2] = state.dX_src[:, 2].mean()
        st[3] = torch.abs(state.dX_src[:, 0]).median()
        st[4] = torch.abs(state.dX_src[:, 1]).median()
        st[5] = torch.abs(state.dX_src[:, 2]).median()
        st[6] = torch.sqrt(
            state.dX_src[:, 0] ** 2 +
            state.dX_src[:, 1] ** 2 +
            state.dX_src[:, 2] ** 2
        ).max()
        st[7] = torch.quantile(
            torch.sqrt(
                state.dX_src[:, 0] ** 2 +
                state.dX_src[:, 1] ** 2 +
                state.dX_src[:, 2] ** 2
            ), 0.90,
        ).item()

        stats_cpu = st.detach().cpu().numpy()
        epoch_time = time.time() - epoch_start_time

        # one compact epoch line + relative change over windows
        # Maintain snapshot queue and compute relative changes for N and N2
        dX_cpu = state.dX_src.detach().cpu()
        param_snapshots.append(dX_cpu)
        rel_change_N = float('nan')
        rel_change_N2 = float('nan')
        if rel_window > 0 and len(param_snapshots) > rel_window:
            prev1 = param_snapshots[-(rel_window + 1)]
            num1 = torch.linalg.norm(dX_cpu - prev1).item()
            den1 = torch.linalg.norm(prev1).item() + 1e-12
            rel_change_N = num1 / den1
        if rel_window2 > 0 and len(param_snapshots) > rel_window2:
            prev2 = param_snapshots[-(rel_window2 + 1)]
            num2 = torch.linalg.norm(dX_cpu - prev2).item()
            den2 = torch.linalg.norm(prev2).item() + 1e-12
            rel_change_N2 = num2 / den2

        # Log metrics to wandb if enabled
        if wandb_logger:
            metrics = extract_metrics_from_stats_tensor(st)
            metrics.update({
                "loss": total_loss_mean,
                "mad_p": mad_p.item(),
                "mad_s": mad_s.item(),
                "epoch_time": epoch_time,
                "learning_rate": sampler.param_groups[0]['lr'],
                "noise_enabled": int(bool(sampler.param_groups[0].get('add_noise', sampler.param_groups[0].get('noise_scale', 1.0) > 0.0))),
                "relative_change_N": rel_change_N,
                "relative_change_N2": rel_change_N2,
                "samples_collected": len(state.samples)
            })
            wandb_logger.log_phase4_metrics(epoch, metrics)

        print(
            f"{epoch}/{state.params['phase4_epochs']} "
            f"L: {total_loss_mean:.4e} |ΔX|: {stats_cpu[3]:.3e} |ΔY|: {stats_cpu[4]:.3e} |ΔZ|: {stats_cpu[5]:.3e} "
            f"|ΔR|_max: {stats_cpu[6]:.3e} |ΔR|_90: {stats_cpu[7]:.3e} "
            f"MAD_p: {mad_p.item():.3f} MAD_s: {mad_s.item():.3f} time: {epoch_time:.1f}s "
            f"lr: {sampler.param_groups[0]['lr']:.2e} "
            f"relΔ(N={rel_window}): {rel_change_N:.3e} relΔ(N2={rel_window2}): {rel_change_N2:.3e}"
        )

        # checkpointing / sample saves
        checkpoint_interval = state.params.get("checkpoint_interval", 50)
        if epoch % checkpoint_interval == 0 and epoch > 0 and not skip_saving_first_epoch:
            save_checkpoint(
                state.params,
                sampler,
                epoch,
                state.N,
                state.dX_src,
                state.samples,
                state.stats_tensor,
                phase="phase4",
                global_step_count=state.global_step_count,
            )
            print("Begin saving samples")
            state.sample_count = save_samples_periodic(
                state.params, state.origins0, state.X_src,
                state.samples, state.projector, state.sample_count
            )
            print("Finished saving samples")
            
            # Log sample collection metrics to wandb
            if wandb_logger:
                wandb_logger.log_sample_metrics(state.sample_count, {
                    "samples_saved": len(state.samples),
                    "total_samples_collected": state.sample_count
                })
            
            state.sample_count += 1
            state.samples = []

        if skip_saving_first_epoch:
            skip_saving_first_epoch = False

    # --- final saves ---
    state.sample_count = save_samples_periodic(
        state.params, state.origins0, state.X_src, state.samples, state.projector, state.sample_count
    )
    print(f"Saving final batch of {len(state.samples)} samples...")
    state.sample_count = save_samples_periodic(
        state.params, state.origins0, state.X_src, state.samples, state.projector, state.sample_count
    )
    print(f"Final samples saved to batch_{state.sample_count}")

    epoch = state.params["phase4_epochs"] - 1
    print(f"Saving final checkpoint at epoch {epoch}...")
    save_checkpoint(state.params, sampler, epoch, state.N, state.dX_src, [], state.stats_tensor, phase="phase4", global_step_count=state.global_step_count)  # type: ignore[arg-type]
    print("Final checkpoint saved.")



def autotune_sampling_step(state: LocateState, probe_steps=1500, max_halves=4, target_snr=0.10):
    sampler = state.sampler; assert sampler is not None
    drift = LossDrift(win=2000, rel_slope=0.01)

    for attempt in range(max_halves+1):
        # short probe
        steps = 0
        for j in range(0, min(probe_steps, state.N // state.batch_size_sgld + 1)):
            i0 = j * state.batch_size_sgld
            i1 = min(i0 + state.batch_size_sgld, state.N)
            sampler.zero_grad(set_to_none=True)
            loss = posterior_loss(
                state.II[i0:i1,:], state.YY[i0:i1],
                state.X_src, state.dX_src, state.model,
                state.prior_event, state.prior_centroid,
                state.scale_theta[0], state.scale_theta[1],
                state.N, normalize=True
            )
            loss.backward()
            sampler.step()
            drift.push(float(loss))
            steps += 1

        plateau = drift.ok()
        print(f"[probe] lr={sampler.param_groups[0]['lr']:.3e} steps={steps} plateau={plateau}")

        if plateau:
            print("[probe] sampling step accepted.")
            return

        # halve and retry
        sampler.set_lr(sampler.param_groups[0]['lr'] * 0.5)
        print("[probe] halving lr and retrying…")

    print("[probe] reached max_halves; continuing with conservative lr.")

class LossDrift:
    def __init__(self, win=2000, rel_slope=0.01):
        self.buf = collections.deque(maxlen=win)
        self.rel_slope = rel_slope  # acceptable slope per 1k steps as fraction of window std
    def push(self, v: float):
        self.buf.append(float(v))
    def ok(self):
        if len(self.buf) < self.buf.maxlen: return True
        y = np.asarray(self.buf, dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        m = np.polyfit(x, y, 1)[0] * 1000.0  # slope per 1k steps
        return abs(m) <= self.rel_slope * (y.std() + 1e-12)

@torch.no_grad()
def sgld_snr(opt):
    # Deprecated with simplified SGLD; kept for API compatibility if imported elsewhere.
    return float('nan'), float('nan')

@torch.no_grad()
def sgld_drift_noise(opt):
    # Deprecated with simplified SGLD; kept for API compatibility if imported elsewhere.
    return 0.0, 0.0, 0.0, 0.0, 0.0, float('nan')


def locate_all(
    params: dict,
    origins0: pl.DataFrame,
    dtimes: pl.DataFrame,
    model: nn.Module,
    device: torch.device,
    wandb_logger=None,
) -> None:
    """Run the full SPIDER location pipeline (warmup + SGLD sampling)."""
    state = _build_initial_state(params, origins0, dtimes, model, device)
    phase, start_epoch, skip_saving_first_epoch, ckpt = _resume_or_initialize(state)

    # Phase 1: MAP estimation with Adam optimizer
    if phase == "phase1":
        if wandb_logger:
            wandb_logger.start_phase("phase1")
        _phase1_map_warmup(state, start_epoch, wandb_logger)
        phase = "phase2"

    # Set up sampler and optionally load state if resuming from SGLD phases
    sampler = _setup_sampler(state)
    if ckpt is not None and phase in {"phase2", "phase3", "phase4"}:
        try:
            sampler.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        except Exception as e:
            print(f"Warning: could not load sampler state: {e}")
        # One-time fix: enforce current JSON hyperparameters after resume
        try:
            # Normalize/migrate param group keys depending on sampler class
            from spider.optim.sgld import pSGLD as _pSGLD_cls
            if isinstance(sampler, _pSGLD_cls):
                for g in sampler.param_groups:
                    # Map SGLDAdaptiveDrift keys if present
                    if 'beta2' in g and 'beta' not in g:
                        g['beta'] = float(g['beta2'])
                    if 'precondition' in g and 'preconditioning' not in g:
                        g['preconditioning'] = bool(g['precondition'])
                    # Ensure required keys exist
                    g.setdefault('beta', float(state.params.get('sgld_beta', state.params.get('sgld_beta2', 0.99))))
                    g.setdefault('preconditioning', bool(state.params.get('sgld_preconditioning', True)))
                    # Noise mapping
                    if 'add_noise' not in g:
                        ns = float(g.get('noise_scale', 0.0))
                        g['add_noise'] = bool(ns > 0.0)
                    # Core numeric fields
                    g.setdefault('n_obs', int(state.N))
                    g.setdefault('eps', float(state.params.get('sgld_eps', 1e-5)))
                    g.setdefault('lr', float(state.params.get('lr_sgld', 1e-3)))

            # Always take lr from JSON on restart
            lr_json = float(state.params.get("lr_sgld", sampler.param_groups[0]['lr']))
            if hasattr(sampler, "set_lr"):
                sampler.set_lr(lr_json)  # type: ignore[attr-defined]
            else:
                for g in sampler.param_groups:
                    g['lr'] = lr_json
            # For phase2/phase3 restarts, ensure noise is off at entry
            if phase in {"phase2", "phase3"}:
                for g in sampler.param_groups:
                    if 'add_noise' in g:
                        g['add_noise'] = False
                    g['noise_scale'] = 0.0
        except Exception as e:
            print(f"Warning: could not apply JSON params on resume: {e}")

    # Phase 2
    if phase == "phase2":
        if wandb_logger:
            wandb_logger.start_phase("phase2")
        # Transfer preconditioning state from Adam to SGLD if coming from phase 1
        if ckpt is None or ckpt.get("phase") == "phase1":
            from spider.optim.sgld import transplant_v_from_adam
            transplant_v_from_adam(state.optimizer, sampler)
            print("Transferred preconditioning state from Adam to SGLD")
        
        _phase2_preconditioner(state, wandb_logger)
        _set_lr_from_state(sampler)
        phase = "phase3"

    # Phase 3
    if phase == "phase3":
        if wandb_logger:
            wandb_logger.start_phase("phase3")
        _phase3_noise_ramp(state, wandb_logger)
        phase = "phase4"

    # Phase 4
    if phase == "phase4":
        if wandb_logger:
            wandb_logger.start_phase("phase4")
    # Restart epoch counting at 0 when entering phase 4 from earlier phases.
    # Only preserve epoch if we are resuming directly from a phase 4 checkpoint.
    ckpt_phase = str(ckpt.get("phase", "")) if ckpt is not None else ""
    ckpt_phase_norm = ckpt_phase.strip().lower().replace(" ", "")
    resuming_direct_phase4 = (phase == "phase4" and ckpt_phase_norm == "phase4")
    sgld_start_epoch = start_epoch if resuming_direct_phase4 else 0
    _phase4_sampling(state, sgld_start_epoch, skip_saving_first_epoch, wandb_logger)
    return


