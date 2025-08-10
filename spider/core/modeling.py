import os
import time
from pyproj import Proj
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from spider.optim.sgld import SGLDAdamPrecond, suggest_lr_from_state, print_phase2_diagnostics, Phase3Stopper, \
    phase3_step_control, sgld_heartbeat_prestep_params, snapshot_tensors, heartbeat_poststep_from, transplant_v_from_adam
from spider.io.checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint_files
from spider.io.samples import save_samples_periodic, get_next_sample_count, clear_samples_file


def compute_travel_times(idx, y, X_src, ΔX_src, model):
    """
    Compute predicted travel times for differential time pairs.

    Args:
        idx: Event pair indices [N, 2]
        y: Differential time data [N, 5] (dt, X, Y, Z, phase)
        X_src: Base source coordinates [M, 4]
        ΔX_src: Source coordinate perturbations [M, 4]
        model: Neural network model

    Returns:
        dt_pred: Predicted differential times
    """
    # Compute perturbed source positions
    X_src1 = X_src[idx[:, 0]] + ΔX_src[idx[:, 0]]
    X_src2 = X_src[idx[:, 1]] + ΔX_src[idx[:, 1]]
    X_rec = y[:, 1:]

    # Batch model evaluation for both sources
    batch_input = torch.cat([
        torch.cat((X_src1[:, :3], X_rec), dim=1),
        torch.cat((X_src2[:, :3], X_rec), dim=1)
    ], dim=0)

    # Single model forward pass
    T_pred = model(batch_input).squeeze()
    batch_size = X_src1.shape[0]
    T1_pred = T_pred[:batch_size]
    T2_pred = T_pred[batch_size:]

    # Compute differential times
    dt_pred = (T2_pred + X_src2[:, 3]) - (T1_pred + X_src1[:, 3])
    return dt_pred


def likelihood_loss(idx, y, X_src, ΔX_src, model, σ_p, σ_s):
    """
    Compute likelihood loss for differential time observations.

    Args:
        idx: Event pair indices
        y: Differential time data
        X_src: Base source coordinates
        ΔX_src: Source coordinate perturbations
        model: Neural network model
        σ_p, σ_s: Phase uncertainties

    Returns:
        Loss value
    """
    dt_obs = y[:, 0]
    dt_pred = compute_travel_times(idx, y, X_src, ΔX_src, model)

    # Phase-dependent uncertainty
    phase_mask = y[:, 4] < 0.5  # P-wave mask
    denom = torch.where(phase_mask, σ_p, σ_s)

    return F.huber_loss(dt_obs / denom, dt_pred / denom, reduction="mean")


def prior_loss_event(ΔX_src, prior_event):
    """Compute event prior loss."""
    return -prior_event.log_prob(ΔX_src).sum()


def prior_loss_centroid(ΔX_src, prior_centroid):
    """Compute global centroid prior loss."""
    global_centroid = ΔX_src.mean(dim=0)
    return -prior_centroid.log_prob(global_centroid).sum() * ΔX_src.shape[0]


def prior_loss(ΔX_src, prior_event, prior_centroid, N):
    """Compute total prior loss."""
    return prior_loss_event(ΔX_src, prior_event) / N + prior_loss_centroid(ΔX_src, prior_centroid) / N


def posterior_loss(idx, y, X_src, ΔX_src, model, prior_event, prior_centroid, σ_p, σ_s, N, normalize=False):
    """
    Compute total loss combining likelihood and prior terms.

    Args:
        idx: Event pair indices
        y: Differential time data
        X_src: Base source coordinates
        ΔX_src: Source coordinate perturbations
        model: Neural network model
        prior_event: Event prior distribution
        prior_centroid: Centroid prior distribution
        σ_p, σ_s: Phase uncertainties
        N: Number of observations
        normalize: Whether to normalize the loss

    Returns:
        Total loss value
    """
    ℓL = likelihood_loss(idx, y, X_src, ΔX_src, model, σ_p, σ_s)
    ℓp = prior_loss(ΔX_src, prior_event, prior_centroid, N)

    if normalize:
        return ℓL + ℓp
    else:
        return N * ℓL + ℓp


def compute_residuals(idx, y, X_src, ΔX_src, model):
    """
    Compute residuals between observed and predicted differential times.

    Args:
        idx: Event pair indices
        y: Differential time data
        X_src: Base source coordinates
        ΔX_src: Source coordinate perturbations
        model: Neural network model

    Returns:
        Residuals (observed - predicted)
    """
    dt_obs = y[:, 0]
    dt_pred = compute_travel_times(idx, y, X_src, ΔX_src, model)
    return dt_obs - dt_pred


def write_output(origins0, X_src1, unc_src, projector):
    """
    Write output with projected coordinates converted back to lat/lon.

    Args:
        origins0: Original origins DataFrame
        X_src1: Source coordinates in projected space
        unc_src: Source uncertainties
        projector: PyProj projector for coordinate conversion

    Returns:
        DataFrame with updated coordinates and uncertainties
    """
    origins = origins0.clone()

    # Add numeric columns from NumPy arrays
    origins = origins.with_columns([
        pl.Series("T_src", X_src1[:, 3]),
        pl.Series("depth", X_src1[:, 2]),
        pl.Series("X", X_src1[:, 0]),
        pl.Series("Y", X_src1[:, 1]),
        pl.Series("unc_x", unc_src[:, 0]),
        pl.Series("unc_y", unc_src[:, 1]),
        pl.Series("unc_z", unc_src[:, 2]),
    ])

    # Convert projected coordinates back to longitude and latitude
    lons = []
    lats = []
    for i in range(len(origins)):
        lon, lat = projector(origins["X"][i], origins["Y"][i], inverse=True)
        lons.append(lon)
        lats.append(lat)

    origins = origins.with_columns([
        pl.Series(lons).alias("longitude"),
        pl.Series(lats).alias("latitude")
    ])

    return origins


def compute_residuals_full(II, YY, X_src, ΔX_src, model, bs, N):
    """
    Compute residuals for all observations using batched processing.

    Args:
        II: Event pair indices
        YY: Differential time data
        X_src: Base source coordinates
        ΔX_src: Source coordinate perturbations
        model: Neural network model
        bs: Base batch size
        N: Number of observations

    Returns:
        Residuals for all observations
    """
    residuals = torch.zeros_like(YY[:, -1])
    with torch.no_grad():
        # Use larger batch size for residual computation to reduce overhead
        residual_batch_size = min(bs * 4, N)

        for j in range(0, N // residual_batch_size + 1):
            i_start = j * residual_batch_size
            i_end = min([i_start + residual_batch_size, N])
            residuals[i_start:i_end] = compute_residuals(
                II[i_start:i_end, :], YY[i_start:i_end], X_src, ΔX_src, model
            )
    return residuals


def med_abs_dev(x):
    """Compute median absolute deviation."""
    return np.median(np.abs(x - np.median(x)))


def med_abs_dev_torch(x):
    """Compute median absolute deviation using PyTorch."""
    return torch.median(torch.abs(x - torch.median(x)))


def shuffle_data(x, y):
    """Shuffle data arrays together."""
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


def locate_all(params, origins0, dtimes, model, device):
    evid_to_row = {}

    count = 0
    for row in origins0.iter_rows(named=True):
        evid_to_row[row['evid']] = count
        count += 1

    # Create evid1_idx and evid2_idx columns
    evid1_idx = [evid_to_row[x] for x in dtimes['evid1']]
    evid2_idx = [evid_to_row[x] for x in dtimes['evid2']]
    dtimes = dtimes.with_columns([
        pl.Series(evid1_idx).alias("evid1_idx"),
        pl.Series(evid2_idx).alias("evid2_idx"),
        pl.Series(np.arange(dtimes.shape[0])).alias("arid"),
    ])

    projector = Proj(proj='laea', lat_0=params["lat_min"], lon_0=params["lon_min"], datum='WGS84', units='km')
    XX, YY = projector(origins0["longitude"].to_numpy(), origins0["latitude"].to_numpy())
    X_src = torch.zeros(origins0.shape[0], 4, dtype=torch.float32, device=device)
    X_src[:,0] = torch.tensor(XX, dtype=torch.float32, device=device)
    X_src[:,1] = torch.tensor(YY, dtype=torch.float32, device=device)
    X_src[:,2] = torch.tensor(origins0["depth"].to_numpy(), dtype=torch.float32, device=device)
    ΔX_src = torch.zeros(origins0.shape[0], 4, dtype=torch.float32, device=device)
    ΔX_src.requires_grad_()
    ΔX_src = torch.nn.Parameter(ΔX_src)

    # Use int64 for indexing to avoid invalid GPU indexing (int32 can corrupt memory)
    II = torch.tensor(dtimes[["evid1_idx", "evid2_idx"]].to_numpy(), dtype=torch.int64, device=device)
    YY = torch.tensor(dtimes[["dt", "X", "Y", "Z", "phase"]].to_numpy(), dtype=torch.float32, device=device)

    model = nn.DataParallel(model, device_ids=params["devices"])

    batch_size_warmup = params["batch_size_warmup"]
    batch_size_sgld = params["batch_size_sgld"]
    N = dtimes.shape[0]
    η_sgld = params["lr_sgld"]
    η_warmup = params["lr_warmup"]

    prior_event = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(len(params["prior_event_std"]), device=device, dtype=torch.float32),
                    covariance_matrix=torch.diag(torch.tensor(params["prior_event_std"], device=device, dtype=torch.float32)**2))
    prior_centroid = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(len(params["prior_centroid_std"]), device=device, dtype=torch.float32),
                    covariance_matrix=torch.diag(torch.tensor(params["prior_centroid_std"], device=device, dtype=torch.float32)**2))

    optimizer = torch.optim.Adam([ΔX_src], lr=η_warmup)
    scale_θ = torch.tensor(params["phase_unc"], device=device, dtype=torch.float32)

    print("Phase 1: MAP estimation with Adam optimizer (noise free)")
    stats_tensor = torch.zeros(8, device=device)

    for epoch in range(params["n_warmup_epochs"]):
        epoch_start_time = time.time()
        total_loss = []
        for j in range(0, N // batch_size_warmup + 1):
            i_start = j * batch_size_warmup
            i_end = min([i_start + batch_size_warmup, N])

            optimizer.zero_grad()
            loss = posterior_loss(
                II[i_start:i_end,:],
                YY[i_start:i_end],
                X_src,
                ΔX_src,
                model,
                prior_event,
                prior_centroid,
                scale_θ[0],
                scale_θ[1],
                N,
                normalize=True
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss.append(loss.item())

        total_loss = torch.tensor(total_loss)

        with torch.no_grad():
            if epoch % 100 == 0:
                X_src1 = (X_src + ΔX_src).cpu().detach().numpy()
                origins = write_output(origins0, X_src1, X_src.cpu().detach().numpy(), projector)
                origins.write_csv("{}_MAP.csv".format(params["catalog_outfile"]))
                X_src1 = None

            if epoch % 10 == 0 or epoch == params["n_warmup_epochs"] - 1:
                residuals = compute_residuals_full(II, YY, X_src, ΔX_src, model, batch_size_warmup, N)
                idx_p = torch.nonzero(YY[:,4] < 0.5)
                mad_p = med_abs_dev_torch(residuals[idx_p])
                idx_s = torch.nonzero(YY[:,4] > 0.5)
                mad_s = med_abs_dev_torch(residuals[idx_s])
            else:
                mad_p = torch.tensor(0.0, device=device)
                mad_s = torch.tensor(0.0, device=device)

        epoch_time = time.time() - epoch_start_time

        stats_tensor[0] = ΔX_src[:,0].mean()
        stats_tensor[1] = ΔX_src[:,1].mean()
        stats_tensor[2] = ΔX_src[:,2].mean()
        stats_tensor[3] = torch.abs(ΔX_src[:,0]).median()
        stats_tensor[4] = torch.abs(ΔX_src[:,1]).median()
        stats_tensor[5] = torch.abs(ΔX_src[:,2]).median()
        stats_tensor[6] = torch.sqrt(ΔX_src[:,0]**2 + ΔX_src[:,1]**2 + ΔX_src[:,2]**2).max()
        stats_tensor[7] = torch.quantile(torch.sqrt(ΔX_src[:,0]**2 + ΔX_src[:,1]**2 + ΔX_src[:,2]**2), 0.90).item()

        stats_cpu = stats_tensor.cpu().detach().numpy()
        print("it %d" % epoch,
              "L: %.4e" % total_loss.mean(),
              "|ΔX|: %.3e" % stats_cpu[3],
              "|ΔY|: %.3e" % stats_cpu[4],
              "|ΔZ|: %.3e" % stats_cpu[5],
              "|ΔR|_max: %.3e" % stats_cpu[6],
              "|ΔR|_90: %.3e" % stats_cpu[7],
              "MAD_p: %.3f" % mad_p.item(),
              "MAD_s: %.3f" % mad_s.item(),
              "time: %.1fs" % epoch_time)

    if params["n_warmup_epochs"] > 0:
        X_src1 = (X_src + ΔX_src).cpu().detach().numpy()
        origins = write_output(origins0, X_src1, X_src.cpu().detach().numpy(), projector)
        origins.write_csv("{}_MAP.csv".format(params["catalog_outfile"]))
        X_src1 = None

    # Initialize sampling variables
    count = 0
    # sampler = SGLD([ΔX_src], N, lr=η_sgld, preconditioning=True)
    sampler = SGLDAdamPrecond(
        [ΔX_src],
        lr=η_sgld,
        beta2=0.999,
        eps=1e-5,
        freeze_preconditioner=False,     # adapt v_t during burn-in
        normalized_loss=True, n_data=N,
        mode='scale_grad',               # fallback: 'scale_lr'
        max_drift_norm=None,             # e.g., 50.0 if needed
        max_rel_step=None,               # e.g., 1e-3 if needed
    )
    for g in sampler.param_groups:
        g['mode'] = 'scale_lr'        # drift uses g_raw, optimizer multiplies step by N internally
        g['normalized_loss'] = True   # (as you have)
        g['n_data'] = N

    transplant_v_from_adam(optimizer, sampler)
    samples = []
    sample_count = 0

    # Pre-compute phase indices for faster residual statistics
    idx_p = torch.nonzero(YY[:,4] < 0.5).squeeze()
    idx_s = torch.nonzero(YY[:,4] > 0.5).squeeze()

    # Pre-allocate tensors for statistics to avoid repeated allocations
    stats_tensor = torch.zeros(8, device=device)  # [mean_dx, mean_dy, mean_dz, mean_abs_dx, mean_abs_dy, mean_abs_dz, max_resid]

    # Check if we should reset batch numbers before loading checkpoint
    reset_batch_numbers = params.get("reset_batch_numbers", False)
    if reset_batch_numbers:
        print("Resetting batch numbers to 0 (reset_batch_numbers=True)")
        # Clear checkpoint files when resetting batch numbers
        clear_checkpoint_files(params)
        # Optionally clear the samples file when resetting
        clear_samples_on_reset = params.get("clear_samples_on_reset", False)
        if clear_samples_on_reset:
            clear_samples_file(params)
        # Force fresh start by setting checkpoint to None
        start_epoch = 0
        checkpoint_ΔX_src = None
        checkpoint_samples = None
        checkpoint_stats = None
        sample_count = 0
    else:
        # Check for existing checkpoint and resume if available
        start_epoch, _, checkpoint_ΔX_src, checkpoint_samples, checkpoint_stats = load_checkpoint(params, sampler, device)
        if checkpoint_ΔX_src is not None:
            print(f"Resuming from epoch {start_epoch}")
            sample_count = get_next_sample_count(params)
        else:
            start_epoch = 0
            # Continue from existing batch numbers if samples file exists
            sample_count = get_next_sample_count(params)
            if sample_count > 0:
                print(f"Continuing from batch number {sample_count} (existing samples file found)")
            else:
                print("Starting with batch number 0 (no existing samples file)")

    # Apply checkpoint data if available
    resumed_from_checkpoint = checkpoint_ΔX_src is not None
    if resumed_from_checkpoint:
        ΔX_src = checkpoint_ΔX_src
        # Discard any in-memory samples from the checkpoint so counts start at 0 post-resume
        samples = []
        stats_tensor = checkpoint_stats if checkpoint_stats is not None else stats_tensor

    skip_saving_first_epoch = resumed_from_checkpoint

    sampler.set_adapt_only(False)
    sampler.set_noise_scale(0.0)
    for g in sampler.param_groups:
        g['max_drift_norm'] = 50.0
        g['max_rel_step'] = 1e-3

    print("Phase 2: deterministic preconditioned drift (still no noise)")
    count = 0
    while count < 200:
        for j in range(0, N // batch_size_sgld + 1):
            i_start = j * batch_size_sgld
            i_end = min([i_start + batch_size_sgld, N])

            sampler.zero_grad(set_to_none=True)
            loss = posterior_loss(II[i_start:i_end,:], YY[i_start:i_end], X_src, ΔX_src, model, prior_event, prior_centroid, scale_θ[0], scale_θ[1], N, normalize=True)
            loss.backward()
            print_phase2_diagnostics(sampler)
            sampler.step()
            count += 1

    eta_base, *_ = suggest_lr_from_state(sampler)
    for g in sampler.param_groups:
        g['lr'] = eta_base
        g['eps'] = 1e-4

    print("Phase 3: warm up noise 0→1")

    stopper = Phase3Stopper(window=100, tol=0.10)
    ramp_len = 500  # noise 0→1 over 500 steps

    while True:
        epoch_start_time = time.time()
        total_loss = []
        for j in range(0, N // batch_size_sgld + 1):

            noise_scale = min(1.0, (count+1)/ramp_len)
            sampler.set_noise_scale(noise_scale)

            i_start = j * batch_size_sgld
            i_end = min([i_start + batch_size_sgld, N])

            sampler.zero_grad(set_to_none=True)
            loss = posterior_loss(II[i_start:i_end,:], YY[i_start:i_end], X_src, ΔX_src, model, prior_event, prior_centroid, scale_θ[0], scale_θ[1], N, normalize=True)
            loss.backward()

            done, base_lr, info = phase3_step_control(
                sampler, step_idx=count, ramp_len=ramp_len, stopper=stopper,
                # tweak these two for more/less movement:
                u_abs=1e-5, sigma_abs=1e-3,
                verbose=True
            )

            sampler.step()
            count += 1

        if done:
            break

    sampler.freeze_preconditioner()
    sampler.set_noise_scale(1.0)
    final_lr = float(torch.median(torch.tensor(list(stopper.hist))))
    for g in sampler.param_groups: g['lr'] = final_lr

    print("Phase 4: full SGLD sampling with frozen preconditioner")
    for epoch in range(start_epoch, params["n_sgld_epochs"]):
        epoch_start_time = time.time()
        total_loss = []

        for j in range(0, N // batch_size_sgld + 1):
            i_start = j * batch_size_sgld
            i_end = min([i_start + batch_size_sgld, N])

            sampler.zero_grad(set_to_none=True)
            loss = posterior_loss(II[i_start:i_end,:], YY[i_start:i_end], X_src, ΔX_src, model, prior_event, prior_centroid, scale_θ[0], scale_θ[1], N, normalize=True)
            loss.backward()

            # prev = snapshot_tensors([ΔX_src])
            # sgld_heartbeat_prestep_params([ΔX_src], sampler, loss_value=loss.item(), phase="auto")

            # STEP
            sampler.step()

            # AFTER step
            # heartbeat_poststep_from(prev, [ΔX_src])

            with torch.no_grad():
                total_loss.append(loss.item())
                if count % params["save_every_n"] == 0:
                    # Fix: Use clone() to ensure we have a clean copy
                    samples.append(ΔX_src.cpu().detach().clone())
            count += 1

            # Periodic memory cleanup
            if count % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Compute statistics efficiently on GPU
        with torch.no_grad():
            # Only compute residuals every few epochs to save computation
            if epoch % 10 == 0 or epoch == params["n_sgld_epochs"] - 1:
                residuals = compute_residuals_full(II, YY, X_src, ΔX_src, model, batch_size_sgld, N)
                idx_p = torch.nonzero(YY[:,4] < 0.5).squeeze()
                idx_s = torch.nonzero(YY[:,4] > 0.5).squeeze()
                mad_p = med_abs_dev_torch(residuals[idx_p])
                mad_s = med_abs_dev_torch(residuals[idx_s])
            else:
                mad_p = torch.tensor(0.0, device=device)
                mad_s = torch.tensor(0.0, device=device)

            # Single CPU transfer for all statistics
            stats_cpu = stats_tensor.cpu().numpy()
            total_loss_mean = torch.tensor(total_loss).mean().item()
        total_loss = torch.tensor(total_loss)

        epoch_time = time.time() - epoch_start_time

        # Update stats tensor with parameters
        stats_tensor[0] = ΔX_src[:,0].mean()
        stats_tensor[1] = ΔX_src[:,1].mean()
        stats_tensor[2] = ΔX_src[:,2].mean()
        stats_tensor[3] = torch.abs(ΔX_src[:,0]).median()
        stats_tensor[4] = torch.abs(ΔX_src[:,1]).median()
        stats_tensor[5] = torch.abs(ΔX_src[:,2]).median()
        stats_tensor[6] = torch.sqrt(ΔX_src[:,0]**2 + ΔX_src[:,1]**2 + ΔX_src[:,2]**2).max()
        stats_tensor[7] = torch.quantile(torch.sqrt(ΔX_src[:,0]**2 + ΔX_src[:,1]**2 + ΔX_src[:,2]**2), 0.90).item()

        # Single CPU transfer for all statistics
        stats_cpu = stats_tensor.cpu().detach().numpy()

        print("iter %d samp=%d N=%d" % (epoch, len(samples), N),
              "L: %.4e" % total_loss_mean,
              "|ΔX|: %.3e" % stats_cpu[3],
              "|ΔY|: %.3e" % stats_cpu[4],
              "|ΔZ|: %.3e" % stats_cpu[5],
              "|ΔR|_max: %.3e" % stats_cpu[6],
              "|ΔR|_90: %.3e" % stats_cpu[7],
              "MAD_p: %.3f" % mad_p.item(),
              "MAD_s: %.3f" % mad_s.item(),
              "time: %.1fs" % epoch_time)

        # Save checkpoint and samples periodically
        checkpoint_interval = params.get("checkpoint_interval", 50)
        if epoch % checkpoint_interval == 0 and epoch > 0 and not skip_saving_first_epoch:
            # Save checkpoint
            save_checkpoint(params, sampler, epoch, N, ΔX_src, samples, stats_tensor)
            print("Begin saving samples")
            sample_count = save_samples_periodic(params, origins0, X_src, samples, projector, sample_count)
            print("Finished saving samples")
            sample_count += 1
            samples = []

        # After first epoch post-resume, allow saving again
        if skip_saving_first_epoch:
            skip_saving_first_epoch = False

    sample_count = save_samples_periodic(params, origins0, X_src, samples, projector, sample_count)

    # Save any remaining samples if we have enough
    print(f"Saving final batch of {len(samples)} samples...")
    sample_count = save_samples_periodic(params, origins0, X_src, samples, projector, sample_count)
    print(f"Final samples saved to batch_{sample_count}")

    # Save final checkpoint for potential resumption
    print(f"Saving final checkpoint at epoch {epoch}...")
    save_checkpoint(params, sampler, epoch, N, ΔX_src, [], stats_tensor)  # Empty samples list since they're already saved
    print(f"Final checkpoint saved. You can resume from epoch {epoch} by running again with the same parameters.")

    return
