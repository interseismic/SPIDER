import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


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


"""Modeling primitives used by the location pipeline.

The locate pipeline is implemented in `spider.core.locate`. This file keeps the
low-level math and utilities (losses, residuals, output conversion) separate
from orchestration.
"""


def locate_all(params, origins0, dtimes, model, device):
    """Deprecated. Use `spider.core.locate.locate_all`.

    This proxy keeps backward compatibility for imports from modeling.
    """
    from .locate import locate_all as _locate_all
    return _locate_all(params, origins0, dtimes, model, device)
