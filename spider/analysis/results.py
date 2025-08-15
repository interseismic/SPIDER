from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


AllowedFields = Set[str]


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _choose_device(explicit_device: Optional[str]) -> torch.device:
    if explicit_device is not None:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EventSamplesSummary:
    """
    Container for per-event summary and centered samples.

    Fields are optional and computed on demand based on `include`.
    - lats, lons, deps: numpy arrays of shape (n_events,)
    - X, Y, Z: centered numpy arrays of shape (n_events, n_samples_after_burnin)
    - cat_dd: pandas DataFrame with per-event means, spreads, and evid
    """

    lats: Optional[np.ndarray] = None
    lons: Optional[np.ndarray] = None
    deps: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None
    cat_dd: Optional[pd.DataFrame] = None

    @staticmethod
    def compute(
        event_samples: Dict[str, Any],
        burn_in: int = 0,
        include: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        events_chunk_size: int = 2048,
        dtype: torch.dtype = torch.float32,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        thin: int = 1,
    ) -> "EventSamplesSummary":
        """Compute EventSamplesSummary with optional burn-in and thinning.
        
        Args:
            event_samples: Dictionary containing sample data
            burn_in: Number of initial samples to discard per event
            include: Subset of fields to compute
            device: Override device for computation
            events_chunk_size: Number of events to process in each chunk
            dtype: Data type for torch tensors
            show_progress: Whether to show progress bar
            progress_desc: Description for progress bar
            thin: Thinning factor - keep every nth sample after burn-in (default: 1 = no thinning)
        """
        # Determine which outputs to compute
        if include is None:
            include_set: AllowedFields = {"lats", "lons", "deps", "X", "Y", "Z", "cat_dd"}
        else:
            include_set = set(include)

        if thin < 1:
            raise ValueError(f"thin must be >= 1, got {thin}")

        target_device = _choose_device(device)

        # Determine shapes and allocate outputs lazily to minimize peak memory
        # Interpret burn_in as number of initial SAMPLES to discard per event (not events)
        lat_src = event_samples.get("latitude")
        n_events_total = (_as_numpy(lat_src)).shape[0]
        start_event = 0
        n_events = n_events_total

        # Resolve total number of samples and apply sample burn-in
        # We infer n_samples_total from whichever coord is available among X/Y/Z
        sample_source = None
        for key in ("X", "Y", "Z", "latitude", "longitude", "depth"):
            if key in event_samples and event_samples[key] is not None:
                sample_source = key
                break
        if sample_source is None:
            raise KeyError("Event samples must include at least one of 'X','Y','Z' to determine sample dimension")
        n_samples_total = _as_numpy(event_samples[sample_source]).shape[1]
        start_sample = int(burn_in)
        if start_sample < 0:
            raise ValueError(f"burn_in must be >= 0 (got {start_sample})")
        if start_sample >= n_samples_total:
            raise ValueError(f"burn_in={start_sample} >= number of samples per event ({n_samples_total}). Nothing left to summarize.")
        
        # Calculate samples after burn-in and thinning
        n_samples_after_burnin = n_samples_total - start_sample
        n_samples_after = (n_samples_after_burnin + thin - 1) // thin  # Ceiling division

        # Identify needs
        need_lats = ("lats" in include_set) or ("cat_dd" in include_set)
        need_lons = ("lons" in include_set) or ("cat_dd" in include_set)
        need_deps = ("deps" in include_set) or ("cat_dd" in include_set)
        need_X = ("X" in include_set) or ("cat_dd" in include_set)
        need_Y = ("Y" in include_set) or ("cat_dd" in include_set)
        need_Z = ("Z" in include_set) or ("cat_dd" in include_set)

        # Preallocate CPU outputs
        out = EventSamplesSummary()
        if need_lats:
            lats_all = np.empty((n_events,), dtype=np.float32)
        else:
            lats_all = None
        if need_lons:
            lons_all = np.empty((n_events,), dtype=np.float32)
        else:
            lons_all = None
        if need_deps:
            deps_all = np.empty((n_events,), dtype=np.float32)
        else:
            deps_all = None

        # Determine second dimension (samples) for X/Y/Z to preallocate centered arrays if requested
        n_samples = None
        if need_X:
            n_samples = n_samples_after
            X_centered_all = np.empty((n_events, n_samples_after), dtype=np.float32) if "X" in include_set else None
            X_mean_all = np.empty((n_events,), dtype=np.float32)
            sigma_x_all = np.empty((n_events,), dtype=np.float32) if "cat_dd" in include_set else None
        else:
            X_centered_all = X_mean_all = sigma_x_all = None
        if need_Y:
            if n_samples is None:
                n_samples = n_samples_after
            Y_centered_all = np.empty((n_events, n_samples_after), dtype=np.float32) if "Y" in include_set else None
            Y_mean_all = np.empty((n_events,), dtype=np.float32)
            sigma_y_all = np.empty((n_events,), dtype=np.float32) if "cat_dd" in include_set else None
        else:
            Y_centered_all = Y_mean_all = sigma_y_all = None
        if need_Z:
            if n_samples is None:
                n_samples = n_samples_after
            Z_centered_all = np.empty((n_events, n_samples_after), dtype=np.float32) if "Z" in include_set else None
            Z_mean_all = np.empty((n_events,), dtype=np.float32)
            sigma_z_all = np.empty((n_events,), dtype=np.float32) if "cat_dd" in include_set else None
        else:
            Z_centered_all = Z_mean_all = sigma_z_all = None

        # Helper to fetch chunk to GPU tensor (without keeping large tensors resident)
        def get_gpu_chunk(src: Any, s: int, e: int, apply_burn_in: bool = True, apply_thinning: bool = True) -> torch.Tensor:
            if src is None:
                return None  # type: ignore[return-value]
            if isinstance(src, torch.Tensor):
                chunk = src[s:e]
                if apply_burn_in and start_sample > 0:
                    chunk = chunk[:, start_sample:]
                if apply_thinning and thin > 1:
                    chunk = chunk[:, ::thin]
                return chunk.to(target_device, non_blocking=True).to(dtype)
            arr = _as_numpy(src)
            chunk = arr[s:e]
            if apply_burn_in and start_sample > 0:
                chunk = chunk[:, start_sample:]
            if apply_thinning and thin > 1:
                chunk = chunk[:, ::thin]
            return torch.as_tensor(chunk, device=target_device, dtype=dtype)

        # Iterate over event chunks
        with torch.no_grad():
            iterator = range(0, n_events, events_chunk_size)
            if show_progress:
                total_chunks = (n_events + events_chunk_size - 1) // events_chunk_size
                iterator = tqdm(iterator, total=total_chunks, desc=progress_desc or "Computing summaries", leave=False)

            for out_start in iterator:
                ev_start = start_event + out_start
                ev_end = min(start_event + out_start + events_chunk_size, start_event + n_events)
                out_end = out_start + (ev_end - ev_start)

                # Location fields
                if need_lats:
                    lat_chunk = get_gpu_chunk(event_samples["latitude"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                    lats_chunk = lat_chunk.mean(dim=1)
                    lats_all[out_start:out_end] = lats_chunk.detach().cpu().numpy()
                    del lat_chunk, lats_chunk
                if need_lons:
                    lon_chunk = get_gpu_chunk(event_samples["longitude"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                    lons_chunk = lon_chunk.mean(dim=1)
                    lons_all[out_start:out_end] = lons_chunk.detach().cpu().numpy()
                    del lon_chunk, lons_chunk
                if need_deps:
                    dep_chunk = get_gpu_chunk(event_samples["depth"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                    deps_chunk = dep_chunk.mean(dim=1)
                    deps_all[out_start:out_end] = deps_chunk.detach().cpu().numpy()
                    del dep_chunk, deps_chunk

                # X/Y/Z - need thinning for centered arrays
                if need_X:
                    X_chunk = get_gpu_chunk(event_samples["X"], ev_start, ev_end, apply_burn_in=True, apply_thinning=True)
                    X_mean_chunk = X_chunk.mean(dim=1)
                    X_mean_all[out_start:out_end] = X_mean_chunk.detach().cpu().numpy()
                    if "X" in include_set:
                        centered = X_chunk - X_mean_chunk[:, None]
                        X_centered_all[out_start:out_end, :] = centered.detach().cpu().numpy()
                    if "cat_dd" in include_set:
                        q_hi = torch.quantile(X_chunk, 0.995, dim=1)
                        q_lo = torch.quantile(X_chunk, 0.005, dim=1)
                        sigma_x_all[out_start:out_end] = (0.5 * (q_hi - q_lo)).detach().cpu().numpy()
                    del X_chunk, X_mean_chunk
                if need_Y:
                    Y_chunk = get_gpu_chunk(event_samples["Y"], ev_start, ev_end, apply_burn_in=True, apply_thinning=True)
                    Y_mean_chunk = Y_chunk.mean(dim=1)
                    Y_mean_all[out_start:out_end] = Y_mean_chunk.detach().cpu().numpy()
                    if "Y" in include_set:
                        centered = Y_chunk - Y_mean_chunk[:, None]
                        Y_centered_all[out_start:out_end, :] = centered.detach().cpu().numpy()
                    if "cat_dd" in include_set:
                        q_hi = torch.quantile(Y_chunk, 0.995, dim=1)
                        q_lo = torch.quantile(Y_chunk, 0.005, dim=1)
                        sigma_y_all[out_start:out_end] = (0.5 * (q_hi - q_lo)).detach().cpu().numpy()
                    del Y_chunk, Y_mean_chunk
                if need_Z:
                    Z_chunk = get_gpu_chunk(event_samples["Z"], ev_start, ev_end, apply_burn_in=True, apply_thinning=True)
                    Z_mean_chunk = Z_chunk.mean(dim=1)
                    Z_mean_all[out_start:out_end] = Z_mean_chunk.detach().cpu().numpy()
                    if "Z" in include_set:
                        centered = Z_chunk - Z_mean_chunk[:, None]
                        Z_centered_all[out_start:out_end, :] = centered.detach().cpu().numpy()
                    if "cat_dd" in include_set:
                        q_hi = torch.quantile(Z_chunk, 0.995, dim=1)
                        q_lo = torch.quantile(Z_chunk, 0.005, dim=1)
                        sigma_z_all[out_start:out_end] = (0.5 * (q_hi - q_lo)).detach().cpu().numpy()
                    del Z_chunk, Z_mean_chunk

                torch.cuda.empty_cache() if target_device.type == "cuda" else None

        # Assign outputs
        if need_lats:
            out.lats = lats_all
        if need_lons:
            out.lons = lons_all
        if need_deps:
            out.deps = deps_all
        if need_X:
            out.X = X_centered_all
        if need_Y:
            out.Y = Y_centered_all
        if need_Z:
            out.Z = Z_centered_all

        if "cat_dd" in include_set:
            evid_src = event_samples["event_ids"]
            evid = evid_src.detach().cpu().numpy() if isinstance(evid_src, torch.Tensor) else np.asarray(evid_src)
            evid_series = pd.Series(evid)
            try:
                evid_series = evid_series.astype(int)
            except Exception:
                pass

            # Build DataFrame
            data = {}
            if need_lats:
                data["latitude"] = lats_all
            if need_lons:
                data["longitude"] = lons_all
            if need_deps:
                data["depth"] = deps_all
            if need_X:
                data["X"] = X_mean_all
                if sigma_x_all is not None:
                    data["sigma_x"] = sigma_x_all
            if need_Y:
                data["Y"] = Y_mean_all
                if sigma_y_all is not None:
                    data["sigma_y"] = sigma_y_all
            if need_Z:
                data["Z"] = Z_mean_all
                if sigma_z_all is not None:
                    data["sigma_z"] = sigma_z_all

            data["evid"] = evid_series.values[start_event:start_event + n_events]
            out.cat_dd = pd.DataFrame(data=data)

        return out


def compute_effective_sample_size(
    summary: EventSamplesSummary,
    max_lag: Optional[int] = None,
    method: str = "autocorr",
    show_progress: bool = False,
    device: Optional[str] = None,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Compute effective sample size (ESS) for each event from EventSamplesSummary.
    
    Parameters
    ----------
    summary : EventSamplesSummary
        Summary object containing centered samples (X, Y, Z)
    max_lag : int, optional
        Maximum lag for autocorrelation computation. If None, uses min(1000, n_samples//4)
    method : str, default "autocorr"
        Method for computing ESS. Currently only "autocorr" is supported.
    show_progress : bool, default False
        Whether to show progress bar for large datasets
    device : str, optional
        Device to use for computation ("cuda", "cpu", or None for auto-detect)
    batch_size : int, default 1000
        Number of events to process in each batch (for GPU memory management)
        
    Returns
    -------
    np.ndarray
        Array of shape (n_events,) containing ESS for each event.
        ESS is computed as the minimum ESS across X, Y, Z parameters.
        
    Notes
    -----
    The effective sample size is computed using autocorrelation analysis.
    For each parameter (X, Y, Z), we compute the autocorrelation function
    and use it to estimate the integrated autocorrelation time, which
    gives us the ESS as n_samples / (1 + 2 * sum of autocorrelations).
    
    The final ESS for each event is the minimum across all parameters,
    which provides a conservative estimate of the effective sample size.
    
    This function uses PyTorch for GPU acceleration when available.
    """
    
    if method != "autocorr":
        raise ValueError(f"Method '{method}' not supported. Only 'autocorr' is currently supported.")
    
    # Check that we have the required data
    if summary.X is None or summary.Y is None or summary.Z is None:
        raise ValueError("EventSamplesSummary must include X, Y, Z centered samples to compute ESS")
    
    n_events, n_samples = summary.X.shape
    
    # Set default max_lag if not provided
    if max_lag is None:
        max_lag = min(1000, n_samples // 4)
    
    # Choose device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to PyTorch tensors and move to device
    X_tensor = torch.tensor(summary.X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(summary.Y, dtype=torch.float32, device=device)
    Z_tensor = torch.tensor(summary.Z, dtype=torch.float32, device=device)
    
    # Initialize ESS array on device
    ess_per_event = torch.zeros(n_events, dtype=torch.float32, device=device)
    
    # Process in batches to manage GPU memory
    iterator = range(0, n_events, batch_size)
    if show_progress:
        total_batches = (n_events + batch_size - 1) // batch_size
        iterator = tqdm(iterator, total=total_batches, desc="Computing ESS", leave=False)
    
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, n_events)
        batch_size_actual = batch_end - batch_start
        
        # Get batch of samples
        x_batch = X_tensor[batch_start:batch_end, :]  # (batch_size, n_samples)
        y_batch = Y_tensor[batch_start:batch_end, :]
        z_batch = Z_tensor[batch_start:batch_end, :]
        
        # Compute ESS for each parameter in the batch
        ess_x = _compute_ess_batch_torch(x_batch, max_lag)
        ess_y = _compute_ess_batch_torch(y_batch, max_lag)
        ess_z = _compute_ess_batch_torch(z_batch, max_lag)
        
        # Take the minimum ESS across parameters (conservative estimate)
        ess_per_event[batch_start:batch_end] = torch.minimum(
            torch.minimum(ess_x, ess_y), ess_z
        )
        
        # Clear GPU cache if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Return as numpy array
    return ess_per_event.cpu().numpy()


def _compute_ess_batch_torch(chains: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Compute effective sample size for a batch of MCMC chains using PyTorch.
    
    Parameters
    ----------
    chains : torch.Tensor
        Tensor of shape (batch_size, n_samples) containing MCMC samples
    max_lag : int
        Maximum lag for autocorrelation computation
        
    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size,) containing ESS for each chain
    """
    batch_size, n_samples = chains.shape
    
    # Remove mean to get centered samples
    chains_centered = chains - chains.mean(dim=1, keepdim=True)
    
    # Compute variance for each chain
    var = chains_centered.var(dim=1, unbiased=True)  # (batch_size,)
    
    # Handle zero variance case
    zero_var_mask = var == 0
    ess = torch.full((batch_size,), float(n_samples), dtype=torch.float32, device=chains.device)
    
    # Only compute ESS for chains with non-zero variance
    if zero_var_mask.all():
        return ess
    
    # Get chains with non-zero variance
    valid_chains = chains_centered[~zero_var_mask]  # (valid_batch_size, n_samples)
    valid_var = var[~zero_var_mask]  # (valid_batch_size,)
    
    # Compute autocorrelation function for all valid chains
    acf = torch.zeros(valid_chains.shape[0], max_lag + 1, dtype=torch.float32, device=chains.device)
    acf[:, 0] = 1.0  # Autocorrelation at lag 0 is always 1
    
    # Compute autocorrelations for all lags at once
    for lag in range(1, max_lag + 1):
        if lag >= n_samples:
            break
        
        # Compute autocorrelation at this lag for all chains
        numerator = torch.sum(valid_chains[:, :-lag] * valid_chains[:, lag:], dim=1)
        denominator = (n_samples - lag) * valid_var
        
        # Avoid division by zero
        acf[:, lag] = torch.where(
            denominator != 0,
            numerator / denominator,
            torch.zeros_like(numerator)
        )
    
    # Compute integrated autocorrelation time
    # Use only positive autocorrelations to avoid negative ESS
    positive_acf = acf[:, 1:max_lag+1]
    positive_acf = torch.clamp(positive_acf, min=0)  # Set negative values to 0
    
    # Sum autocorrelations (multiply by 2 for two-sided sum)
    integrated_autocorr = 1.0 + 2.0 * torch.sum(positive_acf, dim=1)
    
    # Compute effective sample size
    valid_ess = n_samples / integrated_autocorr
    
    # Ensure ESS is not negative or larger than n_samples
    valid_ess = torch.clamp(valid_ess, min=1.0, max=float(n_samples))
    
    # Put results back in the original tensor
    ess[~zero_var_mask] = valid_ess
    
    return ess


def _compute_ess_single_chain(chain: np.ndarray, max_lag: int) -> float:
    """
    Compute effective sample size for a single MCMC chain (CPU version).
    
    Parameters
    ----------
    chain : np.ndarray
        Array of shape (n_samples,) containing the MCMC samples
    max_lag : int
        Maximum lag for autocorrelation computation
        
    Returns
    -------
    float
        Effective sample size
    """
    n_samples = len(chain)
    
    # Remove mean to get centered samples
    chain_centered = chain - np.mean(chain)
    
    # Compute variance
    var = np.var(chain_centered, ddof=1)
    
    if var == 0:
        return float(n_samples)  # No variance means perfect sampling
    
    # Compute autocorrelation function
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0  # Autocorrelation at lag 0 is always 1
    
    for lag in range(1, max_lag + 1):
        if lag >= n_samples:
            break
        
        # Compute autocorrelation at this lag
        numerator = np.sum(chain_centered[:-lag] * chain_centered[lag:])
        denominator = (n_samples - lag) * var
        
        if denominator == 0:
            acf[lag] = 0
        else:
            acf[lag] = numerator / denominator
    
    # Compute integrated autocorrelation time
    # Use only positive autocorrelations to avoid negative ESS
    positive_acf = acf[1:max_lag+1]
    positive_acf = positive_acf[positive_acf > 0]
    
    if len(positive_acf) == 0:
        return float(n_samples)  # No positive autocorrelation means independent samples
    
    # Sum autocorrelations (multiply by 2 for two-sided sum)
    integrated_autocorr = 1.0 + 2.0 * np.sum(positive_acf)
    
    # Compute effective sample size
    ess = n_samples / integrated_autocorr
    
    # Ensure ESS is not negative or larger than n_samples
    ess = max(1.0, min(ess, float(n_samples)))
    
    return ess


def compute_ess_summary(
    summary: EventSamplesSummary,
    max_lag: Optional[int] = None,
    method: str = "autocorr",
    show_progress: bool = False,
    device: Optional[str] = None,
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Compute effective sample size summary statistics.
    
    Parameters
    ----------
    summary : EventSamplesSummary
        Summary object containing centered samples
    max_lag : int, optional
        Maximum lag for autocorrelation computation
    method : str, default "autocorr"
        Method for computing ESS
    show_progress : bool, default False
        Whether to show progress bar
    device : str, optional
        Device to use for computation ("cuda", "cpu", or None for auto-detect)
    batch_size : int, default 1000
        Number of events to process in each batch (for GPU memory management)
        
    Returns
    -------
    dict
        Dictionary containing ESS statistics:
        - 'ess_per_event': array of ESS for each event
        - 'mean_ess': mean ESS across all events
        - 'median_ess': median ESS across all events
        - 'min_ess': minimum ESS across all events
        - 'max_ess': maximum ESS across all events
        - 'std_ess': standard deviation of ESS across all events
        - 'n_events': number of events
        - 'n_samples': number of samples per event
    """
    
    ess_per_event = compute_effective_sample_size(
        summary, 
        max_lag=max_lag, 
        method=method, 
        show_progress=show_progress,
        device=device,
        batch_size=batch_size
    )
    
    return {
        'ess_per_event': ess_per_event,
        'mean_ess': np.mean(ess_per_event),
        'median_ess': np.median(ess_per_event),
        'min_ess': np.min(ess_per_event),
        'max_ess': np.max(ess_per_event),
        'std_ess': np.std(ess_per_event),
        'n_events': len(ess_per_event),
        'n_samples': summary.X.shape[1] if summary.X is not None else 0
    }


