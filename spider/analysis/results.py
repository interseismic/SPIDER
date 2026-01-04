from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set, Union

import numpy as np
import torch
from tqdm.auto import tqdm

# Pandas is optional. Some environments (e.g. mismatched numpy/pandas wheels) cannot import it.
# We keep analysis utilities usable without pandas by importing it lazily.
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]


AllowedFields = Set[str]

StrOrSeq = Union[str, Sequence[str]]


def _normalize_str_sequence(x: Optional[StrOrSeq]) -> Optional[Sequence[str]]:
    """
    Normalize optional string-or-sequence inputs.

    - None -> None
    - "foo" -> ["foo"] (treat a bare string as a single item, not an iterable of chars)
    - ["foo","bar"] -> as-is
    """
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return x


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
    - X, Y, Z, T: centered numpy arrays of shape (n_events, n_samples_after_burnin)
    - cat_dd: pandas DataFrame with per-event means, spreads, and evid
    """

    lats: Optional[np.ndarray] = None
    lons: Optional[np.ndarray] = None
    deps: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None
    # Time offset samples ("delta_t" in the HDF5 samples store), centered per event.
    # Units are seconds.
    T: Optional[np.ndarray] = None
    cat_dd: Optional[Any] = None

    @staticmethod
    def compute(
        event_samples: Dict[str, Any],
        burn_in: int = 0,
        include: Optional[StrOrSeq] = None,
        uncertainty_metrics: Optional[StrOrSeq] = None,
        compute_map: bool = False,
        map_bins: int = 256,
        add_wasserstein: bool = False,
        wasserstein_config_path: Optional[str] = None,
        wasserstein_prior_std: Optional[Sequence[float]] = None,
        wasserstein_dims: str = "0,1,2",
        wasserstein_device: str = "cpu",
        wasserstein_jitter: float = 1e-10,
        device: Optional[str] = None,
        events_chunk_size: int = 2048,
        dtype: torch.dtype = torch.float32,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        thin: int = 1,
        use_map_if_available: bool = True,
    ) -> "EventSamplesSummary":
        """Compute EventSamplesSummary with optional burn-in and thinning.
        
        Args:
            event_samples: Dictionary containing sample data
            burn_in: Number of initial samples to discard per event
            include: Subset of fields to compute
            uncertainty_metrics: Which uncertainty metrics to compute for cat_dd (if requested).
                Defaults to ["sigma"]. Supported values:
                - "sigma": half-width of central 99% interval (0.5*(q0.995 - q0.005))
                - "std": standard deviation
                - "mae": mean absolute deviation from the mean
                - "mad": median absolute deviation from the median
                - "iqr": half of interquartile range (0.5*(q0.75 - q0.25))
                - "qhw_<p>": half-width of central p interval, e.g., "qhw_0.95"
            compute_map: Whether to compute per-parameter MAP (mode) from samples for X/Y/Z and lat/lon/dep.
                If True, adds columns: X_map, Y_map, Z_map, longitude_map, latitude_map, depth_map.
            map_bins: Number of bins to use for histogram-based mode estimation (default: 256).
            device: Override device for computation
            events_chunk_size: Number of events to process in each chunk
            dtype: Data type for torch tensors
            show_progress: Whether to show progress bar
            progress_desc: Description for progress bar
            thin: Thinning factor - keep every nth sample after burn-in (default: 1 = no thinning)
            use_map_if_available: If True (default) and map_* fields exist, use them for catalog locations.
                                  If False, force computing posterior means even if map_* fields exist.
        """
        # Determine which outputs to compute
        include = _normalize_str_sequence(include)
        uncertainty_metrics = _normalize_str_sequence(uncertainty_metrics)
        if include is None:
            include_set: AllowedFields = {"lats", "lons", "deps", "X", "Y", "Z", "cat_dd"}
        else:
            include_set = set(include)

        # Pandas-backed outputs require pandas.
        # Keep the rest of the analysis pipeline usable even if pandas can't import.
        if ("cat_dd" in include_set) and (pd is None):
            raise ImportError(
                "pandas is required for include='cat_dd', but pandas failed to import "
                "(likely numpy/pandas binary mismatch in this environment)."
            )

        if thin < 1:
            raise ValueError(f"thin must be >= 1, got {thin}")

        # Parse uncertainty metrics (only used if "cat_dd" is requested)
        # Default to ["sigma"] for backward compatibility
        if "cat_dd" in include_set:
            if uncertainty_metrics is None or len(list(uncertainty_metrics)) == 0:
                metrics_list = ["sigma"]
            else:
                metrics_list = list(uncertainty_metrics)
        else:
            metrics_list = []

        # Sanitize and parse metrics (support dynamic qhw_<p>)
        # Build: 
        # - fixed_metrics: set[str] among {"sigma","std","mae","mad","iqr"}
        # - qhw_specs: dict[name -> p]
        fixed_allowed = {"sigma", "std", "mae", "mad", "iqr"}
        fixed_metrics = set()
        qhw_specs: Dict[str, float] = {}
        for m in metrics_list:
            m_str = str(m).strip().lower()
            if m_str in fixed_allowed:
                fixed_metrics.add(m_str)
                continue
            if m_str.startswith("qhw_"):
                try:
                    p_str = m_str.split("qhw_", 1)[1]
                    p_val = float(p_str)
                    if not (0.0 < p_val < 1.0):
                        raise ValueError
                except Exception:
                    raise ValueError(f"Invalid uncertainty metric '{m}'. Expected 'qhw_<p>' with 0<p<1.")
                # Sanitize name for column: replace '.' with 'p'
                key = f"qhw_{str(p_val).replace('.', 'p')}"
                qhw_specs[key] = p_val
                continue
            raise ValueError(f"Unsupported uncertainty metric '{m}'.")

        # Helper: get all metric names we will materialize into columns
        # Keep "sigma" for backward compatibility
        all_metric_names: Sequence[str] = list(sorted(fixed_metrics)) + list(sorted(qhw_specs.keys()))
        # Ensure stable order and place "sigma" first if present
        if "sigma" in all_metric_names:
            names_wo_sigma = [n for n in all_metric_names if n != "sigma"]
            all_metric_names = ["sigma"] + names_wo_sigma

        target_device = _choose_device(device)

        # Determine shapes and allocate outputs lazily to minimize peak memory
        # Interpret burn_in as number of initial SAMPLES to discard per event (not events)
        lat_src = event_samples.get("latitude")
        if lat_src is None:
            raise ValueError(
                "event_samples is missing key 'latitude'. "
                "This usually means no samples were loaded (e.g. `read_all_samples(...)` returned an empty dict). "
                "If you only ran Phase 1 (`spider locate-map`) there are no `batch_*` groups yet; run `spider sample` "
                "to generate posterior samples. If you intended to summarize MAP-only outputs, ensure the samples HDF5 "
                "exists and contains map_* datasets (map_latitude/map_longitude/map_depth)."
            )
        lat_arr = _as_numpy(lat_src)
        if getattr(lat_arr, "ndim", 0) < 1:
            raise ValueError(
                f"event_samples['latitude'] has invalid shape {getattr(lat_arr, 'shape', None)}. "
                "Expected at least 1D with shape (n_events, n_samples). "
                "This can happen when the samples store has no batches or was not found."
            )
        n_events_total = int(lat_arr.shape[0])
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
        samp_arr = _as_numpy(event_samples[sample_source])
        if getattr(samp_arr, "ndim", 0) < 2:
            raise ValueError(
                f"event_samples['{sample_source}'] has invalid shape {getattr(samp_arr, 'shape', None)}. "
                "Expected 2D array with shape (n_events, n_samples)."
            )
        n_samples_total = int(samp_arr.shape[1])
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
        need_T = ("T" in include_set)

        # Preallocate CPU outputs
        out = EventSamplesSummary()
        if need_lats:
            lats_all = np.empty((n_events,), dtype=np.float32)
        else:
            lats_all = None
        # Optional per-parameter MAP arrays for lat/lon/dep
        lat_map_all = np.empty((n_events,), dtype=np.float32) if (need_lats and compute_map) else None
        if need_lons:
            lons_all = np.empty((n_events,), dtype=np.float32)
        else:
            lons_all = None
        lon_map_all = np.empty((n_events,), dtype=np.float32) if (need_lons and compute_map) else None
        if need_deps:
            deps_all = np.empty((n_events,), dtype=np.float32)
        else:
            deps_all = None
        dep_map_all = np.empty((n_events,), dtype=np.float32) if (need_deps and compute_map) else None

        # Determine second dimension (samples) for X/Y/Z to preallocate centered arrays if requested
        n_samples = None
        if need_X:
            n_samples = n_samples_after
            X_centered_all = np.empty((n_events, n_samples_after), dtype=np.float32) if "X" in include_set else None
            X_mean_all = np.empty((n_events,), dtype=np.float32)
            # Uncertainty metrics for X
            metric_arrays_X: Dict[str, np.ndarray] = {
                metric_name: np.empty((n_events,), dtype=np.float32)
                for metric_name in all_metric_names
            } if "cat_dd" in include_set and len(all_metric_names) > 0 else {}
            X_map_all = np.empty((n_events,), dtype=np.float32) if compute_map else None
        else:
            X_centered_all = X_mean_all = None
            metric_arrays_X = {}
            X_map_all = None
        if need_Y:
            if n_samples is None:
                n_samples = n_samples_after
            Y_centered_all = np.empty((n_events, n_samples_after), dtype=np.float32) if "Y" in include_set else None
            Y_mean_all = np.empty((n_events,), dtype=np.float32)
            # Uncertainty metrics for Y
            metric_arrays_Y: Dict[str, np.ndarray] = {
                metric_name: np.empty((n_events,), dtype=np.float32)
                for metric_name in all_metric_names
            } if "cat_dd" in include_set and len(all_metric_names) > 0 else {}
            Y_map_all = np.empty((n_events,), dtype=np.float32) if compute_map else None
        else:
            Y_centered_all = Y_mean_all = None
            metric_arrays_Y = {}
            Y_map_all = None
        if need_Z:
            if n_samples is None:
                n_samples = n_samples_after
            Z_centered_all = np.empty((n_events, n_samples_after), dtype=np.float32) if "Z" in include_set else None
            Z_mean_all = np.empty((n_events,), dtype=np.float32)
            # Uncertainty metrics for Z
            metric_arrays_Z: Dict[str, np.ndarray] = {
                metric_name: np.empty((n_events,), dtype=np.float32)
                for metric_name in all_metric_names
            } if "cat_dd" in include_set and len(all_metric_names) > 0 else {}
            Z_map_all = np.empty((n_events,), dtype=np.float32) if compute_map else None
        else:
            Z_centered_all = Z_mean_all = None
            metric_arrays_Z = {}
            Z_map_all = None

        if need_T:
            if n_samples is None:
                n_samples = n_samples_after
            T_centered_all = np.empty((n_events, n_samples_after), dtype=np.float32)
            T_mean_all = np.empty((n_events,), dtype=np.float32)
        else:
            T_centered_all = T_mean_all = None

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

        # Optional MAP fields (if present in samples dict, we prefer them over sample means IF use_map_if_available is True)
        map_lon = event_samples.get("map_longitude", None) if use_map_if_available else None
        map_lat = event_samples.get("map_latitude", None) if use_map_if_available else None
        map_dep = event_samples.get("map_depth", None) if use_map_if_available else None

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
                    if map_lat is not None:
                        lats_all[out_start:out_end] = _as_numpy(map_lat)[ev_start:ev_end].astype(np.float32)
                    else:
                        lat_chunk = get_gpu_chunk(event_samples["latitude"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                        lats_chunk = lat_chunk.mean(dim=1)
                        lats_all[out_start:out_end] = lats_chunk.detach().cpu().numpy()
                        # Compute MAP from samples if requested (independent of presence of map_*)
                        if compute_map:
                            # Use the same chunk or fetch if map_lat existed
                            lat_for_mode = lat_chunk
                            lat_modes = _compute_modes_from_chunk(lat_for_mode, map_bins)
                            lat_map_all[out_start:out_end] = lat_modes
                        del lat_chunk, lats_chunk
                    # If we used map_lat above, we still compute MAP from samples if requested
                    if map_lat is not None and compute_map:
                        lat_chunk = get_gpu_chunk(event_samples["latitude"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                        lat_modes = _compute_modes_from_chunk(lat_chunk, map_bins)
                        lat_map_all[out_start:out_end] = lat_modes
                        del lat_chunk
                if need_lons:
                    if map_lon is not None:
                        lons_all[out_start:out_end] = _as_numpy(map_lon)[ev_start:ev_end].astype(np.float32)
                    else:
                        lon_chunk = get_gpu_chunk(event_samples["longitude"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                        lons_chunk = lon_chunk.mean(dim=1)
                        lons_all[out_start:out_end] = lons_chunk.detach().cpu().numpy()
                        if compute_map:
                            lon_modes = _compute_modes_from_chunk(lon_chunk, map_bins)
                            lon_map_all[out_start:out_end] = lon_modes
                        del lon_chunk, lons_chunk
                    if map_lon is not None and compute_map:
                        lon_chunk = get_gpu_chunk(event_samples["longitude"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                        lon_modes = _compute_modes_from_chunk(lon_chunk, map_bins)
                        lon_map_all[out_start:out_end] = lon_modes
                        del lon_chunk
                if need_deps:
                    if map_dep is not None:
                        deps_all[out_start:out_end] = _as_numpy(map_dep)[ev_start:ev_end].astype(np.float32)
                    else:
                        dep_chunk = get_gpu_chunk(event_samples["depth"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                        deps_chunk = dep_chunk.mean(dim=1)
                        deps_all[out_start:out_end] = deps_chunk.detach().cpu().numpy()
                        if compute_map:
                            dep_modes = _compute_modes_from_chunk(dep_chunk, map_bins)
                            dep_map_all[out_start:out_end] = dep_modes
                        del dep_chunk, deps_chunk
                    if map_dep is not None and compute_map:
                        dep_chunk = get_gpu_chunk(event_samples["depth"], ev_start, ev_end, apply_burn_in=True, apply_thinning=False)
                        dep_modes = _compute_modes_from_chunk(dep_chunk, map_bins)
                        dep_map_all[out_start:out_end] = dep_modes
                        del dep_chunk

                # X/Y/Z - need thinning for centered arrays
                if need_X:
                    X_chunk = get_gpu_chunk(event_samples["X"], ev_start, ev_end, apply_burn_in=True, apply_thinning=True)
                    X_mean_chunk = X_chunk.mean(dim=1)
                    X_mean_all[out_start:out_end] = X_mean_chunk.detach().cpu().numpy()
                    if "X" in include_set:
                        centered = X_chunk - X_mean_chunk[:, None]
                        X_centered_all[out_start:out_end, :] = centered.detach().cpu().numpy()
                    if compute_map:
                        X_modes = _compute_modes_from_chunk(X_chunk, map_bins)
                        X_map_all[out_start:out_end] = X_modes
                    if "cat_dd" in include_set and len(all_metric_names) > 0:
                        # Compute uncertainties requested for X
                        # Precompute stats as needed
                        if "std" in fixed_metrics:
                            std_vals = torch.std(X_chunk, dim=1, unbiased=False)
                        if "mae" in fixed_metrics:
                            mae_vals = torch.mean(torch.abs(X_chunk - X_mean_chunk[:, None]), dim=1)
                        if "mad" in fixed_metrics:
                            x_med = torch.quantile(X_chunk, 0.5, dim=1)
                            mad_vals = torch.quantile(torch.abs(X_chunk - x_med[:, None]), 0.5, dim=1)
                        if "iqr" in fixed_metrics:
                            q75 = torch.quantile(X_chunk, 0.75, dim=1)
                            q25 = torch.quantile(X_chunk, 0.25, dim=1)
                            iqr_vals = 0.5 * (q75 - q25)
                        # 'sigma' is defined as half-width of central 99% interval
                        if "sigma" in fixed_metrics:
                            q_hi = torch.quantile(X_chunk, 0.995, dim=1)
                            q_lo = torch.quantile(X_chunk, 0.005, dim=1)
                            sigma_vals = 0.5 * (q_hi - q_lo)
                        # qhw_<p> specs
                        qhw_vals: Dict[str, torch.Tensor] = {}
                        for metric_name, p_val in qhw_specs.items():
                            tail = (1.0 - p_val) / 2.0
                            q_hi = torch.quantile(X_chunk, 1.0 - tail, dim=1)
                            q_lo = torch.quantile(X_chunk, tail, dim=1)
                            qhw_vals[metric_name] = 0.5 * (q_hi - q_lo)
                        # Store to arrays
                        if "std" in fixed_metrics:
                            metric_arrays_X["std"][out_start:out_end] = std_vals.detach().cpu().numpy()
                        if "mae" in fixed_metrics:
                            metric_arrays_X["mae"][out_start:out_end] = mae_vals.detach().cpu().numpy()
                        if "mad" in fixed_metrics:
                            metric_arrays_X["mad"][out_start:out_end] = mad_vals.detach().cpu().numpy()
                        if "iqr" in fixed_metrics:
                            metric_arrays_X["iqr"][out_start:out_end] = iqr_vals.detach().cpu().numpy()
                        if "sigma" in fixed_metrics:
                            metric_arrays_X["sigma"][out_start:out_end] = sigma_vals.detach().cpu().numpy()
                        for metric_name, vals in qhw_vals.items():
                            metric_arrays_X[metric_name][out_start:out_end] = vals.detach().cpu().numpy()
                    del X_chunk, X_mean_chunk
                if need_Y:
                    Y_chunk = get_gpu_chunk(event_samples["Y"], ev_start, ev_end, apply_burn_in=True, apply_thinning=True)
                    Y_mean_chunk = Y_chunk.mean(dim=1)
                    Y_mean_all[out_start:out_end] = Y_mean_chunk.detach().cpu().numpy()
                    if "Y" in include_set:
                        centered = Y_chunk - Y_mean_chunk[:, None]
                        Y_centered_all[out_start:out_end, :] = centered.detach().cpu().numpy()
                    if compute_map:
                        Y_modes = _compute_modes_from_chunk(Y_chunk, map_bins)
                        Y_map_all[out_start:out_end] = Y_modes
                    if "cat_dd" in include_set and len(all_metric_names) > 0:
                        if "std" in fixed_metrics:
                            std_vals = torch.std(Y_chunk, dim=1, unbiased=False)
                        if "mae" in fixed_metrics:
                            mae_vals = torch.mean(torch.abs(Y_chunk - Y_mean_chunk[:, None]), dim=1)
                        if "mad" in fixed_metrics:
                            y_med = torch.quantile(Y_chunk, 0.5, dim=1)
                            mad_vals = torch.quantile(torch.abs(Y_chunk - y_med[:, None]), 0.5, dim=1)
                        if "iqr" in fixed_metrics:
                            q75 = torch.quantile(Y_chunk, 0.75, dim=1)
                            q25 = torch.quantile(Y_chunk, 0.25, dim=1)
                            iqr_vals = 0.5 * (q75 - q25)
                        if "sigma" in fixed_metrics:
                            q_hi = torch.quantile(Y_chunk, 0.995, dim=1)
                            q_lo = torch.quantile(Y_chunk, 0.005, dim=1)
                            sigma_vals = 0.5 * (q_hi - q_lo)
                        qhw_vals = {}
                        for metric_name, p_val in qhw_specs.items():
                            tail = (1.0 - p_val) / 2.0
                            q_hi = torch.quantile(Y_chunk, 1.0 - tail, dim=1)
                            q_lo = torch.quantile(Y_chunk, tail, dim=1)
                            qhw_vals[metric_name] = 0.5 * (q_hi - q_lo)
                        if "std" in fixed_metrics:
                            metric_arrays_Y["std"][out_start:out_end] = std_vals.detach().cpu().numpy()
                        if "mae" in fixed_metrics:
                            metric_arrays_Y["mae"][out_start:out_end] = mae_vals.detach().cpu().numpy()
                        if "mad" in fixed_metrics:
                            metric_arrays_Y["mad"][out_start:out_end] = mad_vals.detach().cpu().numpy()
                        if "iqr" in fixed_metrics:
                            metric_arrays_Y["iqr"][out_start:out_end] = iqr_vals.detach().cpu().numpy()
                        if "sigma" in fixed_metrics:
                            metric_arrays_Y["sigma"][out_start:out_end] = sigma_vals.detach().cpu().numpy()
                        for metric_name, vals in qhw_vals.items():
                            metric_arrays_Y[metric_name][out_start:out_end] = vals.detach().cpu().numpy()
                    del Y_chunk, Y_mean_chunk
                if need_Z:
                    Z_chunk = get_gpu_chunk(event_samples["Z"], ev_start, ev_end, apply_burn_in=True, apply_thinning=True)
                    Z_mean_chunk = Z_chunk.mean(dim=1)
                    Z_mean_all[out_start:out_end] = Z_mean_chunk.detach().cpu().numpy()
                    if "Z" in include_set:
                        centered = Z_chunk - Z_mean_chunk[:, None]
                        Z_centered_all[out_start:out_end, :] = centered.detach().cpu().numpy()
                    if compute_map:
                        Z_modes = _compute_modes_from_chunk(Z_chunk, map_bins)
                        Z_map_all[out_start:out_end] = Z_modes
                    if "cat_dd" in include_set and len(all_metric_names) > 0:
                        if "std" in fixed_metrics:
                            std_vals = torch.std(Z_chunk, dim=1, unbiased=False)
                        if "mae" in fixed_metrics:
                            mae_vals = torch.mean(torch.abs(Z_chunk - Z_mean_chunk[:, None]), dim=1)
                        if "mad" in fixed_metrics:
                            z_med = torch.quantile(Z_chunk, 0.5, dim=1)
                            mad_vals = torch.quantile(torch.abs(Z_chunk - z_med[:, None]), 0.5, dim=1)
                        if "iqr" in fixed_metrics:
                            q75 = torch.quantile(Z_chunk, 0.75, dim=1)
                            q25 = torch.quantile(Z_chunk, 0.25, dim=1)
                            iqr_vals = 0.5 * (q75 - q25)
                        if "sigma" in fixed_metrics:
                            q_hi = torch.quantile(Z_chunk, 0.995, dim=1)
                            q_lo = torch.quantile(Z_chunk, 0.005, dim=1)
                            sigma_vals = 0.5 * (q_hi - q_lo)
                        qhw_vals = {}
                        for metric_name, p_val in qhw_specs.items():
                            tail = (1.0 - p_val) / 2.0
                            q_hi = torch.quantile(Z_chunk, 1.0 - tail, dim=1)
                            q_lo = torch.quantile(Z_chunk, tail, dim=1)
                            qhw_vals[metric_name] = 0.5 * (q_hi - q_lo)
                        if "std" in fixed_metrics:
                            metric_arrays_Z["std"][out_start:out_end] = std_vals.detach().cpu().numpy()
                        if "mae" in fixed_metrics:
                            metric_arrays_Z["mae"][out_start:out_end] = mae_vals.detach().cpu().numpy()
                        if "mad" in fixed_metrics:
                            metric_arrays_Z["mad"][out_start:out_end] = mad_vals.detach().cpu().numpy()
                        if "iqr" in fixed_metrics:
                            metric_arrays_Z["iqr"][out_start:out_end] = iqr_vals.detach().cpu().numpy()
                        if "sigma" in fixed_metrics:
                            metric_arrays_Z["sigma"][out_start:out_end] = sigma_vals.detach().cpu().numpy()
                        for metric_name, vals in qhw_vals.items():
                            metric_arrays_Z[metric_name][out_start:out_end] = vals.detach().cpu().numpy()
                    del Z_chunk, Z_mean_chunk

                if need_T:
                    if "delta_t" not in event_samples or event_samples["delta_t"] is None:
                        raise KeyError("Requested include contains 'T' but event_samples is missing 'delta_t'")
                    T_chunk = get_gpu_chunk(event_samples["delta_t"], ev_start, ev_end, apply_burn_in=True, apply_thinning=True)
                    T_mean_chunk = T_chunk.mean(dim=1)
                    if T_mean_all is not None:
                        T_mean_all[out_start:out_end] = T_mean_chunk.detach().cpu().numpy()
                    if T_centered_all is not None:
                        centered = T_chunk - T_mean_chunk[:, None]
                        T_centered_all[out_start:out_end, :] = centered.detach().cpu().numpy()
                    del T_chunk, T_mean_chunk

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
        if need_T:
            out.T = T_centered_all

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
            # Internal: stable per-event row index aligned to sample arrays (n_events,)
            # This allows downstream utilities (e.g., calibration) to align cat_dd rows to
            # summary.X/Y/Z even if additional merges later duplicate rows.
            data["_event_row"] = np.arange(n_events, dtype=np.int64)
            if need_lats:
                data["latitude"] = lats_all
                if compute_map and lat_map_all is not None:
                    data["latitude_map"] = lat_map_all
            if need_lons:
                data["longitude"] = lons_all
                if compute_map and lon_map_all is not None:
                    data["longitude_map"] = lon_map_all
            if need_deps:
                data["depth"] = deps_all
                if compute_map and dep_map_all is not None:
                    data["depth_map"] = dep_map_all
            if need_X:
                data["X"] = X_mean_all
                # Add uncertainty metrics for X
                for metric_name in all_metric_names:
                    col_name = "sigma_x" if metric_name == "sigma" else f"{metric_name}_x"
                    data[col_name] = metric_arrays_X[metric_name]
                if compute_map and X_map_all is not None:
                    data["X_map"] = X_map_all
            if need_Y:
                data["Y"] = Y_mean_all
                for metric_name in all_metric_names:
                    col_name = "sigma_y" if metric_name == "sigma" else f"{metric_name}_y"
                    data[col_name] = metric_arrays_Y[metric_name]
                if compute_map and Y_map_all is not None:
                    data["Y_map"] = Y_map_all
            if need_Z:
                data["Z"] = Z_mean_all
                for metric_name in all_metric_names:
                    col_name = "sigma_z" if metric_name == "sigma" else f"{metric_name}_z"
                    data[col_name] = metric_arrays_Z[metric_name]
                if compute_map and Z_map_all is not None:
                    data["Z_map"] = Z_map_all

            data["evid"] = evid_series.values[start_event:start_event + n_events]
            out.cat_dd = pd.DataFrame(data=data)

            # Optional: merge per-event prior↔posterior Wasserstein distance (Gaussian-approx)
            if bool(add_wasserstein):
                try:
                    from .prior_posterior_wasserstein import (
                        compute_event_wasserstein_from_samples,
                        DIM_NAMES,
                        load_event_prior_std,
                    )
                except Exception as e:
                    raise ImportError("Failed to import Wasserstein utilities") from e

                if wasserstein_prior_std is not None:
                    prior_std_np = np.asarray([float(x) for x in wasserstein_prior_std], dtype=np.float64)
                    if prior_std_np.shape != (4,) or not np.all(prior_std_np > 0.0):
                        raise ValueError(
                            f"wasserstein_prior_std must be 4 positive floats for (dX,dY,dZ,dt), got: {wasserstein_prior_std!r}"
                        )
                else:
                    if not wasserstein_config_path:
                        raise ValueError(
                            "add_wasserstein=True requires either wasserstein_config_path (to read model.priors.event.params.std) "
                            "or wasserstein_prior_std=[sx,sy,sz,st]."
                        )
                    prior_std_np = load_event_prior_std(str(wasserstein_config_path))

                res = compute_event_wasserstein_from_samples(
                    samples=event_samples,
                    prior_std=prior_std_np.tolist(),
                    burn_in=int(burn_in),
                    thin=int(thin),
                    dims=str(wasserstein_dims),
                    device=str(wasserstein_device),
                    jitter=float(wasserstein_jitter),
                )

                w2_event_ids = res["event_ids"]
                w2_series = pd.Series(w2_event_ids)
                try:
                    w2_series = w2_series.astype(int)
                except Exception:
                    pass

                dims_list = list(res.get("meta", {}).get("dims", list(range(res["post_mean"].shape[1]))))
                dim_names = [DIM_NAMES.get(int(d), f"d{int(d)}") for d in dims_list]

                w2_df = pd.DataFrame({"evid": w2_series.values, "w2_xyz": res["w2"]})
                mu = np.asarray(res["post_mean"])
                std = np.asarray(res["post_std"])
                w2_1d = np.asarray(res["w2_1d"])
                for j, name in enumerate(dim_names):
                    w2_df[f"w2_{name}"] = w2_1d[:, j]
                    w2_df[f"mu_{name}"] = mu[:, j]
                    w2_df[f"std_{name}"] = std[:, j]

                out.cat_dd = out.cat_dd.merge(w2_df, on="evid", how="left", copy=False)

        return out


def _compute_modes_from_chunk(samples_chunk: torch.Tensor, bins: int) -> np.ndarray:
    """
    Compute a histogram-based mode estimate per row of a 2D samples tensor.
    Returns a numpy array of shape (n_rows,) with mode estimates.
    Vectorized implementation.
    """
    m, n = samples_chunk.shape
    device = samples_chunk.device
    
    # 1. Compute min and max per row
    mins, _ = samples_chunk.min(dim=1)
    maxs, _ = samples_chunk.max(dim=1)
    ranges = maxs - mins
    
    # Handle singular ranges
    singular_mask = ranges <= 1e-9
    ranges = torch.where(singular_mask, torch.tensor(1.0, device=device, dtype=ranges.dtype), ranges)
    
    # 2. Normalize samples to [0, bins)
    # Use (bins - eps) to ensure max value falls into last bin
    # We want floor((val - min) / range * bins)
    # Clamp to [0, bins-1] to be safe
    bin_indices = ((samples_chunk - mins.unsqueeze(1)) / ranges.unsqueeze(1) * bins).long()
    bin_indices = bin_indices.clamp(0, bins - 1)
    
    # 3. Find mode of bin indices per row
    # torch.mode works on the last dimension
    mode_indices, _ = torch.mode(bin_indices, dim=1)
    
    # 4. Map back to continuous values (midpoint of the bin)
    # bin_width = range / bins
    # val = min + (mode_index + 0.5) * bin_width
    mode_vals = mins + (mode_indices.float() + 0.5) * (ranges / bins)
    
    # Handle singular rows: just return the min (which equals max)
    mode_vals = torch.where(singular_mask, mins, mode_vals)
    
    return mode_vals.detach().cpu().numpy()


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
    
    # Choose device ("cuda" if available unless explicitly overridden)
    target_device = _choose_device(device)

    # IMPORTANT: Do NOT move full X/Y/Z to CUDA up front.
    # Even though we "batch" later, pre-loading all events onto GPU can OOM.
    # Instead, materialize only per-batch slices on the target device.
    ess_per_event = np.empty(n_events, dtype=np.float32)
    
    # Process in batches to manage GPU memory
    iterator = range(0, n_events, batch_size)
    if show_progress:
        total_batches = (n_events + batch_size - 1) // batch_size
        iterator = tqdm(iterator, total=total_batches, desc="Computing ESS", leave=False)
    
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, n_events)
        batch_size_actual = batch_end - batch_start
        
        # Get batch of samples (materialize directly on target device)
        # Using `as_tensor` avoids an extra copy on CPU; on CUDA it will copy the slice.
        x_batch = torch.as_tensor(summary.X[batch_start:batch_end, :], dtype=torch.float32, device=target_device)
        y_batch = torch.as_tensor(summary.Y[batch_start:batch_end, :], dtype=torch.float32, device=target_device)
        z_batch = torch.as_tensor(summary.Z[batch_start:batch_end, :], dtype=torch.float32, device=target_device)
        
        # Compute ESS for each parameter in the batch
        ess_x = _compute_ess_batch_torch(x_batch, max_lag)
        ess_y = _compute_ess_batch_torch(y_batch, max_lag)
        ess_z = _compute_ess_batch_torch(z_batch, max_lag)
        
        # Take the minimum ESS across parameters (conservative estimate)
        ess_batch = torch.minimum(torch.minimum(ess_x, ess_y), ess_z)
        ess_per_event[batch_start:batch_end] = ess_batch.detach().to("cpu").numpy()
        
        # Clear GPU cache if using CUDA
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    
    return ess_per_event


def _compute_ess_batch_torch(chains: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Compute effective sample size for a batch of MCMC chains using PyTorch and FFT.
    Vectorized implementation.
    
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
    device = chains.device
    
    # Remove mean to get centered samples
    chains_centered = chains - chains.mean(dim=1, keepdim=True)
    
    # Compute variance for each chain
    var = chains_centered.var(dim=1, unbiased=True)  # (batch_size,)
    
    # Handle zero variance case
    zero_var_mask = var <= 1e-12
    # If var is 0, ESS is effectively n_samples
    
    # FFT Padding: next power of 2 >= 2*n_samples to avoid circular correlation
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_samples)))
    
    # Compute FFT (real-to-complex)
    fft_x = torch.fft.rfft(chains_centered, n=n_fft, dim=1)
    
    # Compute Power Spectral Density
    psd = fft_x * torch.conj(fft_x)
    
    # Inverse FFT to get Autocorrelation (complex-to-real)
    # The result of irfft is unnormalized autocorrelation
    acf_full = torch.fft.irfft(psd, n=n_fft, dim=1)[:, :max_lag+1]
    
    # Normalize: ACF[0] should be 1.0
    # acf_full[:, 0] corresponds to sum(x^2), which is (n-1)*var roughly.
    acf0 = acf_full[:, 0:1]
    
    # Avoid division by zero
    rho = torch.where(
        acf0 > 1e-12,
        acf_full / acf0,
        torch.zeros_like(acf_full)
    )
    
    # Use only positive autocorrelations (truncate at first negative or just keep positives?)
    # Standard practice often truncates at first negative or uses Geyer's monotone sequence.
    # The previous implementation effectively just ignored negative values:
    # "Use only positive autocorrelations to avoid negative ESS"
    # positive_acf = positive_acf[positive_acf > 0] -> this was filtering out non-positives
    # Let's replicate: clamp min=0
    
    positive_rho = rho[:, 1:] # Drop lag 0
    positive_rho = torch.clamp(positive_rho, min=0)
    
    # Sum autocorrelations (multiply by 2 for two-sided sum)
    integrated_autocorr = 1.0 + 2.0 * torch.sum(positive_rho, dim=1)
    
    # Compute effective sample size
    ess = n_samples / integrated_autocorr
    
    # Ensure ESS is not negative or larger than n_samples
    ess = torch.clamp(ess, min=1.0, max=float(n_samples))
    
    # Handle zero variance
    ess = torch.where(zero_var_mask, torch.tensor(float(n_samples), device=device, dtype=ess.dtype), ess)
    
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
        - 'ess_per_event_x': array of ESS for X for each event
        - 'ess_per_event_y': array of ESS for Y for each event
        - 'ess_per_event_z': array of ESS for Z for each event
        - 'ess_per_event_t': array of ESS for T (delta_t) for each event
        - 'ess_per_event_xyzt_min': conservative ESS per event (min across x,y,z,t)
        - 'mean_ess': mean ESS across all events
        - 'median_ess': median ESS across all events
        - 'min_ess': minimum ESS across all events
        - 'max_ess': maximum ESS across all events
        - 'std_ess': standard deviation of ESS across all events
        - 'n_events': number of events
        - 'n_samples': number of samples per event
    """
    
    if method != "autocorr":
        raise ValueError(f"Method '{method}' not supported. Only 'autocorr' is currently supported.")

    # Check that we have the required data
    if summary.X is None or summary.Y is None or summary.Z is None or summary.T is None:
        raise ValueError(
            "EventSamplesSummary must include centered samples X, Y, Z, T to compute per-dimension ESS. "
            "Build the summary with include=['X','Y','Z','T'] (T is derived from the samples key 'delta_t')."
        )

    n_events, n_samples = summary.X.shape
    if summary.Y.shape != (n_events, n_samples) or summary.Z.shape != (n_events, n_samples) or summary.T.shape != (n_events, n_samples):
        raise ValueError(
            f"Unexpected summary sample shapes: X{summary.X.shape} Y{summary.Y.shape} Z{summary.Z.shape} T{summary.T.shape}"
        )

    # Set default max_lag if not provided
    if max_lag is None:
        max_lag = min(1000, n_samples // 4)

    target_device = _choose_device(device)

    ess_x = np.empty((n_events,), dtype=np.float32)
    ess_y = np.empty((n_events,), dtype=np.float32)
    ess_z = np.empty((n_events,), dtype=np.float32)
    ess_t = np.empty((n_events,), dtype=np.float32)

    iterator = range(0, n_events, batch_size)
    if show_progress:
        total_batches = (n_events + batch_size - 1) // batch_size
        iterator = tqdm(iterator, total=total_batches, desc="Computing ESS (x,y,z,t)", leave=False)

    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, n_events)
        x_batch = torch.as_tensor(summary.X[batch_start:batch_end, :], dtype=torch.float32, device=target_device)
        y_batch = torch.as_tensor(summary.Y[batch_start:batch_end, :], dtype=torch.float32, device=target_device)
        z_batch = torch.as_tensor(summary.Z[batch_start:batch_end, :], dtype=torch.float32, device=target_device)
        t_batch = torch.as_tensor(summary.T[batch_start:batch_end, :], dtype=torch.float32, device=target_device)

        ex = _compute_ess_batch_torch(x_batch, int(max_lag))
        ey = _compute_ess_batch_torch(y_batch, int(max_lag))
        ez = _compute_ess_batch_torch(z_batch, int(max_lag))
        et = _compute_ess_batch_torch(t_batch, int(max_lag))

        ess_x[batch_start:batch_end] = ex.detach().to("cpu").numpy()
        ess_y[batch_start:batch_end] = ey.detach().to("cpu").numpy()
        ess_z[batch_start:batch_end] = ez.detach().to("cpu").numpy()
        ess_t[batch_start:batch_end] = et.detach().to("cpu").numpy()

        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    # Preserve the legacy conservative ESS per event: min across spatial dims only.
    ess_xyz_min = np.minimum(np.minimum(ess_x, ess_y), ess_z)
    # New: min across x/y/z/t (useful when t mixing is the bottleneck).
    ess_xyzt_min = np.minimum(ess_xyz_min, ess_t)

    def _stats(v: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "std": float(np.std(v)),
        }

    stats_xyz = _stats(ess_xyz_min)
    out: Dict[str, Any] = {
        # Legacy keys (min across X/Y/Z)
        "ess_per_event": ess_xyz_min,
        "mean_ess": stats_xyz["mean"],
        "median_ess": stats_xyz["median"],
        "min_ess": stats_xyz["min"],
        "max_ess": stats_xyz["max"],
        "std_ess": stats_xyz["std"],
        "n_events": int(n_events),
        "n_samples": int(n_samples),

        # Per-dimension ESS arrays
        "ess_per_event_x": ess_x,
        "ess_per_event_y": ess_y,
        "ess_per_event_z": ess_z,
        "ess_per_event_t": ess_t,
        "ess_per_event_xyzt_min": ess_xyzt_min,
    }

    # Per-dimension summary stats (explicit names for easy plotting/logging)
    for name, arr in (("x", ess_x), ("y", ess_y), ("z", ess_z), ("t", ess_t), ("xyzt_min", ess_xyzt_min)):
        st = _stats(arr)
        out[f"mean_ess_{name}"] = st["mean"]
        out[f"median_ess_{name}"] = st["median"]
        out[f"min_ess_{name}"] = st["min"]
        out[f"max_ess_{name}"] = st["max"]
        out[f"std_ess_{name}"] = st["std"]

    return out


