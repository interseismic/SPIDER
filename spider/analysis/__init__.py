from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    # Only for type checkers; avoids importing heavy deps at runtime.
    from .results import EventSamplesSummary as EventSamplesSummary  # noqa: F401


def compute_cat_dd_and_xyz(
    event_samples: Dict[str, Any],
    burn_in: int = 0,
    include: Optional[Union[str, Sequence[str]]] = None,
    uncertainty_metrics: Optional[Union[str, Sequence[str]]] = None,
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
    """Compute and return an EventSamplesSummary, optionally selecting which fields to compute.

    Args:
        event_samples: Dictionary containing sample data
        burn_in: Number of initial samples to discard per event
        include: any subset of {"lats", "lons", "deps", "X", "Y", "Z", "cat_dd"}; None -> all
        uncertainty_metrics: Which uncertainty metrics to compute for cat_dd. Defaults to ["sigma"].
            Supported:
            - "sigma": half-width of central 99% interval (0.5*(q0.995 - q0.005)) [backward compatible]
            - "std": standard deviation
            - "mae": mean absolute deviation from the mean
            - "mad": median absolute deviation from the median
            - "iqr": half of interquartile range, 0.5*(q0.75 - q0.25)
            - "qhw_<p>": half-width of central p interval, e.g., "qhw_0.95"
        compute_map: If True, compute mode (MAP) per-parameter for X/Y/Z and lat/lon/dep from samples,
                     and include as additional columns: X_map, Y_map, Z_map, longitude_map, latitude_map, depth_map.
        map_bins: Number of bins to use for histogram-based mode estimation (default: 256).
        device: override device for computation (e.g., "cuda" or "cpu").
        events_chunk_size: Number of events to process in each chunk
        dtype: Data type for torch tensors
        show_progress: Whether to show progress bar
        progress_desc: Description for progress bar
        thin: Thinning factor - keep every nth sample after burn-in (default: 1 = no thinning)
        use_map_if_available: If True (default) and map_* fields exist, use them for catalog locations.
                              If False, force computing posterior means even if map_* fields exist.
    """
    # Lazy import to avoid importing heavy binary deps (pandas) when users only want
    # lightweight analysis utilities (e.g., graph diagnostics) in constrained envs.
    from .results import EventSamplesSummary

    return EventSamplesSummary.compute(
        event_samples,
        burn_in=burn_in,
        include=include,
        uncertainty_metrics=uncertainty_metrics,
        compute_map=compute_map,
        map_bins=map_bins,
        add_wasserstein=add_wasserstein,
        wasserstein_config_path=wasserstein_config_path,
        wasserstein_prior_std=wasserstein_prior_std,
        wasserstein_dims=wasserstein_dims,
        wasserstein_device=wasserstein_device,
        wasserstein_jitter=wasserstein_jitter,
        device=device,
        events_chunk_size=events_chunk_size,
        dtype=dtype,
        show_progress=show_progress,
        progress_desc=progress_desc,
        thin=thin,
        use_map_if_available=use_map_if_available,
    )


def compute_effective_sample_size(*args, **kwargs):
    from .results import compute_effective_sample_size as _impl

    return _impl(*args, **kwargs)


def compute_ess_summary(*args, **kwargs):
    from .results import compute_ess_summary as _impl

    return _impl(*args, **kwargs)


def calibrate_event_posteriors_against_truth(*args, **kwargs):
    """
    Lazy import wrapper for calibration utilities.

    See `spider.analysis.calibration.calibrate_event_posteriors_against_truth`.
    """
    from .calibration import calibrate_event_posteriors_against_truth as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "compute_cat_dd_and_xyz",
    "compute_effective_sample_size",
    "compute_ess_summary",
    "calibrate_event_posteriors_against_truth",
    "EventSamplesSummary",
]


def __getattr__(name: str):
    """
    Lazy attribute access for heavy optional analysis objects.

    This keeps `import spider.analysis` lightweight (avoids eager pandas import),
    while still supporting `from spider.analysis import EventSamplesSummary`.
    """
    if name == "EventSamplesSummary":
        try:
            from .results import EventSamplesSummary as _EventSamplesSummary
        except Exception as e:  # pandas/numpy binary mismatch etc.
            raise ImportError(
                "Cannot import `EventSamplesSummary` because its dependencies failed to import "
                "(typically pandas / binary wheels). Import `spider.analysis.results` directly "
                "once your environment is fixed."
            ) from e
        return _EventSamplesSummary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
