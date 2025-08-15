from typing import Dict, Any, Optional, Sequence
import torch

from .results import EventSamplesSummary, compute_effective_sample_size, compute_ess_summary


def compute_cat_dd_and_xyz(
    event_samples: Dict[str, Any],
    burn_in: int = 0,
    include: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    events_chunk_size: int = 2048,
    dtype: torch.dtype = torch.float32,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
    thin: int = 1,
) -> EventSamplesSummary:
    """Compute and return an EventSamplesSummary, optionally selecting which fields to compute.

    Args:
        event_samples: Dictionary containing sample data
        burn_in: Number of initial samples to discard per event
        include: any subset of {"lats", "lons", "deps", "X", "Y", "Z", "cat_dd"}; None -> all
        device: override device for computation (e.g., "cuda" or "cpu").
        events_chunk_size: Number of events to process in each chunk
        dtype: Data type for torch tensors
        show_progress: Whether to show progress bar
        progress_desc: Description for progress bar
        thin: Thinning factor - keep every nth sample after burn-in (default: 1 = no thinning)
    """
    return EventSamplesSummary.compute(
        event_samples,
        burn_in=burn_in,
        include=include,
        device=device,
        events_chunk_size=events_chunk_size,
        dtype=dtype,
        show_progress=show_progress,
        progress_desc=progress_desc,
        thin=thin,
    )


__all__ = ['compute_cat_dd_and_xyz', 'EventSamplesSummary', 'compute_effective_sample_size', 'compute_ess_summary']
