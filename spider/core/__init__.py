"""Core algorithms and data preparation."""

from .data import (
    prepare_stations,
    spatial_cat_subset,
    break_edges_longer_than,
    filter_min_unique_phase_per_event,
    filter_min_rows_per_unordered_pair,
    prepare_input_dfs,
)
from .modeling import (
    compute_travel_times,
    likelihood_loss,
    prior_loss_event,
    prior_loss_centroid,
    prior_loss,
    posterior_loss,
    compute_residuals,
    compute_residuals_full,
    med_abs_dev,
    med_abs_dev_torch,
    shuffle_data,
    write_output,
)

# NOTE: Do NOT import `.locate` at module import time.
# `spider.core.locate` depends on `spider.io.samples`, and `spider.io.samples` is used
# independently (e.g. in notebooks) without needing the full locate pipeline.
# Importing locate here can create circular import chains.
def locate_all(*args, **kwargs):
    from .locate import locate_all as _locate_all  # local import to avoid circular deps
    return _locate_all(*args, **kwargs)

__all__ = [
    "prepare_stations",
    "spatial_cat_subset",
    "break_edges_longer_than",
    "filter_min_unique_phase_per_event",
    "filter_min_rows_per_unordered_pair",
    "prepare_input_dfs",
    "compute_travel_times",
    "likelihood_loss",
    "prior_loss_event",
    "prior_loss_centroid",
    "prior_loss",
    "posterior_loss",
    "compute_residuals",
    "compute_residuals_full",
    "med_abs_dev",
    "med_abs_dev_torch",
    "shuffle_data",
    "locate_all",
    "write_output",
]


