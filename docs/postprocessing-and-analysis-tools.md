# Post-processing and Analysis Tools

This page summarizes SPIDER post-processing utilities for reading samples, building summaries, diagnostics, and plotting.

## 1) Load samples

Primary reader:

- `spider.io.samples.read_all_samples(...)`

Example:

```python
from spider.io.samples import read_all_samples

samples = read_all_samples(
    {"samples_outfile": "SPIDER_samples_final.h5"},
    backend="numpy",
    thin=5,
)
```

Returned core arrays:

- `event_ids`
- `longitude`, `latitude`, `depth`
- `X`, `Y`, `Z`, `delta_t`

Useful metadata keys for chain-aware analysis:

- `_batch_names`, `_batch_boundaries`, `_batch_slices`
- `_sample_chain_idx`
- `_chain_segments`, `_chain_slices`, `_chain_indices`

Map-only behavior:

- If no `batch_*` groups exist but root `map_*` datasets exist, `read_all_samples` returns a single-sample map-only structure.

Related I/O helpers:

- `merge_samples_hdf5(...)` (merge multi-chain sample files)
- `read_growclust_bootstrap(...)` (convert GrowClust bootstrap outputs into SPIDER-like sample dict)

## 2) Build event summaries

High-level entrypoint:

- `spider.analysis.compute_cat_dd_and_xyz(...)`

This returns an `EventSamplesSummary` with centered sample arrays and optional catalog summary table.

Example:

```python
from spider.analysis import compute_cat_dd_and_xyz

summary = compute_cat_dd_and_xyz(
    samples,
    burn_in=1000,
    include=["X", "Y", "Z", "T", "cat_dd"],
    uncertainty_metrics=["sigma", "std", "mad", "qhw_0.95"],
)
```

Key options:

- `include`: choose outputs (`X`, `Y`, `Z`, `T`, `lats`, `lons`, `deps`, `cat_dd`)
- `burn_in`, `thin`
- `compute_map` / `map_bins` for histogram mode estimates
- `add_wasserstein` to append per-event prior-vs-posterior Wasserstein diagnostics

## 3) ESS diagnostics

ESS utilities in `spider.analysis.results`:

- `compute_effective_sample_size(summary, ...)`
- `compute_ess_summary(summary, ...)`

`compute_ess_summary` returns per-event and aggregate ESS metrics, including:

- `ess_per_event_x`, `ess_per_event_y`, `ess_per_event_z`, `ess_per_event_t`
- conservative `ess_per_event_xyzt_min`

These are useful for identifying under-mixed events and uneven exploration.

## 4) Calibration against truth

Calibration utility:

- `spider.analysis.calibration.calibrate_event_posteriors_against_truth(...)`

This compares posterior locations to a truth catalog and computes:

- per-event errors (`err_x_km`, `err_y_km`, `err_z_km`, `dr_km`)
- normalized errors and Mahalanobis diagnostics
- coverage diagnostics for requested confidence levels

Use when synthetic truth or trusted reference catalogs are available.

## 5) Wasserstein prior-vs-posterior diagnostics

Module:

- `spider.analysis.prior_posterior_wasserstein`

Main routines:

- `compute_event_wasserstein(...)` (from HDF5 samples + config)
- `compute_event_wasserstein_from_samples(...)` (from in-memory sample dict)

CLI-style usage:

```bash
python -m spider.analysis.prior_posterior_wasserstein \
  --samples SPIDER_samples_final.h5 \
  --config SPIDER.json \
  --burn 0.2 \
  --thin 5 \
  --dims 0,1,2
```

## 6) Sticky-event diagnosis utility

Module:

- `spider.analysis.diagnose_sticky_events`

Purpose:

- Join ESS, posterior spread, and graph degree to highlight suspicious events (high ESS but unusually small posterior spread).

Example:

```bash
python -m spider.analysis.diagnose_sticky_events \
  --params SPIDER.json \
  --burn-in 1000 \
  --thin 5 \
  --max-lag 256 \
  --top 50 \
  --out-csv sticky_events.csv
```

## 7) Spatial semivariogram analysis

Module:

- `spider.analysis.spatial`

Key functions:

- `compute_receiver_ratio_semivariograms(...)`
- `plot_receiver_ratio_semivariograms(...)`

These help inspect receiver-side spatial structure and station-related behavior.

## 8) Graph partition analysis utilities

Module:

- `spider.analysis.graph_partition`

Key functions:

- `partition_graph_greedy_unionfind(...)`
- `partition_graph_recursive_bisection(...)`
- `partition_graph_disjoint_blocks(...)`

These are helpful for inspecting and prototyping event-graph partitioning strategies used by shared-event workflows.

## 9) Plotting tools

Main plotting API (`spider.plotting`):

- `plot_event_distributions(...)`
- `plot_event_chains(...)`
- `plot_uncertainty_histograms(...)`
- `plot_event_marginal_hist2d(...)`
- `plot_noise_scale_posterior_vs_prior(...)`

Additional 2D KDE marginal helper (currently defined in `spider.plotting.events`):

- `plot_event_marginal_kde2d(...)`

Example:

```python
from spider.plotting import (
    plot_event_distributions,
    plot_event_chains,
    plot_uncertainty_histograms,
)
from spider.plotting.events import plot_event_marginal_kde2d

fig, ax = plot_event_distributions(summary, coords=("X", "Y", "Z"))
fig, ax = plot_event_chains(summary, coords=("X", "Y", "Z", "T"))
fig, ax = plot_uncertainty_histograms(summary, coords=("X", "Y", "Z", "T"))
fig, axes = plot_event_marginal_kde2d(samples, event_index=0, coords=("X", "Y", "Z"))
```

## 10) Typical post-processing workflow

1. Read samples with thinning (`read_all_samples`).
2. Build summary (`compute_cat_dd_and_xyz`) including `X/Y/Z/T` and `cat_dd`.
3. Compute ESS summary and inspect low-tail events.
4. Generate chain and marginal plots.
5. (Optional) run calibration and Wasserstein diagnostics for deeper quality checks.
