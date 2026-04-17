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

## 4) Wasserstein prior-vs-posterior diagnostics

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

## 5) Plotting tools

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

## 6) Typical post-processing workflow

1. Read samples with thinning (`read_all_samples`).
2. Build summary (`compute_cat_dd_and_xyz`) including `X/Y/Z/T` and `cat_dd`.
3. Compute ESS summary and inspect low-tail events.
4. Generate chain and marginal plots.
5. (Optional) run calibration and Wasserstein diagnostics for deeper quality checks.
