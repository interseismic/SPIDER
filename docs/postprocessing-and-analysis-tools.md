# Post-processing and Analysis Tools

This page summarizes the Python utilities for reading posterior samples, building per-event
summaries, checking convergence, and plotting. Everything here runs offline on the HDF5 samples
file (`io.samples_outfile`) and the params JSON; no GPU is required, although most functions accept
`device="cuda"` for speed. The file layout itself is documented in {doc}`outputs-and-formats`, and
the CLI steps that produce it in {doc}`cli-workflow`.

## 0) One command: `spider summarize`

For the common case (posterior means, spreads and ESS per event, written as a catalog CSV) there is
a single CLI step that runs the read -> summary -> ESS pipeline described in sections 1 to 3:

```bash
spider summarize SPIDER.json --burn-in 1000 --thin 1 --device cpu --out SPIDER_posterior.csv
```

It prints event and sample counts, the burn-in/thin used, median and p90 posterior std in x/y/z/T,
and the ESS summary, and writes `<io.catalog_outfile>_posterior.csv` by default (columns: `evid`,
`latitude`, `longitude`, `depth`, `X`, `Y`, `Z`, `delta_t`, `std_x/y/z/t`, `sigma_x/y/z`,
`ess_x/y/z/t`, `ess_xyzt_min`, `n_samples`). `--chain-mode auto|single` controls whether merged
`sample-multi` chains are treated per chain for ESS. It fails with a clear message if the samples
store is missing, empty, or MAP-only. Flags and exit codes are documented in {doc}`cli-workflow`.
Use the Python API below when you need other uncertainty metrics, MAP-from-samples columns,
Wasserstein diagnostics, plots, or the centered sample arrays themselves.

## 1) Load samples

Primary reader:

```python
from spider.io import read_all_samples   # also spider.io.samples.read_all_samples

samples = read_all_samples("SPIDER.json", backend="numpy", thin=5)
```

`params` may be a path to the params JSON, the nested config dict, or a materialized dict; only
`io.samples_outfile` (or a flat `samples_outfile`) is read from it, so a minimal
`{"io": {"samples_outfile": "SPIDER_samples.h5"}}` works. Other arguments: `backend`
(`"numpy"` | `"torch"`), `device`, `dtype`, `pin_memory` (torch only), `thin` (keep every n-th
sample, `>= 1`). If the file does not exist the function logs `Samples store not found` and returns
`{}` — check for that before indexing.

Returned arrays, all `(n_events, n_samples)` unless noted:

- `event_ids` — `(n_events,)` strings, in file order.
- `X`, `Y`, `Z`, `delta_t` — sampled **offsets** ΔX from the initial catalog (km, km, km, s).
  Add the initial catalog position to obtain absolute coordinates.
- `depth` — absolute depth (km).
- `longitude`, `latitude` — absolute; populated only if lon/lat conversion was enabled when the
  samples were written (otherwise zeros).
- `log_sigma_p`, `log_sigma_s` — `(n_samples,)`, present only in files written by older versions
  that learned noise scales.
- `map_longitude`, `map_latitude`, `map_depth` — `(n_events,)` MAP locations, when Phase 1 wrote
  them.

Chain/batch provenance metadata (useful for chain-aware diagnostics and external R-hat):

- `_batch_names`, `_batch_sample_counts`, `_batch_boundaries`, `_batch_slices`,
  `_batch_chain_idx`, `_batch_chain_file`, `_batch_source_batch`, `_batch_source_batch_idx`
- `_sample_chain_idx` — `(n_samples,)` chain index of every sample (`-1` if unknown)
- `_chain_segments`, `_chain_slices`, `_chain_indices`, `_chain_sample_counts`

Map-only behavior: if no `batch_*` groups exist but root `map_*` datasets do (e.g. after
`locate-map` only), `read_all_samples` returns a single-sample map-only structure.

Related helpers in `spider.io.samples` (not re-exported by `spider.io`):

- `merge_samples_hdf5(*, out_path, in_paths, overwrite=True)` — merge per-chain sample files into
  one, preserving chain provenance in the batch attributes. `spider sample-multi` calls this
  automatically unless `--no-merge` is given.
- `read_growclust_bootstrap(path, params, *, backend="numpy", device=None, dtype=..., pin_memory=False, thin=1)`
  — convert a GrowClust bootstrap file into a SPIDER-like sample dict for side-by-side comparison;
  `params` must provide `lat_min`/`lon_min` or `model.domain`.

## 2) Build event summaries

High-level entrypoint:

```python
from spider.analysis import compute_cat_dd_and_xyz

summary = compute_cat_dd_and_xyz(
    samples,
    burn_in=1000,
    thin=1,
    include=["X", "Y", "Z", "T", "cat_dd"],
    uncertainty_metrics=["sigma", "std", "mad", "qhw_0.95"],
    device="cpu",
    show_progress=True,
)
cat_dd = summary.cat_dd       # pandas DataFrame, one row per event
```

`compute_cat_dd_and_xyz(...)` is a thin wrapper around `EventSamplesSummary.compute(...)` and
returns an `EventSamplesSummary` with fields `lats`, `lons`, `deps` (`(n_events,)` posterior
means), `X`, `Y`, `Z`, `T` (`(n_events, n_kept_samples)` **centered** samples, km / s),
`cat_dd`, and the chain metadata `sample_chain_idx`, `chain_slices`, `chain_segments` carried
over from the samples dict.

Key options:

- `include`: any subset of `lats`, `lons`, `deps`, `X`, `Y`, `Z`, `T`, `cat_dd`; `None` selects
  everything **except** `T`. Request `T` explicitly (derived from `delta_t`) — `compute_ess_summary`
  requires `include=["X", "Y", "Z", "T"]`. `cat_dd` needs pandas.
- `uncertainty_metrics` (columns added to `cat_dd` per axis, e.g. `sigma_x`, `std_y`,
  `qhw_0.95_z`): `sigma` (default; half-width of the central 99 % interval), `std`, `mae`,
  `mad`, `iqr` (half interquartile range), `qhw_<p>` (half-width of the central `p` interval).
- `burn_in` (samples dropped per event), `thin`.
- `compute_map` / `map_bins`: add histogram-mode columns `X_map`, `Y_map`, `Z_map`,
  `longitude_map`, `latitude_map`, `depth_map`.
- `use_map_if_available` (default `True`): use the Phase-1 MAP locations stored in the file for
  `longitude`/`latitude`/`depth` in `cat_dd`; set `False` to force posterior means.
- `add_wasserstein` (+ `wasserstein_config_path` or `wasserstein_prior_std`, `wasserstein_dims`,
  `wasserstein_device`, `wasserstein_jitter`): append per-event prior-vs-posterior Wasserstein
  columns (section 4).
- `device`, `events_chunk_size` (2048), `dtype`, `show_progress`, `progress_desc`.

`cat_dd` columns: `evid`, `_event_row`, `latitude`, `longitude`, `depth`, `X`, `Y`, `Z`
(posterior mean offsets, km), one column per requested metric and axis, and the `*_map` columns
when `compute_map=True`. There are no per-axis columns for `T`; use `summary.T` directly.

## 3) ESS diagnostics

```python
from spider.analysis import compute_ess_summary, compute_effective_sample_size

ess = compute_ess_summary(summary, max_lag=None, device="cpu", batch_size=1000)
ess_xyz_min = compute_effective_sample_size(summary)   # (n_events,) min over X/Y/Z
```

`compute_ess_summary` (autocorrelation method, FFT-based) returns a dict with:

- per-event arrays: `ess_per_event_x`, `ess_per_event_y`, `ess_per_event_z`, `ess_per_event_t`,
  the conservative `ess_per_event_xyzt_min`, and the legacy `ess_per_event` (min over X/Y/Z);
- summary statistics `mean_ess_<d>`, `median_ess_<d>`, `min_ess_<d>`, `max_ess_<d>`,
  `std_ess_<d>` for `d` in `x`, `y`, `z`, `t`, `xyzt_min` (plus the legacy `mean_ess`, ...,
  `std_ess` for the X/Y/Z minimum);
- bookkeeping: `n_events`, `n_samples`, `chain_mode`, `n_chains`, `chain_sample_counts`.

`max_lag` defaults to `min(1000, n_samples // 4)`. When the summary carries `sample_chain_idx`
with at least two chains (merged `sample-multi` output), ESS is computed per chain and summed
(`chain_mode = "multi_chain_sum"`); otherwise `chain_mode = "single_chain"`.

What to look at: the low tail of `ess_per_event_xyzt_min`, not the median. Events with ESS an
order of magnitude below the median are usually the weakly constrained ones (few differential
times); see the preconditioner-floor limitation in {doc}`sampler-health` and the
`diagnose_sticky_events` tool in section 7, which joins per-event ESS with data coverage. SPIDER
does not compute split-R-hat; use ArviZ or similar on the raw samples with `_sample_chain_idx`.

## 4) Wasserstein prior-vs-posterior diagnostics

Module `spider.analysis.prior_posterior_wasserstein` (Gaussian-approximation W2 between the event
prior `N(0, diag(std^2))` and each event's posterior):

- `compute_event_wasserstein(*, samples_outfile, config_path, burn="0", thin=1, dims="0,1,2,3", device="cpu", jitter=1e-10)`
  — reads the HDF5 and the prior std from `model.priors.event.params.std` in the config.
- `compute_event_wasserstein_from_samples(*, samples, prior_std, burn_in=0, thin=1, dims="0,1,2,3", device="cpu", jitter=1e-10)`
  — same computation on an in-memory sample dict.

Both return `{"w2": (n_events,), "event_ids": ..., "meta": {...}}`. Command-line form:

```bash
python -m spider.analysis.prior_posterior_wasserstein \
  --samples SPIDER_samples.h5 --config SPIDER.json \
  --burn 0.2 --thin 5 --dims 0,1,2 --out_csv w2.csv
```

`--burn` accepts a sample count or a fraction in `[0, 1)`; `--spatial_only` is shorthand for
`--dims 0,1,2`. The summary line (`median`, `p90`, `p99`, `max`) is a quick way to see how far the
posterior has moved from the prior as a function of burn-in. Note that with the hierarchical prior
the effective prior is the learned per-component precision, not `params.std`, so treat these
numbers as relative.

## 5) Plotting tools

Main plotting API (`spider.plotting`); all functions accept either the raw samples dict or an
`EventSamplesSummary`, and return `(fig, ax)` / `(fig, axes)`:

- `plot_event_distributions(samples_data, n_rows=6, n_cols=4, coords=("X","Y","Z"), burn_in=0, units="meters", kde_bw="silverman", event_indices=None, random_select=False, xlim=None, ...)`
  — per-event 1-D KDE marginals on a grid of events.
- `plot_event_chains(samples_data, n_rows=2, n_cols=2, coords=("X","Y","Z"), event_indices=None, event_ids=None, random_select=False, burn_in=0, thin=1, units="meters", ...)`
  — trace plots; the first thing to look at when ESS is low.
- `plot_event_marginal_hist2d(samples_data, event_index=None, event_id=None, coords=("X","Y","Z"), burn_in=0, units="km", xlim=(-0.1, 0.1), bin_width=0.003, ...)`
  — 1-D marginals plus 2-D histograms with contours for one event.
- `plot_uncertainty_histograms(df, columns, labels=None, colors=None, scale=1.0, xlim=(0, 600), cumulative=True, density=True, output_path=None, ...)`
  — cumulative histograms of `cat_dd` columns (e.g. `["sigma_x", "sigma_y", "sigma_z"]`);
  `scale=1000` converts km to metres.
- `plot_noise_scale_posterior_vs_prior(params, out_path, ...)` — only meaningful for sample files
  from older versions that learned noise scales.
- `build_catalog_posterior_mean(samples_data)` — Polars catalog of posterior means (or `cat_dd`
  when given a summary).

Additional 2-D KDE helper (defined in `spider.plotting.events`, not re-exported):

- `plot_event_marginal_kde2d(samples_data, event_index=None, event_id=None, coords=("X","Y","Z"), burn_in=0, units="km", xlim=(-0.1, 0.1), grid_size=100, ...)`

Example:

```python
from spider.plotting import (
    plot_event_distributions, plot_event_chains, plot_event_marginal_hist2d,
    plot_uncertainty_histograms,
)
from spider.plotting.events import plot_event_marginal_kde2d

fig, ax = plot_event_distributions(summary, coords=("X", "Y", "Z"), units="meters", xlim=(-600, 600))
fig, ax = plot_event_chains(summary, coords=("X", "Y", "Z", "T"), n_rows=8, n_cols=4)
fig, ax = plot_uncertainty_histograms(summary.cat_dd, ["sigma_x", "sigma_y", "sigma_z"],
                                      scale=1000.0, xlim=(0, 600))
fig, axes = plot_event_marginal_hist2d(samples, event_index=0, coords=("X", "Y", "Z"))
fig, axes = plot_event_marginal_kde2d(samples, event_index=0, coords=("X", "Y", "Z"))
```

`example/plot_results.ipynb` in the repository walks through this sequence on the example run.

## 6) Residual diagnostics from a finished run

```bash
spider analyze-resid my_params.json --device 0 \
  --bundle checkpoints/phase2_bundle.pth \
  --use-latest-checkpoint \
  --plot-dir ./diagnostics
```

Rebuilds the state from the Phase-2 bundle (no MAP rerun) and runs the residual diagnostics
enabled under `observability.diagnostics`: `shared_event_re_tau` (method-of-moments estimate of
the shared-event scales, printed), `event_pair_station_corr` (PNG in `--plot-dir`), and
`truth_catalog` for truth-referenced residuals. Without any of them enabled the command only prints
the bundle summary. `--use-latest-checkpoint` evaluates at the most recent sampling checkpoint
instead of the MAP. Residual QQ plots and histograms also go to `--plot-dir` when enabled. The
former `--plot-variograms` / `--no-plot-variograms` flags have been removed. Key lists and
interpretation are in {doc}`diagnostics-catalog`.

## 7) Other analysis utilities

- `spider.analysis.calibrate_event_posteriors_against_truth(summary, truth_catalog, *, lat0, lon0, post_cat_dd=None, evid_col="evid", truth_lon_col="longitude", truth_lat_col="latitude", truth_depth_col="depth", use_map_if_available=False, use_full_cov_from_samples=True, dedupe_evid=True, coverages=(0.5, 0.9, 0.95))`
  — posterior calibration against a truth catalog (synthetic tests): per-event errors, z-scores,
  Mahalanobis distances and coverage of the chi-square ellipsoids. `lat0`/`lon0` must be the run's
  `model.domain.lat_min`/`lon_min`.
- `python -m spider.analysis.diagnose_sticky_events --params <params> [--burn-in N] [--thin N] [--max-lag 256] [--device cpu] [--ess-device cuda] [--top 50] [--out-csv FILE]`
  — joins per-event graph degree (number of differential times, from the filtered dtimes table),
  posterior `std_xyz_km` and `ess_xyzt_min`, prints `corr(degree, ess_xyzt_min)` and the most
  suspicious events (high ESS with implausibly low spread), and optionally writes the table. It
  is also the quickest way to plot ESS against data coverage.
- `spider.analysis.graph_partition.{partition_graph_greedy_unionfind, partition_graph_recursive_bisection, partition_graph_disjoint_blocks}`
  — split a large event graph into disjoint blocks for independent runs.
- `spider.analysis.spatial.compute_receiver_ratio_semivariograms(...)` /
  `plot_receiver_ratio_semivariograms(res_P, res_S, out_path)` — station-level semivariograms of
  scaled residual means; low-level (takes the `II`/`YY`/`X_src`/`dX_src` tensors of a run state).
- `spider.diagnostics.fim.{compute_block_fim, filter_unstable_events, analyze_fim_stability}` —
  block Fisher-information diagnostics on a live `LocateState`; internal/developer use.
- `spider.diagnostics.online_ess.compute_online_ess(*, samples, event_idx, dims=(0,1,2), window=512, max_lag=None)`
  — the routine behind the `ess_online/*` metrics; usable on a list of `(n_events, 4)` ΔX
  snapshots.
- Repository scripts: `scripts/dtimes_csv_to_ascii_pairs.py` (SPIDER dtimes CSV → legacy ASCII
  pair blocks), `scripts/smoke_test_synth_init_catalog.py`,
  `scripts/canary_shared_event_re_backend.py` (backend consistency check).
- Maintenance CLIs: `python -m spider.tools.audit_reachability` (static import-reachability audit
  of the package, to spot stale modules),
  `python -m spider.tools.clean_worktree` (remove local artifacts such as `wandb/`, plots,
  checkpoints; opt-in).

## 8) Typical post-processing workflow

Quick path: `spider summarize <params> --burn-in N` (section 0) covers steps 1 to 3 and 5 below
with the default metrics. The manual sequence:

1. Read samples with thinning (`read_all_samples(..., thin=...)`).
2. Build the summary with `compute_cat_dd_and_xyz(..., include=["X","Y","Z","T","cat_dd"])`,
   choosing `burn_in` from the trace plots (or the Wasserstein-vs-burn-in curve).
3. Compute `compute_ess_summary` and inspect the low tail of `ess_per_event_xyzt_min`; plot the
   chains of the worst events.
4. Generate marginal plots for representative events and the uncertainty histograms for the
   catalog.
5. Export `summary.cat_dd` (posterior means/MAP plus per-axis uncertainty) as the relocated
   catalog.
6. Optionally run `analyze-resid`, the calibration routine (synthetic tests) and the Wasserstein
   diagnostics for deeper quality checks.
