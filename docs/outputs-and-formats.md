# Outputs and File Formats

This page documents the main artifacts produced by SPIDER runs.

## Output paths (`io` block)

- `io.catalog_outfile` — **prefix** for the Phase-1 MAP catalog CSVs (used by `locate-map` and
  `locate-full` only; `sample` writes no catalog).
- `io.samples_outfile` — HDF5 posterior sample store (Phases 2–4; MAP locations are also written
  into it at the end of Phase 1).
- `io.checkpoint_dir` — checkpoints and, by default, the Phase-2 bundle.

## Catalog output (`catalog_outfile`)

Phase 1 (MAP) writes:

- `<catalog_outfile>_MAP.csv` — final MAP catalog (also refreshed every 100 epochs during Phase 1
  as a progress snapshot, so a partially written file exists while the MAP is still running). The
  snapshot is written by the same routine as the final file, so during the second MAP pass it
  already carries the `dropped_after_phase1` rows and column described below.
- `<catalog_outfile>_MAP_pass1.csv` — only when `inference.sampler.phase1_two_pass.enabled` is
  `true`: a copy of the pass-1 MAP catalog, preserved before the second pass overwrites `_MAP.csv`.

Columns: every column of `io.catalog_infile` is preserved (e.g. `mag`), with `longitude`,
`latitude`, `depth` overwritten at the MAP solution, plus appended `T_src` (origin-time
correction in seconds — the `time` column is *not* shifted), `X`, `Y` (local LAEA km, origin at
`model.domain.lon_min`/`lat_min`) and `unc_x`, `unc_y`, `unc_z` (always `NaN`; uncertainties come
from the posterior samples, not this file).

When the post-Phase-1 filters leave events with no dtimes (or in components smaller than
`model.filters.events.min_events_per_cluster`), those events are removed from the second MAP pass,
the Phase-2 bundle and the samples, but stay in `_MAP.csv` at their last MAP location with an extra
column `dropped_after_phase1` (`1` for dropped events, `0` otherwise). The column is only present
when at least one event was dropped; the log reports the counts as
`Post-Phase-1 event filter: dropping N/M events (...)`.

Nothing in SPIDER reads `_MAP.csv` back — resume uses checkpoints or the Phase-2 bundle.

## Samples HDF5 (`samples_outfile`)

Sampling output is stored in HDF5 and written in `batch_*` groups, one group per flush
(`io.sample_write_interval` epochs, plus a final flush). Within Phase 4 every
`io.save_every_n`-th minibatch step is kept as one sample.

Root-level attributes/datasets:

- root attrs: `n_events`, `event_ids_json` (JSON list of event IDs), `map_present` (set once MAP
  locations have been written at the end of Phase 1); merged multi-chain files add `n_chains`
  and `chain_files_json`
- root datasets (written at the end of Phase 1): `map_longitude`, `map_latitude`, `map_depth`

### Batch groups

Each `batch_<index>` group contains 2D `(n_events, n_samples_in_batch)` datasets:

- `X`, `Y`, `Z`, `delta_t` — sampled **offsets** ΔX from the initial catalog (km, km, km, s)
- `depth` — absolute depth (km)
- `longitude`, `latitude` — absolute; populated only when lon/lat conversion is enabled at write
  time (`samples_store_lonlat`, default on; otherwise the datasets exist but are zero)

Files written by older SPIDER versions may also carry 1D datasets `log_sigma_p`, `log_sigma_s`
(learned noise scales). Current runs use fixed `phase_unc` and do not write them.

Batch attrs: `n_events`, `event_ids_json`, `sample_count`, and best-effort `global_step_count`,
`epoch`, `phase`, `wall_time_s`.

### Reader behavior (`read_all_samples`)

`spider.io.samples.read_all_samples(params, backend='numpy'|'torch', thin=1, ...)` accepts a
config path, a nested config dict or a materialized params dict and:

- concatenates all `batch_*` groups in sorted order,
- supports thinning (`thin`),
- includes batch/chain provenance metadata in the returned dict:
  `_batch_names`, `_batch_boundaries`, `_batch_slices`, `_batch_chain_idx`, `_batch_chain_file`,
  `_sample_chain_idx`, `_chain_segments`, `_chain_slices`, `_chain_indices`,
- returns a MAP-only pseudo-chain (`n_samples=1`) when only root `map_*` datasets exist.

Reader tuning keys (`read_samples_mismatch_mode` in `{skip,min,error}`,
`read_samples_legacy_mode` in `{drop,keep,error}`) and writer keys (`samples_store_lonlat`,
`hdf5_compression`, `hdf5_compression_opts`, `hdf5_shuffle`) are read from the mapping handed to
`read_all_samples`/the writer; they are **not** part of the validated `io` schema, so set them on
a dict you construct for post-processing.

## Checkpoints (`checkpoint_dir`)

`checkpoint_<phase>_epoch_<n>.pth`, written every `io.checkpoint_interval` epochs. Payload:
`phase`, `epoch`, `N`, `ΔX_src`, `stats_tensor`, `optimizer_state_dict`, `optimizer_type`,
`global_step_count`, plus optional `noise_log_scale` and `event_precision_matrix`. A `samples` key
is present for format compatibility but is always an empty list — the samples HDF5 is the
sample-of-record.

Loading: the latest checkpoint is selected by modification time and tensors are remapped to the
requested device. `locate-map` only resumes Phase-1 checkpoints (a newer non-Phase-1 checkpoint in
the directory is an error); with `inference.runtime.reset_batch_numbers: true` existing
checkpoints are cleared first, and with `clear_samples_on_reset: true` the samples file is deleted
too.

## Phase-2 bundle artifacts

Written only by `locate-map` (default `<checkpoint_dir>/phase2_bundle.pth`, override with
`--bundle-out`). `locate-full` does not write a bundle.

Sidecar tables next to the bundle: `phase2_bundle.pth.origins0.parquet`,
`phase2_bundle.pth.dtimes.parquet` (zstd-compressed Parquet; large dtime tables are kept out of
the pickle on purpose).

Payload (`bundle_version = 3`; the loader accepts 1–3): `bundle_version`, `created_unix_s`,
`origins0_parquet`, `dtimes_parquet`, `dX_src`, `residual_filter_applied` (bool; lets `sample`
skip re-applying the residual filter to already-filtered dtimes), and, when the hierarchical event
prior is enabled, `event_precision_matrix` (the learned `(K, 4, 4)` per-component precision at the
end of Phase 1, adopted by `sample` so Phases 2–4 do not fall back to the base `params.std` until
the first Wishart update). A bundle without it, used with the hierarchical prior on, triggers the
warning `Bundle carries no hierarchical precision ...; Regenerate the bundle with spider locate-map`.
Older bundles may additionally carry `params`, `noise_log_scale`, `phase1_optimizer_state_dict`,
`global_step_count`; current runs deliberately omit them and rebuild all runtime state from the
current config.

Important: the bundled dtimes are the **post-Phase-1** dtimes. The residual filter
(`model.filters.residual.phase: after_phase1`, the default), the pair/station-ratio filter (when
`ratio_filter_phase: after_phase1`) and the linearization filter are applied at the MAP locations
at the end of Phase 1, and events left without data are removed, so the bundle — and every
`sample` run started from it — sees the filtered dataset. With two-pass MAP the second pass runs on
this filtered dataset and the bundle stores the pass-2 solution.

## Multi-chain merged sample files

When chain outputs are merged (`sample-multi` without `--no-merge`), batch attrs may include
`chain_idx`, `chain_file`, `source_batch`, `source_batch_idx`. `read_all_samples` preserves these
through metadata keys so analysis code can perform chain-aware diagnostics.

## Reading the per-epoch console line

```text
[spider][INFO][RUN] phase1 | 137/1000 | L=9.1731e+00 | dx=3.536e-01 | dy=3.024e-01 | dz=7.149e-01 | dT=1.208e-01 | dr_max=3.464e+00 | dr_90=2.212e+00 | t=2.0s | precond=none noise=off lr=0.001 | sigma_p=0.0100 sigma_s=0.0100
```

The phase label is `phase1`, `phase1-pass2` (optional second MAP pass), `phase2`, `phase3` or
`phase4`. `L` is the edge-count-weighted mean loss over the epoch; `dx/dy/dz/dT` are median |Δ|
per component (km, s); `dr_max`/`dr_90` are the max / 90th-percentile displacement magnitude
(km); `t` is the wall time of the epoch. If `dr_max` sits at the norm of
`inference.safety.max_abs_dX` epoch after epoch, events are pinned at the clamp. Phase 3 appends
`ramp=<0..1>` (noise ramp progress).

If any minibatch is skipped by the finiteness guards, the epoch line itself gains a
`skipped=k/M` field after the sampler status (k skipped out of M batches seen; absent when k is 0),
and you also get:

```text
[spider][INFO][RUN] phase1 | 42/1000 | L=9.1731e+00 | ... | t=2.0s | precond=none noise=off lr=0.001 | skipped=3/128 | sigma_p=0.0100 sigma_s=0.0100
[spider][WARN][RUN] epoch 42: skipped 3/128 batches (non-finite loss=1, non-finite grad=2); parameters unchanged for those batches.
```

and, when every batch was skipped, the additional text "ALL batches skipped -> optimization is
frozen. Typical cause: a NaN gradient from a source that has converged onto a receiver ...". In
that case `L` is reported as `nan` (not `0.0`), and the epoch metrics carry `skipped_batches`
(also logged to W&B).

## W&B metrics

When `observability.wandb.enabled` is `true`, the per-epoch metrics are logged to Weights &
Biases, gated by the metric groups in `observability.diagnostics.wandb.groups`
({doc}`diagnostics-catalog`). Two groups relate directly to the artifacts above:

- `core`: the fields of the console line (`dr_max`, `dr_90`, medians, loss, `skipped_batches`,
  `n_batches`).
- `priors`: hierarchical prior diagnostics and, when the centroid prior is enabled, per-component
  centroid diagnostics `centroid/n_components`, `centroid/main_dx|dy|dz|dt` (mean shift of the
  largest component), `centroid/max_dr`, `centroid/max_abs_dt`, `centroid/rms_dr_weighted`,
  `centroid/z_max` and `centroid/frac_components_z_gt3` (component means measured against their
  prior width `std/sqrt(n)`), `centroid/neg_logp`. The same numbers are printed to the console as
  `Component centroids: K=...` every `display_precond_every` epochs. Note that the example config
  ships with the `priors` group off.
