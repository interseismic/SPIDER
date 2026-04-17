# Outputs and File Formats

This page documents the main artifacts produced by SPIDER runs.

## Output paths (`io` block)

Primary output keys:

- `io.catalog_outfile`
- `io.samples_outfile`
- `io.checkpoint_dir`

These are consumed across `locate-map`, `sample`, and `locate-full`.

## Catalog output (`catalog_outfile`)

SPIDER writes MAP catalog output as:

- `<catalog_outfile>_MAP.csv`

This is the post-MAP event catalog representation used by downstream analysis and resume workflows.

## Samples HDF5 (`samples_outfile`)

Sampling output is stored in HDF5 and written in `batch_*` groups.

Root-level attributes/datasets:

- root attrs:
  - `n_events`
  - `event_ids_json` (JSON list of event IDs)
- optional root datasets:
  - `map_longitude`
  - `map_latitude`
  - `map_depth`

### Batch groups

Each group named `batch_<index>` contains:

- 2D datasets shaped `(n_events, n_samples_in_batch)`:
  - `longitude`
  - `latitude`
  - `depth`
  - `delta_t`
  - `X`
  - `Y`
  - `Z`
- optional 1D datasets (length `n_samples_in_batch`):
  - `log_sigma_p`
  - `log_sigma_s`
- batch attrs (best-effort provenance):
  - `n_events`
  - `event_ids_json`
  - optional `global_step_count`, `epoch`, `phase`, `wall_time_s`

### Reader behavior (`read_all_samples`)

`spider.io.samples.read_all_samples(...)`:

- concatenates all `batch_*` groups in sorted order,
- supports thinning (`thin`),
- includes batch/chain provenance metadata in returned dict:
  - `_batch_names`, `_batch_boundaries`, `_batch_slices`
  - `_sample_chain_idx`, `_chain_segments`, `_chain_slices`
- can return MAP-only pseudo-chain (`n_samples=1`) when only root MAP datasets exist.

Useful guard options:

- `read_samples_mismatch_mode` in `{skip,min,error}` for inconsistent event counts across batches.
- `read_samples_legacy_mode` in `{drop,keep,error}` for mixed legacy/modern batch provenance.

## Checkpoints (`checkpoint_dir`)

Checkpoints are saved as:

- `checkpoint_<phase>_epoch_<n>.pth`

Typical payload fields:

- `phase`
- `epoch`
- `N`
- `ΔX_src`
- `samples`
- `stats_tensor`
- `optimizer_state_dict`
- `optimizer_type`
- `global_step_count`
- optional:
  - `noise_log_scale`
  - `event_precision_matrix`

Loading behavior:

- latest checkpoint is selected by modification time.
- tensors are remapped to requested device at load.

## Phase-2 bundle artifacts

Phase transition bundle path (default under checkpoint dir):

- `phase2_bundle.pth`

Sidecar tables written next to the bundle:

- `phase2_bundle.pth.origins0.parquet`
- `phase2_bundle.pth.dtimes.parquet`

Bundle payload includes at minimum:

- `bundle_version`
- `origins0_parquet`
- `dtimes_parquet`
- `dX_src`

Optional payload:

- `params`
- `noise_log_scale`
- `phase1_optimizer_state_dict`
- `global_step_count`

## Multi-chain merged sample files

When chain outputs are merged, batch attrs may include:

- `chain_idx`
- `chain_file`
- `source_batch`
- `source_batch_idx`

`read_all_samples` preserves these through metadata keys so analysis code can perform chain-aware diagnostics.
