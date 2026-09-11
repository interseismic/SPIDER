# API Reference (High Level)

SPIDER is primarily CLI-driven. The Python API is available for workflow integration; it expects
validated/materialized params (see the configuration API below).

## Core pipeline

- `spider.core.data.prepare_input_dfs(params, *, model=None, device=None)`
  - Returns `(stations, dtimes, origins)` Polars DataFrames after all data-prep filters.
  - `model`/`device` are required if a filter is configured to run at data-prep time
    (`model.filters.residual.phase: before` or `linearization_error.phase: before`).
- `spider.core.locate.locate_map(params, origins0, dtimes, model, device, wandb_logger=None, *, bundle_out=None)`
  - Phase 1 (MAP) only, including the optional second pass; writes `<catalog_outfile>_MAP.csv`
    and, with `bundle_out`, the Phase-2 bundle.
- `spider.core.locate.locate_sample_from_bundle(...)`
  - Phases 2–4 starting from a bundle (what `spider sample` calls).
- `spider.core.locate.locate_all(params, origins0, dtimes, model, device, wandb_logger=None)`
  - Legacy single-process full pipeline (Phases 1–4).

## Configuration API

- `spider.core.config_v2.load_config(raw, mode=None)` / `load_config_file(path, mode=None)` →
  `ResolvedConfig`
- `spider.core.config_v2.resolve_config(cfg, mode=None)`, `apply_defaults(cfg)`,
  `build_runtime_map(cfg)`
- `spider.core.config_v2.to_legacy_runtime_params(resolved, *, profile="all", require_priors=True)`
  (`profile` is `"all"` or `"synth"`)

## I/O, checkpoints and bundles

- `spider.io.samples.read_all_samples(params, backend="numpy", ..., thin=1)`
- `spider.io.samples.merge_samples_hdf5(...)`, `save_map_locations(...)`
- `spider.io.phase_bundle.save_phase2_bundle(...)` / `load_phase2_bundle(...)`
- `spider.io.checkpoint.save_checkpoint(...)` / `load_checkpoint(params, device)`

## Analysis utilities

- `spider.analysis.compute_cat_dd_and_xyz(...)` → `spider.analysis.results.EventSamplesSummary`
- `spider.analysis.compute_ess_summary(...)`, `compute_effective_sample_size(...)`
- `spider.analysis.calibrate_event_posteriors_against_truth(...)`
- plotting helpers under `spider.plotting`

See {doc}`postprocessing-and-analysis-tools` for usage examples.
