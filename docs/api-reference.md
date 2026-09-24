# API Reference (High Level)

SPIDER is primarily CLI-driven (`spider ...`, equivalently `python -m spider ...`; see
{doc}`cli-workflow`). The Python API below is what the CLI calls and is available for workflow
integration. It is intentionally low-level: the core pipeline expects **validated and
materialized** params (see "Configuration API"), Polars DataFrames for the input tables, and a
`torch.device`. This page is hand-written (no autodoc); signatures were checked against the code
at the time of writing — when in doubt, read the docstrings in the module.

## Configuration API (`spider.core.config_v2`)

- `load_config(raw, mode=None)` / `load_config_file(path, mode=None)` → `ResolvedConfig`.
  Validates the canonical JSON shape (`io`, `model`, `inference`, `observability`, optional
  `synth`; unknown keys in key-checked blocks are rejected) and resolves defaults. `mode` is one
  of `locate-map`, `sample`, `sample-multi`, `analyze-resid`, `locate-full`, `synth` and enables
  mode-specific checks.
- `validate_config(raw, mode=None)`, `parse_canonical_config(raw)`, `validate_cross_field_config(...)`
  — the validation steps individually; raise `ConfigError` (a list of `ConfigIssue` with `path`,
  `code`, `message`).
- `resolve_config(cfg, mode=None)`, `apply_defaults(cfg)`, `build_runtime_map(cfg)`.
- `to_legacy_runtime_params(resolved, *, profile="all", require_priors=True)` → flat `dict`
  consumed by the runtime (`profile` is `"all"` or `"synth"`). Prior blocks are validated here by
  `spider.core.priors_config.validate_and_materialize_priors` (strict, no implicit defaults).

The one-liner the CLI uses:

```python
from spider.core.config_v2 import load_config, to_legacy_runtime_params
params = to_legacy_runtime_params(load_config(json.load(open("SPIDER.json")), mode="sample"))
```

`spider validate-config <params> --print-resolved` prints the same materialized dict. Start new
configs from `example/SPIDER_example.json` or `spider/examples/params_template.json`.

## Core pipeline

- `spider.core.eikonet_loader.load_eikonet_model(params, device)` → travel-time surrogate
  (`eikonet` package, or `gridtt` tables when `model.eikonet.model_kind == "gridtt"`).
- `spider.core.data.prepare_input_dfs(params, *, model=None, device=None)` →
  `(stations, dtimes, origins)` Polars DataFrames after all data-prep filters. `model`/`device` are
  required only if a filter is configured to run at data-prep time
  (`model.filters.residual.phase: before` or `linearization_error.phase: before`).
- `spider.core.locate.locate_map(params, origins0, dtimes, model, device, wandb_logger=None, *, bundle_out=None)`
  — Phase 1 (MAP) only, including the optional second pass and the post-Phase-1 filters; writes
  `<catalog_outfile>_MAP.csv` (and `_MAP_pass1.csv`) and, with `bundle_out`, the Phase-2 bundle.
- `spider.core.locate.locate_sample_from_bundle(*, params, bundle_path, model, device, wandb_logger=None)`
  — Phases 2–4 starting from a bundle (what `spider sample` calls).
- `spider.core.locate.locate_all(params, origins0, dtimes, model, device, wandb_logger=None)` —
  legacy single-process full pipeline (Phases 1–4); writes no bundle.
- `spider.core.analyze_resid.analyze_resid_from_bundle(*, params, bundle_path, model, device, plot_variograms=True, plot_dir=None, use_latest_checkpoint=False)`
  — what `spider analyze-resid` calls.
- `spider.utils.wandb_logger.init_wandb_if_enabled(params)` → `WandbLogger | None`, to pass as
  `wandb_logger`.

## I/O, checkpoints and bundles

- `spider.io.read_all_samples(params, backend="numpy", device=None, dtype=torch.float32, pin_memory=False, thin=1)`
  — read all `batch_*` groups of `io.samples_outfile` (see
  {doc}`postprocessing-and-analysis-tools` for the returned keys).
- `spider.io.samples.merge_samples_hdf5(*, out_path, in_paths, overwrite=True)`,
  `spider.io.samples.read_growclust_bootstrap(path, params, ...)`,
  `spider.io.samples.save_map_locations(params, origins0, X_src, dX_src, projector)`.
- `spider.io.phase_bundle.save_phase2_bundle(*, path, origins0, dtimes, dX_src, residual_filter_applied=False, event_precision_matrix=None, ...)`
  / `load_phase2_bundle(*, path)` → `Phase2Bundle` (fields `origins0`, `dtimes`, `dX_src`,
  `residual_filter_applied`, `event_precision_matrix`, plus legacy `params`, `noise_log_scale`,
  `phase1_optimizer_state_dict`, `global_step_count`).
- `spider.io.checkpoint.save_checkpoint(params, optimizer, epoch, N, ΔX_src, samples, stats_tensor, phase, global_step_count=0, noise_log_scale=None, event_precision_matrix=None)`,
  `load_checkpoint(params, device)`, `clear_checkpoint_files(params)`,
  `prune_checkpoints_after_phase1(params)` → `(deleted, kept)`.
- `spider.io.synth.synth_initial_catalog_from_truth(truth_catalog, *, lat0, lon0, event_prior_std, seed=42, spatial_only=False, clip_domain=True, z_min=None, z_max=None)`
  → `SynthCatalogResult` (used by `spider synth`).

## Analysis and plotting

- `spider.analysis.compute_cat_dd_and_xyz(...)` → `spider.analysis.results.EventSamplesSummary`
  (also importable lazily as `spider.analysis.EventSamplesSummary`).
- `spider.analysis.compute_ess_summary(summary, ...)`,
  `spider.analysis.compute_effective_sample_size(summary, ...)`.
- `spider.analysis.calibrate_event_posteriors_against_truth(summary, truth_catalog, *, lat0, lon0, ...)`.
- `spider.analysis.prior_posterior_wasserstein.{compute_event_wasserstein, compute_event_wasserstein_from_samples}`.
- `spider.analysis.graph_partition.{partition_graph_greedy_unionfind, partition_graph_recursive_bisection, partition_graph_disjoint_blocks}`.
- `spider.diagnostics.online_ess.compute_online_ess(...)`, `spider.diagnostics.fim.*`.
- Plotting helpers under `spider.plotting` (`plot_event_distributions`, `plot_event_chains`,
  `plot_uncertainty_histograms`, `plot_event_marginal_hist2d`, `plot_noise_scale_posterior_vs_prior`,
  `build_catalog_posterior_mean`) and `spider.plotting.events.plot_event_marginal_kde2d`.

See {doc}`postprocessing-and-analysis-tools` for usage examples and return values.
