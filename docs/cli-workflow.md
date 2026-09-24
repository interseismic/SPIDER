# CLI Workflow

SPIDER CLI entrypoint (`python -m spider` is equivalent):

```bash
spider --help
```

All commands run as a single process. Launching them under `torchrun` (`WORLD_SIZE > 1`) is
rejected with `no longer supports torchrun/DDP`; use `sample-multi --devices ...` for multi-GPU
work.

Every run command requires a device flag. `--device` accepts a CUDA id (`--device 0`),
`cpu` (case-insensitive) or `-1` for the CPU, and also `cuda:N`. Any negative id means CPU.
If CUDA is unavailable SPIDER warns `CUDA is not available; running on CPU.` and falls back to
the CPU. The flag overrides `inference.compute.devices` in the config. `sample-multi` takes the
same specifiers as a comma-separated list, e.g. `--devices 0,1` or `--devices cpu`. An invalid
specifier is rejected at parse time with `Invalid device specifier ...: expected a CUDA id
(e.g. 0), 'cuda:<int>', 'cpu' or -1`.

## Main commands

```bash
# Schema validation (add --print-resolved to dump the resolved runtime config).
# --mode is one of: locate-map, sample, sample-multi, analyze-resid, locate-full, synth
spider validate-config my_params.json --mode sample

# Phase 1 only (MAP). Writes <catalog_outfile>_MAP.csv and the Phase-2 bundle.
spider locate-map my_params.json --device 0 [--bundle-out PATH] [--shift-guard [--shift-guard-factor F]]

# Phases 2-4 only (sampling). Requires the bundle from locate-map.
spider sample my_params.json --device 0 [--bundle PATH] [--shift-guard [--shift-guard-factor F]]

# Multi-GPU independent chains from the same bundle (chains merged into io.samples_outfile unless --no-merge)
spider sample-multi my_params.json --devices 0,1,2,3 [--bundle PATH] [--chains N] [--seed0 0] [--out-dir DIR] [--dry-run] [--no-merge]

# Residual diagnostics from an existing bundle (no MAP rerun)
spider analyze-resid my_params.json --device 0 [--bundle PATH] [--use-latest-checkpoint] [--plot-dir DIR]

# Posterior catalog CSV (per-event mean, std, ESS) from io.samples_outfile; runs on the CPU by default
spider summarize my_params.json [--burn-in N] [--thin K] [--out PATH] [--device cpu|N] [--chain-mode auto|single]

# Full pipeline in one process (Phase 1 + Phases 2-4) [legacy; prefer locate-map + sample]
spider locate-full my_params.json --device 0

# Synthetic differential-time dataset matching the config (requires a top-level `synth` block)
spider synth my_params.json --device 0
```

Flag notes:

- `--bundle-out` / `--bundle` default to `<io.checkpoint_dir>/phase2_bundle.pth`.
- `--shift-guard` aborts the run as soon as any event moves more than
  `--shift-guard-factor` (default 5.0) times `model.priors.event.params.std`, printing the
  observations responsible. Useful for catching a wrong `dt` sign or a bad station.
- `analyze-resid --use-latest-checkpoint` evaluates the diagnostics at the most recent checkpoint
  in `io.checkpoint_dir` (for example after `sample`) instead of at the bundle's MAP state.
  Diagnostic plots (residual QQ plots and histograms, event-pair/station correlation PNGs) are
  written to the bundle directory unless `--plot-dir` is given. The former
  `--plot-variograms` / `--no-plot-variograms` flags have been removed (they had no effect).
- `locate-full` does **not** write a Phase-2 bundle, so it cannot be followed by `sample`.
- `validate-config` exits non-zero on schema errors. Misspelled or unknown keys inside the
  pass-through blocks (which the runtime reads with defaults, so they would otherwise be ignored
  silently) are reported as non-fatal
  `[spider][WARN][CFGV2] [warning] <path> (unknown_key): Unknown key(s) ignored by the runtime: ...`
  lines after the `validation OK` line; the exit status stays 0.

## `summarize`

`spider summarize <params.json>` is the one-command post-processing step. It reads only
`io.samples_outfile` and `io.catalog_outfile` from the config, so it also works on a minimal
`{"io": {...}}` file. It runs `read_all_samples` -> `compute_cat_dd_and_xyz` ->
`compute_ess_summary` and:

- prints the number of events and samples per event, the burn-in/thin used, the median and p90
  posterior standard deviation in x/y/z (km) and t (s), and the ESS summary (chain mode, number of
  chains, median and minimum `ess_xyzt_min`, per-dimension medians, plus the list of scalar keys
  returned by `compute_ess_summary`);
- writes a CSV, default `<io.catalog_outfile>_posterior.csv` (next to `<catalog_outfile>_MAP.csv`),
  with one row per event: `evid`, `latitude`, `longitude`, `depth` (Phase-1 MAP location when the
  store carries `map_*` datasets, otherwise the posterior mean), `X`, `Y`, `Z`, `delta_t`
  (posterior mean offsets, km / s), `std_x`, `std_y`, `std_z`, `std_t`, `sigma_x`, `sigma_y`,
  `sigma_z` (half-width of the central 99 % interval), `ess_x`, `ess_y`, `ess_z`, `ess_t`,
  `ess_xyzt_min`, `n_samples`.

Flags:

- `--burn-in N`: samples discarded per event, counted in raw (unthinned) samples. `--thin K`:
  keep every K-th remaining sample. At least 2 samples per event must remain.
- `--out PATH`: CSV path (directories are created).
- `--device cpu|N`: device for the summary and ESS computation; default `cpu`. The samples are
  read into host memory in full (no read-time thinning), so very large stores need RAM.
- `--chain-mode auto|single`: with `auto` (default) a store merged from `sample-multi` chains
  gets ESS computed per chain and summed (`chain_mode=multi_chain_sum` in the log); `single`
  ignores chain boundaries.

Exit status is 1 with a `[WARN][SUMMARY]` message when the samples store is missing, when
`read_all_samples` finds no samples, or when the store only holds Phase-1 MAP locations (run
`sample` first); 2 for invalid `--burn-in`/`--thin` values.

Deprecated aliases: `spider locate` == `sample`, `spider locate-multi` == `sample-multi`
(both print a deprecation warning; `locate` has no `--bundle` flag, so the bundle must be at
`<checkpoint_dir>/phase2_bundle.pth`).

## `sample-multi` details

`sample-multi` reads the config, writes one temporary per-chain config
(`spider_chain<i>_*.json` in `--out-dir`, default: the directory of the config file) and starts
one `spider sample --device <dev> --bundle <bundle>` subprocess per chain. Per chain `i`:

- device: `--devices` entries are used round-robin (`--chains` defaults to the number of devices);
- `inference.runtime.seed` = `--seed0 + 1000003 * i`;
- `io.samples_outfile` = `<stem>_chain<i>.h5`;
- `io.checkpoint_dir` = `<checkpoint_dir>/chain<i>`;
- `observability.wandb.run_name` gets the suffix `_chain<i>`.

When all chains exit successfully the per-chain stores are merged into the original
`io.samples_outfile` (existing file overwritten) unless `--no-merge` is passed. `--dry-run` prints
the chain plan without starting anything.

## Typical run order

1. Validate the config (`validate-config`).
2. Run `locate-map`. With `inference.sampler.phase1_two_pass.enabled` (on in the example configs)
   this runs MAP, applies the post-MAP filters at the MAP locations, drops events left without
   data, and runs MAP again; the log labels the second pass `phase1-pass2`.
3. Inspect residuals and diagnostics: `spider analyze-resid my_params.json --device 0`. Note that
   with `model.filters.residual.phase: after_phase1` (the default) the residual outlier filter has
   already run at the end of Phase 1, so the bundle contains the filtered dtimes.
4. Run `sample` (or `sample-multi`).
5. Run `spider summarize my_params.json --burn-in N` to get the posterior catalog CSV and ESS
   summary; for plots and custom analysis use the Python API on `io.samples_outfile`
   (see {doc}`postprocessing-and-analysis-tools`).

## Reading the log

Every line is prefixed `[spider][<LEVEL>][<SECTION>]`. Sections you will see in order during
`locate-map`: `CFGV2` (validation), `DATA` and `FILTER` (input counts and each filter's effect),
`RUN` (per-epoch progress), `PRIORS` (hierarchical and centroid prior updates), `BUNDLE`
(`Wrote Phase-2 bundle -> ...`). `sample` starts with `RUN` (`Running SPIDER sampling from
bundle=...`), `PRIORS` (`Adopted hierarchical event-prior precision from bundle ...`) and, if the
residual filter is on, `FILTER` (`Residual filter already applied to bundle dtimes; skipping ...`).

### Phase-1 log lines

The MAP start is announced as
`Phase 1: MAP (optimizer=adam) | lr_schedule=<constant|cosine(T_max=..., eta_min=...)> | ...`.
Each epoch then prints `phase1 | <epoch>/<total> | L=<loss> | dx=... | dr_max=... | ...`
({doc}`outputs-and-formats` explains each field). Other lines to expect:

- `Hierarchical event prior updated: std≈[sx,sy,sz,st] corr_zt≈... [largest of K clusters, n_events=...]`
  every `display_precond_every` epochs once `hyper.update.start_epoch` is reached.
- `Component centroids: K=... | main (n=...) shift [...] km ... | worst z=...` at the same cadence
  when the centroid prior is enabled.
- `Post-Phase-1 event filter: dropping N/M events (... with no remaining dtimes, ... in components <
  min_events_per_cluster=...)` or `... all events keep >=1 dtime ...` after the post-MAP filters.
- `Phase 1 second pass: epochs=... warm_start=... dtimes=... (post-MAP filters applied)` when
  two-pass MAP is enabled.
- A warning `skipped N/M batches (non-finite loss=..., non-finite grad=...)` means the finiteness
  guards rejected batches that epoch; if *all* batches are skipped the loss is reported as `nan`
  and the optimization is frozen (typical cause: a source that converged onto a receiver).
