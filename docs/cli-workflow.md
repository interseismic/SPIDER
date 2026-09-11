# CLI Workflow

SPIDER CLI entrypoint (`python -m spider` is equivalent):

```bash
spider --help
```

All commands run as a single process. Launching them under `torchrun` (`WORLD_SIZE > 1`) is
rejected; use `sample-multi --devices ...` for multi-GPU work.

## Main commands

```bash
# Schema validation (add --print-resolved to dump the resolved runtime config)
spider validate-config my_params.json --mode sample

# Phase 1 only (MAP). Writes <catalog_outfile>_MAP.csv and the Phase-2 bundle.
spider locate-map my_params.json --device 0 [--bundle-out PATH] [--shift-guard [--shift-guard-factor F]]

# Phases 2-4 only (sampling). Requires the bundle from locate-map.
spider sample my_params.json --device 0 [--bundle PATH] [--shift-guard [--shift-guard-factor F]]

# Multi-GPU independent chains from the same bundle (chains merged into io.samples_outfile unless --no-merge)
spider sample-multi my_params.json --devices 0,1,2,3 [--bundle PATH] [--chains N] [--seed0 0] [--out-dir DIR] [--dry-run] [--no-merge]

# Residual diagnostics from an existing bundle (no MAP rerun)
spider analyze-resid my_params.json --device 0 [--bundle PATH] [--use-latest-checkpoint] [--no-plot-variograms] [--plot-dir DIR]

# Full pipeline in one process (Phase 1 + Phases 2-4) [legacy; prefer locate-map + sample]
spider locate-full my_params.json --device 0

# Synthetic differential-time dataset matching the config (requires a top-level `synth` block)
spider synth my_params.json --device 0
```

Deprecated aliases: `spider locate` == `sample`, `spider locate-multi` == `sample-multi`
(both print a deprecation warning; `locate` has no `--bundle` flag, so the bundle must be at
`<checkpoint_dir>/phase2_bundle.pth`).

## Typical run order

1. Validate the config (`validate-config`).
2. Run `locate-map`. With `inference.sampler.phase1_two_pass.enabled` (on in the example configs)
   this runs MAP, applies the post-MAP filters at the MAP locations, and runs MAP again; the log
   labels the second pass `phase1-pass2`.
3. Inspect residuals and diagnostics: `spider analyze-resid my_params.json --device 0`. Note that
   with `model.filters.residual.phase: after_phase1` (the default) the residual outlier filter has
   already run at the end of Phase 1, so the bundle contains the filtered dtimes.
4. Run `sample` (or `sample-multi`).
5. Analyze posterior samples from `io.samples_outfile` (see {doc}`postprocessing-and-analysis-tools`).

## Phase-1 log lines

Each MAP epoch prints `phase1 | <epoch>/<total> | L=<loss> | dx=... | dr_max=... | ...`. A warning
`skipped N/M batches (non-finite loss=..., non-finite grad=...)` means the finiteness guards
rejected batches that epoch; if *all* batches are skipped the loss is reported as `nan` and the
optimization is frozen (typical cause: a source that converged onto a receiver).
