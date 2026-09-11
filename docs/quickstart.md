# Quickstart

## Installation

From the repository root:

```bash
pip install -e /path/to/eikonet
pip install -e .
```

Optional extras:

```bash
pip install -e '.[wandb]'
```

## Validate configuration first

```bash
spider validate-config example/SPIDER_example.json --mode sample
```

Add `--print-resolved` to dump the resolved runtime configuration and see which defaults were
applied (for example `io.sample_write_interval <- io.checkpoint_interval`,
`inference.runtime.seed <- 0`).

## Minimal workflow

Use the example config in `example/SPIDER_example.json`.

```bash
# Phase 1 (MAP): writes <io.catalog_outfile>_MAP.csv and <io.checkpoint_dir>/phase2_bundle.pth
spider locate-map example/SPIDER_example.json --device 0

# Phases 2-4 (sampling): reads the bundle written by locate-map (override with --bundle PATH)
spider sample example/SPIDER_example.json --device 0
```

Running `sample` before `locate-map` fails with a "Phase-2 bundle not found" error.

The example config runs Phase 1 with a cosine MAP learning-rate schedule
(`inference.sampler.lr_schedule.phase1`) and a second MAP pass
(`inference.sampler.phase1_two_pass`), so you will see two Phase-1 passes in the log and both
`<catalog_outfile>_MAP_pass1.csv` and `<catalog_outfile>_MAP.csv` on disk. See
{doc}`configuration-reference` to disable either.

Multi-GPU independent chains (run `locate-map` first):

```bash
spider sample-multi example/SPIDER_example.json --devices 0,1,2,3
```

Per-chain sample stores are merged into `io.samples_outfile` unless `--no-merge` is passed.

## Outputs

Paths are derived from the `io` block:

- `<io.catalog_outfile>_MAP.csv` — Phase-1 (MAP) catalog. `io.catalog_outfile` is a prefix, not a
  literal output path.
- `<io.catalog_outfile>_MAP_pass1.csv` — pass-1 MAP catalog, written only when
  `inference.sampler.phase1_two_pass.enabled` is `true`.
- `<io.checkpoint_dir>/phase2_bundle.pth` (+ `.origins0.parquet`, `.dtimes.parquet` sidecars) —
  Phase-2 bundle written by `locate-map` and consumed by `sample` / `sample-multi` / `analyze-resid`.
- `io.samples_outfile` — posterior sample store (HDF5), written when `io.write_samples` is `true`.
- `io.checkpoint_dir` — periodic checkpoints.

See {doc}`outputs-and-formats` for file layouts.
