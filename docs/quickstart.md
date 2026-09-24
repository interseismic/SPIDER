# Quickstart

## What you need

- A trained EikoNet travel-time model (a `.pt` state dict) for your study region. EikoNet is a
  separate package; SPIDER loads the checkpoint through `model.model_file` and the architecture
  settings in `model.eikonet`.
- Three CSV files: an event catalog, a station table and a differential-time table. Column
  requirements and the differential-time sign convention are in {doc}`data-formats`.
- A JSON config. Start from `spider/examples/params_template.json` (placeholder paths, W&B off)
  or `example/SPIDER_example.json` (a complete synthetic example). Both validate against the
  current schema; {doc}`config-walkthrough` goes through the template key by key and lists the
  values to change first, {doc}`configuration` explains the layout.

The repository ships a complete, self-contained example in `example/`: a synthetic catalog
(`events_from_prior.csv`), a station table (`station.csv`), differential times
(`dtimes_with_noise_from_truth.csv`), a 64-wide EikoNet checkpoint for the example domain
(`model_state_dict.pt`, 150 KB) and two configs (`SPIDER_example.json` for pSGLD,
`SPIDER_example_sghmc.json` for SGHMC). It runs as checked out; no paths need editing and W&B is
off.

A GPU is strongly recommended but not required: `--device -1` runs on the CPU.

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

This installs the `spider` command; `python -m spider` is equivalent.

## Validate the configuration first

```bash
spider validate-config example/SPIDER_example.json --mode sample
```

A valid config prints `Config v2 validation OK` followed by one `default applied: ...` line per
default that was filled in (for example `io.sample_write_interval <- io.checkpoint_interval`,
`inference.runtime.seed <- 0`). An invalid config prints the offending key path and a code, for
example `inference.sampler (unknown_key): Unknown key(s): foo`, and exits with status 1.

`--mode` accepts the name of the command you intend to run (`locate-map`, `sample`,
`sample-multi`, `analyze-resid`, `locate-full`, `synth`). Add `--print-resolved` to dump the
resolved runtime configuration as JSON.

Validation does not need a GPU and does not read your data files.

## Minimal workflow

The paths inside `example/SPIDER_example.json` are relative to the `example/` directory
(`./model_state_dict.pt`, `./station.csv`, ...), so run from there. Nothing needs editing: the
config points at the shipped checkpoint and has `observability.wandb.enabled: false`. (If you
turn W&B on and initialisation fails, SPIDER logs `Disabling wandb due to init failure` and
continues without it.)

```bash
cd example

# Phase 1 (MAP): writes <io.catalog_outfile>_MAP.csv and <io.checkpoint_dir>/phase2_bundle.pth
spider locate-map SPIDER_example.json --device 0

# Phases 2-4 (sampling): reads the bundle written by locate-map (override with --bundle PATH)
spider sample SPIDER_example.json --device 0
```

Running `sample` before `locate-map` fails with
`Phase-2 bundle not found: <checkpoint_dir>/phase2_bundle.pth`.

The shipped config runs 10 000 MAP epochs twice (two-pass MAP) and 100 000 sampling epochs; on a
single GPU the MAP takes minutes and the sampling phase hours. For a first smoke test copy the
config and reduce `inference.sampler.epochs_per_phase` (for example `[30, 5, 5, 20]`); the whole
pipeline then finishes in about a minute. With so few Phase-1 epochs the hierarchical prior
update (which starts at epoch 200) does not fire, so the `Hierarchical event prior updated` line
described below only appears in the full-length run.

Once `sample` has finished, summarise the posterior into a per-event catalog with

```bash
spider summarize SPIDER_example.json --burn-in 1000 --thin 10 --out SPIDER_summary.csv
```

(`--burn-in` and `--thin` count samples per event; `--out` defaults to
`<io.catalog_outfile>_posterior.csv`).
{doc}`config-walkthrough` has the full first-run checklist of log lines to look for.

`--device` is required on both commands. It takes a CUDA device id, or `-1` for the CPU. If CUDA
is not available SPIDER warns `CUDA is not available; running on CPU.` and continues.

### What `locate-map` does

1. Reads the CSVs and applies the data-prep filters (`[DATA]` and `[FILTER]` log lines such as
   `Initial dtimes n=...`, `Final origins n=...`).
2. Runs the MAP optimisation (Adam). Each epoch prints a `phase1 | <epoch>/<total> | L=... |
   dr_max=...` line. The example config uses a cosine learning-rate schedule
   (`inference.sampler.lr_schedule.phase1`), announced as
   `Phase 1: MAP (optimizer=adam) | lr_schedule=cosine(T_max=..., eta_min=...)`.
3. Applies the post-MAP filters at the relocated positions (residual outlier filter,
   linearization-error filter) and removes events that are left with no data
   (`Post-Phase-1 event filter: ...`).
4. Because the example enables `inference.sampler.phase1_two_pass`, it then runs MAP a second
   time on the filtered data (`Phase 1 second pass: epochs=... warm_start=true ...`; epoch lines
   are labelled `phase1-pass2`). The pass-1 catalog is kept as `<catalog_outfile>_MAP_pass1.csv`.
5. Writes the Phase-2 bundle: `Wrote Phase-2 bundle -> <checkpoint_dir>/phase2_bundle.pth`.

The example also enables the hierarchical event prior (`model.priors.event.hyper`). The fixed
prior width `model.priors.event.params.std` only acts for the first `update.start_epoch` epochs of
the MAP; after that the event-shift scale is learned from the catalog, per connected
component, and logged as `Hierarchical event prior updated: std≈[...]`. The learned scale is
stored in the bundle, and `sample` reports `Adopted hierarchical event-prior precision from bundle`
when it starts. This is the recommended setup; {doc}`probability-model` explains why a single
fixed prior cannot serve both the MAP and the sampler.

### Multi-GPU independent chains

Run `locate-map` once, then:

```bash
spider sample-multi SPIDER_example.json --devices 0,1,2,3
```

This launches one `spider sample` process per chain (default: one chain per device) with a
different seed each, writes per-chain stores `<samples_outfile stem>_chain<i>.h5`, and merges them
into `io.samples_outfile` unless `--no-merge` is passed. Details are in {doc}`cli-workflow`.

## Outputs

Paths are derived from the `io` block:

- `<io.catalog_outfile>_MAP.csv` — Phase-1 (MAP) catalog. `io.catalog_outfile` is a prefix, not a
  literal output path (`"./SPIDER_out.cat"` gives `SPIDER_out.cat_MAP.csv`).
- `<io.catalog_outfile>_MAP_pass1.csv` — pass-1 MAP catalog, written only when
  `inference.sampler.phase1_two_pass.enabled` is `true`.
- `<io.checkpoint_dir>/phase2_bundle.pth` (+ `.origins0.parquet`, `.dtimes.parquet` sidecars) —
  Phase-2 bundle written by `locate-map` and consumed by `sample` / `sample-multi` / `analyze-resid`.
- `io.samples_outfile` — posterior sample store (HDF5), written when `io.write_samples` is `true`.
  `locate-map` also stores the MAP locations in it.
- `io.checkpoint_dir` — periodic checkpoints (`checkpoint_<phase>_epoch_<n>.pth`).

See {doc}`outputs-and-formats` for file layouts and {doc}`postprocessing-and-analysis-tools` for
reading the samples.

## First-run problems

- `Unknown key(s)` from `validate-config`: the config has a key the schema does not know. Keys are
  case-sensitive; compare with `spider/examples/params_template.json` ({doc}`config-walkthrough`).
- `ValueError` from `eikonet.io.load_model` listing mismatched tensor shapes: the `model.eikonet`
  architecture keys do not match the checkpoint. Copy `n_hidden`, `n_blocks`, `n_fourier`,
  `phase_emb_dim` and `use_fourier` from the EikoNet training config ({doc}`EIKONET_TRAINING`).
- `Phase-2 bundle not found`: run `locate-map` first, or pass `--bundle PATH`.
- `dr_max` in the `phase1` lines sits at the norm of `inference.safety.max_abs_dX` every epoch:
  events are pinned at the shift clamp. Widen the clamp or check the differential-time sign
  convention ({doc}`data-formats`).
- `skipped N/M batches (non-finite loss=..., non-finite grad=...)`: see {doc}`troubleshooting`.
