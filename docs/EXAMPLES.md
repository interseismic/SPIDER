# Examples

All commands below assume an installed `spider` console script (`python -m spider ...` is
equivalent) and a config that passes `spider validate-config params.json`.

## Check a config before running

```bash
spider validate-config path/to/params.json --mode locate-map
spider validate-config path/to/params.json --print-resolved   # also dump the resolved runtime keys
```

The validator lists every default it applied. Only some blocks reject unknown keys
(`model.filters`, `model.likelihoods`, `model.eikonet`, `inference.batching` and
`observability.diagnostics` silently ignore typos), so check the resolved output for anything you
are unsure about.

## Run a single chain

Phase 1 (MAP) and Phases 2–4 (sampling) are separate commands. `locate-map` writes the
Phase-2 bundle that `sample` consumes.

```bash
# Phase 1 (MAP): writes <io.catalog_outfile>_MAP.csv and <io.checkpoint_dir>/phase2_bundle.pth
spider locate-map path/to/params.json --device 0

# Phases 2-4 (sampling): reads the bundle (override with --bundle PATH)
spider sample path/to/params.json --device 0
```

`--device` is a CUDA index (a negative value forces CPU) and is mandatory; the config's
`inference.compute.devices` is ignored at run time. `spider locate` is a deprecated alias for
`sample` (Phases 2–4 only).

The bundle carries only the filtered dtimes, the MAP shifts and the learned hierarchical prior
precision; sampler settings are always re-read from the JSON, so you can edit
`inference.sampler` (learning rates, epochs, `phase_unc` of the `sample` block, `tau_s`, ...)
and re-run `spider sample` without repeating the MAP.

## Run multiple chains (multi-GPU)

Run `locate-map` first, then launch independent sampling chains from the same bundle.
`--devices` is required; chains are assigned to devices round-robin and default to one chain
per device.

```bash
spider sample-multi path/to/params.json --devices 0,1,2,3 [--chains N] [--seed0 0] \
    [--bundle PATH] [--out-dir DIR] [--dry-run] [--no-merge]
```

Each chain `i` is a separate `spider sample` subprocess with its own
`inference.runtime.seed = seed0 + 1000003*i`, samples file `<samples_outfile>_chain<i>.h5`,
checkpoint directory `<checkpoint_dir>/chain<i>` and W&B run name `<run_name>_chain<i>`; the
generated per-chain param files land in `--out-dir` (default: the directory of `params.json`).
When all chains finish, the per-chain stores are merged into `io.samples_outfile` (each merged
`batch_*` group records `chain_idx` and `source_batch`) unless `--no-merge` is passed.
`--dry-run` prints the chain plan without starting anything. (`spider locate-multi` is a
deprecated alias for `sample-multi`.)

## Import HypoDD differential times (sign convention)

SPIDER predicts `T(evid2) - T(evid1)`. HypoDD `dt.cc` / `dt.ct` files store `t1 - t2`, so a
catalog built from them must be sign-flipped on the way in:

```json
{
  "model": {
    "filters": {
      "dtimes": { "remove_duplicates": true, "max_abs_input_dt": 99999,
                  "dtime_thin_frac": 1.0, "flip_dt_sign": true, "cc_min": 0.0 }
    }
  }
}
```

A wrong sign shows up immediately: the `Initial residual stats (ΔX=0)` line printed before
Phase 1 reports residual medians far above `phase_unc`, and the MAP spreads the catalog instead of
sharpening it.

## Recommended event prior (hierarchical Wishart hyperprior)

Use a loose fixed Gaussian only as the starting point and let the Wishart hyperprior learn the
per-component shift scale; keep the centroid prior tight to fix each component's translation
gauge. This is what the shipped example configs do.

```json
{
  "model": {
    "priors": {
      "event": {
        "enabled": true, "type": "gaussian", "params": { "std": [5.0, 5.0, 5.0, 2.0] },
        "hyper": {
          "enabled": true, "type": "wishart_precision",
          "params": { "df": 10.0, "scale_std": [0.5, 0.5, 0.5, 0.2] },
          "update": { "every_epochs": 10, "start_epoch": 200, "min_cluster_events": 100 }
        }
      },
      "centroid": { "enabled": true, "type": "gaussian", "params": { "std": [0.01, 0.01, 0.01, 0.01] } }
    }
  }
}
```

Watch the `[PRIORS] Hierarchical event prior updated: std≈[...]` lines during Phase 1: the learned
standard deviations should settle at a few hundred metres / a few hundredths of a second. The
learned precision is stored in the bundle; if `spider sample` warns
`Bundle carries no hierarchical precision`, regenerate the bundle with `locate-map`.

## Enable Weights & Biases

W&B settings live under `observability`. `project_name` and `run_name` are required keys, and
`project_name` may not be `null` when `enabled` is `true`.

```json
{
  "observability": {
    "wandb": {
      "enabled": true,
      "project_name": "my-project",
      "run_name": "my-run"
    }
  }
}
```

Then run `spider locate-map` / `spider sample` as usual. Install the optional dependency with
`pip install -e '.[wandb]'`. Metric families can be restricted with
`observability.diagnostics.wandb.groups` (for example `{"core": true, "priors": true}`).

## Two-pass MAP with a cosine learning-rate schedule

The shipped example configs enable both. Phase 1 runs once, the post-MAP filters are applied at
the MAP locations (pair/station ratio, residual outlier filter, linearization error), events left
without data are dropped, the optimizer is rebuilt, and MAP runs again on the filtered data.

```json
{
  "inference": {
    "sampler": {
      "lr_schedule": {"phase1": {"type": "cosine", "eta_min": 1.0e-5}},
      "phase1_two_pass": {"enabled": true, "epochs": null, "warm_start": true}
    }
  },
  "model": {
    "filters": {
      "residual": {"enabled": true, "method": "abs", "abs_max": 0.15, "phase": "after_phase1"}
    }
  }
}
```

Outputs: `<io.catalog_outfile>_MAP_pass1.csv` (after pass 1) and `<io.catalog_outfile>_MAP.csv`
(final). Set `phase1_two_pass.enabled` to `false` for a single pass. Use `method: "abs"` for the
residual filter with two passes: a MAD threshold re-derives itself from already-cleaned residuals
on the second application.

## Read the MAP catalog

`<io.catalog_outfile>_MAP.csv` keeps every input column, replaces `longitude`/`latitude`/`depth`
by the MAP values, and adds `T_src` (origin-time shift in seconds; the `time` column is not
modified), `X`/`Y` (local km frame), NaN placeholders `unc_x`/`unc_y`/`unc_z`, and, when the
post-Phase-1 filter removed anything, `dropped_after_phase1` (1 = event left without usable
dtimes after the post-MAP filters; it is absent from the bundle and from the samples).

```python
import polars as pl
cat = pl.read_csv("path/to/output_catalog_MAP.csv")
if "dropped_after_phase1" in cat.columns:
    print(cat.filter(pl.col("dropped_after_phase1") == 1).shape[0], "events dropped after Phase 1")
    cat = cat.filter(pl.col("dropped_after_phase1") == 0)
```

## Resume from checkpoints

Checkpoints are written every `io.checkpoint_interval` epochs to `io.checkpoint_dir` as
`checkpoint_<phase>_epoch_<n>.pth`. Set `inference.runtime.reset_batch_numbers` to `false`
(`true` deletes existing checkpoints and starts fresh; with `clear_samples_on_reset` also the
HDF5 store).

- `locate-map` resumes the latest Phase-1 checkpoint (the cosine schedule is advanced to the
  resume epoch); if the latest checkpoint is from Phases 2–4 it aborts with an explanatory error.
- Runs with `inference.sampler.phase1_two_pass` enabled are intended for fresh runs; resuming into
  the second pass is not supported.
- `sample` never resumes from a checkpoint: it always starts Phase 2 at epoch 0 from the bundle.
  With `reset_batch_numbers=false` it does continue the `batch_<k>` numbering of an existing
  samples file, so repeated `sample` runs append to the same store.
- `locate-full` resumes whichever phase the latest checkpoint belongs to.

## Residual diagnostics without re-running the MAP

```bash
spider analyze-resid path/to/params.json --device 0 [--bundle PATH] \
    [--use-latest-checkpoint] [--no-plot-variograms] [--plot-dir DIR]
```

Evaluates residuals at the MAP state stored in the bundle (or at the latest checkpoint with
`--use-latest-checkpoint`, e.g. after sampling) and writes variogram PNGs to `--plot-dir`
(default: the bundle directory).

## Read samples and summarize

```python
import json
from spider.io.samples import read_all_samples
from spider.analysis.results import EventSamplesSummary

with open("path/to/params.json", "r") as f:
    params = json.load(f)

samples = read_all_samples({"samples_outfile": params["io"]["samples_outfile"]}, thin=1)
summary = EventSamplesSummary.compute(samples, burn_in=0)
print(summary)
```

`read_all_samples` concatenates all `batch_*` groups (and returns the root-level MAP datasets when
no batches exist yet, e.g. after `locate-map` only). Event order matches the bundle's event table,
i.e. the rows of `_MAP.csv` with `dropped_after_phase1 == 0`.

## Train an EikoNet for a new region

See `docs/EIKONET_TRAINING.md`. In short: write a 1-D `velmod.csv`, copy a recipe such as
`cahuilla/experiments/eikonet_h64_fine/eikonet.json` (64-wide network, axisymmetric FMM labels),
run `eikonet-train --config eikonet.json --device cuda:0`, check the depth derivative with
`python maunaloa/experiments/check_gradients.py eikonet.json`, then mirror `x_max`, `y_max`,
`n_hidden`, `n_blocks`, `n_fourier`, `use_fourier`, `phase_emb_dim`, `t2_log_scale` into
`model.eikonet` and `z_min`/`z_max` into `model.domain`.

## Example configs in this repo

These validate against the strict schema (`spider validate-config <path> --mode sample`):

- `example/SPIDER_example.json` — pSGLD sampler; hierarchical event prior (df 10, start_epoch 200,
  min_cluster_events 100), tight centroid prior, Laplace MAP loss, collapsed shared-event likelihood
  with `tau_s = [0.033, 0.06]`, two-pass MAP with cosine schedule, phases `[10000, 1000, 1000, 100000]`.
  It ships with a 64-wide EikoNet (`example/model_state_dict.pt`, `n_hidden 64`) and W&B off, so it
  runs from a fresh clone without edits.
- `example/SPIDER_example_sghmc.json` — same setup with the SGHMC sampler (`sghmc_alpha = 0.1`).
- `spider/examples/params_template.json` — placeholder-path template with the same prior and
  filter settings, phases `[2000, 0, 10, 10000]` (Phase 1 long enough for the hierarchical prior's
  `start_epoch 200`), SGHMC, and a 64-wide `model.eikonet` block.
