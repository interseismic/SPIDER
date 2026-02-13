# SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

SPIDER is a Python toolkit for probabilistic earthquake relocation using differential travel times, neural travel‑time prediction, and scalable MCMC sampling. It combines a fast surrogate travel‑time model (EikoNet) with a multi‑phase inference pipeline to estimate event locations with uncertainty.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input data formats](#input-data-formats)
- [Configuration](#configuration)
- [CLI workflow](#cli-workflow)
- [Likelihoods and correlated residuals](#likelihoods-and-correlated-residuals)
- [Samplers](#samplers)
- [Batching and performance](#batching-and-performance)
- [Diagnostics](#diagnostics)
- [Python API](#python-api)
- [Citation](#citation)

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[wandb]'
```

## Quick start

1) Copy a nested config and edit paths:

```bash
cp spider/examples/params_template.json my_params.json
```

2) Run Phase 1 (MAP), then sample:

```bash
python -m spider locate-map my_params.json --device 0
python -m spider sample my_params.json --device 0
```

3) Multi‑GPU independent chains:

```bash
python -m spider sample-multi my_params.json --devices 0,1,2,3
```

## Input data formats

SPIDER reads CSVs via Polars. Required columns:

### Event catalog (`io.catalog_infile`)

- `evid` (event id)
- `longitude`, `latitude`, `depth`
- `time` (parseable datetime string)

### Stations (`io.station_file`)

- `network`, `station`
- `longitude`, `latitude`
- Optional: `depth` (missing values treated as 0.0)

### Differential times (`io.dtime_file`)

- `network`, `station`
- `evid1`, `evid2`
- `dt`
- `phase` ("P"/"S" or 0/1; normalized to 0=P, 1=S)
- Optional: `cc`

## Configuration

SPIDER uses a strict **nested JSON** schema. The validator is implemented in:

- `spider/core/config_schema.py`
- `spider/core/priors_config.py`

Key sections:

### `io`

Paths and output settings:

- `dtime_file`, `station_file`, `catalog_infile`
- `catalog_outfile`, `samples_outfile`, `checkpoint_dir`
- `checkpoint_interval`, `save_every_n`, `write_samples`

### `model`

- `model_file`: EikoNet checkpoint
- `domain`: `lon_min`, `lat_min`, `z_min`, `z_max`, `scale`
- `priors`: event and centroid priors
- `likelihood`: residual model and correlated residual options
- `filters`: dtimes/events/residual filters

### `inference`

- `compute.devices`: list of GPU device ids
- `sampler`: backend and hyperparameters
- `batching`: batch sizes and optional event‑batching
- `diagnostics`: logging and post‑hoc diagnostics

## CLI workflow

The CLI entrypoint is `python -m spider`:

```bash
python -m spider --help
```

Core commands:

```bash
# Phase 1 (MAP) -> writes <checkpoint_dir>/phase2_bundle.pth
python -m spider locate-map my_params.json --device 0

# Phase 2–4 (sampling) from the Phase‑2 bundle
python -m spider sample my_params.json --device 0

# Full pipeline (Phase 1–4)
python -m spider locate-full my_params.json --device 0

# Multi‑GPU independent chains
python -m spider sample-multi my_params.json --devices 0,1,2,3
```

## Likelihoods and correlated residuals

`model.likelihood.type` supports:

- `gaussian` / `l2` (alias: `mse`)
- `laplace` / `l1` / `mae`
- `huber`
- `student_t` (requires `model.likelihood.student_t.nu`)
- `correlated` / `correlated_gaussian`

Key fields:

- `phase_unc`: per‑phase noise standard deviation `[P, S]`
- `sigma_distance_linear`: optional distance‑dependent sigma

### Shared‑event correlated residuals

Enable the collapsed shared‑event random‑effects model:

```json
"model": {
  "likelihood": {
    "type": "correlated_gaussian",
    "shared_event_re": {
      "enabled": true,
      "grouping": "station_phase",
      "tau_s": [0.03, 0.04],
      "solver": "pcg_sparse"
    }
  }
}
```

Important options:

- `grouping`: `station_phase`
- `tau_s`: per‑phase shared‑event scale
- `solver`: PCG‑based solvers (`pcg_sparse`)
- `whitening`: optional PCG whitening preconditioner
- `edge_weighting`: distance‑based edge weights (`distance_power`)

## Samplers

`inference.sampler.backend` supports:

- `psgld`
- `sghmc`

Common settings:

- `lr`: per‑phase learning rates
- `temperature`: target temperature
- `preconditioning`: RMSProp‑style preconditioning

## Batching and performance

`inference.batching.standard` controls Phase‑2/4 batch sizes:

- `warmup`: batch size for Phase‑2 drift
- `sgld`: batch size for Phase‑4 sampling
- `shuffle`: shuffle rows per epoch

Optional event‑level batching:

```json
"inference": {
  "batching": {
    "event_batches": {
      "enabled": true,
      "events_per_batch": 100,
      "max_edges_per_batch": 1000000
    }
  }
}
```

## Diagnostics

`inference.diagnostics` controls:

- W&B logging groups
- Residual distribution diagnostics
- Shared‑event correlation diagnostics (`shared_event_legcorr2d`)
- Online ESS (optional)

## Python API

Common analysis helpers:

```python
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz

samples = read_all_samples(params, thin=5)
summary = compute_cat_dd_and_xyz(samples, burn_in=100)
```

Useful modules:

- `spider.io`: reading/writing, samples, checkpoints
- `spider.analysis`: diagnostics and calibration
- `spider.plotting`: visualization helpers
- `spider.core`: inference pipeline and likelihoods

## Citation

If you use SPIDER in your research, please cite:

```bibtex
@software{spider2026,
  title={SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation},
  year={2026},
  url={https://github.com/interseismic/SPIDER}
}
```
## SPIDER

SPIDER (**S**calable **P**robabilistic **I**nference for **D**ifferential **E**arthquake **R**elocation) relocates earthquakes from differential travel times using:

- A fast travel‑time surrogate (EikoNet, PyTorch)
- A multi‑phase inference pipeline (MAP warmup → preconditioning drift → noise ramp → sampling)
- GPU acceleration and optional Weights & Biases logging

This repo contains the `spider/` Python package and its CLI.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[wandb]'
```

## Quick start (CLI)

1) Copy a nested config and edit paths:

- `spider/examples/params_template.json`
- or a project config (e.g., `yifan_redo/SPIDER_yifan.json`)

2) Run the full pipeline on one GPU:

```bash
python -m spider locate-full path/to/params.json --device 0
```

3) Recommended workflow: run Phase 1 once, then sample one or more chains:

```bash
# Phase 1 (MAP) -> writes <checkpoint_dir>/phase2_bundle.pth by default
python -m spider locate-map path/to/params.json --device 0

# Phase 2–4 (sampling) from the Phase‑2 bundle
python -m spider sample path/to/params.json --device 0
```

4) Multiple independent chains across GPUs:

```bash
python -m spider locate-map path/to/params.json --device 0
python -m spider sample-multi path/to/params.json --devices 0,1,2,3
```

See all commands and options:

```bash
python -m spider --help
```

## Input data formats (CSV)

SPIDER reads CSVs via **Polars** and expects:

- **Event catalog** (`io.catalog_infile`)
  - Required: `evid`, `longitude`, `latitude`, `depth`, `time`
  - `time` must be parseable as a datetime string

- **Stations** (`io.station_file`)
  - Required: `network`, `station`, `longitude`, `latitude`
  - Optional: `depth` (missing values treated as 0.0)

- **Differential times** (`io.dtime_file`)
  - Required: `network`, `station`, `evid1`, `evid2`, `dt`, `phase`
  - Optional: `cc`
  - `phase` may be `"P"/"S"` or `0/1` (normalized to `0=P`, `1=S`)

## Outputs

Configured under the `io` block:

- `io.catalog_outfile`: relocated catalog output
- `io.samples_outfile`: HDF5 MCMC samples (Phase 4)
- `io.checkpoint_dir`: phase‑tagged checkpoints and bundles

## Configuration guide (nested JSON)

Configs are **strict nested blocks**. Start from a template and modify.

Top‑level sections you will typically edit:

### `io`
Paths, checkpoints, and sample saving.

### `model`
- `model.model_file`: EikoNet checkpoint
- `model.domain`: spatial bounds and scale (`lon_min`, `lat_min`, `z_min`, `z_max`, `scale`)
- `model.priors`: event and centroid priors
- `model.likelihood`: residual model and correlated errors
- `model.filters`: dtimes/events/residual filters

### `inference`
- `inference.compute`: device list (single‑device commands also accept `--device`)
- `inference.sampler`: sampler backend and hyperparameters
- `inference.batching`: batch sizes and optional event‑batching
- `inference.diagnostics`: logging, W&B metrics, residual diagnostics

## Likelihood models

`model.likelihood.type` supports:

- `gaussian` / `l2` (alias: `mse`)
- `laplace` / `l1` / `mae`
- `huber`
- `student_t` (with `model.likelihood.student_t.nu`)
- `correlated` / `correlated_gaussian` (requires `shared_event_re.enabled=true`)

Key fields:

- `phase_unc`: per‑phase noise standard deviation `[P, S]`
- `sigma_distance_linear`: optional distance‑dependent sigma (linear in separation)

### Shared‑event correlated residuals

Enable with:

```json
"model": {
  "likelihood": {
    "type": "correlated_gaussian",
    "shared_event_re": { "enabled": true, "grouping": "station_phase", ... }
  }
}
```

Important options:

- `grouping`: `station_phase` (default in most configs)
- `tau_s`: per‑phase shared‑event scale
- `solver`: `pcg_sparse` (CPU) + optional GPU batching
- `whitening`: optional PCG whitening preconditioner
- `edge_weighting`: distance‑based edge weights (`distance_power` with scale/power)

## Samplers

`inference.sampler.backend` supports:

- `psgld`
- `sghmc`
- `adaptive_sghmc`
- `sgnht`
- `adsgld_adam`

Common settings:

- `lr`: per‑phase learning rates
- `temperature`: target temperature
- `preconditioning`: RMSProp‑style preconditioning

## Batching

`inference.batching.standard` controls Phase‑2/4 batch sizes:

- `warmup`: batch size for Phase‑2 drift
- `sgld`: batch size for Phase‑4 sampling
- `shuffle`: shuffle rows per epoch

Optional event‑level batching:

```json
"inference": { "batching": { "event_batches": { "enabled": true, ... } } }
```

## Diagnostics and logging

`inference.diagnostics` controls:

- W&B groups (`wandb.groups`)
- Residual distribution diagnostics
- Shared‑event correlation diagnostics (`shared_event_legcorr2d`)
- ESS/online diagnostics

## Python API (analysis/plotting)

For scripting and analysis:

- `spider.io.samples.read_all_samples`
- `spider.analysis.compute_cat_dd_and_xyz`
- `spider.plotting` helpers

## Repository structure

- `spider/`: core package
- `spider/core/`: inference + model
- `spider/io/`: reading/writing and samples
- `spider/analysis/`: diagnostics and calibration
- `spider/plotting/`: plotting utilities
- `spider/examples/`: minimal config template


