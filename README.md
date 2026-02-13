# SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

SPIDER is a Python toolkit for probabilistic earthquake relocation using differential travel times, neural travel‑time prediction, and scalable MCMC sampling. It combines a fast surrogate travel‑time model (EikoNet) with a multi‑phase inference pipeline to estimate event locations with uncertainty.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [EikoNet training](#eikonet-training)
- [Input data formats](#input-data-formats)
- [Configuration](#configuration)
- [CLI workflow](#cli-workflow)
- [Likelihoods and correlated residuals](#likelihoods-and-correlated-residuals)
- [Samplers](#samplers)
- [Batching and performance](#batching-and-performance)
- [Diagnostics](#diagnostics)
- [W\&B outputs](#wandb-outputs)
- [Learning rate tuning (variance ratio)](#learning-rate-tuning-variance-ratio)
- [Python API](#python-api)
- [Example configuration](#example-configuration)
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

## EikoNet training

SPIDER expects a trained EikoNet travel‑time model (`model.model_file`). Train it once for your velocity model and spatial domain.

Example `eikonet.json`:

```json
{
  "velmod_file": "/path/to/velmod.csv",
  "lon_min": -117.5,
  "lat_min": 33.0,
  "z_min": -5.0,
  "z_max": 80.0,
  "scale": 400.0,
  "model_file": "/path/to/model_state_dict.pt",
  "train_batch_size": 512,
  "val_batch_size": 10000,
  "n_train": 1000000,
  "n_test": 2000000,
  "n_epochs": 1000,
  "lr": 1e-3
}
```

Train (script in repo root):

```bash
python eikonet_train_1D.py eikonet.json
```

Velocity model CSV format:

```csv
depth,vs,vp
-5.0,3.3,6.0
0.0,3.3,6.0
5.0,3.4,6.1
10.0,3.5,6.2
...
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

## W&B outputs

Enable W&B with:

```json
"wandb": {
  "enabled": true,
  "project_name": "spider_runs",
  "run_name": "my_run"
}
```

Metric groups are controlled by `inference.diagnostics.wandb.groups`. Common groups:

- `core`: total loss, likelihood, priors
- `noise`: phase noise and variance‑related metrics
- `sampler`: sampler diagnostics (e.g., drift/noise ratios)
- `precond`: preconditioner stats (RMSProp moments)
- `resid_rms`: residual RMS by phase
- `corr_error`: correlated‑residual diagnostics (shared‑event RE)

If you do not see a group, check `inference.diagnostics.wandb.groups` in your config.

## Learning rate tuning (variance ratio)

For SGHMC‑style samplers, SPIDER logs a variance‑ratio diagnostic:

- `grad_noise_to_langevin_*`: ratio of minibatch‑gradient noise variance to injected Langevin noise variance.

Target range for stable sampling is typically **~0.1–1.0**. Use this heuristic:

- **Ratio > 1.0**: reduce `inference.sampler.lr` (Phase‑4) by 2–5×.
- **Ratio < 0.1**: increase `inference.sampler.lr` by 2–5×.

Keep batch size, temperature, and preconditioning fixed while tuning `lr`. Once the
ratio is in range, you can fine‑tune for mixing speed.

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

## Example configuration

This is a real, working nested config with paths shortened for readability:

```json
{
  "io": {
    "dtime_file": "/path/to/dtimes.csv",
    "station_file": "/path/to/stations.csv",
    "catalog_infile": "/path/to/events.csv",
    "catalog_outfile": "/path/to/SPIDER_out.cat",
    "samples_outfile": "/path/to/SPIDER_samples.h5",
    "checkpoint_dir": "/path/to/checkpoints/",
    "checkpoint_interval": 100,
    "save_every_n": 10,
    "write_samples": true
  },
  "wandb": {
    "enabled": true,
    "project_name": "spider_paper",
    "run_name": "run_001"
  },
  "model": {
    "model_file": "/path/to/model_state_dict.pt",
    "domain": {
      "lon_min": -119.6127,
      "lat_min": 33.8856,
      "z_min": -2.0,
      "z_max": 30.0,
      "scale": 400.0
    },
    "priors": {
      "event": {
        "enabled": true,
        "type": "gaussian",
        "params": { "std": [0.5, 0.5, 0.5, 0.2] }
      },
      "centroid": {
        "enabled": true,
        "type": "gaussian",
        "params": { "std": [0.01, 0.01, 0.01, 0.01] }
      }
    },
    "likelihood": {
      "type": "correlated_gaussian",
      "phase_unc": [0.02, 0.03],
      "shared_event_re": {
        "enabled": true,
        "grouping": "station_phase",
        "tau_s": [0.033, 0.06],
        "max_nodes_per_group": 25000,
        "max_rows_per_group": 1500000,
        "solver": "pcg_sparse",
        "pcg_max_iters": 20,
        "pcg_tol": 5e-3,
        "gpu_max_groups_per_batch": 128,
        "gpu_max_edges_per_batch": 1500000,
        "gpu_reuse_pcg_init": true,
        "whitening": {
          "enabled": true,
          "solver": "pcg",
          "pcg_batched": true,
          "pcg_bucket_nodes": [4096, 16384, 25000],
          "pcg_max_iters": 100,
          "pcg_tol": 3e-4,
          "pcg_min_iters": 2,
          "precompute": true,
          "precompute_device": "gpu",
          "edge_weighting": "distance_power",
          "edge_weight_power": 0.25,
          "edge_weight_scale_km": 40.0,
          "edge_weight_global_scale": 1.0,
          "edge_weight_normalize": true,
          "edge_weight_eps_km": 1e-3
        }
      }
    },
    "filters": {
      "dtimes": {
        "remove_duplicates": true,
        "max_abs_input_dt": 99999,
        "dtime_thin_frac": 1.0,
        "flip_dt_sign": false,
        "cc_min": 0.0
      },
      "events": {
        "min_dtimes": 1,
        "min_unique_phase_per_event": 8,
        "min_dtimes_per_pair": 1,
        "min_event_degree": 3,
        "min_events_per_cluster": 0,
        "ratio_filter_phase": "before"
      },
      "residual": {
        "enabled": false,
        "method": "mad",
        "mad_sigma": 3.0,
        "abs_max": 99999
      }
    }
  },
  "inference": {
    "compute": { "devices": [1, 3] },
    "sampler": {
      "backend": "psgld",
      "epochs_per_phase": [1000, 100, 100, 100000],
      "lr": [1e-3, 1e-2, 1e-2, 1e-2],
      "dt_lr_mult": 0.2,
      "temperature": 1.0,
      "eps": 1e-5,
      "preconditioning": { "enabled": true, "type": "rmsprop" },
      "beta": 0.99
    },
    "batching": {
      "standard": { "warmup": 1000000, "sgld": 1000000, "shuffle": false },
      "event_batches": { "enabled": false, "events_per_batch": 50, "max_edges_per_batch": 200000 }
    },
    "diagnostics": {
      "pair_count_stats_enable": true,
      "display_precond_every": 50,
      "wandb": { "enabled": true }
    }
  }
}
```

See the full file for all diagnostics and runtime options: `yifan_redo/SPIDER_yifan.json`.

## Citation

If you use SPIDER in your research, please cite:

```bibtex
@misc{ross2026spiderscalableprobabilisticinference,
  title={SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation},
  author={Zachary E. Ross and John D. Wilding and Kamyar Azizzadenesheli and Aitaro Kato},
  year={2026},
  eprint={2508.12117},
  archivePrefix={arXiv},
  primaryClass={physics.geo-ph},
  url={https://arxiv.org/abs/2508.12117}
}
```
