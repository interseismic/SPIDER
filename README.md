# SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

SPIDER is a Python toolkit for probabilistic earthquake relocation using differential travel times, neural travel‑time prediction, and scalable MCMC sampling. It combines a fast surrogate travel‑time model (EikoNet) with a multi‑phase inference pipeline to estimate event locations with uncertainty.

This is a brand new codebase. Please be patient with us as we work to making this usable by the broader scientific community.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [EikoNet training](#eikonet-training)
- [Input data formats](#input-data-formats)
- [Configuration](#configuration)
- [Additional configuration blocks](#additional-configuration-blocks)
- [CLI workflow](#cli-workflow)
- [Likelihoods and correlated residuals](#likelihoods-and-correlated-residuals)
- [Samplers](#samplers)
- [Batching and performance](#batching-and-performance)
- [Diagnostics](#diagnostics)
- [WandB outputs](#wandb-outputs)
- [Learning rate tuning (variance ratio)](#learning-rate-tuning-variance-ratio)
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

Use the lightweight example in `./example`:

```bash
# Phase 1 (MAP)
spider locate-map example/SPIDER_example.json --device 0

# Phase 2–4 (sampling)
spider sample example/SPIDER_example.json --device 0
```

Outputs are written to the paths defined in `example/SPIDER_example.json`.

Multi‑GPU independent chains:

```bash
spider sample-multi example/SPIDER_example.json --devices 0,1,2,3
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
- `likelihoods`: phase‑specific residual models (locate_map vs sample)
- `filters`: dtimes/events/residual filters

### `inference`

- `compute.devices`: list of GPU device ids
- `sampler`: backend and hyperparameters
- `batching`: batch sizes and optional event‑batching
- `diagnostics`: logging and post‑hoc diagnostics

## Additional configuration blocks

These are commonly used in real configs but not exhaustively listed above:

### Likelihood extras (sample)

- `model.likelihoods.sample.sigma_distance_linear`: distance‑dependent sigma (linear in separation)
- `model.likelihoods.sample.shared_event_re.whitening`: PCG whitening preconditioner options
- `model.likelihoods.sample.shared_event_re.edge_weighting`: distance‑based weights (`distance_power`)

### Filters

`model.filters.dtimes` (row‑level filters):

- `remove_duplicates`: drop duplicate dt rows.
- `max_abs_input_dt`: discard rows with |dt| above this threshold.
- `dtime_thin_frac`: random thinning fraction (0–1) to subsample dtimes.
- `flip_dt_sign`: if true, multiply dt by −1 (for convention changes).
- `cc_min`: drop rows with cross‑correlation below this threshold.

`model.filters.events` (event‑level filters):

- `min_dtimes`: minimum number of dt rows per event.
- `min_unique_phase_per_event`: minimum unique station‑phase picks per event.
- `min_dtimes_per_pair`: minimum dt rows per event pair.
- `min_event_degree`: minimum graph degree for each event.
- `min_events_per_cluster`: minimum cluster size to keep a component.
- `ratio_filter_phase`: whether to apply ratio filters `before` or `after` other filters.
- `linearization_error`: optional filter block for large linearization errors.

`model.filters.residual` (residual outliers, usually Phase‑2 only):

- `enabled`: turn the residual outlier filter on/off.
- `method`: outlier method (e.g., `mad`).
- `mad_sigma`: MAD threshold in standard‑deviation units.
- `abs_max`: absolute residual cap.

### Sampler details

- `inference.sampler.epochs_per_phase`: per‑phase epoch counts `[map, warmup, burnin, sample]`.
- `inference.sampler.dt_lr_mult`: learning‑rate scale for the dt parameter.
- `inference.sampler.eps`: numerical stabilizer for preconditioning.
- `inference.sampler.beta`: RMSProp/EMA decay for preconditioning stats.
- `inference.sampler.sghmc_alpha`: SGHMC friction (only for `backend="sghmc"`).
- `inference.sampler.preconditioning`: RMSProp config block (`enabled`, `type`).

### Diagnostics

- `inference.diagnostics.wandb.groups`: metric group switches
- `inference.diagnostics.shared_event_legcorr2d`: correlated‑residual diagnostics
- `inference.diagnostics.resid_distribution`: residual histograms/QQ
- `inference.diagnostics.shared_event_re_tau`: tau grid search
- `inference.diagnostics.resid_scalar_metrics`: binned residual metrics
- `inference.diagnostics.truth_catalog`: optional truth catalog for eval

### Runtime and safety

- `inference.runtime`: logging cadence, cache clearing, checkpoint behavior
- `inference.safety.max_abs_dX`: clamp on hypocenter step sizes

## CLI workflow

The CLI entrypoint is `spider`:

```bash
spider --help
```

Core commands:

```bash
# Phase 1 (MAP) -> writes <checkpoint_dir>/phase2_bundle.pth
spider locate-map my_params.json --device 0

# Phase 2–4 (sampling) from the Phase‑2 bundle
spider sample my_params.json --device 0

# Full pipeline (Phase 1–4)
spider locate-full my_params.json --device 0

# Multi‑GPU independent chains
spider sample-multi my_params.json --devices 0,1,2,3
```

## Likelihoods and correlated residuals

`model.likelihoods` provides separate likelihoods for Phase 1 (`locate_map`) and Phases 2–4 (`sample`).

### Base residual model

The correlated Gaussian likelihood uses per‑phase noise (for `model.likelihoods.sample`):

- `model.likelihoods.sample.type`: residual distribution (use `correlated_gaussian`).
- `phase_unc`: per‑phase noise standard deviation `[P, S]` applied to residuals.
- `sigma_distance_linear`: optional distance‑dependent sigma (linear in event‑pair separation) to broaden uncertainty for wide pairs.

## Priors

`model.priors` controls optional priors over events and the centroid:

`model.priors.event`:

- `enabled`: enable/disable event priors.
- `type`: prior family (currently `gaussian`).
- `params.std`: per‑event standard deviations for `[x, y, z, dt]` (units follow your domain).

`model.priors.centroid`:

- `enabled`: enable/disable centroid prior.
- `type`: prior family (currently `gaussian`).
- `params.std`: centroid standard deviations for `[x, y, z, dt]`.

### Shared‑event correlated residuals (whitening assumed ON)

For correlated Gaussian residuals, SPIDER uses a collapsed shared‑event random‑effects model,
grouped by station‑phase. In the current workflow, **whitening is assumed enabled** to
accelerate PCG solves and improve conditioning.

Enable:

```json
"model": {
  "likelihoods": {
    "locate_map": {
      "type": "laplace",
      "phase_unc": [0.02, 0.03]
    },
    "sample": {
      "type": "correlated_gaussian",
      "shared_event_re": {
        "enabled": true,
        "grouping": "station_phase",
        "tau_s": [0.03, 0.04],
        "max_nodes_per_group": 25000,
        "max_rows_per_group": 1500000,
        "solver": "pcg_sparse",
        "pcg_max_iters": 20,
        "pcg_tol": 5e-3,
        "gpu_max_groups_per_batch": 128,
        "gpu_max_edges_per_batch": 1500000,
        "gpu_reuse_pcg_init": true,
        "edge_weighting": "distance_power",
        "edge_weight_power": 0.25,
        "edge_weight_scale_km": 40.0,
        "edge_weight_global_scale": 1.0,
        "edge_weight_normalize": true,
        "edge_weight_eps_km": 1e-3,
        "whitening": {
          "enabled": true,
          "solver": "pcg",
          "pcg_batched": true,
          "pcg_bucket_nodes": [4096, 16384, 25000],
          "pcg_max_iters": 100,
          "pcg_tol": 3e-4,
          "pcg_min_iters": 2,
          "precompute": true,
          "precompute_device": "gpu"
        }
      }
    }
  }
}
```

Key pieces:

- **Grouping**: `grouping="station_phase"` groups residuals by station/phase for shared‑event correlations.
- **Shared‑event scale**: `tau_s` per phase sets the shared‑event random‑effect scale.
- **PCG solver**: `solver="pcg_sparse"` uses a sparse PCG solve on CPU; GPU batching controls memory.
- **Whitening (assumed ON)**: `shared_event_re.whitening.*` configures the PCG whitening preconditioner.
- **Edge weights**: distance‑based weighting of residual correlations.

Shared‑event correlated residual parameters:

- `shared_event_re.enabled`: turn on/off the correlated residual model (required for `type="correlated_gaussian"`).
- `shared_event_re.grouping`: grouping strategy (`station_phase` is standard).
- `shared_event_re.tau_s`: per‑phase shared‑event scales `[P, S]`.
- `shared_event_re.max_nodes_per_group`: cap group size to control memory/compute.
- `shared_event_re.max_rows_per_group`: cap total residual rows per group.
- `shared_event_re.solver`: linear solver (`pcg_sparse` for CPU PCG).
- `shared_event_re.pcg_max_iters`: PCG iteration cap for the correlated solve.
- `shared_event_re.pcg_tol`: PCG tolerance for the correlated solve.
- `shared_event_re.gpu_max_groups_per_batch`: GPU batching limit for correlated solves.
- `shared_event_re.gpu_max_edges_per_batch`: GPU edge limit per batch.
- `shared_event_re.gpu_reuse_pcg_init`: reuse PCG initial guesses to speed repeated solves.
- `shared_event_re.edge_weighting`: edge‑weight model (`distance_power` for distance‑based scaling).
- `shared_event_re.edge_weight_power`: power for distance‑based edge weights.
- `shared_event_re.edge_weight_scale_km`: distance scale (km) for edge weights.
- `shared_event_re.edge_weight_global_scale`: global multiplier on edge weights.
- `shared_event_re.edge_weight_normalize`: normalize weights to stabilize scaling across groups.
- `shared_event_re.edge_weight_eps_km`: epsilon (km) to avoid divide‑by‑zero in weights.
- `shared_event_re.whitening.enabled`: enable the whitening preconditioner.
- `shared_event_re.whitening.solver`: whitening solver (`pcg`).
- `shared_event_re.whitening.pcg_batched`: batch PCG whitening solves for speed.
- `shared_event_re.whitening.pcg_bucket_nodes`: bucket sizes for batched whitening.
- `shared_event_re.whitening.pcg_max_iters`: PCG iteration cap for whitening.
- `shared_event_re.whitening.pcg_tol`: PCG tolerance for whitening.
- `shared_event_re.whitening.pcg_min_iters`: minimum PCG iterations for whitening.
- `shared_event_re.whitening.precompute`: precompute whitening factors.
- `shared_event_re.whitening.precompute_device`: device for precomputation (`gpu` or `cpu`).

If you disable whitening, increase PCG iterations and expect slower/less stable solves.

## Samplers

`inference.sampler.backend` supports:

- `psgld`
- `sghmc`

Common settings:

- `backend`: sampler choice (`psgld` or `sghmc`).
- `epochs_per_phase`: epochs for phases 1–4 `[map, warmup, burnin, sample]`.
- `lr`: per‑phase learning rates `[phase1, phase2, phase3, phase4]` (scaled per‑obs).
- `dt_lr_mult`: multiplier for the dt parameter learning rate.
- `temperature`: target posterior temperature (1.0 = nominal posterior).
- `eps`: numerical stabilizer for preconditioning updates.
- `beta`: RMSProp/EMA decay for preconditioning statistics.
- `sghmc_alpha`: friction term for SGHMC (only used when `backend="sghmc"`).
- `preconditioning.enabled`: toggle RMSProp‑style preconditioning.
- `preconditioning.type`: preconditioner type (e.g., `rmsprop`).

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

## WandB outputs

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

### Analysis and plotting

The plotting/analysis utilities are typically driven from a summary produced by
`compute_cat_dd_and_xyz`. A minimal workflow:

```python
from spider.io.samples import read_all_samples
from spider.analysis import compute_cat_dd_and_xyz
from spider.plotting import (
    plot_event_distributions,
    plot_event_chains,
    plot_event_marginal_hist2d,
    plot_uncertainty_histograms,
)
from spider.plotting.events import plot_event_marginal_kde2d

samples = read_all_samples(params, thin=5)
summary = compute_cat_dd_and_xyz(samples, burn_in=100, include=["X", "Y", "Z", "T"])

# Posterior spreads for many events
fig, ax = plot_event_distributions(summary, coords=("X", "Y", "Z"), units="meters")

# Per-event chains
fig, ax = plot_event_chains(summary, coords=("X",), n_rows=6, n_cols=4)

# 2D marginal histograms
fig, axes = plot_event_marginal_hist2d(samples_data=samples, event_index=0, coords=("X", "Y", "Z"))

# KDE-based 2D marginals (single event)
fig, axes = plot_event_marginal_kde2d(samples_data=samples, event_index=0, coords=("X", "Y", "Z"))

# Aggregate uncertainty histograms
fig, ax = plot_uncertainty_histograms(summary, coords=("X", "Y", "Z", "T"))
```


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
