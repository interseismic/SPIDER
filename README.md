# SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![Docs](https://readthedocs.org/projects/spider-docs/badge/?version=latest)](https://spider-docs.readthedocs.io/en/latest/)

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

## Documentation site (Sphinx / Read the Docs)

SPIDER includes a dedicated Sphinx documentation site under `docs/`, suitable for Read the Docs.

Build locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Then open:

- `docs/_build/html/index.html`

## Installation

```bash
pip install -e /path/to/eikonet
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

SPIDER expects a trained EikoNet travel‑time model (`model.model_file`) created with the external `eikonet` package. SPIDER no longer trains or loads an in-repo legacy EikoNet implementation.

Example `eikonet.json`:

```json
{
  "velmod_file": "/path/to/velmod.csv",
  "x_max": 400.0,
  "y_max": 400.0,
  "z_min": -5.0,
  "z_max": 80.0,
  "model_file": "/path/to/eikonet_model.pt",
  "batch_size": 512,
  "n_train": 100000,
  "n_test": 100000,
  "n_epochs": 100,
  "lr": 1e-3
}
```

Train with the package CLI:

```bash
eikonet-train --config eikonet.json
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

SPIDER uses a strict **nested JSON** schema. The canonical validator/loader is implemented in:

- `spider/core/config_v2/validate.py`
- `spider/core/config_v2/load.py`
- `spider/core/priors_config.py`

Key sections:

### `io`

Paths and output settings:

- `dtime_file`, `station_file`, `catalog_infile`
- `catalog_outfile`, `samples_outfile`, `checkpoint_dir`
- `checkpoint_interval`, `save_every_n`, `write_samples`

### `model`

- `model_file`: EikoNet checkpoint produced by `eikonet`
- `domain`: `lon_min`, `lat_min`, `z_min`, `z_max`, `scale`
- Optional `eikonet` sub-block (advanced): override loader params like `x_max`, `y_max`, `model_kind`, `n_hidden`, `n_blocks`
- `priors`: event and centroid priors
- `likelihoods`: phase‑specific residual models (locate_map vs sample)
- `filters`: dtimes/events/residual filters

### `inference`

- `compute.devices`: list of GPU device ids
- `sampler`: backend and hyperparameters
- `batching`: batch sizes and optional event‑batching
- `runtime.torch` (optional): torch runtime/perf controls, including:
  - `allow_tf32`: bool
  - `matmul_precision`: `"highest" | "high" | "medium"`
  - `compile_eikonet`: bool (enable `torch.compile` on EikoNet model)
  - `compile_mode`: optional compile mode (`"default"`, `"reduce-overhead"`, `"max-autotune"`, etc.)
  - `compile_backend`: optional compile backend (e.g., `"inductor"`)
  - `compile_dynamic`: optional bool
  - `compile_fullgraph`: optional bool

### `observability`

- `wandb`: run enable/project/name
- `diagnostics`: metric grouping and online diagnostics toggles

## Additional configuration blocks

These are commonly used in real configs but not exhaustively listed above:

### Likelihood extras (sample)

- `model.likelihoods.sample.shared_event_re.solver`: whitening-first PCG solver options
- `model.likelihoods.sample.shared_event_re.edge_weights`: distance‑based edge weighting options

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

- `observability.diagnostics.wandb.groups`: metric group switches
- `observability.diagnostics.ess_online`: optional online ESS/IACT diagnostics

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

# Validate config schema (canonical config_v2)
spider validate-config my_params.json --mode sample
```

## Likelihoods and correlated residuals

`model.likelihoods` provides separate likelihoods for Phase 1 (`locate_map`) and Phases 2–4 (`sample`).

### Base residual model

The correlated Gaussian likelihood uses per‑phase noise (for `model.likelihoods.sample`):

- `model.likelihoods.sample.type`: residual distribution (use `correlated_gaussian`).
- `phase_unc`: per‑phase noise standard deviation `[P, S]` applied to residuals.

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
        "model": {
          "group_by": "station_phase",
          "tau_s": [0.03, 0.04],
          "cluster": { "mode": "none", "k": 1 }
        },
        "limits": {
          "max_nodes": 25000,
          "max_rows": 1500000
        },
        "fallback": {
          "to_diag": true,
          "abort_on_pcg_fallback": false
        },
        "numerics": {
          "jitter0": 1e-8,
          "jitter_max": 1e-3
        },
        "solver": {
          "kind": "pcg",
          "max_iters": 100,
          "min_iters": 2,
          "tol": 3e-4,
          "batched": true,
          "node_bin_edges": [4096, 16384, 25000],
          "warm_start": true,
          "cache_max_entries": 32,
          "prefetch_grouping": false,
          "profile_micro_steps": false,
          "merge_sparse_node_bins": true,
          "min_groups_per_node_bin": 32,
          "max_node_bins_per_node": 4,
          "precompute": {
            "enabled": true,
            "device": "gpu"
          }
        },
        "edge_weights": {
          "mode": "distance_power",
          "power": 0.25,
          "scale_km": 40.0,
          "global_scale": 1.0,
          "normalize": true,
          "eps_km": 1e-3
        },
        "autotune": {
          "enabled": true,
          "observe_epochs": 1,
          "latest_epoch": 2,
          "min_groups": 128,
          "max_node_bins": 8,
          "min_groups_per_node_bin": 24,
          "min_node_bin": 512,
          "min_gain": 0.08,
          "raise_nodes_cap": true,
          "nodes_cap_max": 65536
        },
        "logging": {
          "quiet": true,
          "stats_log_every_epochs": 0
        }
      }
    }
  }
}
```

Key pieces:

- **Model term**: `shared_event_re.model.group_by` defines how residuals are grouped (currently `station_phase`), and `shared_event_re.model.tau_s` sets per-phase shared-event scale.
- **Limits**: `limits.max_nodes` / `limits.max_rows` cap group sizes and directly control fallback risk.
- **PCG solver**: `shared_event_re.solver.*` configures the whitening-first PCG path.
- **Edge weights**: distance‑based weighting of residual correlations.

### PCG whitening setup guide

Use this sequence when standing up a new run:

1. Set `model.likelihoods.sample.type` to `correlated_gaussian` and `shared_event_re.enabled=true`.
2. Keep `shared_event_re.model.group_by="station_phase"` and provide physically sensible `tau_s`.
3. Start with conservative limits (`limits.max_nodes`, `limits.max_rows`) and allow fallback (`fallback.to_diag=true`) while tuning.
4. Use PCG (`solver.kind="pcg"`) and keep `solver.batched=true` unless debugging.
5. Enable warmup autotune (`autotune.enabled=true`) so node caps/bins can adjust early.
6. Turn on periodic stats (`logging.stats_log_every_epochs`) and inspect fallback/convergence counters.

### Parameter reference and tuning intent

`shared_event_re.model`:

- `group_by`: grouping strategy. Current supported value is `station_phase`.
- `tau_s`: per-phase RE scale `[P, S]`. Too small can force near-diagonal behavior (`tau_zero` fallbacks); too large can worsen conditioning.
- `cluster.mode`, `cluster.k`: optional grouping controls for cluster-aware behavior.

`shared_event_re.limits`:

- `max_nodes`: hard cap on group node count. Groups above this cap can fall back to diagonal.
- `max_rows`: hard cap on per-group row count. Groups above this cap can fall back to diagonal.
- Tune these first when you see many fallback reasons `rows_cap` / `nodes_cap`.

`shared_event_re.fallback`:

- `to_diag`: if `true`, problematic groups gracefully use diagonal approximation.
- `abort_on_pcg_fallback`: if `true`, raises immediately when fallback occurs; use for strict debugging/CI, usually `false` in production tuning.

`shared_event_re.numerics`:

- `jitter0`: base stabilizer added to the node-space system.
- `jitter_max`: upper bound for jitter escalation in robust solve paths.
- If you see non-finite/unstable solves, increase `jitter0` modestly before relaxing other controls.

`shared_event_re.solver` (core PCG controls):

- `kind`: keep as `pcg`.
- `max_iters`: upper bound on PCG iterations. Raise if convergence is consistently truncated.
- `min_iters`: force a minimum iteration count (helps avoid over-optimistic early exits on noisy batches).
- `tol`: convergence tolerance. Smaller is more accurate but slower.
- `batched`: enables grouped batched PCG (recommended).
- `node_bin_edges`: bucket edges used by batched PCG. Keep this consistent with observed group sizes so groups stay on the PCG path.
- `warm_start`: can help iterative stability across repeated group structures, but is optional for correctness.
- `cache_max_entries`: cache budget for whitening state. Too small may reduce warm-start reuse.
- `merge_sparse_node_bins`, `min_groups_per_node_bin`, `max_node_bins_per_node`: control sparse-bin consolidation for batched solves.
- `precompute.enabled`, `precompute.device`: optional precompute of whitening structures; does not change target distribution.

`shared_event_re.edge_weights`:

- `mode`: `uniform`, `distance_rbf`, `distance_linear`, or `distance_power`.
- `ell_km`: RBF length scale.
- `scale_km`: scale parameter for distance-power/linear formulations.
- `power`: exponent for `distance_power`.
- `eps_km`: distance floor for numerical safety.
- `global_scale`: global multiplier.
- `normalize`: normalize weights inside each group.

`shared_event_re.autotune` (warmup bucket optimizer):

- `enabled`: turns on one-shot warmup tuning.
- `observe_epochs` / `latest_epoch`: window where tuning is allowed.
- `min_groups`: minimum observed groups before making changes.
- `max_node_bins`: upper bound on node bins in proposed plan.
- `min_groups_per_node_bin`: sparsity threshold used by autotuner scoring.
- `min_node_bin`: smallest allowed node bin edge.
- `min_gain`: required score improvement to accept new bins.
- `raise_nodes_cap`: allows increasing `limits.max_nodes` when observed groups exceed current cap.
- `nodes_cap_max`: hard ceiling for autotuned `max_nodes`.

`shared_event_re.logging`:

- `quiet`: suppresses extra one-off whitening prints.
- `stats_log_every_epochs`: emits periodic per-epoch whitening stats and fallback reasons.

### Recommended convergence-first preset

For most large real-data runs, start with:

- `solver.max_iters: 50-100`
- `solver.tol: 1e-3 to 3e-4`
- `solver.min_iters: 2`
- `solver.batched: true`
- `limits.max_nodes: 25k-100k` depending on memory
- `limits.max_rows: 1.5M-4M`
- `numerics.jitter0: 1e-8` (increase gradually if solves are brittle)
- `autotune.enabled: true` with early window (`observe_epochs: 1`, `latest_epoch: 2`)

Optional operational knobs (`node_bin_edges`, cache, precompute, sparse-bin merge) can be tuned after convergence/fallback behavior is stable.

### How to verify PCG convergence and no-fallback behavior

Enable periodic logging and track these counters:

- `shared_event_re/groups_pcg_mean`
- `shared_event_re/groups_fallback_diag_mean`
- `shared_event_re/max_rows_max`, `shared_event_re/max_nodes_max`
- periodic console fallback summary: `rows_cap`, `nodes_cap`, `tau_zero`

Healthy signs:

- `groups_pcg_mean > 0`.
- `groups_fallback_diag_mean ~ 0` (or very close to zero).
- No repeated warnings that all groups fell back to diagonal.
- Fallback reason counters (`rows_cap`, `nodes_cap`, `tau_zero`) remain zero in steady state.

### Convergence/fallback troubleshooting

`fallback_diag` is high:

- Increase `limits.max_nodes` and/or `limits.max_rows`.
- Check `tau_s` is strictly positive for both phases.
- Temporarily set `fallback.abort_on_pcg_fallback=true` to force immediate failure and inspect causes.

All groups are falling back to diagonal:

- Raise `limits.max_nodes` until `max_nodes_max` is comfortably below the limit.
- Raise `limits.max_rows` until `max_rows_max` is comfortably below the limit.
- Verify `tau_s` has no zeros and is not effectively collapsed by config mistakes.
- Keep `solver.kind="pcg"` and `solver.batched=true`.

PCG convergence appears weak or brittle:

- Raise `max_iters`.
- Tighten `tol` if you need stricter convergence; relax only if solves become numerically fragile.
- Increase `jitter0` modestly for stability.
- Keep `fallback.abort_on_pcg_fallback=true` during debugging to catch failures early.

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
- `preconditioning.type`: preconditioner type (`rmsprop` or `lrd`).
- `preconditioning.lrd.rank` / `mode` / `update_every` / `buffer_size`: LRD controls.

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

`observability.diagnostics` controls:

- W&B logging groups
- Online ESS (optional)

## WandB outputs

Enable W&B with:

```json
"observability": {
  "wandb": {
    "enabled": true,
    "project_name": "spider_runs",
    "run_name": "my_run"
  }
}
```

Metric groups are controlled by `observability.diagnostics.wandb.groups`. Common groups:

- `core`: total loss, likelihood, priors
- `noise`: phase noise and variance‑related metrics
- `sampler`: sampler diagnostics (e.g., drift/noise ratios)
- `precond`: preconditioner stats (RMSProp moments)
- `resid_rms`: residual RMS by phase
- `corr_error`: correlated‑residual diagnostics (shared‑event RE)

If you do not see a group, check `observability.diagnostics.wandb.groups` in your config.

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
@article{ross2026spider,
  title={SPIDER: Scalable probabilistic inference for differential earthquake relocation},
  author={Ross, Zachary E and Wilding, John D and Azizzadenesheli, Kamyar and Kato, Aitaro},
  journal={Journal of Geophysical Research: Solid Earth},
  volume={131},
  number={3},
  pages={e2025JB032769},
  year={2026},
  publisher={Wiley Online Library}
}
```
