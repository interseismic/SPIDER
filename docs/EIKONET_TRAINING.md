# EikoNet Training Guide

This guide explains how to train EikoNet travel-time models for use with SPIDER, and what makes a
model good enough for relocation.

## Overview

EikoNet is a neural network that predicts seismic travel times through a velocity model by
solving the eikonal equation as a physics-informed loss, optionally combined with supervised
travel-time labels from a fast-marching (FMM) solver. SPIDER requires a pre-trained EikoNet
checkpoint for fast, differentiable travel-time evaluation during relocation. Training lives in
the separate `eikonet` package (`~/git/eikonet` in this repository's development setup).

The 1-D model (`EikoNet1D`) factorizes the travel time as `T = T0 * T1`, where `T0` is the
straight-ray time in a homogeneous medium (`vp`/`vs`) and `T1` is a learned correction that
depends only on the horizontal source-receiver distance, the two depths and the phase. It is
therefore exactly translation invariant in the horizontal plane; `x_max`/`y_max` only set the
normalization of the distance input.

## What matters for relocation: derivatives, not just travel times

SPIDER moves events with the *source-position gradient* of `T`. Two different accuracies are at
stake, and they are measured separately by `maunaloa/experiments/check_gradients.py`:

| network | `n_hidden` / `n_blocks` | travel-time rms | dT/dz_src error (fraction of the median derivative) |
|---|---|---|---|
| compact | 16 / 4 | ~10-20 ms | 15-40 % |
| recommended | 64 / 4 | comparable or better | 2-18 % depending on domain size |

A network whose travel times look fine can still have a poor depth derivative, which shows up in
SPIDER as biased relative depths. **Use `n_hidden: 64`, `n_blocks: 4`** (about 36k parameters,
still tiny), train with axisymmetric FMM labels (below), and run
`check_gradients.py` before using a model.

## Prerequisites

- Python 3.10+
- PyTorch (CUDA optional but strongly recommended)
- The `eikonet` package installed with the training extra:

```bash
pip install -e '/path/to/eikonet[train]'
```

The `train` extra pulls in `pandas`, `scikit-fmm` and `eikonalfm`. The FMM label generator needs
`scikit-fmm`; the default `factored` source treatment additionally needs `eikonalfm` and falls
back to a sphere-initialized `scikit-fmm` solve if it is missing.

## Training Configuration

### Recommended recipe (1-D model, 64-wide, axisymmetric labels)

This is `cahuilla/experiments/eikonet_h64_fine/eikonet.json` (a 500 km x 80 km domain; the Mauna
Loa recipe `maunaloa/experiments/eikonet_h64/eikonet.json` is identical apart from `x_max/y_max
200`, `z_max 30`, `r_max 80`). The trainer works directly in a local km frame bounded by `x_max`,
`y_max`, `z_min`, `z_max` -- there are **no** `lon_min`, `lat_min` or `scale` keys (those belong
to SPIDER's `model.domain`, see the integration section).

```json
{
  "velmod_file": "/path/to/velmod.csv",
  "x_max": 500.0,
  "y_max": 500.0,
  "z_min": -5.0,
  "z_max": 80.0,
  "radius_schedule": "cosine",
  "r_min": 0.1,
  "r_max": 100.0,
  "model_file": "model_state_dict.pt",
  "batch_size": 4096,
  "n_train": 1000000,
  "n_test": 1000000,
  "n_epochs": 240,
  "lr": 0.002,
  "n_hidden": 64,
  "n_blocks": 4,
  "n_fourier": 4,
  "phase_emb_dim": 6,
  "loss_w_pde": 1.0,
  "loss_w_tt": 0.2,
  "loss_w_tau": 0.1,
  "supervised_enabled": true,
  "supervised_file": "fmm_labels_axi_0p1km.npz",
  "supervised_train_split": 0.9,
  "supervised_seed": 0,
  "supervised_fraction_train": 0.5,
  "supervised_fraction_test": 0.5,
  "supervised_fraction_schedule": "match_radius",
  "supervised_fraction_min": 0.05,
  "supervised_follow_radius_curriculum": true,
  "supervised_loss": "huber",
  "supervised_huber_delta": 0.05,
  "supervised_use_tau_from_tt": true,
  "supervised_generate_fmm_if_missing": true,
  "supervised_fmm_mode": "axisymmetric",
  "supervised_fmm_axi_dr_km": 0.1,
  "supervised_fmm_axi_dz_km": 0.1,
  "supervised_fmm_train_size": 150000,
  "supervised_fmm_test_size": 60000,
  "supervised_fmm_sources_train": 256,
  "supervised_fmm_sources_test": 96,
  "supervised_fmm_seed": 0,
  "save_best": true,
  "save_every_epoch": false
}
```

On a V100 this takes about 15-20 minutes. The first run also writes the label file
(`supervised_file`) if it does not exist; later runs reuse it. `r_max` should cover the longest
source-receiver horizontal distance in your dtimes (not the whole domain).

### Core Parameter Descriptions

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `velmod_file` | string | 1-D velocity model CSV (`depth,vs,vp`) | one of `velmod_file` / `vel3d_file` required |
| `vel3d_file` | string | 3-D CVM-H style lat/lon/depth velocity file (takes precedence; trains `EikoNet3D`) | -- |
| `model_file` | string | Output checkpoint path (relative paths resolve against the config file) | Required |
| `x_max`, `y_max` | float | Horizontal domain extents (km) for 1-D models; derived from the velocity file for 3-D | `1.0`, `1.0` |
| `z_min`, `z_max` | float | Depth bounds (km); must cover all event depths and station elevations | `0.0`, `z_min + 1.0` |
| `arch` | string | `mlp` \| `hash` \| `reno` \| `hybrid` (SPIDER loads only `mlp`) | `mlp` |
| `batch_size` | int | Training batch size (number of optimizer steps per epoch is `n_train / batch_size`) | `128` |
| `test_batch_size` | int/null | Validation batch size (`null` = architecture-dependent) | `null` |
| `n_train` | int | Training samples per epoch (regenerated each epoch unless `regenerate_each_epoch=false`) | `100000` |
| `n_test` | int | Validation samples | `100000` |
| `n_epochs` | int | Training epochs | `10` |
| `lr` | float | Learning rate | `1e-3` |
| `lr_schedule` | string | `cosine` (or `cosine_annealing`); anything else = constant | `cosine` |
| `lr_min` | float/null | Cosine floor (`null` = `lr * 1e-2`) | `null` |
| `optimizer` | string | `adam` \| `adamw` | `adam` |
| `weight_decay` | float/null | `null` = 0.0 for `adam`, 0.01 for `adamw` (with no-decay grouping for biases/norms/embeddings) | `null` |
| `vp`, `vs` | float | Homogeneous reference velocities for the `T0` factor (km/s). The trainer always builds the network with 6.0 / 3.2 regardless of these keys, so leave them at the defaults (or omitted) in both the training config and SPIDER's `model.eikonet`, otherwise the loaded model differs from the trained one | `6.0`, `3.2` |
| `save_best` | bool | Save the best-validation snapshot as the final model | `true` |
| `save_every_epoch` | bool | Also overwrite `model_file` every epoch | `false` |
| `verbose` / `log_every` | bool / int | Logging controls (overridable with `--quiet` / `--log-every`) | `true` / `1` |
| `device` | string | Fallback device if `--device` / `EIKONET_DEVICE` are not given | `cuda:0` if available |

Gradient clipping is applied unconditionally at `max_norm = 1.0`; there is no `gradient_clip`
key. There is **no multi-GPU training support** -- a `devices` key is ignored; select one GPU with
`--device`.

### Network Architecture Parameters

These determine the state-dict layout and **must be mirrored under `model.eikonet` in the SPIDER
config**, or the checkpoint will not load:

| Parameter | Default | Notes |
|---|---|---|
| `n_hidden` | 128 | width of every hidden layer; **use 64** (see above) |
| `n_blocks` | 5 | residual blocks (2 linear layers each) after the input layer; 4 is enough |
| `n_fourier` | 4 | Fourier frequencies per scalar input (`use_fourier: true`) |
| `use_fourier` | true | |
| `phase_emb_dim` | 8 | P/S embedding width (recipes use 6) |
| `t2_log_scale` | 0.1 | stored in the model; only used by the 3-D `T2` branch |
| `vp`, `vs` | 6.0 / 3.2 | `T0` velocities |

`use_midpoint` and the `hash_*` keys belong to `arch: "hash"` only; `reno_*` / `hybrid_*` to the
other architectures.

### Advanced Parameters

- **PDE gradient:** `pde_gradient` (`autograd` \| `finite_diff`), `pde_fd_step_km` (0.05).
  `arch = reno` or `hybrid` requires `finite_diff` and raises otherwise; for `mlp` the autograd
  double-backward is the default and the most accurate.
- **PDE loss:** `pde_loss` (`l1` \| `l2` \| `huber` \| `log`, default `l1`), `pde_huber_delta` (0.05).
- **Loss weights:** `loss_w_pde` (1.0), `loss_w_tt` (0.0; travel-time labels), `loss_w_tau` (0.0;
  labels for the `T1` factor, derived from `tt` when `supervised_use_tau_from_tt=true`),
  `loss_w_grad` (1.0; active only when the label file carries receiver gradients, which the
  in-package generator always writes). `loss_w_tt`/`loss_w_tau` > 0 require
  `supervised_enabled=true`.
- **Sampling strategy:** `sampling` (`random` \| `weighted` \| `residual`) plus
  `sampling_weight_min/max` and the `residual_*` mixture controls
  (`residual_pool_factor`, `residual_frac_{uniform,residual,gradient,short}`,
  `residual_short_km`, `residual_clip_quantile`, `residual_grad_voxels`). `random` with
  `regenerate_each_epoch: true` (default) is what the recipes use.
- **Radius curriculum:** `radius_schedule` (`none` \| `linear` \| `cosine`), `r_min`, `r_max`,
  `z_src_min`, `z_src_max`. The maximum horizontal source-receiver distance grows from `r_min` to
  `r_max` over the run, which stabilizes early training. Validation sampling follows the same
  curriculum, so `save_best` compares like with like only once the radius has reached `r_max`
  (late in a cosine schedule; this is why the recipes use a long, smooth schedule rather than a
  short one).
- **Supervised / hybrid FMM training:** `supervised_enabled`, `supervised_file`,
  `supervised_train_split` (0.9), `supervised_seed`, `supervised_fraction_train` /
  `supervised_fraction_test` (fraction of each batch drawn from the label pool),
  `supervised_fraction_schedule` (`match_radius` (default) \| `none` \| `linear` \| `cosine`) with
  `supervised_fraction_min`, `supervised_follow_radius_curriculum` (only labels inside the
  current radius are used), `supervised_loss` (`huber` \| `l1` \| `mse`), `supervised_huber_delta`,
  `supervised_use_tau_from_tt`, and the in-package label generator
  (`supervised_generate_fmm_if_missing`, `supervised_fmm_method` (`factored` \| `sphere` \| `point`),
  `supervised_fmm_source_radius_cells`, `supervised_fmm_train_size`, `supervised_fmm_test_size`,
  `supervised_fmm_sources_train`, `supervised_fmm_sources_test`, `supervised_fmm_seed`,
  `supervised_fmm_mode`, `supervised_fmm_axi_dr_km/axi_dz_km`, `supervised_fmm_dx_km/dy_km/dz_km`).
- **3-D velocity smoothing:** `smooth_true3d` (true), `smooth_knn` (8), `smooth_power` (2.0).

### FMM label geometry (`supervised_fmm_mode`)

For a 1-D velocity model the travel time depends only on horizontal offset and the two depths, so
the 3-D eikonal problem reduces *exactly* to a 2-D solve in the `(rho, z)` plane.
`supervised_fmm_mode: "auto"` (default) uses this axisymmetric solve for 1-D models -- one factored
FMM per source depth on a `[-r_max, r_max] x [z_min, z_max]` plane at `supervised_fmm_axi_dr_km` /
`supervised_fmm_axi_dz_km` (default 0.1 km) -- and the full 3-D grid (`grid3d`, spacing
`supervised_fmm_dx_km/dy_km/dz_km`, defaults 8 / 8 / 2 km) only for 3-D models. Requesting
`axisymmetric` with a `vel3d_file` raises. Label error measured against a 0.5 / 0.125 km 3-D
reference on a Hawaii 1-D model (P, 5-30 km paths):

| mode | spacing (km) | label error rms | max | cost per source |
|---|---|---|---|---|
| axisymmetric | 0.1 | 0.6 ms (reference-limited) | 2 ms | 0.14 s per source *depth* |
| grid3d | 1 / 1 / 0.25 | 1.7 ms | 7 ms | 3 s (200 km domain), ~1 min (500 km) |
| grid3d | 2 / 2 / 0.5 | 8 ms | 41 ms | 0.3 s |
| grid3d | 4 / 4 / 1 | 28 ms | 138 ms | < 0.1 s |
| grid3d | 8 / 8 / 2 (default) | worse | | |

Sources are snapped to grid nodes (depth only in axisymmetric mode; `x/y` stay continuous) and the
snapped coordinates are recorded, so no snapping error enters the labels. Coarse labels bias the
network through the `loss_w_tt` / `loss_w_grad` terms -- a 4-8 km grid is not accurate enough for
relocation-grade models -- so for 3-D models keep the grid at <= 1 / 1 / 0.25 km. The domain
extents (`x_max`, `y_max`) only set the network's coordinate normalization for 1-D models; they do
not need to cover the label grid. The label `.npz` stores `x_*`, `tt_*`, `tau_*`, `radius_*` and
`grad_*` arrays for the train and test splits.

## Velocity Model Format

### CSV Structure

Create a CSV file with depth-dependent velocities:

```text
depth,vs,vp
-5.0,3.3,6.0
0.0,3.3,6.0
5.0,3.4,6.1
10.0,3.5,6.2
20.0,3.7,6.4
30.0,3.9,6.6
40.0,4.1,6.8
60.0,4.5,7.2
80.0,4.9,7.6
```

- **`depth`**: depth in kilometers (negative above sea level)
- **`vs`**, **`vp`**: S- and P-wave velocity in km/s

Velocities are interpolated linearly in depth and held constant outside the tabulated range.
Guidelines: cover the full `z_min`-`z_max` range, and include enough depth points for the
interpolation to represent your layered model (sharp discontinuities are fine for the FMM labels
but harder for the PDE loss; the supervised terms carry the network across them).

## Training Process

### Step 1: Prepare the velocity model

Write `velmod.csv` as above (or point `vel3d_file` at a CVM-H style file).

### Step 2: Configure training

Copy the recipe above, set `velmod_file`, the domain (`x_max`, `y_max`, `z_min`, `z_max`),
`r_max` and `model_file`.

### Step 3: Train

```bash
eikonet-train --config eikonet.json --device cuda:0
# equivalent:
python -m eikonet.cli.train --config eikonet.json --device cuda:0
```

Useful flags: `--quiet` (suppress logs), `--log-every N`. `--device` falls back to the
`EIKONET_DEVICE` environment variable, then `params["device"]`, then `cuda:0` if available. A bare
integer is accepted as a CUDA index. The log reports, per epoch, the training/validation PDE loss,
slowness mse and relative errors, the current curriculum radius, and (with labels) the
supervised `tt`/`tau`/`grad` losses.

From Python:

```python
from eikonet.training import train_from_config
train_from_config("eikonet.json", device="cuda:0")
```

### Step 4: Verify loading

Checkpoints are saved as **state dicts**, so loading requires the same geometry and architecture
parameters used for training. `load_model` accepts the training config directly:

```bash
python -c "
import json
from eikonet.io import load_model
params = json.load(open('eikonet.json'))
m = load_model(params['model_file'], params=params, device='cpu')
print('Loaded:', type(m).__name__, sum(p.numel() for p in m.parameters()), 'params')
"
```

`model_kind` is inferred from the config (`3d` when `vel3d_file` is set, `1d` otherwise); SPIDER
passes it explicitly from `model.eikonet.model_kind`.

### Step 5: Validate the depth derivative

```bash
python maunaloa/experiments/check_gradients.py eikonet.json [model_state_dict.pt] \
    [--zs 0.5 8] [--rho 1 40] [--zr -4 0] [--n 4000]
```

The script draws random source depths (`--zs`), horizontal distances (`--rho`) and receiver
depths (`--zr`), solves the exact axisymmetric FMM on a 0.05 km grid for each source depth, and
compares the network's travel time, `dT/dz_src` (by finite differences of the FMM field) and
`dT/drho`. For each phase it prints the travel-time rms error in ms, the `dT/dz_src` rms error in
ms/km and as a percentage of the median FMM derivative, and the same binned by distance. Aim for a
depth-derivative error well below 20 % over the distance range of your dtimes; if it is not, train
longer or wider, not on coarser labels.

## Training Tips

### Throughput

The networks are tiny, so training is bound by data handling and kernel-launch overhead, not by
GPU arithmetic. Two things determine epoch time:

- Batches are served from device-resident tensors (`_TensorLoader`: the six sample tensors are
  moved to the GPU once per epoch and each batch is a contiguous slice or an `index_select` with a
  fresh permutation), and epoch statistics accumulate on the device with a single host read-back
  per epoch. The earlier per-sample `Dataset`/`DataLoader` path cost more than the forward/backward
  pass.
- `batch_size` sets the number of optimizer steps per epoch (each step is a double-backward through
  the eikonal residual). On a V100 with `n_train = 1e6`: `batch_size 512` ~ 18 s/epoch,
  `4096` ~ 4 s/epoch, `16384` < 1 s/epoch. Larger batches need more epochs (or a higher `lr`) to
  reach the same number of steps; `4096` with `n_epochs 240`, `lr 2e-3` is a good compromise for the
  64-wide recipes and completes in ~15-20 minutes.

### Other tips

- **GPU memory:** reduce `batch_size` / `test_batch_size` if you run out of memory.
- **Accuracy:** increase `n_train` and `n_epochs`; the default `lr_schedule: cosine` already
  anneals the learning rate to `lr * 1e-2`.
- **Stability:** lower `lr` (e.g. `5e-4`) if the loss does not converge; the fixed gradient clip
  of 1.0 is always active.
- **Domain:** `x_max`/`y_max` must cover the full extent of your stations and events in the local
  km frame; `z_min`/`z_max` must span all event depths and station elevations. `r_max` should
  cover the longest horizontal source-receiver distance you will evaluate.
- **Quality assurance:** monitor training and validation loss, then run `check_gradients.py`.
  Model file size depends on `arch` (small for `mlp`, much larger for `reno`/`hybrid`).

## Integration with SPIDER

SPIDER loads EikoNet through the external `eikonet` package (`spider/core/eikonet_loader.py`).
In the SPIDER config, mirror the architecture of the trained model:

```json
{
  "model": {
    "model_file": "path/to/trained/model_state_dict.pt",
    "domain": { "lon_min": -119.6, "lat_min": 33.9,
                "z_min": -5.0, "z_max": 80.0, "scale": 500.0 },
    "eikonet": {
      "model_kind": "1d",
      "x_max": 500.0,
      "y_max": 500.0,
      "n_hidden": 64,
      "n_blocks": 4,
      "n_fourier": 4,
      "use_fourier": true,
      "phase_emb_dim": 6,
      "t2_log_scale": 0.1,
      "vp": 6.0,
      "vs": 3.2
    }
  }
}
```

- `model.eikonet.x_max` / `y_max` **must equal** the training `x_max` / `y_max` (they set the
  `r_scale` normalization inside the network). If omitted, SPIDER falls back to
  `model.domain.scale` for both.
- `model.domain.z_min` / `z_max` must match the training `z_min` / `z_max`.
- SPIDER's defaults for omitted architecture keys are the trainer defaults (`n_hidden 128`,
  `n_blocks 5`, `n_fourier 4`, `use_fourier true`, `phase_emb_dim 8`, `t2_log_scale 0.1`,
  `vp 6.0`, `vs 3.2`), so a 64-wide model **must** have `n_hidden`, `n_blocks` and
  `phase_emb_dim` spelled out.
- `model.eikonet.model_kind` accepts `"1d"`, `"3d"`, or `"gridtt"` (the latter loads a GridTT
  travel-time table operator from the separate `gridtt` package instead of an EikoNet;
  `model_file` is then the table directory written by `gridtt build-spider`). **SPIDER cannot
  load `hash`, `reno` or `hybrid` checkpoints**, even though `eikonet` can train them.
- Checkpoints whose tensors do not match the requested architecture (wrong `n_hidden`, or models
  trained with the pre-package `EikoNet` code, with `input_layer`/`blocks` layers) are
  **rejected** by `eikonet.io.load_model` with a `ValueError` listing the mismatched keys. Retrain
  legacy models with the current package; a silent partial load would otherwise leave the network
  untrained.
- `model.domain.lon_min` / `lat_min` are SPIDER's LAEA projection origin; they have no
  counterpart in the training config. Because the 1-D model is translation invariant in the
  horizontal plane, any origin works as long as all projected coordinates stay within
  `x_max`/`y_max` of it.

### Zero-offset sources

The travel-time model adds a squared-distance floor of `1e-6 km^2` (`_DIST_EPS2`) inside the
`sqrt()` of the `T0` and `T1` terms (a 1 m distance floor), so gradients remain finite when a
source coincides exactly with a receiver. Without it, such geometries produce NaN gradients; in
SPIDER this surfaces as the `epoch N: skipped k/m batches ... ALL batches skipped -> optimization
is frozen` warning. Retraining is not required for existing checkpoints: the floor changes travel
times by ~1e-7 relative at realistic source-receiver distances.

---

See also the [configuration reference](configuration-reference.md) for the `model.eikonet`
block and `docs/SPIDER_User_Guide.tex` (section "EikoNet travel-time model").
