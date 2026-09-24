# Config Walkthrough

This page walks through `spider/examples/params_template.json` block by block. Every key is
listed with one line on what it does; each block ends with the values to change first when you
point SPIDER at a new dataset. The template already carries the recommended settings (hierarchical
event prior, tight centroid prior, two-pass MAP with a cosine schedule, residual and
linearization filters, 64-wide EikoNet), so a new run is usually "fix the paths and the domain,
then look at the data filters".

```bash
cp spider/examples/params_template.json my_region.json
# edit paths and domain, then:
spider validate-config my_region.json --mode locate-map
```

For the meaning of each value beyond one line see {doc}`configuration-reference`; for the
statistical model behind the priors and likelihoods see {doc}`probability-model`. A complete,
runnable instance of this template is `example/SPIDER_example.json` ({doc}`quickstart`).

## Block by block

### `io`

```json
"io": {
    "dtime_file": "PATH/TO/dtimes.csv",
    "station_file": "PATH/TO/stations.csv",
    "catalog_infile": "PATH/TO/events.csv",
    "catalog_outfile": "PATH/TO/output_catalog.csv",
    "samples_outfile": "PATH/TO/output_samples.h5",
    "checkpoint_dir": "PATH/TO/checkpoints/",
    "checkpoint_interval": 1000,
    "sample_write_interval": 1000,
    "save_every_n": 10,
    "write_samples": true
}
```

- `dtime_file`, `station_file`, `catalog_infile`: the three input CSVs (columns in
  {doc}`data-formats`). Relative paths are resolved from the directory you run `spider` in.
- `catalog_outfile`: a prefix, not a file name. `locate-map` writes `<prefix>_MAP.csv` (and
  `<prefix>_MAP_pass1.csv` with two-pass MAP).
- `samples_outfile`: HDF5 store for posterior samples; `locate-map` also puts the MAP locations
  in it, and `spider summarize` reads it.
- `checkpoint_dir`: where `phase2_bundle.pth` and `checkpoint_<phase>_epoch_<n>.pth` go.
- `checkpoint_interval`: epochs between checkpoints (all phases).
- `sample_write_interval`: epochs between flushes of buffered samples to `samples_outfile`
  (defaults to `checkpoint_interval`; `0` keeps only the final flush).
- `save_every_n`: in Phase 4, keep one sample every `save_every_n` optimizer steps (minibatches).
- `write_samples`: `false` disables the HDF5 store entirely.

Change first: all six paths. Leave the intervals alone until a run is stable.

### `model.model_file` and `model.eikonet`

```json
"model_file": "PATH/TO/eikonet_model.pt",
"eikonet": {
    "model_kind": "1d",
    "n_hidden": 64,
    "n_blocks": 4,
    "n_fourier": 4,
    "phase_emb_dim": 6,
    "use_fourier": true,
    "t2_log_scale": 0.1
}
```

- `model_file`: the EikoNet state dict (or, for `model_kind: "gridtt"`, a travel-time table
  directory). Trained with the separate `eikonet` package; see {doc}`EIKONET_TRAINING`.
- `model_kind`: `1d` (default) or `3d` for EikoNet checkpoints, `gridtt` for tables.
- `n_hidden`, `n_blocks`, `n_fourier`, `phase_emb_dim`, `use_fourier`, `t2_log_scale`: must
  match the training config exactly. The loader refuses a checkpoint whose tensors do not fit,
  so a mismatch fails at start-up rather than silently loading garbage. Omitted keys fall back to
  the trainer defaults (`n_hidden 128`, `n_blocks 5`, `phase_emb_dim 8`), which is why a 64-wide
  model needs them spelled out.
- `x_max`, `y_max` (not in the template): horizontal normalisation of the network; default to
  `model.domain.scale`. Set them explicitly if the model was trained with a different extent.
- `vp`, `vs` (not in the template): reference velocities of the straight-ray `T0` term; defaults
  `6.0` and `3.2` match the trainer.

Change first: `model_file`, then copy the architecture keys from the `eikonet.json` you trained
with. A 64-wide, 4-block model is the recommended size; 16-wide test models have poor depth
derivatives.

### `model.domain`

```json
"domain": {
    "lon_min": -120.0,
    "lat_min": 35.0,
    "z_min": -5.0,
    "z_max": 30.0,
    "scale": 400.0
}
```

- `lon_min`, `lat_min`: origin of SPIDER's local projection. Put it near the south-west corner
  of your catalog so all projected coordinates are positive and inside `scale`.
- `z_min`, `z_max`: depth range (km, positive down) the EikoNet was trained on. Must match the
  training `z_min`/`z_max`.
- `scale`: horizontal extent (km) used to normalise coordinates; also the fallback for
  `model.eikonet.x_max`/`y_max`. Must match the training `x_max`/`y_max`.

Change first: all five, together with the EikoNet you are loading. The shipped example uses
`lon_min -119.6127, lat_min 33.8856, z -5..80, scale 500` for a model trained on exactly that box.

### `model.priors`

```json
"priors": {
    "event": {
        "enabled": true,
        "type": "gaussian",
        "params": { "std": [5.0, 5.0, 5.0, 2.0] },
        "hyper": {
            "enabled": true,
            "type": "wishart_precision",
            "params": { "df": 10.0, "scale_std": [0.5, 0.5, 0.5, 0.2] },
            "update": { "every_epochs": 10, "start_epoch": 200, "min_cluster_events": 100 }
        }
    },
    "centroid": {
        "enabled": true,
        "type": "gaussian",
        "params": { "std": [0.01, 0.01, 0.01, 0.01] }
    }
}
```

`event` is the prior on each event's shift `[dx, dy, dz, dt]` (km, km, km, s) from its catalog
location; `centroid` pins the mean shift of each connected component.

- `event.enabled`, `event.type`: `gaussian` is the only type.
- `event.params.std`: the fixed Gaussian width. With the hyperprior on it acts only until
  `update.start_epoch`; keep it loose (`[5, 5, 5, 2]`) so the MAP is not pulled back to the
  catalog.
- `event.hyper.enabled`, `event.hyper.type`: turn on the Wishart hyperprior on the shift
  precision (`wishart_precision` is the only type). Recommended on.
- `event.hyper.params.df`: Wishart degrees of freedom; must be `> 3`. `10` is a mild prior that
  lets a few hundred events set the scale.
- `event.hyper.params.scale_std`: prior guess of the shift scale, per component
  (`[0.5, 0.5, 0.5, 0.2]` = 500 m and 0.2 s).
- `event.hyper.update.every_epochs`: cadence of the precision update (`10`).
- `event.hyper.update.start_epoch`: keep the fixed prior for this many epochs of each MAP pass
  before the first update (`200`), so the scale is not estimated from shifts that are still zero.
  `inference.sampler.epochs_per_phase[0]` must be well above this or the update never fires.
- `event.hyper.update.min_cluster_events`: components smaller than this share the precision
  estimated from all events instead of fitting their own (`100`).
- `centroid.enabled`, `centroid.type`, `centroid.params.std`: per-component Gaussian on the mean
  shift, `[0.01, 0.01, 0.01, 0.01]` (10 m, 10 ms). This fixes the translation gauge of every
  component; keep it tight unless you deliberately want components to drift as a whole.

Change first: nothing for a typical local catalog. If your catalog is very rough, widen
`event.params.std` (it only matters before `start_epoch`); if you have many small components,
lower `min_cluster_events`.

### `model.likelihoods`

```json
"likelihoods": {
    "locate_map": { "type": "laplace", "phase_unc": [0.03, 0.03] },
    "sample": {
        "type": "correlated_gaussian",
        "phase_unc": [0.03, 0.03],
        "shared_event_re": {}
    }
}
```

(`shared_event_re` is abbreviated here; its sub-blocks are listed below.) Two likelihood groups: `locate_map` is used in Phase 1, `sample` in Phases 2 to 4.

- `locate_map.type`: `laplace` (robust L1) is recommended for the MAP; `gaussian`, `huber`
  (`huber_delta`) and `student_t` (`student_t.nu`) are also accepted.
- `locate_map.phase_unc`, `sample.phase_unc`: fixed `[sigma_P, sigma_S]` noise scales in seconds.
  These set the residual scale everything else is measured against.
- `sample.type`: `correlated_gaussian` enables the collapsed shared-event random effect below;
  `gaussian` gives the plain diagonal likelihood.
- `sample.shared_event_re.enabled`: turn the random effect on.
- `shared_event_re.model.group_by`: `station_phase` (only supported value).
- `shared_event_re.model.tau_s`: `[tau_P, tau_S]` scale (s) of the per-event, per-station-phase
  shared term. Set both `> 0`; the template uses `[0.03, 0.04]`.
- `shared_event_re.model.cluster.mode`, `cluster.k`: `none` / `1` (no sub-clustering).
- `shared_event_re.limits.max_nodes`, `limits.max_rows`: largest group solved with the correlated
  form; bigger groups fall back to the diagonal likelihood. `25000` / `1500000` cover most
  catalogs.
- `shared_event_re.fallback.to_diag`, `fallback.abort_on_pcg_fallback`: allow (or abort on) the
  diagonal fallback when a group exceeds the caps or PCG fails.
- `shared_event_re.numerics.jitter0`, `numerics.jitter_max`: diagonal jitter for the solve.
- `shared_event_re.solver.*`: PCG settings (`kind: pcg`, `max_iters`, `min_iters`, `tol`,
  `batched`, `node_bin_edges`, `warm_start`, `cache_max_entries`, `prefetch_grouping`,
  `profile_micro_steps`, `merge_sparse_node_bins`, `min_groups_per_node_bin`,
  `max_node_bins_per_node`, `precompute.enabled`, `precompute.device`). Leave them; `autotune`
  adjusts the bins from the observed group sizes. See {doc}`pcg-whitening-convergence`.
- `shared_event_re.edge_weights.*`: `mode` (`uniform` in the template; `distance_power` in the
  example), `ell_km`, `eps_km`, `power`, `scale_km`, `global_scale`, `normalize`.
- `shared_event_re.autotune.*`: `enabled`, `observe_epochs`, `latest_epoch`, `min_groups`,
  `max_node_bins`, `min_groups_per_node_bin`, `min_node_bin`, `min_gain`, `raise_nodes_cap`,
  `nodes_cap_max`.
- `shared_event_re.logging.quiet`, `logging.stats_log_every_epochs`: console verbosity of the
  solver.

Change first: `phase_unc` in both groups (your pick uncertainty, typically 0.02 to 0.05 s for
cross-correlation times) and `tau_s` (comparable to or a little larger than `phase_unc`). Watch
the `[LIKELIHOOD]` fallback statistics on the first sampling epochs before touching `limits` or
`solver`.

### `model.filters`

```json
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
        "min_unique_phase_per_event": 1,
        "min_dtimes_per_pair": 1,
        "min_event_degree": 1,
        "min_events_per_cluster": 3,
        "max_pair_station_ratio": 1.0,
        "ratio_filter_phase": "before",
        "linearization_error": {
            "enabled": true, "phase": "after_phase1", "max_ratio": 0.05,
            "batch_size": 50000, "sample_size": 200000, "log_every_batches": 25
        }
    },
    "residual": { "enabled": true, "method": "abs", "abs_max": 0.15, "phase": "after_phase1" }
}
```

Row filters (`dtimes`):

- `remove_duplicates`: keep one row per (pair, network, station, phase).
- `max_abs_input_dt`: drop rows with `|dt|` at or above this (s); `99999` disables.
- `dtime_thin_frac`: random fraction of rows to keep (`1.0` keeps all).
- `flip_dt_sign`: negate `dt` on input. SPIDER's convention is `dt = t(evid2) - t(evid1)`;
  HypoDD `dt.cc` files need `true`.
- `cc_min`: drop rows whose `cc` column is below this (`0` disables).

Event and pair filters (`events`):

- `min_dtimes`: minimum rows per pair after the other filters.
- `min_unique_phase_per_event`: minimum distinct station-phases per event (the example uses `8`).
- `min_dtimes_per_pair`: minimum rows per pair right after `cc_min`.
- `min_event_degree`: minimum number of partner events (the example uses `3`).
- `min_events_per_cluster`: drop connected components smaller than this, at data prep and again
  after Phase 1 (`3`; a two-event component cannot constrain its own gauge).
- `max_pair_station_ratio`, `ratio_filter_phase`: drop rows whose pair separation exceeds this
  fraction of the pair-to-station distance; `1.0` is effectively off.
- `lat_bounds`, `lon_bounds` (not in the template): optional `[min, max]` catalog subset.
- `linearization_error.enabled`, `.phase`, `.max_ratio`: drop rows whose linearised
  travel-time error exceeds `max_ratio` of the differential time, evaluated at the MAP locations
  (`after_phase1`). `batch_size`, `sample_size`, `log_every_batches` only control the sweep.

Residual filter (`residual`):

- `enabled`, `method`, `abs_max`, `phase`: drop rows with `|residual| > abs_max` (s) at the MAP
  locations (`abs`, `0.15`, `after_phase1`). `method: mad` with `mad_sigma` is the alternative.

Change first: `flip_dt_sign` (check your sign convention), `cc_min` if you have a `cc` column,
then `min_unique_phase_per_event` and `min_event_degree` to trim poorly connected events. Set
`abs_max` to a few times your `phase_unc`.

### `inference.sampler`

```json
"sampler": {
    "backend": "sghmc",
    "epochs_per_phase": [2000, 0, 10, 10000],
    "lr": [0.001, 1e-05, 1e-05, 1e-05],
    "temperature": 1.0,
    "noise_scale_mult": 1.0,
    "sghmc_alpha": 0.01,
    "eps": 1e-08,
    "preconditioning": { "enabled": true, "type": "rmsprop", "include_gamma": false },
    "beta": 0.999,
    "freeze_preconditioner_sampling": false,
    "lr_schedule": { "phase1": { "type": "cosine", "eta_min": 1e-05 } },
    "phase1_two_pass": { "enabled": true, "epochs": null, "warm_start": true }
}
```

- `backend`: `sghmc` or `psgld` (the example uses `psgld`; see {doc}`sampler-options`).
- `epochs_per_phase`: `[Phase 1 MAP, Phase 2 preconditioner warm-up, Phase 3 noise ramp, Phase 4
  sampling]`. Phase 1 must comfortably exceed `priors.event.hyper.update.start_epoch`.
- `lr`: `lr[0]` is the Phase-1 Adam rate; `lr[1..3]` are per-observation sampler step sizes.
- `temperature`: injected-noise temperature (`1.0` = the posterior; `0` = noise off).
- `noise_scale_mult`: extra multiplier on the injected noise in Phases 3 and 4.
- `sghmc_alpha`: SGHMC friction; required key (a number for `sghmc`, `null` allowed for `psgld`).
- `eps`, `beta`: RMSprop damping and second-moment decay of the preconditioner.
- `preconditioning.enabled`, `.type`, `.include_gamma`: `rmsprop` diagonal preconditioner
  (`lrd`/`component_lrd` with a `preconditioning.lrd` sub-block are the alternatives).
- `freeze_preconditioner_sampling`: freeze preconditioner statistics in Phases 3 and 4.
- `lr_schedule.phase1.type`, `.eta_min` (and optional `.T_max`): cosine decay of the Phase-1
  learning rate from `lr[0]` to `eta_min` over the pass. Recommended on.
- `phase1_two_pass.enabled`, `.epochs`, `.warm_start`: run the MAP, apply the post-MAP filters at
  the relocated positions, then run the MAP again (`epochs: null` reuses `epochs_per_phase[0]`).
  Recommended on.
- `reparameterization` (not in the template; `enabled`, `spatial_scale`, `dt_scale`): blocked
  sampler coordinates; the example uses `dt_scale: 0.3`.
- `grad_clip_norm`, `overrides` (not in the template): gradient clipping and per-group sampler
  overrides; leave them out.

Change first: `epochs_per_phase` (enough Phase-1 epochs for the loss to flatten, then Phase-4
length for the number of samples you want) and `lr[0]` if the `phase1` lines show `dr_max`
pinned at the safety clamp. Sampler step sizes come after a healthy MAP.

### `inference.batching`

```json
"batching": {
    "standard": { "warmup": 250000, "sgld": 1000000, "shuffle": false },
    "event_batches": {
        "enabled": false, "events_per_batch": 50, "max_edges_per_batch": 200000,
        "bucket_reorder_all": false, "bucket_reuse_epochs": 1
    }
}
```

- `standard.warmup`: rows per minibatch in Phase 1; `standard.sgld`: rows per minibatch in
  Phases 2 to 4; `standard.shuffle`: reshuffle rows each epoch.
- `event_batches.*`: batch by owner event instead of by row (`enabled`, `events_per_batch`,
  `max_edges_per_batch`, `bucket_reorder_all`, `bucket_reuse_epochs`). Off by default.

Change first: lower `warmup`/`sgld` only if you run out of GPU memory; larger batches mean fewer,
cleaner gradient steps per epoch.

### `inference.safety`

```json
"safety": { "max_abs_dX": [10.0, 10.0, 10.0, 3.0] }
```

- `max_abs_dX`: hard clamp on each event's shift `[dx, dy, dz, dt]` (km, km, km, s). `null`
  disables it.

Change first: only if `dr_max` in the `phase1` log lines sits at the clamp every epoch, and then
check the sign convention before widening it.

### `inference.runtime` and `inference.compute`

```json
"compute": { "devices": [0] },
"runtime": {
    "cuda_empty_cache_every": 0,
    "reset_batch_numbers": true,
    "clear_samples_on_reset": true,
    "min_samples_to_save": 10,
    "verbose": true,
    "cluster_events": false,
    "gauge_projection": { "enabled": false, "mode": "global", "dims": [0, 1, 2],
                          "apply_noise": true, "apply_momentum": true }
}
```

- `compute.devices`: informational; the CLI `--device` flag decides at run time.
- `cuda_empty_cache_every`: call `torch.cuda.empty_cache()` every this many steps (`0` = never).
- `reset_batch_numbers`: `true` starts fresh (clears checkpoints); `false` resumes from the latest
  checkpoint in `io.checkpoint_dir`.
- `clear_samples_on_reset`: delete `io.samples_outfile` at start so samples do not append.
- `min_samples_to_save`, `cluster_events`: accepted, currently no runtime effect.
- `verbose`: per-epoch console lines.
- `seed` (not in the template): base seed, default `0`; `sample-multi` sets it per chain.
- `gauge_projection.enabled`, `.mode`, `.dims`, `.apply_noise`, `.apply_momentum`: project out
  the per-component translation during sampling. Off; the centroid prior handles the gauge.
- `torch` (not in the template): `allow_tf32`, `matmul_precision`, `compile_eikonet` and friends.

Change first: nothing. Set `reset_batch_numbers: false` only when you intend to resume.

### `observability`

```json
"observability": {
    "wandb": { "enabled": false, "project_name": null, "run_name": null },
    "diagnostics": {
        "pair_count_stats_enable": true,
        "sgld_log_gnoise": false,
        "sgld_log_temperature": false,
        "display_precond_every": 50,
        "wandb": { "enabled": false, "groups": { "core": true } },
        "ess_online": { "enabled": false, "every_n_samples": 50, "n_events": 64, "seed": 0,
                        "window": 512, "max_lag": 128, "dims": [0, 1, 2, 3] }
    }
}
```

- `wandb.enabled`, `.project_name`, `.run_name`: Weights & Biases run; `project_name` must be
  set when enabled. Off in the template so a fresh clone runs without an account.
- `diagnostics.pair_count_stats_enable`: log per-event pair counts at start-up.
- `diagnostics.sgld_log_gnoise`, `.sgld_log_temperature`: extra sampler noise diagnostics.
- `diagnostics.display_precond_every`: cadence (epochs) of the console `[PRIORS]` lines
  (hierarchical prior std, component centroids).
- `diagnostics.wandb.enabled`, `.groups`: which metric groups go to W&B (`core`, `noise`,
  `sampler`, `ess_online`, `priors`).
- `diagnostics.ess_online.*`: online effective-sample-size estimate during Phase 4 (`enabled`,
  `every_n_samples`, `n_events`, `seed`, `window`, `max_lag`, `dims`).
- `diagnostics.truth_catalog` (not in the template; used by the synthetic example): `path`,
  `time_source`, `time_ref`, `require_all` for comparing against a known catalog.

Change first: turn on `wandb` (both blocks) if you use W&B; otherwise leave as is.

## New dataset: the short list

1. `io.*` paths.
2. `model.model_file` plus the `model.eikonet` architecture keys and `model.domain`, all copied
   from the EikoNet training config.
3. `model.filters.dtimes.flip_dt_sign` and `cc_min`, then the `events` connectivity thresholds.
4. `model.likelihoods.*.phase_unc` and `shared_event_re.model.tau_s`.
5. `inference.sampler.epochs_per_phase`.

Everything else in the template is the recommended default.

## First run health checklist

1. `spider validate-config my_region.json --mode locate-map` prints `Config v2 validation OK`.
   Fix any `unknown_key` or `missing_key` before running; keys are case-sensitive.
2. `spider locate-map my_region.json --device 0`. In the log, look for:
   - `[DATA] Final origins n=...` and `Final dtimes n=...`: the data filters left you with the
     events you expect.
   - `Phase 1: MAP (optimizer=adam) | lr_schedule=cosine(...)`, then `phase1 | <epoch>/<total> |
     L=... | dr_max=...` lines with `L` decreasing and `dr_max` below the safety clamp.
   - `Hierarchical event prior updated: std≈[...]` every `every_epochs` epochs after
     `start_epoch`; the values should settle at a few hundred metres and a few hundredths of a
     second.
   - `Post-Phase-1 event filter: ...` reporting how many events were dropped after the residual
     and linearization filters, then the `phase1-pass2` epochs.
   - `Wrote Phase-2 bundle -> <checkpoint_dir>/phase2_bundle.pth`.
3. `spider sample my_region.json --device 0`. Early in the log, look for:
   - `Adopted hierarchical event-prior precision from bundle` (the learned prior scale carried
     over from the MAP).
   - `Component centroids: K=... | main (n=...) shift ...` every `display_precond_every` epochs;
     the shifts should stay at the centimetre level with the tight centroid prior.
   - Phase-4 epoch lines without `skipped N/M batches` warnings ({doc}`sampler-health`).
4. Post-process with one command:

   ```bash
   spider summarize my_region.json [--burn-in N] [--thin K] [--out PATH]
   ```

   reads `io.samples_outfile`, discards the first `N` samples per event, keeps every `K`-th, and
   writes a per-event catalog with posterior means, uncertainties and effective sample sizes.
   {doc}`postprocessing-and-analysis-tools` covers the Python API behind it.
