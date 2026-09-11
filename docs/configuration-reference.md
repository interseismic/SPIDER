# Configuration Reference (config_v2)

This page is a field-oriented reference for the current strict schema.

Validation entrypoints:

- `spider validate-config <config.json> --mode sample`
- `spider.core.config_v2.validate.parse_canonical_config()`

The root object only allows:

- `io`
- `model`
- `inference`
- `observability`
- `synth` (optional; consumed only by `spider synth`, see below)

Unknown keys at any strict block level raise a validation error.

## `io` block

Required keys:

- `dtime_file` (string)
- `station_file` (string)
- `catalog_infile` (string)
- `catalog_outfile` (string)
- `samples_outfile` (string)
- `checkpoint_dir` (string)
- `checkpoint_interval` (int, `>= 1`)
- `save_every_n` (int, `>= 1`)
- `write_samples` (bool)

Optional keys:

- `sample_write_interval` (int or null, `>= 0` when set)

## `model` block

Allowed keys:

- `model_file`
- `domain`
- `priors`
- `likelihoods`
- `filters`
- `eikonet`

### `model.domain`

Required:

- `lon_min` (number)
- `lat_min` (number)
- `z_min` (number)
- `z_max` (number, must satisfy `z_max > z_min`)
- `scale` (number, `> 0`)

### `model.priors`

Not key-checked by `validate-config`, but strictly validated at runtime by
`spider/core/priors_config.py` (no implicit defaults; units are km for `x/y/z` and seconds for `t`).

- `event` (**required**): `enabled` (bool), `type` (`gaussian`), `params.std` (list[4]
  `[σx, σy, σz, σt]`, required when `enabled`). A Gaussian prior on each event's shift from its
  initial catalog location.
  - `event.hyper` (optional hierarchical prior): `enabled` (bool), `type` (`wishart_precision`),
    `params.df` (`> 3`), `params.scale_std` (list[4]), `update.every_epochs` (int `>= 1`).
- `centroid` (optional): `enabled`, `type` (`gaussian`), `params.std` (list[4]). A Gaussian prior
  on each connected cluster's mean shift; this is what fixes the translation gauge, so keep it
  tight (for example `0.01`) when loosening `event.params.std`.

Removed keys that are now hard errors: `priors` at the top level (use `model.priors`),
`priors.noise` (noise/σ learning was removed; `phase_unc` is fixed), `priors.laplacian`,
`priors.event.schedule` (priors are active in all phases when enabled),
`priors.event.hyper.update.active_phases`, and the legacy flat keys `prior_event_std`,
`prior_centroid_std`, `noise_prior*`, `hierarchical_*`, `laplacian_*`.

### `model.eikonet`

Pass-through dict read by `spider/core/eikonet_loader.py`; it must match the architecture of the
checkpoint at `model.model_file`.

| key | default | meaning |
|---|---|---|
| `x_max`, `y_max` | `model.domain.scale` | horizontal extent (km) used to normalize coordinates |
| `n_hidden` | 128 | hidden width |
| `n_blocks` | 5 | residual blocks |
| `n_fourier` | 4 | Fourier features per input |
| `use_fourier` | true | enable Fourier encoding |
| `phase_emb_dim` | 8 | phase embedding size |
| `t2_log_scale` | 0.1 | log-scale for the 3-D correction term |
| `vp`, `vs` | 6.0, 3.2 | reference velocities (km/s) for the `T0` factor |
| `model_kind` | `1d` | `1d` or `3d` |

### `model.filters`

Required sub-blocks (validated by the runtime bridge, not by `validate-config`; unknown keys inside
them are silently ignored):

- `dtimes`
- `events`
- `residual`

| key | default | meaning |
|---|---|---|
| `dtimes.remove_duplicates` | false | keep one row per (unordered pair, network, station, phase) |
| `dtimes.max_abs_input_dt` | 99999.0 | drop rows with `abs(dt)` above this (s) |
| `dtimes.dtime_thin_frac` | 1.0 | random thinning fraction of rows kept |
| `dtimes.flip_dt_sign` | false | negate `dt` on input |
| `dtimes.cc_min` | 0.0 | drop rows with `cc` below this |
| `events.min_dtimes` | 1 | minimum rows per unordered pair (applied late, after the ratio/finite filters) |
| `events.min_unique_phase_per_event` | 1 | minimum distinct (station, phase) per event |
| `events.min_dtimes_per_pair` | 1 | minimum rows per unordered pair (applied right after `cc_min`) |
| `events.min_event_degree` | 0 | minimum number of connected events (iterative) |
| `events.min_events_per_cluster` | 0 | drop connected components smaller than this (only when `> 1`) |
| `events.max_pair_station_ratio` | 1.0 | drop pairs whose separation exceeds this fraction of the pair–station distance |
| `events.lat_bounds`, `events.lon_bounds` | null | optional `[min, max]` (degrees) catalog subset applied before everything else; dtimes referencing dropped events are removed |
| `events.ratio_filter_phase` | `before` | `before` (data prep) or `after_phase1` (end of Phase 1). Unlike `residual.phase`, an unrecognized value is not rejected — it silently disables the filter |
| `events.linearization_error.enabled` | false | drop rows whose linearization-error ratio exceeds `max_ratio` |
| `events.linearization_error.phase` | `after_phase1` | `before` or `after_phase1` |
| `events.linearization_error.max_ratio` | (required when enabled) | ratio threshold, e.g. `0.05` |
| `events.linearization_error.batch_size` / `sample_size` / `log_every_batches` | 50000 / 200000 / 25 | sweep batching and logging |
| `residual.enabled` | false | residual outlier filter |
| `residual.method` | `mad` | `mad` or `abs` |
| `residual.mad_sigma` | 6.0 | MAD multiplier (`mad` method) |
| `residual.abs_max` | 99999.0 | absolute residual cap (s) |
| `residual.phase` | `after_phase1` | `before` or `after_phase1` |

`residual.phase` controls when the residual outlier filter runs (default `after_phase1`):

- `after_phase1`: at the end of Phase 1, with residuals evaluated at the MAP locations. Applies to `locate-map` outputs, the Phase-2 bundle, and the optional second MAP pass (see `inference.sampler.phase1_two_pass`).
- `before`: during data preparation, with residuals evaluated at the initial catalog locations (`ΔX = 0`).

With `method: mad` the threshold is `min(abs_max, mad_sigma * MAD)` about the residual median; with `method: abs` it is `abs_max` (seconds).
For a two-pass MAP run, `method: abs` is recommended: `mad` re-derives its threshold from the
already-cleaned residuals on each application and keeps trimming.

If neither path ran (for example `epochs_per_phase[0] = 0`, or a bundle produced before this
option existed), the filter is applied once at the start of Phase 2. Bundles written by
`locate-map` record whether the filter was already applied (`residual_filter_applied`), and
`spider sample` skips re-applying it in that case.

#### Filter execution order

Data preparation (`prepare_input_dfs`):

0. optional catalog subset by `events.lat_bounds` / `events.lon_bounds`
1. restrict dtimes to events present in the catalog
2. attach station coordinates
3. `linearization_error` (when `phase: before`)
4. `residual` (when `phase: before`, residuals at the initial locations)
5. `remove_duplicates`
6. `max_abs_input_dt`
7. `dtime_thin_frac`
8. `flip_dt_sign`
9. `cc_min`
10. `min_unique_phase_per_event`
11. `min_dtimes_per_pair`
12. `max_pair_station_ratio` (when `ratio_filter_phase: before`)
13. drop rows with non-finite `dt` or station coordinates
14. `min_dtimes`
15. `min_event_degree`
16. `min_events_per_cluster`
17. drop catalog events with no remaining dtimes

Note that thinning (`dtime_thin_frac`) runs before `cc_min` and the `min_*` event filters, so it
affects which events survive.

End of Phase 1 (`_finalize_phase1`, after `<catalog_outfile>_MAP.csv` is written), evaluated at
the MAP locations: `max_pair_station_ratio` (when `ratio_filter_phase: after_phase1`) →
`residual` (when `phase: after_phase1`) → `linearization_error` (when `phase: after_phase1`).
With two-pass MAP this sequence runs after each pass.

### `model.likelihoods`

Required sub-blocks (validated by the runtime bridge): `locate_map`, `sample`.

`locate_map` (Phase 1, MAP):

- `type`: `huber` (default), `laplace` (aliases `l1`, `mae`), `l2` (aliases `gaussian`, `mse`), or
  `student_t` (alias `student-t`)
- `phase_unc` (list[2] `[σP, σS]` in seconds, required): fixed per-phase noise scales — noise
  learning was removed, these are not estimated
- `huber_delta` (default 1.0), `student_t.nu` (default 4.0)

`sample` (Phases 2–4):

- `type` must be `correlated_gaussian` (`correlated` is accepted as an alias)
- `phase_unc` (list[2], required), `huber_delta`, `student_t.nu` as above
- `shared_event_re` (below)

## `model.likelihoods.sample.shared_event_re`

Allowed top-level keys:

- `enabled`
- `model`
- `limits`
- `fallback`
- `numerics`
- `station_phase_term`
- `solver`
- `edge_weights`
- `autotune`
- `logging`

Legacy flat keys are rejected (for example `grouping`, `max_nodes_per_group`, `jitter0`, `bucket_nodes`).

### `shared_event_re.model`

- `group_by` (current supported runtime value: `station_phase`)
- `tau_s` (`[P, S]`)
- `cluster.mode`, `cluster.k`

### `shared_event_re.limits`

- `max_nodes`
- `max_rows`

### `shared_event_re.fallback`

- `to_diag`
- `abort_on_pcg_fallback`

### `shared_event_re.numerics`

- `jitter0`
- `jitter_max`

### `shared_event_re.solver`

Canonical keys:

- `kind` (`pcg`)
- `max_iters`
- `min_iters`
- `tol`
- `batched`
- `node_bin_edges`
- `warm_start`
- `cache_max_entries`
- `prefetch_grouping`
- `profile_micro_steps`
- `merge_sparse_node_bins`
- `min_groups_per_node_bin`
- `max_node_bins_per_node`
- `precompute.enabled`
- `precompute.device`

### `shared_event_re.edge_weights`

- `mode`
- `ell_km`
- `eps_km`
- `power`
- `scale_km`
- `global_scale`
- `normalize`

### `shared_event_re.autotune`

- `enabled`
- `observe_epochs`
- `latest_epoch`
- `min_groups`
- `max_node_bins`
- `min_groups_per_node_bin`
- `min_node_bin`
- `min_gain`
- `raise_nodes_cap`
- `nodes_cap_max`

### `shared_event_re.logging`

- `quiet`
- `stats_log_every_epochs`

## `inference` block

Allowed keys:

- `sampler`
- `batching`
- `runtime`
- `safety`
- `compute`

### `inference.sampler`

Required/validated:

- `backend` (`psgld` or `sghmc`)
- `epochs_per_phase` (list of 4 ints, each `>= 0`)
- `lr` (list of 4 positive numbers)
- `temperature` (`>= 0`)
- `beta` (`0 <= beta < 1`)
- `eps` (`> 0`)
- `freeze_preconditioner_sampling` (bool)
- `sghmc_alpha` (required key; number for SGHMC, number/null otherwise)

Optional:

- `noise_scale_mult` (`> 0` when set)
- `grad_clip_norm` (`>= 0` when set; `0` disables). Applies to all phases when set. When unset,
  Phase 1 (Adam/MAP) clips at 100.0 and Phases 2–4 do not clip.
- `reparameterization` (dict)
- `overrides` (dict)
- `lr_schedule` (dict)
- `phase1_two_pass` (dict)

`preconditioning` sub-block:

- `enabled` (bool)
- `type` (`rmsprop`, `lrd`, or `component_lrd` when enabled; aliases `cc_lrd`, `block_lrd`, and `component-lrd` are accepted)
- `include_gamma` (bool)
- `lrd` (dict when present): `rank` (16), `mode` (`svd` | `oja`, default `svd`; `randomized_svd` /
  `stochastic_svd` alias to `svd`, other values fall back to `svd`), `update_every` (20),
  `buffer_size` (64), `eta` (or `oja_eta`, 0.02), `diag_floor` (defaults to `inference.sampler.eps`),
  `target` (`dX_src_only`)

`overrides` sub-block: `core.{lr_mult, temperature_mult, eps, freeze_preconditioner_sampling}`;
other keys are ignored.

`reparameterization` sub-block:

- `enabled` (bool)
- `spatial_scale` (`> 0`, static scale for `x/y/z` in blocked sampler coordinates)
- `dt_scale` (`> 0`, static scale for `dt` in blocked sampler coordinates)

`lr_schedule` sub-block (Phase 1 / MAP learning-rate schedule; default constant `lr[0]`):

- `phase1.type` (`none` or `cosine`)
- `phase1.eta_min` (`>= 0`, final learning rate of the cosine decay; default `0`)
- `phase1.T_max` (`null` or int `>= 1`): cosine horizon in epochs. When `null` it resolves at
  runtime to the epoch count of the current MAP pass — `epochs_per_phase[0]` for pass 1 and
  `phase1_two_pass.epochs` for pass 2.

Example: `"lr_schedule": {"phase1": {"type": "cosine", "eta_min": 1.0e-5}}`

`phase1_two_pass` sub-block (two-pass MAP; default disabled):

- `enabled` (bool)
- `epochs` (`null` or int `>= 1`; epochs for the second pass, default `epochs_per_phase[0]`)
- `warm_start` (bool, default `true`; start the second pass from the pass-1 MAP locations instead of the initial catalog)

When enabled, Phase 1 runs once, the post-MAP filters are applied at the MAP locations
(`model.filters.residual` with `phase: after_phase1`, `max_pair_station_ratio` with
`ratio_filter_phase: after_phase1`, and `linearization_error` with `phase: after_phase1`),
the Adam optimizer (and any `lr_schedule`) is rebuilt, and MAP is run again on the filtered
dtimes. The pass-1 catalog is kept as `<catalog_outfile>_MAP_pass1.csv`; `<catalog_outfile>_MAP.csv`
holds the final result. Not supported under torchrun/DDP, and intended for fresh runs
(`inference.runtime.reset_batch_numbers: true`), not for resuming into the second pass.

### `inference.batching`

Required sub-blocks (validated by the runtime bridge): `standard`, `event_batches`.

Runtime-bridged keys in common use:

- `standard`: `warmup`, `sgld`, `shuffle`
- `event_batches`: `enabled`, `events_per_batch`, `max_edges_per_batch`, `bucket_reorder_all`, `bucket_reuse_epochs`

### `inference.runtime`

Required:

- `cuda_empty_cache_every` (int)
- `reset_batch_numbers` (bool; `true` clears checkpoints and starts fresh, `false` resumes)
- `clear_samples_on_reset` (bool)
- `min_samples_to_save` (int)
- `verbose` (bool)
- `cluster_events` (bool)

Optional (defaults applied at load time and reported by `validate-config --print-resolved`):

- `seed` (int, default `0`)
- `gauge_projection` (dict): `enabled` (false), `mode` (`global` | `cluster`), `dims`
  (default `[0, 1, 2]`), `apply_noise` (true), `apply_momentum` (true)
- `torch` (dict, applied at process start): `allow_tf32` (bool), `matmul_precision`
  (`highest` | `high` | `medium`), `compile_eikonet` (bool, default false), `compile_mode`,
  `compile_backend`, `compile_dynamic`, `compile_fullgraph`

### `inference.safety`

- `max_abs_dX` is `null` or list of 4 numbers `[dx, dy, dz, dt]` (km, km, km, s): per-component
  clamp on each event's shift from its initial location. Events sitting at the clamp appear in the
  Phase-1 log as `dr_max` equal to the clamp norm; widen the clamp rather than leave them pinned.

### `inference.compute`

- `devices` is optional list of device specifiers (int or non-empty string).

## `synth` block (optional)

Pass-through dict read only by `spider synth`; keys: `outfile`, `seed`, `overwrite`,
`apply_filters`, `drop_missing_events`, `event_std`, `true_catalog_infile`,
`true_catalog_outfile`, `init_catalog_outfile`, `init_clip_domain`, `init_spatial_only`,
`rf_amp`, `rf_features`, `rf_len_xy`. Ignored by every other command.

## `observability` block

Allowed keys:

- `wandb`
- `diagnostics`

### `observability.wandb`

Required keys:

- `enabled` (bool)
- `project_name` (non-empty string or null; must be non-null when `enabled` is `true`)
- `run_name` (non-empty string or null)

### `observability.diagnostics`

Accepted as a dict and materialized downstream. Common keys:

- `pair_count_stats_enable`
- `sgld_log_gnoise`
- `sgld_log_temperature`
- `display_precond_every`
- `profile_shared_event_re`
- `wandb` (including `groups`)
- `ess_online`
- `truth_catalog` (`path`, `time_source`, `time_ref`, `require_all`): optional ground-truth catalog
  for synthetic tests

See {doc}`diagnostics-catalog` for behavior and interpretation.
