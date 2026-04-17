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

Schema is pass-through at this layer (validated/materialized downstream). Typical blocks:

- `event`
- `centroid`

### `model.filters`

Expected sub-blocks in runtime bridge:

- `dtimes`
- `events`
- `residual`

Common keys:

- `dtimes`: `remove_duplicates`, `max_abs_input_dt`, `dtime_thin_frac`, `flip_dt_sign`, `cc_min`
- `events`: `min_dtimes`, `min_unique_phase_per_event`, `min_dtimes_per_pair`, `min_event_degree`, `min_events_per_cluster`, `max_pair_station_ratio`, `ratio_filter_phase`, `linearization_error`
- `residual`: `enabled`, `method`, `mad_sigma`, `abs_max`

### `model.likelihoods`

Expected sub-blocks:

- `locate_map`
- `sample`

Current sampling mode expects:

- `model.likelihoods.sample.type = "correlated_gaussian"`

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
- `dt_lr_mult` (`> 0` when set)
- `grad_clip_norm` (`>= 0` when set)
- `overrides` (dict)

`preconditioning` sub-block:

- `enabled` (bool)
- `type` (`rmsprop` or `lrd` when enabled)
- `include_gamma` (bool)
- `lrd` (dict when present)

### `inference.batching`

- `standard` (dict)
- `event_batches` (dict)

Runtime-bridged keys in common use:

- `standard`: `warmup`, `sgld`, `shuffle`
- `event_batches`: `enabled`, `events_per_batch`, `max_edges_per_batch`, `bucket_reorder_all`, `bucket_reuse_epochs`

### `inference.runtime`

Allowed:

- `cuda_empty_cache_every`
- `reset_batch_numbers`
- `clear_samples_on_reset`
- `min_samples_to_save`
- `verbose`
- `cluster_events`
- `seed`
- `gauge_projection`
- `torch`

### `inference.safety`

- `max_abs_dX` is `null` or list of 4 numbers.

### `inference.compute`

- `devices` is optional list of device specifiers (int or non-empty string).

## `observability` block

Allowed keys:

- `wandb`
- `diagnostics`

### `observability.wandb`

Required keys:

- `enabled` (bool)
- `project_name` (non-empty string or null)
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

See {doc}`diagnostics-catalog` for behavior and interpretation.
