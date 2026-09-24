# Configuration Overview

SPIDER is configured with a single nested JSON file. The schema is strict: required keys must be
present, unknown keys in the strictly checked blocks are errors, and there is no compatibility
layer for older flat configs. Check a config with

```bash
spider validate-config my_params.json --mode sample [--print-resolved]
```

Two configs in the repository validate against the current schema and are good starting points:
`spider/examples/params_template.json` (placeholder paths, W&B off) and
`example/SPIDER_example.json` (complete synthetic example).

Code that defines the schema, for reference:

- `spider/core/config_v2/validate.py` — allowed keys, types and ranges for the strict blocks
- `spider/core/config_v2/resolve.py` — the two defaults that are filled in
  (`io.sample_write_interval <- io.checkpoint_interval`, `inference.runtime.seed <- 0`)
- `spider/core/config_v2/legacy_bridge.py` — maps the nested schema onto the runtime parameter
  dict and supplies the defaults for the pass-through blocks
- `spider/core/priors_config.py` — strict validation of `model.priors` (run before every command)

## Top-level blocks

Four top-level blocks are required: `io`, `model`, `inference`, `observability`. A fifth,
`synth`, is optional and used only by `spider synth`. Any other top-level key (for example a
legacy `wandb` or `priors` block) is a hard validation error.

### `io`

Input files (`dtime_file`, `station_file`, `catalog_infile`), output paths (`catalog_outfile`,
`samples_outfile`, `checkpoint_dir`) and cadence controls. All keys except
`sample_write_interval` are required.

- `catalog_outfile` is a **prefix**: Phase 1 writes `<catalog_outfile>_MAP.csv` (and
  `<catalog_outfile>_MAP_pass1.csv` with two-pass MAP).
- `checkpoint_interval` (epochs, >= 1): checkpoint cadence in every phase.
- `sample_write_interval` (epochs, >= 0, default = `checkpoint_interval`): how often buffered
  Phase-4 samples are flushed to the HDF5 store.
- `save_every_n` (steps, >= 1): keep every n-th Phase-4 minibatch step as a posterior sample.
- `write_samples` (bool): set `false` to skip the HDF5 store entirely.

### `model`

- `model_file` — EikoNet checkpoint path (state dict).
- `eikonet` — loader settings for that checkpoint: `model_kind` (`1d` default, `3d`, or `gridtt` for a
  precomputed travel-time table directory from the optional `gridtt` package; see {doc}`configuration-reference`),
  `n_hidden`, `n_blocks`, `n_fourier`, `use_fourier`, `phase_emb_dim`, `t2_log_scale`, `vp`, `vs`,
  and optionally `x_max` / `y_max` (default: `model.domain.scale`). These must match how the
  EikoNet was trained.
- `domain` — projection origin and bounds: `lon_min`, `lat_min`, `z_min`, `z_max` (km,
  `z_max > z_min`), `scale` (km, > 0). All required.
- `priors` — `event` (required) and optional `centroid`; see below.
- `likelihoods` — `locate_map` (MAP misfit) and `sample` (sampling likelihood); see below.
- `filters` — `dtimes`, `events`, `residual` (all three sub-blocks required, keys inside have
  defaults). Execution order is documented in {doc}`data-formats`.

### `inference`

- `sampler` — backend and hyperparameters (`backend`, `epochs_per_phase[4]`, `lr[4]`,
  `temperature`, `beta`, `eps`, `sghmc_alpha`, `freeze_preconditioner_sampling` are required),
  optional `noise_scale_mult`, `grad_clip_norm`, `preconditioning`, `reparameterization`,
  `overrides`, plus the Phase-1 controls `lr_schedule` and `phase1_two_pass`.
- `batching` — `standard` (`warmup`, `sgld`, `shuffle`) and `event_batches`.
- `runtime` — `cuda_empty_cache_every`, `reset_batch_numbers`, `clear_samples_on_reset`,
  `verbose` are required; optional `seed`, `gauge_projection`, `torch`. `min_samples_to_save` and
  `cluster_events` are accepted for backward compatibility and have no effect.
- `safety` — `max_abs_dX`: `null` or a list of four clamp half-widths `[dx, dy, dz, dt]` (km, s);
  optional `shift_guard` (`enabled`, `factor`): abort and print the offending events when a shift
  exceeds `factor` times the event-prior std (debugging aid, default off).
- `compute` — `devices` list. Ignored by the CLI run commands, which take the device from
  `--device` / `--devices` (`sample-multi` rewrites it per chain).

### `observability`

- `wandb` — `enabled`, `project_name`, `run_name` (all three keys required; the names may be
  `null` when disabled, `project_name` must be set when enabled).
- `diagnostics` — diagnostic toggles and the W&B metric-group gates
  (`diagnostics.wandb.groups`), plus the optional end-of-Phase-1 residual diagnostics
  `diagnostics.spatial` (Moran's I) and `diagnostics.pathcorr` (per station-phase correlation vs
  path similarity); see {doc}`diagnostics-catalog` and {doc}`configuration-reference`.

## Strict vs. pass-through blocks

Strictly key-checked by `validate-config` (an unknown key is an error): the root, `io`,
`model` (its six top-level keys), `model.domain`, `inference` and its five sub-blocks' top-level
keys, `inference.sampler` including `preconditioning`, `reparameterization`, `lr_schedule` and
`phase1_two_pass`, `inference.safety`, `observability` and `observability.wandb`, and the
top-level keys of `model.likelihoods.sample.shared_event_re` (which also rejects the old flat
keys such as `tau_s` or `cluster_mode` at that level).

Pass-through (unknown keys are accepted and ignored at runtime): the contents of
`model.likelihoods.*` (other than the checks above and `shared_event_re.solver.kind`, which must be
`pcg` / `pcg_sparse`), `model.filters.*`, `model.eikonet`, `inference.batching.standard` /
`event_batches`, `inference.runtime.gauge_projection` / `torch`, `inference.sampler.overrides` /
`preconditioning.lrd`, and `observability.diagnostics`. A misspelled key in these blocks has no
effect rather than raising an error, but `spider validate-config` reports every key that no code
reads as a non-fatal `[warning] <block> (unknown_key): Unknown key(s) ignored by the runtime: ...`
line, so run it after editing a config. Use `--print-resolved` and the defaults listed in
{doc}`configuration-reference` to check what actually took effect.

`model.priors` is the exception: it is not key-checked by `validate-config` but is strictly
validated at the start of every run command (missing or unknown keys raise
`Invalid priors config at ...`).

## Priors (`model.priors`)

Every prior hyperparameter must be spelled out; there are no implicit defaults.

```json
"priors": {
  "event": {
    "enabled": true,
    "type": "gaussian",
    "params": {"std": [5.0, 5.0, 5.0, 2.0]},
    "hyper": {
      "enabled": true,
      "type": "wishart_precision",
      "params": {"df": 10.0, "scale_std": [0.5, 0.5, 0.5, 0.2]},
      "update": {"every_epochs": 10, "start_epoch": 200, "min_cluster_events": 100}
    }
  },
  "centroid": {
    "enabled": true,
    "type": "gaussian",
    "params": {"std": [0.01, 0.01, 0.01, 0.01]}
  }
}
```

- `event` (required): Gaussian prior on each event's shift `[dx, dy, dz, dt]` from its catalog
  location, `params.std` in km, km, km, s (`type` must be `gaussian`).
- `event.hyper` (optional, **recommended**): learns the event-shift covariance from the data with
  a Wishart prior on the precision matrix, separately for each connected component of the event
  graph. `params.df` must be > 3, `params.scale_std` is the Wishart scale expressed as standard
  deviations, `update.every_epochs` (>= 1) is the update cadence, `update.start_epoch` (default 0)
  keeps the fixed `params.std` prior until that epoch so the estimate is not made from
  `dX ≈ 0`, and `update.min_cluster_events` (default 0) makes components smaller than that share
  the precision estimated from all events. Updates run in every phase while enabled; the value at
  the end of Phase 1 is stored in the Phase-2 bundle and adopted by `sample`.
- `centroid` (optional): Gaussian prior on the **mean shift of each connected component**,
  weighted by the component's event count (so the effective width for a component of `n`
  events is `std / sqrt(n)`). It pins the translational/time gauge that differential times cannot
  constrain. There is no global form; a `scope` key is rejected.
- Removed blocks that now fail fast if present: `event.schedule`, `event.hyper.update.active_phases`,
  `noise`, `laplacian`, and any of the old flat `prior_*` / `hierarchical_*` / `laplacian_*` keys.

## Likelihoods (`model.likelihoods`)

- `locate_map.type` selects the MAP misfit: `huber` (default, `huber_delta` default 1.0),
  `laplace` (aliases `l1`, `mae`), `l2` (aliases `gaussian`, `mse`) or `student_t`
  (`student_t.nu` default 4.0). `locate_map.phase_unc` (`[P, S]` seconds) is required.
- `sample.type` must be `correlated_gaussian` (`correlated` is accepted as an alias); it is the
  only sampling likelihood. `sample.phase_unc` is required. Its `shared_event_re` sub-block
  configures the collapsed shared-event random effect ({doc}`pcg-whitening-convergence`);
  `shared_event_re.enabled` should be set explicitly.

## High-impact settings to validate early

- `model.filters.dtimes.flip_dt_sign`: SPIDER expects `dt = t(evid2) - t(evid1)`. Set `true` for
  HypoDD `dt.cc`-style input. A wrong sign makes every event fight its data; the MAP typically
  ends with `dr_max` pinned at the clamp.
- `model.filters.residual.phase` (`after_phase1` default | `before`): whether the residual outlier
  filter is evaluated at the MAP locations at the end of Phase 1, or at the initial catalog
  locations during data prep. `method` is `mad` (default, `mad_sigma` 6.0) or `abs` (`abs_max`).
- `inference.sampler.backend`: `psgld` or `sghmc` (`sghmc_alpha` must be a positive number for
  `sghmc`; it may be `null` for `psgld`).
- `inference.sampler.preconditioning.include_gamma` (default `false`): adds the pSGLD Γ
  correction drift that compensates for a changing RMSprop preconditioner. Negligible with a slowly
  adapting or frozen preconditioner; both example configs leave it off.
- `inference.sampler.lr_schedule.phase1.type` (`none` default | `cosine`): cosine decay of the
  Phase-1 (MAP) learning rate to `eta_min` (default 0) over `T_max` epochs (default: the epoch
  count of the current MAP pass). Removes end-of-run jitter.
- `inference.sampler.phase1_two_pass.enabled` (default `false`): after the post-MAP filters, rebuild
  the optimizer and run MAP again for `epochs` epochs (`null` = `epochs_per_phase[0]`), starting
  from the pass-1 solution when `warm_start` is `true` (default). Writes
  `<catalog_outfile>_MAP_pass1.csv` in addition to `_MAP.csv`.
- `inference.sampler.grad_clip_norm`: when omitted, Phase 1 clips gradients at norm 100 and the
  sampling phases do not clip; when set, the value applies to all phases (`0` disables clipping).
- `inference.safety.max_abs_dX`: per-component clamp on event shifts. Events pinned at the clamp
  cannot reach their data-preferred positions; widen it if the MAP log shows `dr_max` sitting at
  the clamp value.
- `inference.batching.standard.shuffle` should typically be `true` during production sampling.
- `inference.sampler.freeze_preconditioner_sampling` should match your Phase-4 policy
  ({doc}`sampler-options`).

For shared-event PCG setup and convergence checks, see {doc}`pcg-whitening-convergence`.

For a field-by-field schema reference, see {doc}`configuration-reference`.

For a symbol-level probability-model description with config-to-symbol mapping, see {doc}`probability-model`.
