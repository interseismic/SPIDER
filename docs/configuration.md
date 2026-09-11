# Configuration Overview

SPIDER uses strict nested JSON configuration validated through `config_v2`.

Core modules:

- `spider/core/config_v2/validate.py` — schema (allowed keys, types, ranges)
- `spider/core/config_v2/load.py` — load + resolve defaults
- `spider/core/config_v2/legacy_bridge.py` — maps the nested schema onto the runtime parameter dict

## Top-level blocks

Four top-level blocks are required: `io`, `model`, `inference`, `observability`. A fifth,
`synth`, is optional and used only by `spider synth`. Any other top-level key (for example a
legacy `wandb` or `priors` block) is a hard validation error.

### `io`

Input/output file paths and checkpoint controls. `io.catalog_outfile` is a prefix: Phase 1 writes
`<catalog_outfile>_MAP.csv` (and `_MAP_pass1.csv` with two-pass MAP).

### `model`

- EikoNet checkpoint path (`model_file`) and loader settings (`eikonet`).
- Domain origin/bounds/scale (`domain`).
- Priors (`priors`: `event`, optional `centroid`, optional `event.hyper`).
- Likelihoods for MAP (`likelihoods.locate_map`) and sampling (`likelihoods.sample`).
- Data filters (`filters.dtimes`, `filters.events`, `filters.residual`).

### `inference`

- Sampler backend and hyperparameters, Phase-1 learning-rate schedule and two-pass MAP (`sampler`).
- Batch settings (`batching`).
- Runtime, safety clamp, and device controls (`runtime`, `safety`, `compute`).

### `observability`

- W&B run controls (`wandb`).
- Diagnostics group toggles (`diagnostics`).

## Strict vs. pass-through blocks

Only the blocks listed as strict in {doc}`configuration-reference` are key-checked by
`validate-config`. Inside `model.priors`, `model.likelihoods`, `model.filters`, `model.eikonet`,
`inference.batching.*`, `inference.runtime.{gauge_projection,torch}` and
`observability.diagnostics`, unknown keys are accepted and silently ignored (priors are the
exception: they are strictly validated at runtime). A misspelled key in those blocks therefore has
no effect rather than raising an error — check with `spider validate-config --print-resolved`.

## High-impact settings to validate early

- `model.likelihoods.sample.type` must be `correlated_gaussian` (the only supported sampling
  likelihood; `correlated` is accepted as an alias).
- `model.likelihoods.locate_map.type` selects the MAP misfit: `huber` (default), `laplace`
  (aliases `l1`, `mae`), `l2` (aliases `gaussian`, `mse`), or `student_t`.
- `model.likelihoods.sample.shared_event_re.enabled` should be explicitly set.
- `model.filters.residual.phase` (`after_phase1` default | `before`): where the residual outlier
  filter is evaluated — at the MAP locations at the end of Phase 1, or at the initial catalog
  locations during data prep.
- `inference.sampler.backend` should be either `psgld` or `sghmc`.
- `inference.sampler.lr_schedule.phase1.type` (`none` | `cosine`): cosine decay of the Phase-1
  (MAP) learning rate removes end-of-run jitter.
- `inference.sampler.phase1_two_pass.enabled`: re-runs MAP after the post-MAP filters; writes
  `<catalog_outfile>_MAP_pass1.csv` in addition to `_MAP.csv`.
- `inference.safety.max_abs_dX`: per-component clamp on event shifts `[dx, dy, dz, dt]` (km, s).
  Events pinned at the clamp cannot reach their data-preferred positions; widen it if the MAP log
  shows `dr_max` sitting at the clamp value.
- `inference.batching.standard.shuffle` should typically be `true` during production sampling.
- `inference.sampler.freeze_preconditioner_sampling` should match your Phase 4 policy.

For shared-event PCG setup and convergence checks, see {doc}`pcg-whitening-convergence`.

For a field-by-field schema reference, see {doc}`configuration-reference`.

For a symbol-level probability-model description with config-to-symbol mapping, see {doc}`probability-model`.
