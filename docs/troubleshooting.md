# Troubleshooting

Symptoms are grouped by the log line or error you see. Log lines are quoted without the
`[spider][LEVEL][SECTION]` prefix unless the section matters.

## Optimization and sampling

### `loss=nan` or "ALL batches skipped -> optimization is frozen"

```text
[spider][WARN][RUN] epoch 7: skipped 64/64 batches (non-finite loss=0, non-finite grad=64); parameters unchanged for those batches. ALL batches skipped -> optimization is frozen. Typical cause: a NaN gradient from a source that has converged onto a receiver (zero horizontal offset in the travel-time model); check events located on stations.
```

Every minibatch was discarded by the finiteness guards, so parameters never updated and the epoch
loss is reported as `nan` (older versions misreported it as `0.0`). The `skipped_batches` metric
carries the count to W&B. The most common cause, as the message says, is an event that has
converged onto a receiver: at zero horizontal offset the gradient of the travel-time model's
distance term is undefined. Current EikoNet builds include a small distance floor inside the
`sqrt()` that keeps the gradient finite; if you see this with an older model, check
`<catalog_outfile>_MAP.csv` for events located at or within a few metres of a station and either
retrain the surrogate or remove those events. If only a few batches are skipped the run is still
progressing, but the affected observations are being ignored; a `skipped=k/M` field in the epoch
line (or the `[WARN][RUN] ... skipped k/M` message) that appears every epoch deserves the same
investigation.

### Residual filter with `phase: "before"` drops almost every row

If `Residual filter: dropping N/M dtimes` removes most of the table at data-prep time, check the
sign convention first. `model.filters.dtimes.flip_dt_sign` is applied to the loaded table before
any filter (see {doc}`data-formats`), so HypoDD-style `dt.cc` input (`t1 - t2`) needs
`flip_dt_sign: true`; without it the residuals are `2 * dt` and the filter removes good rows. In
versions before this ordering fix the flip ran after the `before` filters, so `flip_dt_sign: true`
plus `residual.phase: "before"` produced the same symptom; upgrade rather than working around it.

### Events pinned at the safety clamp

If `dr_max` in the Phase-1 log equals the norm of `inference.safety.max_abs_dX` epoch after epoch,
some events are being held at the clamp rather than at their data-preferred positions. Widen the
clamp (e.g. `[10, 10, 10, 3]`); it is a safety net, not a prior. Likewise, if the median shifts
(`dx/dy/dz/dT`) exceed `model.priors.event.params.std`, the fixed event prior is fighting the data
for most of the catalog — loosen it, rely on the centroid prior to fix the gauge, and let the
hierarchical hyperprior (`model.priors.event.hyper`, recommended) learn the realistic scale for the
sampling phases instead of tightening the fixed prior by hand.

If the pinned events appear only in the *sampling* phases (posterior means ~`max_abs_dX` away from
the catalog while the MAP looked fine), check the Phase-1 log for the post-Phase-1 event filter:

```text
[spider][INFO][FILTER] Post-Phase-1 event filter: dropping 12/4310 events (9 with no remaining dtimes, 3 in components < min_events_per_cluster=5); their MAP locations stay in _MAP.csv with dropped_after_phase1=1.
```

Events that lose all their differential times to the post-MAP residual / linearization filters
have no likelihood and random-walk to the clamp under the injected noise, dragging the centroid
prior (and hence every other event in their component, by up to a few hundred metres) with them.
SPIDER drops such events (and events in components smaller than
`model.filters.events.min_events_per_cluster`) before the second MAP pass and the bundle are
written, and flags them in `_MAP.csv` (`dropped_after_phase1 = 1`; the column only appears when at
least one event was dropped). Bundles produced by older versions should be regenerated with
`spider locate-map`. The line
`Post-Phase-1 event filter: all events keep >=1 dtime and satisfy min_events_per_cluster.` is the
healthy outcome.

### Weakly constrained events mix much more slowly than the rest

Per-event ESS that correlates strongly with the number of differential times per event is a
known limitation of the RMSprop preconditioner floor (`inference.sampler.eps` larger than the
typical gradient scale). See "Known limitation" in {doc}`sampler-health` for how to diagnose it and
what can be done with the current code.

### Sampler instability at higher learning rate

If trajectories explode when the preconditioner is frozen or `eps` is lowered:

- make sure the adaptation phase (Phase 2) is long enough before the freeze,
- raise `inference.sampler.eps`,
- use gradient clipping (`inference.sampler.grad_clip_norm`; when unset, Phase 1 clips at 100 and
  Phases 2–4 do not clip),
- lower the learning rate and retune gradually.

### `Shift guard triggered (...)`

`--shift-guard` (on `locate-map`, `sample`, `locate-full`) aborts as soon as any event moves more
than `--shift-guard-factor` (default 5) times `model.priors.event.params.std` and prints the
observations of the offending events. It is a debugging aid for bad input data (wrong station
coordinates, mislabeled phases): inspect the printed rows, fix or filter the data, and rerun
without the flag.

## Priors

### Hierarchical prior not adopted in the sampling phases

`spider sample` should print
`[PRIORS] Adopted hierarchical event-prior precision from bundle: std≈[...] (largest component)`.
If instead you see

```text
[spider][WARN][PRIORS] Bundle precision matrix shape (7, 4, 4) does not match the rebuilt clusters (6, 4, 4); starting from the base event prior until the first Wishart update.
```

the connected components at sampling time differ from those at the end of Phase 1 — usually
because `model.filters.*` or `model.filters.events.min_events_per_cluster` changed between the
`locate-map` and `sample` configs, or the bundle predates the current filters. Rerun `locate-map`
with the final config. Bundles written without the hierarchical prior enabled carry no precision
matrix at all; enabling `event.hyper` only for `sample` runs Phases 2–4 under the loose base
prior until `update.start_epoch`.

### `Hierarchical event prior update failed; leaving P0 unchanged for this epoch.`

The Wishart Gibbs step failed numerically for one epoch. Occasional occurrences are harmless. If
it repeats every update, a component has too few events for a 4×4 precision estimate — raise
`model.priors.event.hyper.update.min_cluster_events` so small components share the pooled
estimate.

### `centroid/z_max` far above 3, or a `[PRIORS] Component centroids:` line with many components beyond 3 sigma

The gauge prior is holding one or more component means against the data. For the main component
this means the whole catalog wants to translate, which the differential data cannot justify —
check the initial catalog and station coordinates. For small components it usually accompanies the
random-walk problem described under "Events pinned at the safety clamp".

## Phase-1 filters and bundles

### Second MAP pass never runs

`inference.sampler.phase1_two_pass.enabled` must be `true` (check with `validate-config
--print-resolved`; the flat key is `phase1_two_pass_enable`). The second pass is labelled
`phase1-pass2` in the console, preceded by `Phase 1 second pass: epochs=... warm_start=...
dtimes=... (post-MAP filters applied)`. It is intended for fresh runs
(`inference.runtime.reset_batch_numbers: true`); resuming into a partially finished Phase 1 still
runs the complete second pass afterwards. The warning
`phase1_two_pass is not supported under torchrun/DDP ...; skipping second pass.` can only appear
when `locate_map()` is called programmatically inside a DDP process — the `spider` CLI itself
refuses to start under `torchrun` (`WORLD_SIZE > 1`).

### Phase-2 bundle contains unfiltered dtimes

The post-Phase-1 filter block is best-effort: any failure is reported as
`[WARN][FILTER] Post-Phase1 filters failed: ...` and Phase 1 still finalizes, so the bundle can
carry the unfiltered dataset. Grep the Phase-1 log for `[WARN][FILTER]` and for the expected
`Residual pre-filter: dropping N / M dtimes (...)` (or `Residual pre-filter: no outliers detected;
keeping all rows.`) and `[FILTER] Applied linearization_error (after_phase1) max_ratio=...; kept
N/M dtimes.` lines. `spider analyze-resid` prints `analyze-resid: loaded bundle=... events=N
dtimes=M`, which tells you what the bundle actually contains.

### `Phase-2 bundle not found: ... Run `python -m spider locate-map <params>` first or pass --bundle PATH.`

`sample`, `sample-multi` and `analyze-resid` read `<io.checkpoint_dir>/phase2_bundle.pth` by
default. Run `spider locate-map <params> --device <id>` first (add `--bundle-out PATH` to put the
bundle elsewhere), or pass `--bundle PATH`. The deprecated `spider locate` alias has no `--bundle`
flag.

### `locate-map can only resume/run Phase 1, but latest checkpoint is phase='phase4'`

The most recent checkpoint in `io.checkpoint_dir` is from Phases 2–4. Either delete the
checkpoints or set `inference.runtime.reset_batch_numbers: true` (the default in the example
config) to start fresh; with `clear_samples_on_reset: true` the samples file is cleared as well.

## Models and dependencies

### `Missing required dependency `eikonet``

The travel-time surrogate is loaded through the external `eikonet` package, which is not a
declared dependency of SPIDER. Install it (`pip install -e /path/to/eikonet` for a local
checkout). For `model.eikonet.model_kind: "gridtt"` the `gridtt` package is required instead.

### `EikoNet state_dict checkpoints require x_max/y_max`

Set `model.eikonet.x_max` / `y_max` (preferred) or `model.domain.scale`; they define the
normalization the network was trained with and must match the training configuration.

### `Checkpoint '...' does not match the requested architecture`

Raised by `eikonet.io.load_model`: the state dict at `model.model_file` was trained with a
different architecture (or with the pre-package EikoNet code). Make
`model.eikonet.{n_hidden, n_blocks, n_fourier, use_fourier, phase_emb_dim, model_kind}` match the
training config, or retrain with the current `eikonet` package. Older loaders accepted such files
while loading no weights at all, so a model that "loads fine" but produces nonsense residuals at
the initial catalog (see the pre-optimization residual summary printed by `locate-map`) is the same
problem.

## Configuration

### Config validation errors

Most failures come from legacy key locations. Run `spider validate-config <params>` (optionally
with `--mode locate-map|sample|sample-multi|analyze-resid|locate-full|synth`) and follow the path
in the error. Frequent ones:

| Error path / message | Fix |
| --- | --- |
| unknown key at `<root>` (`wandb`, `phases`, `priors`, `diagnostics`, ...) | move to `observability.wandb`, `inference.sampler.epochs_per_phase` / `lr`, `model.priors`, `observability.diagnostics` |
| `inference.diagnostics is not allowed` | rename the block to `observability.diagnostics` |
| `inference.sampler.lr`: expected list of 4 | per-phase learning rates, not a scalar |
| `inference.sampler.dt_lr_mult` unknown | use `inference.sampler.reparameterization.dt_scale` |
| `model.likelihoods.sample.shared_event_re`: unknown/legacy keys (`tau_s`, `max_rows_per_group`, `abort_on_pcg_fallback`, ...) | use the `model` / `limits` / `fallback` / `numerics` / `solver` sub-blocks |
| `shared_event_re.autotune`: legacy keys `max_bins`, `min_bin_groups`, `min_bucket_node` | rename to `max_node_bins`, `min_groups_per_node_bin`, `min_node_bin` |
| `priors`: moved | put the block under `model.priors` |
| `priors.noise`: removed | delete it; noise scales are fixed (`phase_unc`) |
| `priors.laplacian`, `priors.event.schedule`, `priors.event.hyper.update.active_phases`: removed | delete them; priors and hyper-updates are active in all phases when enabled |
| `priors.centroid.scope`: removed | delete it; the centroid prior is always per connected component |
| `priors.event.hyper.params.df`: must be > 3 | Wishart degrees of freedom for a 4-D precision |
| `inference.sampler.sghmc_alpha`: missing | the key is required; use `null` for the pSGLD backend |

Which blocks are key-checked: the root (`io`, `model`, `inference`, `observability`, `synth`),
`io`, `model` (top level and `domain`), `model.likelihoods.sample.shared_event_re` (all
sub-blocks), `inference` and all of `inference.sampler`, `inference.runtime`, `inference.safety`,
`observability` and `observability.wandb`. **Not** key-checked, so typos are silently ignored:
`model.filters`, `model.eikonet`, the rest of `model.likelihoods`, `inference.batching.standard`
/ `event_batches`, and `observability.diagnostics`. `model.priors` is not key-checked by
`validate-config` but is strictly validated at runtime by `spider/core/priors_config.py` (no
implicit defaults). Use `--print-resolved` to see what was actually materialized.

### `torch.compile` warning on Python 3.13+

`inference.runtime.torch.compile_eikonet` defaults to `false`, so this only affects runs that
opted in — remove the key or set it to `false`. Related pass-through keys under
`inference.runtime.torch`: `allow_tf32`, `matmul_precision`, `compile_mode`, `compile_backend`,
`compile_dynamic`, `compile_fullgraph`.

### `` `sample` no longer supports torchrun/DDP (WORLD_SIZE=4) ``

All `spider` commands are single-process. Do not launch them with `torchrun`; for multi-GPU work
use `spider sample-multi --devices 0,1,2,3`, which runs one independent chain per device and
merges the sample files.

## Shared-event likelihood

### Shared-event PCG fallback warning

If logs show `shared_event_re fallback reasons rows_cap=... nodes_cap=... tau_zero=...` every
epoch, or `shared_event_re warning: all groups fell back to diagonal`:

- inspect the fallback reason counters (`rows_cap`, `nodes_cap`, `tau_zero`),
- raise `model.likelihoods.sample.shared_event_re.limits.max_rows` and/or `limits.max_nodes`
  (defaults 200000 / 512 are far below typical needs; the `max_rows_all` / `max_nodes_all` values
  in the periodic `shared_event_re stats` line tell you what the caps must exceed),
- verify `shared_event_re.model.tau_s` is positive for both phases,
- set `shared_event_re.fallback.abort_on_pcg_fallback: true` while debugging — the resulting error
  prints the full reason breakdown.

See {doc}`pcg-whitening-convergence` for the full checklist.

## Post-processing

### `read_all_samples` returns `{}` / `Samples store not found`

The path is taken from `io.samples_outfile` (a nested config dict, a params JSON path, or a
materialized dict with `samples_outfile` are all accepted) relative to the current working
directory. Run from the directory the config was written for, or pass an absolute path.

### `compute_ess_summary` raises "must include centered samples X, Y, Z, T"

Build the summary with `include=["X", "Y", "Z", "T"]` (plus `"cat_dd"` if you want the table);
`T` is not part of the default `include` set.

### `spider analyze-resid` prints `no sub-diagnostic is enabled, nothing will be computed`

Nothing is enabled: turn on `observability.diagnostics.shared_event_re_tau.enabled`,
`spatial.enabled`, `pathcorr.enabled` and/or `event_pair_station_corr.enabled` (see
{doc}`diagnostics-catalog`); the message lists the keys. The `resid_distribution`,
`resid_scalar_metrics` and `shared_event_legcorr2d` blocks were removed and are ignored even when
`enabled: true`.
