# Configuration Reference (config_v2)

This page is a field-oriented reference for the current strict schema. Every key, default and
constraint below was checked against the code that enforces it:

- `spider/core/config_v2/validate.py` -- strict key checks per block (unknown keys raise)
- `spider/core/config_v2/legacy_bridge.py` -- how each key materializes to runtime parameters;
  **defaults live here**
- `spider/core/priors_config.py` -- `model.priors`, validated at runtime (no implicit defaults)

Validation entrypoints:

- `spider validate-config <config.json> --mode sample` (add `--print-resolved` to see the defaults
  that were filled in, e.g. `io.sample_write_interval <- io.checkpoint_interval`)
- `spider.core.config_v2.validate.parse_canonical_config()`

The root object only allows:

- `io`
- `model`
- `inference`
- `observability`
- `synth` (optional; consumed only by `spider synth`, see below)

Unknown keys at any strict block level raise a validation error. Two levels of strictness exist:

- **Strict blocks** (`io`, `model` top level, `model.domain`, `inference.*`, `observability`,
  `observability.wandb`, `shared_event_re` top level, `inference.sampler.preconditioning`,
  `reparameterization`, `lr_schedule`, `phase1_two_pass`): unknown keys are rejected by
  `validate-config`.
- **Pass-through blocks** (`model.priors`, `model.likelihoods.*` sub-blocks, `model.filters.*`,
  `model.eikonet`, `inference.batching.*`, `inference.runtime.gauge_projection`,
  `inference.runtime.torch`, `inference.sampler.preconditioning.lrd`, `inference.sampler.overrides`,
  `observability.diagnostics`, `synth`): the block itself must be a dict, but its keys are read by
  the runtime with defaults. A misspelled key inside a pass-through block **does not fail
  validation and has no effect at runtime**; instead `spider validate-config` prints a non-fatal
  `[warning] <block> (unknown_key): Unknown key(s) ignored by the runtime: <keys> | suggestion: Known keys: ...`
  line for every pass-through block that contains a key no code reads (`model.priors` is the
  exception: it is strictly validated at runtime, and `synth` is not checked). Programmatically the
  same list is available as
  `spider.core.config_v2.validate.collect_unknown_key_warnings(raw_config_dict) -> list[str]`
  (or `collect_unknown_key_issues(raw) -> list[ConfigIssue]` with `severity="warning"`); the known-key
  sets mirror what `legacy_bridge.py` and the runtime consumers actually read. Treat every such
  warning as a typo until proven otherwise.

Two complete, recommended configurations ship with the repository:
`example/SPIDER_example.json` and `spider/examples/params_template.json`.

## `io` block

Required keys:

| key | type | meaning |
|---|---|---|
| `dtime_file` | string | differential-time CSV (see {doc}`data-formats`) |
| `station_file` | string | station CSV |
| `catalog_infile` | string | starting catalog CSV |
| `catalog_outfile` | string | output catalog stem; `<catalog_outfile>_MAP.csv` is written at the end of Phase 1 |
| `samples_outfile` | string | HDF5 file receiving Phase-4 samples |
| `checkpoint_dir` | string | directory for checkpoints and the Phase-2 bundle |
| `checkpoint_interval` | int `>= 1` | epochs between checkpoints (all phases) |
| `save_every_n` | int `>= 1` | Phase 4: keep one posterior sample every `save_every_n` **optimizer steps** (minibatches), not epochs |
| `write_samples` | bool | `false` disables sample collection and HDF5 flushes entirely |

Optional keys:

| key | default | meaning |
|---|---|---|
| `sample_write_interval` | `checkpoint_interval` | epochs between flushes of the in-memory sample buffer to `samples_outfile` (`>= 0` when set; `0` disables periodic flushes, leaving only the final flush at the end of Phase 4) |

## `model` block

Allowed keys:

- `model_file` (required, string): EikoNet state dict, or the GridTT table directory when
  `eikonet.model_kind = "gridtt"`
- `domain` (required)
- `priors`
- `likelihoods`
- `filters`
- `eikonet`

### `model.domain`

Strict block. Required:

- `lon_min`, `lat_min` (degrees): origin of the local `laea` projection in which all `x/y` are km
- `z_min`, `z_max` (km, must satisfy `z_max > z_min`): depth range of the travel-time model
- `scale` (km, `> 0`): horizontal extent used to normalize EikoNet inputs (fallback for
  `eikonet.x_max`/`y_max`)

### `model.priors`

Not key-checked by `validate-config`, but strictly validated at runtime by
`spider/core/priors_config.py`: every listed key is required unless marked optional, there are no
implicit defaults for prior scales, and removed keys raise. Units are km for `x/y/z` and seconds
for `t`.

- `event` (**required**): `enabled` (bool), `type` (`gaussian`), `params` (dict, always required;
  `params.std` = list[4] `[σx, σy, σz, σt]`, required when `enabled`). A Gaussian prior on each
  event's shift from its initial catalog location. **Use it together with `event.hyper` (below); a
  fixed `params.std` alone is not recommended.** With the hierarchical prior enabled, `params.std`
  only acts until the first Wishart update (epoch `update.start_epoch` of Phase 1), so keep it loose
  (several km, seconds) -- a tight fixed prior fights the differential data and blurs fine structure
  in the MAP.
  - `event.hyper` (**recommended** hierarchical prior; optional block): `enabled` (bool),
    `type` (`wishart_precision`), `params.df` (number, **`> 3`**), `params.scale_std` (list[4]),
    `update.every_epochs` (int `>= 1`, required when enabled), `update.start_epoch` (int `>= 0`,
    default 0: epochs to keep the fixed `params.std` Gaussian before the first Wishart update -- use a
    few hundred so the scale is not estimated from ΔX≈0), `update.min_cluster_events` (int `>= 0`,
    default 0: connected components with fewer events than this share the precision estimated from
    all events instead of their own -- a handful of ΔX cannot pin a 4×4 precision, and a small
    component that drifts loosens its own prior and runs away). The learned precision is one 4×4
    matrix per connected component of the pair graph; `df` is the pseudo-count weight of
    `scale_std` (the Wishart prior mean corresponds to a diagonal precision `1/scale_std²`).

    What happens at runtime: at the end of every epoch with `epoch >= start_epoch` and
    `epoch % every_epochs == 0`, in **every phase** (both MAP passes and Phases 2-4), each component's
    precision is redrawn from its conjugate Wishart posterior given the current shifts of its events
    (a Gibbs step). Every `observability.diagnostics.display_precond_every` epochs the console shows
    `[PRIORS] Hierarchical event prior updated: std≈[…] corr_zt≈… [largest of K clusters, n_events=…]`
    -- the implied standard deviations of the largest component, which is what most events feel. The
    learned matrices are saved in every checkpoint and in the Phase-2 bundle written by `locate-map`;
    `spider sample` adopts them (`[PRIORS] Adopted hierarchical event-prior precision from bundle`) and
    warns if the bundle predates the feature.

    Why: the MAP wants a loose prior, but the sampling phases need a realistic one, otherwise the
    injected noise walks weakly constrained events away from the data. A single fixed `params.std`
    cannot be both. The hierarchical prior learns the event-shift scale from the catalog itself
    (typically a few hundred metres) and carries it from the Phase-2 bundle into sampling, so no
    hand tuning is needed. Recommended block:

    ```json
    "event": {
      "enabled": true, "type": "gaussian", "params": {"std": [5.0, 5.0, 5.0, 2.0]},
      "hyper": {
        "enabled": true, "type": "wishart_precision",
        "params": {"df": 10.0, "scale_std": [0.5, 0.5, 0.5, 0.2]},
        "update": {"every_epochs": 10, "start_epoch": 200, "min_cluster_events": 100}
      }
    }
    ```

- `centroid` (optional block): `enabled` (bool), `type` (`gaussian`), `params` (dict; `params.std`
  list[4], required when `enabled`). A Gaussian prior on the **mean shift of each connected
  component**, with log-density weighted by the component's event count `n_k` (so the effective
  prior width of a component mean is `std/sqrt(n_k)`). It fixes each component's translation gauge,
  which pure differential data leave undetermined, so keep it tight (for example `0.01`) when
  loosening `event.params.std`. Components that share no data are not coupled; there is no global
  form. When `observability.diagnostics.wandb.groups.priors` is on, the per-epoch `centroid/*`
  metrics are logged (`centroid/n_components`, `centroid/main_dx|dy|dz|dt`, `centroid/max_dr`,
  `centroid/max_abs_dt`, `centroid/rms_dr_weighted`, `centroid/z_max`,
  `centroid/frac_components_z_gt3`, `centroid/neg_logp`), and every `display_precond_every` epochs the
  console prints `[PRIORS] Component centroids: K=… | main (n=…) shift […] km … s | worst z=…`.

Removed keys that are now hard errors: `priors` at the top level (use `model.priors`),
`priors.noise` (noise/σ learning was removed; `phase_unc` is fixed), `priors.laplacian`,
`priors.event.schedule` (priors are active in all phases when enabled),
`priors.event.hyper.update.active_phases`, `priors.centroid.scope` (the centroid prior is always per
connected component), and the legacy flat keys `prior_event_enable`, `prior_centroid_enable`,
`prior_noise_enable`, `prior_laplacian_enable`, `prior_laplacian_phase4_only`, `prior_event_std`,
`prior_centroid_std`, `hierarchical_event_prior`, `hierarchical_prior_dof`, `noise_prior`,
`noise_prior_loc`, `noise_prior_scale`, and every `laplacian_*` key.

### `model.eikonet`

Pass-through dict read by `spider/core/eikonet_loader.py`; it must match the architecture of the
checkpoint at `model.model_file`. Unknown keys are ignored at runtime and reported as warnings by
`validate-config`.

| key | default | meaning |
|---|---|---|
| `model_kind` | `1d` | `1d` or `3d` (EikoNet checkpoints), or `gridtt` (precomputed per-station eikonal tables; `model_file` is then the table directory written by `gridtt build-spider`, and the remaining keys in this table are not used) |
| `x_max`, `y_max` | `model.domain.scale` | horizontal extent (km) used to normalize coordinates |
| `n_hidden` | 128 | hidden width |
| `n_blocks` | 5 | residual blocks |
| `n_fourier` | 4 | Fourier features per input |
| `use_fourier` | true | enable Fourier encoding |
| `phase_emb_dim` | 8 | phase embedding size |
| `t2_log_scale` | 0.1 | log-scale for the 3-D correction term |
| `vp`, `vs` | 6.0, 3.2 | reference velocities (km/s) for the `T0` factor |

`z_min`/`z_max` are always taken from `model.domain`.

### `model.filters`

Required sub-blocks (validated by the runtime bridge, not by `validate-config`; unknown keys inside
them are ignored at runtime and reported as warnings by `validate-config`):

- `dtimes`
- `events`
- `residual`

| key | default | meaning |
|---|---|---|
| `dtimes.remove_duplicates` | false | keep one row per (unordered pair, network, station, phase) |
| `dtimes.max_abs_input_dt` | 99999.0 | drop rows with `abs(dt) >= ` this (s); `<= 0` disables |
| `dtimes.dtime_thin_frac` | 1.0 | random thinning fraction of rows kept (`< 1` activates; fixed seed 42, so repeatable) |
| `dtimes.flip_dt_sign` | false | negate `dt` on input (use when your dtime file follows the opposite sign convention; SPIDER predicts `dt = T(event2) - T(event1)`) |
| `dtimes.cc_min` | 0.0 | drop rows with `cc <` this (`<= 0` disables) |
| `events.min_dtimes` | 1 | minimum rows per unordered pair, applied late (after the ratio/finite filters); iterated to a fixed point |
| `events.min_unique_phase_per_event` | 1 | minimum distinct (network, station, phase) per event, counted across all its pairs |
| `events.min_dtimes_per_pair` | 1 | minimum rows per unordered pair, applied right after `cc_min` |
| `events.min_event_degree` | 0 | minimum number of distinct partner events (iterative pruning until stable) |
| `events.min_events_per_cluster` | 0 | drop connected components smaller than this (only when `> 1`). Re-applied at the end of Phase 1 after the post-MAP row filters; events left with no dtimes are always dropped there (see below) |
| `events.max_pair_station_ratio` | 1.0 | drop rows whose 3-D pair separation `R_ev` exceeds this fraction of the 3-D distance `D_es` from the pair centroid to the station (`<= 0` disables) |
| `events.lat_bounds`, `events.lon_bounds` | null | optional `[min, max]` (degrees, `min < max`) catalog subset applied before everything else; dtimes referencing dropped events are removed |
| `events.ratio_filter_phase` | `before` | `before` (data prep) or `after_phase1` (end of Phase 1). Unlike `residual.phase`, an unrecognized value is not rejected -- it silently disables the filter |
| `events.linearization_error.enabled` | false | drop rows whose linearization-error ratio exceeds `max_ratio` |
| `events.linearization_error.phase` | `after_phase1` | `before` or `after_phase1` |
| `events.linearization_error.max_ratio` | (required when enabled) | ratio threshold, e.g. `0.05` |
| `events.linearization_error.batch_size` / `sample_size` / `log_every_batches` | 50000 / 200000 / 25 | sweep batching and logging |
| `residual.enabled` | false | residual outlier filter |
| `residual.method` | `mad` | `mad` or `abs` (any other value behaves as `mad`) |
| `residual.mad_sigma` | 6.0 | MAD multiplier (`mad` method) |
| `residual.abs_max` | 99999.0 | absolute residual cap (s); `<= 0` means no cap |
| `residual.phase` | `after_phase1` | `before` or `after_phase1` (other values are rejected) |

`residual.phase` controls when the residual outlier filter runs (default `after_phase1`):

- `after_phase1`: at the end of Phase 1, with residuals evaluated at the MAP locations. Applies to `locate-map` outputs, the Phase-2 bundle, and the optional second MAP pass (see `inference.sampler.phase1_two_pass`).
- `before`: during data preparation, with residuals evaluated at the initial catalog locations (`ΔX = 0`).

With `method: abs` a row is kept when `|r| <= abs_max`. With `method: mad` the threshold is
`min(abs_max, mad_sigma * MAD)` applied to `|r - median(r)|`; if the MAD is zero or non-finite the
filter falls back to `abs_max`. The console reports the decision, e.g.
`[FILTER] Residual filter: dropping 1234/567890 dtimes (thr=0.150s (abs_max), method=abs).`
For a two-pass MAP run, `method: abs` is recommended: `mad` re-derives its threshold from the
already-cleaned residuals on each application and keeps trimming.

If neither path ran (for example `epochs_per_phase[0] = 0`, or a bundle produced before this
option existed), the filter is applied once at the start of Phase 2. Bundles written by
`locate-map` record whether the filter was already applied (`residual_filter_applied`), and
`spider sample` skips re-applying it in that case
(`[FILTER] Residual filter already applied to bundle dtimes; skipping re-application at Phase 2 start.`).

#### Filter execution order

Data preparation (`prepare_input_dfs`); each `[FILTER]`/`[DATA]` log line reports the row counts:

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
17. de-duplicate catalog rows by `evid` (first row wins), then drop catalog events with no remaining dtimes

Note that thinning (`dtime_thin_frac`) runs before `cc_min` and the `min_*` event filters, so it
affects which events survive.

End of Phase 1 (`_finalize_phase1`, after `<catalog_outfile>_MAP.csv` is written), evaluated at
the MAP locations: `max_pair_station_ratio` (when `ratio_filter_phase: after_phase1`) →
`residual` (when `phase: after_phase1`) → `linearization_error` (when `phase: after_phase1`) →
post-Phase-1 event filter (below). With two-pass MAP this sequence runs after each pass.

The row filters that run at the MAP solution can leave events with no dtimes at all, or in
components smaller than `min_events_per_cluster`. Such events are removed from the event set before
the optional second MAP pass and the Phase-2 bundle: with no likelihood, the preconditioned sampler
noise walks them to the `inference.safety.max_abs_dX` clamp box and, through the centroid prior,
shifts every other event. Their MAP locations are kept in `<catalog_outfile>_MAP.csv` with
`dropped_after_phase1 = 1` (current events carry `0`; see {doc}`outputs-and-formats`). The log
reports the count under
`[FILTER] Post-Phase-1 event filter: dropping N/M events (a with no remaining dtimes, b in components < min_events_per_cluster=…)`
and the new graph size (`events=… dtimes=… components=…`). This step is skipped under torchrun/DDP
(the event set must stay identical across ranks).

### `model.likelihoods`

Required sub-blocks (validated by the runtime bridge): `locate_map`, `sample`. Unknown keys inside
them are ignored at runtime and reported as warnings by `validate-config`.

`locate_map` (Phase 1, MAP):

- `type`: `huber` (default), `laplace` (aliases `l1`, `mae`), `l2` (aliases `gaussian`, `mse`), or
  `student_t` (aliases `student-t`, `studentt`). **An unrecognized value silently falls through to
  Huber.** The example configs use `laplace`.
- `phase_unc` (list[2] `[σP, σS]` in seconds, required): fixed per-phase noise scales -- noise
  learning was removed, these are not estimated
- `huber_delta` (default 1.0, in units of standardized residual `r/σ`), `student_t.nu` (default 4.0)

`sample` (Phases 2-4):

- `type` must be `correlated_gaussian` (`correlated` is accepted as an alias); anything else is a
  hard error
- `phase_unc` (list[2], required)
- `huber_delta`, `student_t.nu`: parsed but **unused** (the sampling likelihood is always Gaussian)
- `shared_event_re` (below)

## `model.likelihoods.sample.shared_event_re`

The collapsed shared-event random-effect likelihood used in Phases 2-4 (see
{doc}`probability-model` for the model and {doc}`pcg-whitening-convergence` for the solver). The
top level of this block is strict; the sub-blocks are pass-through (unknown keys are ignored at
runtime and reported as `validate-config` warnings, except the legacy keys listed at the end, which
raise, and `solver.kind`, whose value is checked).

Allowed top-level keys: `enabled`, `model`, `limits`, `fallback`, `numerics`, `station_phase_term`,
`solver`, `edge_weights`, `autotune`, `logging`.

`enabled` (default `false`): with `false`, the `sample` likelihood reduces to an independent
Gaussian with the fixed `phase_unc` scales.

### `shared_event_re.model`

| key | default | meaning |
|---|---|---|
| `group_by` | `station_phase` | the only supported value (`phase` raises at run time) |
| `tau_s` | `[0.0, 0.0]` | `[τP, τS]` (s), shared-event random-effect scales; a scalar is broadcast to both phases. **Set both `> 0`**: groups with `τ = 0` fall back to the diagonal likelihood and are counted as `tau_zero` in the fallback statistics |
| `cluster.mode` | `none` | `none`, `component` (restrict the correlated solve to within connected components) or `dd_khop` (non-overlapping k-hop balls on the pair graph); cross-cluster rows then enter as independent Gaussians |
| `cluster.k` | 1 | hop radius for `dd_khop` |

### `shared_event_re.limits`

| key | default | meaning |
|---|---|---|
| `max_nodes` | 512 | largest group (distinct events) solved with the correlated form; bigger groups fall back to diagonal (raise this: the examples use 25000) |
| `max_rows` | 200000 | largest group (rows) solved with the correlated form |

### `shared_event_re.fallback`

| key | default | meaning |
|---|---|---|
| `to_diag` | true | allow a group that exceeds the caps or fails PCG to fall back to the diagonal likelihood |
| `abort_on_pcg_fallback` | false | raise instead of falling back (the error message lists the fallback reasons: `rows_cap`, `nodes_cap`, `tau_zero`) |

### `shared_event_re.numerics`

| key | default | meaning |
|---|---|---|
| `jitter0` | 1e-8 | initial diagonal jitter added to the node system |
| `jitter_max` | 1e-3 | jitter ceiling |

### `shared_event_re.station_phase_term`

Optional additive collapsed station-phase random effect: `enabled` (default `false`), `tau_s`
(`[τP, τS]`, default `[0.0, 0.0]`).

### `shared_event_re.solver`

| key | default | meaning |
|---|---|---|
| `kind` | `pcg` | `pcg` (aliases `pcg_sparse`, `pcg-sparse`; case-insensitive). **Any other value is rejected by `validate-config`** (`enum_error` listing the accepted names), because the likelihood code (`spider/core/modeling.py`, `compute_likelihood_loss`) only enters the correlated whitening path for these names and would otherwise fall back to the independent Gaussian without a warning. Older configs with `chol` must be changed to `pcg` |
| `max_iters` | 50 | PCG iteration cap |
| `min_iters` | 2 | minimum PCG iterations |
| `tol` | 1e-3 | relative residual tolerance |
| `batched` | true | solve groups of similar size together |
| `node_bin_edges` | `[512, 1024, 2048, 4096, 8192, 16384, 32768]` | size bins for batched solves (may be re-tuned by `autotune`) |
| `warm_start` | false | reuse the previous solution as the PCG starting point |
| `cache_max_entries` | 8 | cached grouping plans |
| `prefetch_grouping` | false | build the grouping plan before the whitening call |
| `profile_micro_steps` | false | detailed timing of the PCG sub-steps |
| `merge_sparse_node_bins` | true | merge bins that hold too few groups |
| `min_groups_per_node_bin` | 32 | threshold for merging |
| `max_node_bins_per_node` | 4 | cap on bins a node size can be spread over |
| `precompute.enabled` | false | precompute whitening structures once per epoch |
| `precompute.device` | `gpu` | device for the precompute |

### `shared_event_re.edge_weights`

| key | default | meaning |
|---|---|---|
| `mode` | `uniform` | `uniform`, `distance_rbf`, `distance_linear`, or `distance_power` (distance-dependent weights on the incidence operator, evaluated at the current event locations) |
| `ell_km` | 1.0 | RBF length scale (km) |
| `eps_km` | 1e-3 | distance floor (km) |
| `power` | 1.0 | exponent for `distance_power` |
| `scale_km` | 1.0 | distance scale (km) for `distance_linear` / `distance_power` |
| `global_scale` | 1.0 | multiplier on all weights |
| `normalize` | false | normalize weights within each group |

### `shared_event_re.autotune`

Re-tunes `solver.node_bin_edges` (and optionally `limits.max_nodes`) from the observed group-size
histogram during the first sampling epochs.

| key | default | meaning |
|---|---|---|
| `enabled` | true | |
| `observe_epochs` | 1 | epochs of statistics to gather first |
| `latest_epoch` | 2 | last epoch at which a re-tune may happen |
| `min_groups` | 128 | minimum number of groups seen before tuning |
| `max_node_bins` | 8 | maximum number of bins proposed |
| `min_groups_per_node_bin` | 24 | |
| `min_node_bin` | 512 | smallest bin edge |
| `min_gain` | 0.08 | required relative improvement of the bucket-plan score |
| `raise_nodes_cap` | true | allow raising `limits.max_nodes` |
| `nodes_cap_max` | 65536 | ceiling for the raised cap |

### `shared_event_re.logging`

| key | default | meaning |
|---|---|---|
| `quiet` | true | suppress the one-time `[shared_event_re] …` activation/whitening lines |
| `stats_log_every_epochs` | 0 | epoch cadence of whitening statistics (`0` disables) |

### Rejected legacy keys

These raise a `legacy_key` error with a pointer to the replacement:

- top level: `grouping`, `cluster_mode`, `cluster_k`, `tau_s`, `max_nodes_per_group`,
  `max_rows_per_group`, `fallback_to_diag`, `abort_on_pcg_fallback`, `jitter0`, `jitter_max`,
  `station_phase_re`
- `solver`: `bucket_nodes`, `merge_sparse_edge_bins`, `min_groups_per_edge_bin`,
  `max_edge_bins_per_node`
- `autotune`: `max_bins`, `min_bin_groups`, `min_bucket_node`

## `inference` block

Allowed keys:

- `sampler`
- `batching`
- `runtime`
- `safety`
- `compute`

`inference.diagnostics` is rejected (use `observability.diagnostics`).

### `inference.sampler`

Strict block. Required:

| key | constraint | meaning |
|---|---|---|
| `backend` | `psgld` or `sghmc` | sampler for Phases 2-4 (Phase 1 always uses Adam) |
| `epochs_per_phase` | list of 4 ints, each `>= 0` | epochs for Phase 1 (MAP), 2 (drift-only preconditioner warm-up), 3 (noise ramp), 4 (sampling) |
| `lr` | list of 4 numbers, each `> 0` | `lr[0]` is the Phase-1 Adam learning rate (used as is); `lr[1..3]` are **per-observation** sampler step sizes: the optimizer receives `lr[i] / N` where `N` is the number of dtimes, because the drift uses `N × (minibatch-mean gradient)` |
| `temperature` | `>= 0` | injected-noise temperature; noise std scales as `sqrt(T)`; `0` turns noise off |
| `beta` | `0 <= beta < 1` | RMSprop second-moment decay for the preconditioner |
| `eps` | `> 0` | RMSprop damping `G = 1/(eps + sqrt(v))` |
| `freeze_preconditioner_sampling` | bool | freeze preconditioner statistics in **Phases 3 and 4** (Phase 2 always adapts). `true` gives a time-homogeneous sampling kernel; Phase 4 warns when it is `false` with preconditioning on |
| `sghmc_alpha` | required key; number for `sghmc` (finite, `> 0`, and `<= 1` for the SGHMC class); number or `null` for `psgld` | SGHMC friction |

Optional:

| key | default | meaning |
|---|---|---|
| `noise_scale_mult` | 1.0 (`> 0` when set) | extra multiplier on the injected noise in Phases 3 and 4 |
| `grad_clip_norm` | unset | global gradient-norm clip on ΔX (`>= 0`; `0` disables). **When set it applies to all phases.** When unset, Phase 1 (Adam) clips at 100.0 and Phases 2-4 do not clip |
| `preconditioning` | `{}` | see below |
| `reparameterization` | `{}` | see below |
| `overrides` | `{}` | see below |
| `lr_schedule` | `{}` | see below |
| `phase1_two_pass` | `{}` | see below |

`preconditioning` (strict sub-block):

- `enabled` (bool, default `false`; `false` forces type `none` regardless of `type`)
- `type` (`rmsprop`, `lrd`, `component_lrd`, or `block_gn` when enabled; aliases `cc_lrd`, `block_lrd`, and
  `component-lrd` are accepted). With `block_gn`, `lr[1..3]` are dimensionless natural steps and
  values `>= 2` are rejected; `inference.runtime.gauge_projection.enabled` must be `true` (and
  `dims` must be `[0, 1, 2, 3]` when the centroid prior is enabled); see {doc}`sampler-options`.
- `include_gamma` (bool, default **`false`**): add the pSGLD Γ correction drift (pSGLD + `rmsprop`
  only). Γ is the diagonal approximation to the derivative of the preconditioner,
  `Γ = -(1-β) ḡ √v / (ε + √v)²`, computed from the minibatch-mean gradient; in exact pSGLD it
  compensates for the preconditioner changing along the trajectory. With a slowly adapting (or
  frozen, see `freeze_preconditioner_sampling`) RMSprop preconditioner the term is negligible and
  only adds cost, so it is off by default; both shipped example configs also set it to `false`.
  Set `true` to recover the full pSGLD drift when the preconditioner is adapting quickly.
- `lrd` (dict when present; **keys are not whitelisted**): `rank` (16), `mode` (`svd` | `oja`,
  default `svd`; `randomized_svd` / `stochastic_svd` alias to `svd`, other values fall back to
  `svd`), `update_every` (20), `buffer_size` (64), `eta` (or `oja_eta`, 0.02), `diag_floor`
  (defaults to `inference.sampler.eps`), `target` (`dX_src_only`)
- `block_gn` (strict sub-block, used with `type: block_gn`): `fallback_std` (4 positive numbers
  `[km, km, km, s]` or `null`), `min_std` (4 non-negative numbers or `null`),
  `refresh_every_epochs` (int `>= 1` or `null`), `refresh_on_prior_update` (bool, default
  `true`), `batch_size` (int `>= 1`, default 262144), `jitter` (`>= 0`, default 1e-6)

`overrides` (dict): `core.{lr_mult, temperature_mult, eps, freeze_preconditioner_sampling}`.
`core` is the name of the single sampler parameter group (the event shifts); other group names are
ignored. `lr_mult` multiplies the phase learning rate after the `/N` scaling, `temperature_mult`
multiplies `temperature`, `eps` replaces `eps`, and `freeze_preconditioner_sampling` is OR-ed with
the global flag (it can freeze, never unfreeze).

`reparameterization` (strict sub-block):

- `enabled` (bool, default `false`)
- `spatial_scale` (`> 0`, default 1.0): static scale for `x/y/z` in blocked sampler coordinates
- `dt_scale` (`> 0`, default 1.0): static scale for `dt`

See {doc}`sampler-options` for what the scales do. The example configs use `dt_scale: 0.3` with
pSGLD. Ignored (with a warning) under `preconditioning.type: block_gn`.

`lr_schedule` (strict sub-block; Phase-1 / MAP learning-rate schedule; default constant `lr[0]`):

- `phase1.type` (`none` or `cosine`)
- `phase1.eta_min` (`>= 0`, final learning rate of the cosine decay; default `0`)
- `phase1.T_max` (`null` or int `>= 1`): cosine horizon in epochs. When `null` it resolves at
  runtime to the epoch count of the current MAP pass -- `epochs_per_phase[0]` for pass 1 and
  `phase1_two_pass.epochs` for pass 2.

Phase 1 logs the resolved schedule:
`Phase 1: MAP (optimizer=adam) | lr_schedule=cosine(T_max=10000, eta_min=1e-05) | …`. On resume the
schedule is fast-forwarded to the resumed epoch.

Example: `"lr_schedule": {"phase1": {"type": "cosine", "eta_min": 1.0e-5}}`

`phase1_two_pass` (strict sub-block; two-pass MAP; default disabled):

- `enabled` (bool)
- `epochs` (`null` or int `>= 1`; epochs for the second pass, default `epochs_per_phase[0]`)
- `warm_start` (bool, default `true`; start the second pass from the pass-1 MAP locations instead of the initial catalog)

When enabled, Phase 1 runs once, the post-MAP filters are applied at the MAP locations
(`model.filters.residual` with `phase: after_phase1`, `max_pair_station_ratio` with
`ratio_filter_phase: after_phase1`, `linearization_error` with `phase: after_phase1`, then the
post-Phase-1 event filter), the Adam optimizer (and any `lr_schedule`) is rebuilt, and MAP is run
again on the filtered dtimes. The console announces
`[RUN] Phase 1 second pass: epochs=… warm_start=… dtimes=… (post-MAP filters applied)` and labels
the second pass's epoch lines `phase1-pass2`. The pass-1 catalog is kept as
`<catalog_outfile>_MAP_pass1.csv`; `<catalog_outfile>_MAP.csv` holds the final result. The second
pass is skipped when `epochs_per_phase[0] = 0`, and (with a warning) under torchrun/DDP. It is
intended for fresh runs (`inference.runtime.reset_batch_numbers: true`), not for resuming into the
second pass.

### `inference.batching`

Strict at the top level (`standard`, `event_batches`, both required dicts); keys inside are
pass-through with these defaults:

| key | default | meaning |
|---|---|---|
| `standard.warmup` | 10000 | rows per minibatch in Phase 1 (also the batch size of the residual-filter sweeps; the linearization filter has its own `batch_size`) |
| `standard.sgld` | 10000 | rows per minibatch in Phases 2-4 |
| `standard.shuffle` | true | reshuffle rows every epoch (seeded by `runtime.seed + epoch`). Phase 4 warns when `false`: a fixed batch order makes the gradient noise periodic |
| `event_batches.enabled` | false | batch by owner events instead of rows (not supported under torchrun/DDP) |
| `event_batches.events_per_batch` | -- | events per batch (required when enabled) |
| `event_batches.max_edges_per_batch` | -- | row cap per batch |
| `event_batches.bucket_reorder_all` | false | rebuild the row order for every bucket generation |
| `event_batches.bucket_reuse_epochs` | 0 | epochs to reuse a bucket assignment |

The example configs use `warmup`/`sgld` of 250 000 to 1 000 000 rows; the shared-event solve groups
rows by station-phase within each batch, so larger batches produce larger, better-conditioned
groups.

### `inference.runtime`

Strict block. Required:

| key | meaning |
|---|---|
| `cuda_empty_cache_every` | int `>= 0`; call `torch.cuda.empty_cache()` every this many optimizer steps (`0` disables) |
| `reset_batch_numbers` | bool; `true` clears checkpoints and starts fresh, `false` resumes from the latest checkpoint in `checkpoint_dir` |
| `clear_samples_on_reset` | bool; delete `samples_outfile` when starting (or, with `reset_batch_numbers: false`, when resuming) so new samples do not append to an old file |
| `verbose` | bool |

Optional (defaults applied at load time and reported by `validate-config --print-resolved`):

- `min_samples_to_save` (int `>= 0`) and `cluster_events` (bool): **accepted for backward
  compatibility; no effect.** Nothing in the runtime reads them (connected components are always
  computed). They may be omitted; when present they are still type-checked.

- `seed` (int, default `0`): base seed for per-epoch batch permutations and the Phase-4 drift
  diagnostic subset; change it for independent chains
- `gauge_projection` (pass-through dict): `enabled` (default `false`), `mode` (`global` |
  `cluster`, default `global`; other values fall back to `global`), `dims` (default `[0, 1, 2]`;
  indices into `[x, y, z, t]`), `apply_noise` (true), `apply_momentum` (true). When enabled, the
  mean over events (globally, or per connected component with `cluster`) of the selected dimensions
  is removed from the gradient before the preconditioner statistics, from the injected noise and
  momentum in the samplers, and, decisively, from the parameters themselves after every optimizer
  or sampler step in all phases (the frame is re-centered exactly). The last step is what makes it
  a hard constraint: Adam and the RMSprop-preconditioned samplers rescale each element by its own
  running statistics, so a zero-mean gradient alone does not give a zero-mean update and the frames
  would still drift. With `mode: cluster` and `dims: [0,1,2,3]` each connected component keeps its
  input-catalog frame in position and origin time, independent of how many events or dtimes it has;
  the centroid prior then has no effect (its term is identically zero). This is the recommended way
  to fix the gauge when the absolute frame of the input catalog is to be preserved (the convention
  GrowClust and HypoDD also use); the soft centroid prior's strength relative to the data scales
  with dtimes per event and does not hold the frame on large datasets. Log check: the
  `[PRIORS] Component centroids` line should report zero shifts.
- `torch` (pass-through dict, applied at process start): `allow_tf32` (bool),
  `matmul_precision` (`highest` | `high` | `medium`), `compile_eikonet` (bool, default false),
  `compile_mode`, `compile_backend`, `compile_dynamic`, `compile_fullgraph`

### `inference.safety`

Strict block with two keys (`max_abs_dX`, `shift_guard`):

- `max_abs_dX`: `null` or list of 4 numbers `[dx, dy, dz, dt]` (km, km, km, s): per-component
  clamp on each event's shift from its initial location, applied after **every** optimizer step in
  all phases (including Phase 1). Entries that are `<= 0` or non-finite disable the clamp for that
  dimension. An event pinned at the clamp shows up as a `dr_max` value in the epoch log that stops
  changing; widen the clamp rather than leave events pinned, and check why the data do not hold
  them (see the post-Phase-1 event filter above). The example configs use `[10, 10, 10, 3]`.
- `shift_guard` (optional strict sub-block): `enabled` (bool, default `false`), `factor` (number
  `> 0`, default `5.0`). When enabled, after every epoch of every phase (`_shift_guard_check` in
  `spider/core/locate.py`) each event's absolute shift `|ΔX|` is compared with
  `factor * model.priors.event.params.std` per dimension `[x, y, z, t]`. If any event exceeds the
  threshold in any dimension, the run prints
  `Shift guard triggered (<phase> epoch <n>): k event(s) exceeded <factor>×prior_event_std`, lists the
  offending events with their observations and residuals at the initial locations, and exits. It is
  a debugging aid for runaway events (wrong `dt` sign, bad station coordinates); leave it off in
  production. The block materializes to the runtime keys `shift_guard_enable` /
  `shift_guard_factor`; the CLI flags `--shift-guard` / `--shift-guard-factor` on `locate-full`,
  `locate-map` and `sample` are applied afterwards and therefore override the config when given.

### `inference.compute`

- `devices`: optional list of device specifiers, each an int (CUDA index; `-1` = CPU) or a
  non-empty string (`"cpu"`, `"cuda:0"`, or a digit string). Empty or omitted lets the CLI choose.

## `synth` block (optional)

Pass-through dict read only by `spider synth` (all keys optional, defaults in parentheses):
`outfile` (required for output), `seed` (42), `overwrite` (true), `apply_filters` (true),
`drop_missing_events` (false), `event_std` (perturbation std for the initial catalog),
`true_catalog_infile`, `true_catalog_outfile`, `init_catalog_outfile`, `init_clip_domain` (true),
`init_spatial_only` (false), `rf_amp` (0.02), `rf_features` (256), `rf_len_xy` (10.0). Ignored by
every other command.

## `observability` block

Allowed keys:

- `wandb`
- `diagnostics`

### `observability.wandb`

Strict block. Required keys:

- `enabled` (bool)
- `project_name` (non-empty string or null; must be non-null when `enabled` is `true`)
- `run_name` (non-empty string or null)

### `observability.diagnostics`

Pass-through dict. Keys read by the current runtime:

| key | default | meaning |
|---|---|---|
| `pair_count_stats_enable` | false | log per-event pair-count statistics at start-up |
| `display_precond_every` | 50 | epoch cadence of the console `[PRIORS]` lines (hierarchical-prior std estimate, component centroids) and preconditioner displays |
| `profile_shared_event_re` | false | per-epoch `shared_event_re/*` timing and workload metrics |
| `wandb.enabled` | true | master switch for W&B metric groups (independent of `observability.wandb.enabled`, which opens the run) |
| `wandb.groups` | all | dict of `{group: bool}` or list of names. Names consumed by the runtime: `core`, `noise`, `sampler`, `ess_online`, `priors` (the `centroid/*` metrics). Other names found in configs (`precond`, `fixed_eval`, `resid_rms`, `latent_field`, `corr_error`) are accepted but have no gating call site |
| `ess_online` | disabled | `enabled`, `every_n_samples`, `n_events`, `seed`, `window`, `max_lag`, `dims` (default `[0, 1, 2]`); needs the `ess_online` W&B group |
| `truth_catalog` | -- | `path`, `time_source` (`map` \| `initial` \| `truth`), `time_ref` (`min` \| `median` \| `first`), `require_all`: optional ground-truth catalog for synthetic tests |
| `spatial` | disabled | end-of-Phase-1 spatial residual autocorrelation diagnostic (Moran's I); keys below |
| `pathcorr` | disabled | end-of-Phase-1 path-similarity residual correlation diagnostic; keys below |
| `sgld_log_gnoise`, `sgld_log_temperature` | -- | **accepted for backward compatibility; no effect** (not materialized; the sampler metrics are emitted whenever the `sampler` W&B group is on) |

Unknown keys anywhere in this block (including inside `wandb`, `ess_online`, `spatial`, `pathcorr`)
are reported as `validate-config` warnings; the `truth_catalog`, `shared_event_re_tau` and
`event_pair_station_corr` sub-blocks are accepted as opaque dicts.

#### `observability.diagnostics.spatial`

One-shot test for spatially correlated residuals at the MAP solution
(`spider/diagnostics/spatial.py`, `run_spatial_diag_end_phase1`). It runs **at the end of Phase 1**
(in `locate-full` / `locate-map`, after the post-MAP filters and the post-Phase-1 event filter, so
after each pass of a two-pass MAP) and again in `spider analyze-resid` on the bundle. It is skipped
silently unless `enabled` is `true`; failures are logged and never abort the run.

What it computes: standardized residuals `r/σ` for every row at the current locations, a robust
per-event scalar `r̄_e` (median of the signed contributions of event `e`, `-r` when `e` is the
first event of the pair and `+r` when it is the second; events with fewer than `min_obs`
contributions are set to 0), then Moran's I of `r̄` over the graph of unique observed event pairs
with Gaussian distance weights `exp(-0.5 (d/ℓ)²)` (`ℓ` = median pair distance, 100 km fallback),
and a permutation p-value from `permutations` random relabelings. Console output (section `DIAG`):
`Spatial diag (end Phase 1): Moran's I=0.0123 (z=1.20, p=0.115) | events=… edges=… kernel_ell≈… km perms=199`.
A large positive I with small p means neighboring events share residual sign, i.e. unmodeled
path/velocity structure (or a station term) that a per-row independent likelihood does not capture.

| key | default | meaning |
|---|---|---|
| `enabled` | false | run the diagnostic |
| `min_obs` | 5 | minimum residual contributions for an event to get a non-zero `r̄_e` |
| `permutations` | 199 | number of permutations for the p-value |
| `seed` | 0 | RNG seed for the permutations |

#### `observability.diagnostics.pathcorr`

Per station-phase residual correlation versus path similarity
(`spider/diagnostics/pathcorr.py`, `run_pathcorr_diag_end_phase1`). Runs at the same two points as
`spatial` (end of Phase 1 and in `analyze-resid`), only when `enabled` is `true`; failures are
logged and never abort the run.

What it computes: for each `(network, station, phase)` group, a per-event median standardized
residual `r_i` (events with fewer than `min_obs_per_event` contributions are dropped), per-event
path features (epicentral distance `d_i`, azimuth `θ_i`, depth `z_i`), then up to `max_pairs`
random event pairs per group with a robustly scaled path dissimilarity
`Δπ = sqrt((Δd/σd)² + (Δθ/σθ)² + (Δz/σz)²)`. The correlation estimate
`ρ̂(Δπ) = E[r_i r_j] / Var(r)` is binned in `Δπ` with width `bin_width` up to `max_pi` (default:
99th percentile of `Δπ`). Output: a CSV with one row per station-phase (`n_events`, `pairs_used`,
`sigma_d`, `sigma_theta`, `sigma_z`, `rho0` = correlation in the first bin, `pi_at_0.1` = first bin
centre where `ρ̂ <= 0.1`) written to `outfile` (default
`<io.catalog_outfile>_pathcorr_phase1.csv`), plus console lines
`PathCorr: wrote per-station–phase summary to …`, `PathCorr: groups=… median rho0=…` and a table of
the `top_k_print` groups with the largest `rho0`. A `rho0` well above 0 for many groups indicates
shared path effects between events with similar ray paths, i.e. the motivation for the
shared-event random-effect likelihood and a guide for `shared_event_re.model.tau_s` and
`edge_weights`.

| key | default | meaning |
|---|---|---|
| `enabled` | false | run the diagnostic |
| `min_obs_per_event` | 3 | minimum contributions per (station-phase, event) |
| `seed` | 0 | RNG seed for pair sampling |
| `max_pairs` | 50000 | maximum sampled event pairs per station-phase |
| `bin_width` | 0.5 | width of the `Δπ` bins |
| `top_k_print` | 8 | number of groups printed in the console table |
| `max_pi` | 99th percentile of `Δπ` | upper edge of the `Δπ` binning; only set when configured |
| `outfile` | `<catalog_outfile>_pathcorr_phase1.csv` | CSV path for the per-group summary |

See {doc}`diagnostics-catalog` for behavior and interpretation.
