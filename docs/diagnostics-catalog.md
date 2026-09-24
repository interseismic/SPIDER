# Diagnostics Catalog

This page maps the `observability.diagnostics` settings to runtime behavior and lists every metric
and console line SPIDER emits, with interpretation guidance. Metric names below are exactly the keys
written to Weights & Biases (W&B); console lines are quoted as they appear in the log, with the
`[spider][LEVEL][SECTION]` prefix omitted.

## Where diagnostics are configured

- `observability.wandb.{enabled, project_name, run_name}` — turns W&B on/off. Nothing in this
  catalog reaches W&B unless `enabled` is `true` **and** the `wandb` package initializes (an init
  failure only prints `[W&B] Disabling wandb due to init failure: ...` and the run continues).
- `observability.diagnostics.*` — everything else on this page.

Two validation facts worth knowing:

- Keys under `observability.diagnostics` are **not** schema-checked: a misspelled key is silently
  ignored. Use `spider validate-config <params> --print-resolved` and search the output for the
  materialized flat keys (`display_precond_every`, `ess_online_*`, `_wandb_diag_groups`, ...) to
  confirm what was picked up.
- Putting the block under `inference.diagnostics` is a hard error
  (`inference.diagnostics is not allowed; use observability.diagnostics`).

## Configuration keys

| Key | Default | Effect |
| --- | --- | --- |
| `pair_count_stats_enable` | `false` | Prints per-event pair/station count statistics when the state is built. |
| `display_precond_every` | `50` | Cadence, in epochs, of the two periodic `[PRIORS]` console summaries: the hierarchical event-prior line and the component-centroid line (both below). Raise it if the log is too chatty. |
| `profile_shared_event_re` | `false` | Enables shared-event whitening timing/workload profiling and gates every `shared_event_re/*` W&B metric. Keep it off unless you are tuning the PCG solver (see {doc}`pcg-whitening-convergence`). |
| `wandb.enabled` | `true` | Master switch for *all* W&B metric groups (independent of `observability.wandb.enabled`). |
| `wandb.groups` | `{"all"}` | Which metric groups to emit; list (`["core", "sampler"]`) or dict (`{"core": true, ...}`). `all` or `*` enables everything. See "W&B group gating". |
| `ess_online.*` | disabled | Online ESS during sampling; see "Online ESS". |
| `truth_catalog` | — | Ground-truth catalog for `spider analyze-resid` (`path`, `time_source`, `time_ref`, `require_all`); `truth_locations` is accepted as a legacy alias. Used when a sub-diagnostic sets `residuals_at: "truth"`. |
| `shared_event_re_tau` | disabled | `analyze-resid` sub-diagnostic; see below. |
| `event_pair_station_corr` | disabled | `analyze-resid` sub-diagnostic; see below. |

Accepted but with **no effect**:

- `sgld_log_gnoise`, `sgld_log_temperature` — the gradient-noise and temperature metrics are now
  emitted whenever the sampler exposes them and the `sampler` group is on.
- `resid_distribution`, `resid_scalar_metrics`, `shared_event_legcorr2d` — removed diagnostics;
  the blocks are parsed and then hard-disabled at runtime, so `enabled: true` does nothing.
- `hierarchical_log_every_epochs` — superseded by `display_precond_every`.

## Optimization-health metrics (always computed, every phase)

These come from the epoch runner and are printed in the per-epoch console line (see
{doc}`outputs-and-formats`) and logged to W&B under the `core` group.

- `loss` — edge-count-weighted mean loss over the epoch's minibatches; `nan` when no batch
  contributed a step.
- `skipped_batches` — minibatches dropped by the finiteness guards (non-finite loss or gradient);
  `n_batches` is the number of minibatches seen in the epoch. Any non-zero `skipped_batches`
  appends `skipped=k/M` to the per-epoch console line and also prints
  `[WARN][RUN] epoch N: skipped k/M batches (non-finite loss=a, non-finite grad=b); parameters unchanged for those batches.`
  If `k == M` the message continues with `ALL batches skipped -> optimization is frozen. ...` and
  `loss` is `nan`. This is the fastest way to catch a run that looks converged but is not
  updating; see {doc}`troubleshooting`.
- `epoch_time` (s), `permute_time` (s, time spent building the epoch's batch permutation).
- Spatial shift statistics over all events (km): `dx_med_abs`, `dy_med_abs`, `dz_med_abs`
  (median |Δ| per axis, printed as `dx=`, `dy=`, `dz=`), `dr_90`, `dr_max` (90th percentile and
  maximum of the 3-D shift norm). Compare `dr_max` with the norm of `inference.safety.max_abs_dX`:
  equality epoch after epoch means events are pinned at the clamp.
- `dx_mean`, `dy_mean`, `dz_mean` — mean shift over *all* events. With several connected
  components this says little (the centroid prior acts per component); use `centroid/*` instead.
- Origin-time shift statistics (s): `dt_mean`, `dt_med_abs` (printed as `dT=`), `dt_max_abs`,
  `dt_p90_abs`.
- Centered origin-time statistics (s): `dt_centered_std`, `dt_centered_med_abs`,
  `dt_centered_p90_abs` — same after subtracting each connected component's mean ΔT (or the
  global mean when no component labels exist). A pure differential-time likelihood is blind to a
  common time shift, so the raw `dt_*` values can drift slowly while these stay put; the centered
  values measure the *relative* timing that the data actually constrain.

Phases 2–4 add, under the `sampler` group: `learning_rate` (of the first parameter group) and
`noise_enabled` (`0`/`1`; expect `0` in Phase 2, where the sampler runs as a deterministic
preconditioned drift, and `1` in Phases 3–4). Under the `noise` group: `noise/sigma_p`,
`noise/sigma_s`, `noise/log_sigma_p`, `noise/log_sigma_s` (the fixed `phase_unc` values;
noise-scale learning was removed, so these are constants that document the run). Phase 1 logs
`learning_rate` and `noise/*` as well, without group gating.

Every W&B record also carries `phase` (`phase1` ... `phase4`), `phase_epoch` and `global_step`.
Metric names are *not* prefixed per phase; filter on `phase` in W&B when comparing phases.

## Hierarchical event prior (`[PRIORS]` console lines)

With `model.priors.event.hyper.enabled: true` (recommended; see {doc}`configuration-reference`),
the Wishart update runs every `update.every_epochs` epochs in all phases, and every
`display_precond_every` epochs the implied standard deviations are printed:

```text
[spider][INFO][PRIORS] Hierarchical event prior updated: std≈[0.31,0.29,0.52,0.08] corr_zt≈-0.12 [largest of 7 clusters, n_events=4210]
```

- `std≈[σx,σy,σz,σt]` (km, km, km, s) is the learned prior width for the reported component;
  `corr_zt` is the depth/origin-time correlation. With several components the **largest** one is
  reported (the mean over components would be dominated by tiny clusters).
- On real catalogs the learned widths settle at a few hundred metres. If they keep shrinking towards
  zero, the update started too early (`update.start_epoch`) or a small component is estimating its
  own precision from too few events (`update.min_cluster_events`).
- `Hierarchical event prior update failed; leaving P0 unchanged for this epoch.` is a warning that
  the Gibbs step hit a numerical problem; occasional occurrences are harmless, persistent ones mean
  the precision has become ill-conditioned.

When `spider sample` starts from a bundle written with the hierarchical prior, it prints
`Adopted hierarchical event-prior precision from bundle: std≈[...] (largest component)`. If instead
you see `Bundle precision matrix shape ... does not match the rebuilt clusters ...; starting from the
base event prior until the first Wishart update`, the connected components differ from the MAP run
(different filters/config); regenerate the bundle with `spider locate-map`.

## Centroid (gauge) prior metrics (`priors` group)

Emitted every epoch when `model.priors.centroid.enabled` is true and the `priors` W&B group is on.
The centroid prior pins each connected component's mean shift with prior width `std / sqrt(n_k)`
(the log-density is weighted by the component's event count `n_k`), so these report the component
means and how far they sit from that prior:

- `centroid/n_components` — number of connected components in the sampled event set.
- `centroid/main_dx`, `centroid/main_dy`, `centroid/main_dz` (km), `centroid/main_dt` (s) — mean
  shift of the largest component (the catalog's gauge).
- `centroid/max_dr` — largest 3-D norm of any component's mean shift (km); a satellite component
  drifting as a rigid body shows up here first.
- `centroid/max_abs_dt` — largest absolute component mean origin-time shift (s).
- `centroid/rms_dr_weighted` — event-weighted rms of the component mean shifts (km).
- `centroid/z_max` — largest z-score of a component mean against its prior width; values well
  above 3 mean the data are fighting the gauge prior for that component.
- `centroid/frac_components_z_gt3` — fraction of components beyond 3 sigma in any coordinate.
- `centroid/neg_logp` — the centroid prior's total negative log-density (unnormalized, before the
  `1/N` objective scaling).

Every `display_precond_every` epochs the same quantities are printed as

```text
[spider][INFO][PRIORS] Component centroids: K=7 | main (n=4210) shift [+0.0012,-0.0008,+0.0031] km +0.0004 s | max |shift| 0.0410 km | worst z=2.3 (comp n=12), 0/7 components beyond 3 sigma
```

Reading it: with `params.std = 0.01` km and `n = 4210`, the main component's prior width is
`0.01/sqrt(4210) ≈ 0.15 m`, so a shift of a few millimetres is expected. A small component
(`n = 12`) has prior width `2.9 m`; if its `z` grows past 3 and `max |shift|` climbs epoch after
epoch, the events in it have too little data to hold their frame (see the post-Phase-1 event filter
in {doc}`troubleshooting`). The centroid prior has no global form and no `scope` key — a
`priors.centroid.scope` entry is a validation error.

## W&B group gating

Configured under `observability.diagnostics.wandb.{enabled, groups}`. Group names consumed by the
current runtime:

| Group | Metrics |
| --- | --- |
| `core` | all optimization-health metrics above; it is also the outer gate for Phases 2–4 (with `core` off, no per-epoch W&B record is written in those phases, whatever the other groups say). Phase 1 ignores the group switches. |
| `noise` | `noise/*` |
| `sampler` | `learning_rate`, `noise_enabled`, `grad_noise_to_langevin_*`, `t_eff_var_over_target*` |
| `ess_online` | `ess_online/*` (also requires `ess_online.enabled`) |
| `priors` | `centroid/*` |

Names commonly present in older configs (`precond`, `fixed_eval`, `resid_rms`, `latent_field`,
`corr_error`) are accepted but have no gating call site today. `shared_event_re/*` metrics are not
group-gated; they are controlled by `profile_shared_event_re`.

Defaults: if `observability.diagnostics.wandb` is omitted, diagnostics are enabled and the group set
is `{"all"}`.

## Sampler diagnostics (`sampler` group, Phases 2–4)

Emitted when the active sampler backend exposes the corresponding statistics:

- `grad_noise_to_langevin_dr` — median over the `core` parameter group of the ratio *(update
  variance caused by minibatch gradient noise) / (injected Langevin noise variance)* for the
  spatial coordinates. Both pSGLD and SGHMC provide it.
- `grad_noise_to_langevin_dt` — the same ratio for the origin-time coordinate.
- `t_eff_var_over_target`, `t_eff_var_over_target_gm` — variance-based effective temperature over
  the target temperature (median and geometric mean). **SGHMC only**: the pSGLD backend has no
  momentum, returns `nan`, and the keys are therefore omitted.

Interpretation: the `grad_noise_to_langevin_*` ratios should be well below 1 — the minibatch
gradient noise then perturbs the chain much less than the deliberately injected noise, which is the
regime in which SGLD-type samplers approximate the posterior. Values near or above 1 mean the
minibatch is too small or the learning rate too high for the current temperature. For SGHMC,
`t_eff_var_over_target` near 1 (roughly 0.8–1.2) indicates the friction/noise balance is right.
Both are heuristics; see {doc}`sampler-health`.

## Shared-event whitening metrics (`shared_event_re/*`)

Emitted per epoch only when `profile_shared_event_re: true` **and**
`model.likelihoods.sample.shared_event_re.enabled: true`:

Workload and fallback (the ones to watch):

- `shared_event_re/groups_mean`, `groups_pcg_mean`, `groups_fallback_diag_mean` — mean number of
  station-phase groups per batch, and how many of them were solved by PCG versus fell back to the
  diagonal likelihood. Healthy: `groups_fallback_diag_mean` ≈ 0.
- `shared_event_re/max_rows_max`, `max_nodes_max` — largest group seen (rows = dtimes, nodes =
  events). Compare with `limits.max_rows` / `limits.max_nodes`.
- `shared_event_re/station_phase_groups`, `station_phase_quad` — group count and the (negative)
  rank-one correction that the optional `station_phase_term` adds to the shared-event quadratic
  (see {doc}`probability-model`). When the term is enabled each batch runs a second PCG solve, whose
  time is included in `whitening_solve_ms_sum`.

Timing (ms):

- `shared_event_re/time_ms_mean`, `time_ms_sum`, `time_batches`
- `shared_event_re/whitening_solve_ms_sum`, `whitening_cache_build_ms_sum`
- `shared_event_re/whitening_cache_hits`, `whitening_cache_misses`
- `shared_event_re/whitening_pcg_bucket_assign_ms_sum`, `whitening_pcg_bucket_merge_ms_sum`,
  `whitening_pcg_pack_ms_sum`, `whitening_pcg_kernel_ms_sum`, `whitening_pcg_matvec_ms_sum`,
  `whitening_pcg_vecops_ms_sum`, `whitening_pcg_unpack_ms_sum`, `whitening_pcg_leftover_ms_sum`

Console lines (`[LIKELIHOOD]` section), independent of W&B:

- once per run: `shared_event_re stats groups=G pcg=P fallback=F max_rows=R max_nodes=N tau_s=[...]`
  and `shared_event_re whitening groups=... chol=... fallback=... max_rows=... max_nodes=...`;
- every `shared_event_re.logging.stats_log_every_epochs` epochs (default `0` = off):
  `shared_event_re stats epoch=E grouping=station_phase groups=G pcg=P fallback=F rows_cap=a nodes_cap=b tau0=c max_rows=... max_nodes=... max_rows_all=... max_nodes_all=...`
  (`*_all` are the largest groups in the whole dataset, i.e. what the caps must exceed);
- once per epoch while any group falls back:
  `shared_event_re fallback reasons rows_cap=a nodes_cap=b tau_zero=c`;
- `shared_event_re auto-tune: max_nodes_per_group -> N` when `autotune.raise_nodes_cap` (default
  `true`) lifts the node cap automatically, up to `autotune.nodes_cap_max` (default 65536);
- `shared_event_re warning: all groups fell back to diagonal; increase model.likelihoods.sample.shared_event_re.limits.max_nodes or limits.max_rows`:
  the correlated likelihood is effectively off for this batch.

See {doc}`pcg-whitening-convergence` for the tuning playbook.

## Online ESS (`ess_online` group)

Configured under `observability.diagnostics.ess_online`:

| Key | Meaning |
| --- | --- |
| `enabled` | master switch (all other keys are forced to `0` when `false`) |
| `every_n_samples` | recompute every N stored posterior samples |
| `n_events` | size of the random event subset that is tracked (chosen once, with `seed`) |
| `window` | number of most recent samples used (at least 8) |
| `max_lag` | maximum autocorrelation lag (`0` → `min(1000, window/4)`) |
| `dims` | which ΔX columns to use, subset of `[0, 1, 2, 3]` = X, Y, Z, T (default `[0, 1, 2]`) |

The computation runs only while posterior samples are being collected (Phase 4; Phases 2–3 store
no samples) and only if the `ess_online` W&B group is on. Metrics:

- `ess_online/ess_median`, `ess_p10`, `ess_p90` — per-event ESS across the tracked events and
  dims, over the `window` most recent samples.
- `ess_online/tau_median`, `tau_p10`, `tau_p90` — integrated autocorrelation time (samples per
  independent draw); `ess ≈ window / tau`.
- `ess_online/ess_median_dim{d}`, `ess_p10_dim{d}`, `ess_p90_dim{d}`, `tau_median_dim{d}`,
  `tau_p10_dim{d}`, `tau_p90_dim{d}` — the same per dimension `d` in `dims`.
- `ess_online/n_samples`, `n_events`, `n_dims`, `max_lag` — bookkeeping.
- `ess_online/elapsed_s`, `delta_s`, `ess_per_s_median`, `ess_per_s_delta_median`,
  `samples_per_s` — wall-clock throughput; `ess_per_s_median` is the number to maximize when
  trading learning rate against thinning.

Interpretation: `tau_median` of a few tens is typical for well-constrained events; an `ess_p10`
that stays an order of magnitude below `ess_median` points to a subpopulation of poorly mixing
events (usually the weakly constrained ones; see the preconditioner-floor limitation in
{doc}`sampler-health`). Because only `n_events` random events are tracked, use the offline
`compute_ess_summary` on the finished HDF5 for the full picture.

## `analyze-resid` sub-diagnostics

`spider analyze-resid <params> --device <id>` rebuilds the state from the Phase-2 bundle and runs
the diagnostics enabled under `observability.diagnostics`. It prints
`analyze-resid: enabled diagnostics: ...` or, with nothing enabled,
`analyze-resid: no sub-diagnostic is enabled, nothing will be computed. Enable one or more of
observability.diagnostics.shared_event_re_tau.enabled, ...spatial.enabled, ...pathcorr.enabled,
...event_pair_station_corr.enabled.` and exits after the bundle summary. Turn on at least one of:

- `shared_event_re_tau: {enabled: true, ...}` — method-of-moments estimates of the shared-event
  random-effect scales `tau_s` (P and S, in seconds and ms) from the residuals, printed as
  `tau_event: P=... S=...` (plus per-cluster and log-det variants). Optional keys: `residuals_at`
  (`map` | `initial` | `truth`), `n_rows` (200000), `seed`, `batch_size` (50000),
  `sigma_quantile` (0.2), `sigma_source` (`quantile`), `min_events_per_cluster`,
  `min_rows_per_group` (500), `method` (`moments`), `hierarchical` (defaults to the
  likelihood's setting). Use the result to set `shared_event_re.model.tau_s`.
- `event_pair_station_corr: {enabled: true, ...}` — correlation of residuals for the same event
  pair between pairs of stations, written as `<plot_dir>/<outfile_prefix>_<phase>.png`. Optional
  keys: `n_rows` (300000), `batch_size`, `seed`, `top_k_stations` (30), `min_common_pairs` (50),
  `standardize_by_sigma` (true), `residuals_at`, `residual_variant` (`base` | `corr_error`),
  `outfile_prefix`.
- `spatial: {enabled: true, ...}` and `pathcorr: {enabled: true, ...}` — the end-of-Phase-1
  residual diagnostics (Moran's I of station residuals; path-correlation of residuals), re-run on
  the bundle's MAP solution.
- `truth_catalog: {path, time_source, time_ref, require_all}` — enables `residuals_at: "truth"`
  in the blocks above (synthetic tests).

`--plot-dir` sets where PNGs go (default: the bundle's directory). `--use-latest-checkpoint`
evaluates the residuals at the most recent sampling checkpoint instead of the MAP.
The `--plot-variograms` / `--no-plot-variograms` flags are accepted for compatibility but no
variogram output is currently produced by this command. See
{doc}`postprocessing-and-analysis-tools` for the offline analysis API.

## Practical logging profile

For production sampling with useful observability and moderate overhead:

- keep `core`, `sampler`, `noise` and `priors` enabled;
- enable `ess_online` only when actively tuning chains (it stacks `window × n_events` samples on
  the CPU every `every_n_samples` samples);
- keep `profile_shared_event_re: false` unless investigating whitening behavior; use
  `shared_event_re.logging.stats_log_every_epochs` for a cheap console summary instead;
- leave `display_precond_every` at 50 (or higher for long runs);
- always watch `skipped_batches` and the `[WARN][RUN] ... skipped` lines.
