# Diagnostics Catalog

This page maps diagnostics settings to runtime behavior and key metrics.

## Configuration location

Primary controls:

- `observability.wandb.*`
- `observability.diagnostics.*`

Important: W&B metric group switches are only effective when W&B runtime is enabled. Keys under
`observability.diagnostics` are not schema-validated, so a typo is silently ignored. Configuring
diagnostics under `inference.diagnostics` is a hard validation error — use
`observability.diagnostics`.

## Core diagnostics switches

- `observability.diagnostics.pair_count_stats_enable`
  - Enables per-event pair/station count stats at state build time.
- `observability.diagnostics.display_precond_every`
  - Cadence (in epochs) for the hierarchical event-prior summary line
    ("Hierarchical event prior updated: std≈..."). Falls back to `hierarchical_log_every_epochs`
    (10) when unset.
- `observability.diagnostics.profile_shared_event_re`
  - Enables shared-event timing/workload profiling. This switch also gates the `shared_event_re/*`
    W&B metrics below; with it off (the recommended default) those metrics are not emitted.
- `observability.diagnostics.truth_catalog` (`path`, `time_source`, `time_ref`, `require_all`)
  - Truth catalog used by `spider analyze-resid` for truth-referenced residuals. `truth_locations`
    is accepted as a legacy alias.
- `observability.diagnostics.event_pair_station_corr`, `observability.diagnostics.shared_event_re_tau`
  - `analyze-resid` sub-diagnostics, each with an `enabled` flag.
- `observability.diagnostics.eikonet_v1d`
  - Optional 1-D speed estimate diagnostic.

Removed (parsed but hard-disabled at runtime): `resid_distribution`, `resid_scalar_metrics`,
`shared_event_legcorr2d`. No longer used (accepted, no effect): `sgld_log_gnoise`,
`sgld_log_temperature` — the gradient-noise and effective-temperature metrics are emitted whenever
the active sampler exposes them and the `sampler` W&B group is on.

## Optimization-health metrics (always on)

- `loss` — edge-count-weighted mean loss for the epoch; `nan` when no batch contributed a step.
- `skipped_batches` — number of minibatches dropped by the finiteness guards (non-finite loss or
  gradients) in that epoch. Any non-zero value also produces a `[WARN][RUN]` line; if it equals
  the batch count the optimizer is frozen for that epoch (commonly a NaN gradient from an event
  sitting exactly on a receiver). Track this alongside `loss` — it is the fastest way to catch a
  run that looks "converged" but is not updating.
- Phase-1 shift statistics: `dx_med_abs`, `dy_med_abs`, `dz_med_abs`, `dt_med_abs`, `dr_90`,
  `dr_max` (compare `dr_max` with the `inference.safety.max_abs_dX` clamp).

## W&B group gating

Configured under:

- `observability.diagnostics.wandb.enabled`
- `observability.diagnostics.wandb.groups`

Common group names seen in configs:

- `core`
- `noise`
- `sampler`
- `ess_online`
- `precond`
- `fixed_eval`
- `resid_rms`
- `priors`
- `latent_field`
- `corr_error`

Defaults: if `observability.diagnostics.wandb` is omitted, diagnostics are enabled and the group
set is `{"all"}` (every group passes). `all` or `*` in the group list enables everything.

Group names consumed by the current runtime: `core`, `noise`, `sampler`, `ess_online`. Other names
commonly present in configs (`precond`, `fixed_eval`, `resid_rms`, `priors`, `latent_field`,
`corr_error`) are accepted but currently have no gating call site.

## Sampler diagnostics metrics

When the optimizer exposes diagnostics methods, SPIDER logs:

- `grad_noise_to_langevin_dr` (median over the `core` parameter group; formerly named `hypocenter`)
- `grad_noise_to_langevin_dt`
- `t_eff_var_over_target`
- `t_eff_var_over_target_gm`

Interpretation quick guide:

- `grad_noise_to_langevin_dr` and `grad_noise_to_langevin_dt` compare minibatch gradient-noise variance to injected Langevin-noise variance for spatial and `dt` directions, respectively.
- `t_eff_var_over_target` near 1 suggests temperature calibration is closer to target.

See also {doc}`sampler-health`.

## Shared-event whitening diagnostics metrics

Emitted only when `profile_shared_event_re` is `true` and shared-event whitening is enabled:

- `shared_event_re/groups_pcg_mean`
- `shared_event_re/groups_fallback_diag_mean`
- `shared_event_re/max_rows_max`
- `shared_event_re/max_nodes_max`
- `shared_event_re/time_ms_sum`
- `shared_event_re/whitening_solve_ms_sum`
- `shared_event_re/whitening_cache_hits`
- `shared_event_re/whitening_cache_misses`
- `shared_event_re/whitening_pcg_pack_ms_sum`
- `shared_event_re/whitening_pcg_kernel_ms_sum`
- `shared_event_re/whitening_pcg_unpack_ms_sum`
- `shared_event_re/whitening_pcg_leftover_ms_sum`

For convergence/no-fallback triage, prioritize:

- `groups_pcg_mean`
- `groups_fallback_diag_mean`
- `max_rows_max`, `max_nodes_max`
- fallback reason counters in logs (`rows_cap`, `nodes_cap`, `tau_zero`)

See {doc}`pcg-whitening-convergence`.

## Online ESS diagnostics (`ess_online`)

Configured under `observability.diagnostics.ess_online`:

- `enabled`
- `every_n_samples`
- `n_events`
- `seed`
- `window`
- `max_lag`
- `dims`

Common derived metrics:

- `ess_online/ess_median`
- `ess_online/ess_per_s_median`, `ess_online/ess_per_s_delta_median`
- `ess_online/samples_per_s`, `ess_online/n_samples`
- `ess_online/elapsed_s`, `ess_online/delta_s`

All knobs are forced to `0` unless `enabled` is `true`; `dims` defaults to `[0, 1, 2]`.

## Practical logging profile

For production sampling with useful observability and moderate overhead:

- keep `core`, `sampler`, and `noise` enabled,
- enable `ess_online` only when actively tuning chains,
- keep `profile_shared_event_re=false` unless investigating whitening behavior,
- increase `display_precond_every` if logs are too verbose,
- always watch `skipped_batches` (see above).

For post-run residual diagnostics use `spider analyze-resid <params> --device <id>`; see
{doc}`postprocessing-and-analysis-tools`.
