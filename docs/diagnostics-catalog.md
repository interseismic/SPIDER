# Diagnostics Catalog

This page maps diagnostics settings to runtime behavior and key metrics.

## Configuration location

Primary controls:

- `observability.wandb.*`
- `observability.diagnostics.*`

Important: W&B metric group switches are only effective when W&B runtime is enabled.

## Core diagnostics switches

- `observability.diagnostics.pair_count_stats_enable`
  - Enables pair-count stats diagnostics.
- `observability.diagnostics.sgld_log_gnoise`
  - Enables additional gradient-noise logging paths.
- `observability.diagnostics.sgld_log_temperature`
  - Enables additional temperature logging paths.
- `observability.diagnostics.display_precond_every`
  - Controls cadence for some preconditioner console diagnostics.
- `observability.diagnostics.profile_shared_event_re`
  - Enables shared-event profiling summaries in epoch logging.

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

At current runtime, explicit group-gating is most actively used for:

- `core`
- `noise`
- `sampler`
- `ess_online`

Other names may be consumed by logger-side code paths depending on run mode and config.

## Sampler diagnostics metrics

When the optimizer exposes diagnostics methods, SPIDER logs:

- `grad_noise_to_langevin_med`
- `grad_noise_to_langevin_gm`
- `grad_noise_to_langevin_med_dt`
- `grad_noise_to_langevin_gm_dt`
- per-group variants (for example `grad_noise_to_langevin_med_hypocenter`)
- `t_eff_var_over_target`
- `t_eff_var_over_target_gm`
- `grad_noise_var_med`
- `langevin_noise_var_med`

Interpretation quick guide:

- `grad_noise_to_langevin_*` compares minibatch gradient-noise variance to injected Langevin-noise variance.
- `t_eff_var_over_target` near 1 suggests temperature calibration is closer to target.

See also {doc}`sampler-health`.

## Shared-event whitening diagnostics metrics

Common metrics emitted during runs include:

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
- `ess_online/ess_per_s_median`
- `ess_online/samples_per_s`
- `ess_online/elapsed_s`

## Practical logging profile

For production sampling with useful observability and moderate overhead:

- keep `core`, `sampler`, and `noise` enabled,
- enable `ess_online` only when actively tuning chains,
- keep `profile_shared_event_re=false` unless investigating whitening behavior,
- increase `display_precond_every` if logs are too verbose.
