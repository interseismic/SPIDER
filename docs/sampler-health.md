# Sampler Health Checks

This page summarizes practical checks for pSGLD/SGHMC behavior in Phases 2–4 (the same metrics
are logged in Phase 3, where they are expected to be off-target while the noise ramps).

## First check: is the sampler actually moving?

Before tuning anything, confirm no epoch reports skipped batches:

```text
[spider][WARN][RUN] epoch 42: skipped 128/128 batches (non-finite loss=0, non-finite grad=128); parameters unchanged for those batches. ALL batches skipped -> optimization is frozen.
```

`skipped_batches > 0` (metric) or a `loss` of `nan` means some or all minibatches were discarded
by the finiteness guards, so the chain is partly or entirely frozen — which looks identical to
"too cold" exploration but has a different cause (typically an event that has converged onto a
receiver, producing a NaN travel-time gradient). Fix that before touching `sghmc_alpha`, `eps` or
the learning rate.

Also confirm the sampler state in the epoch line itself, e.g.
`precond=rmsprop noise=on(scale=1,T=1) lr=1e-07`. `noise=off` in Phase 4 means no sampling is
happening.

## Core diagnostics

- `t_eff_var_over_target` (temperature calibration proxy)
- `grad_noise_to_langevin_dr` / `grad_noise_to_langevin_dt` (gradient-noise vs injected-noise
  balance for spatial vs `dt`)
- per-event ESS (`compute_ess_summary`, or the online `ess_online/*` metrics)

The first two are emitted to W&B only (group `sampler`, and only when W&B is enabled at runtime);
they do not appear on the console.

## Suggested targets

- `t_eff_var_over_target`: around `0.8-1.2`
- `grad_noise_to_langevin_dr`: around `0.1-1.0` (order of magnitude)
- `grad_noise_to_langevin_dt`: around `0.1-1.0` (order of magnitude)

These are heuristics, not strict guarantees.

## If exploration looks too cold

- Reduce SGHMC friction (`sghmc_alpha`, valid range `(0, 1]`) if overly damped.
- Verify `inference.batching.standard.shuffle=true`.
- Revisit preconditioner freeze timing (`freeze_preconditioner_sampling`).
- Adjust `eps` and learning rate conservatively.

## Robust validation protocol

Single-chain ESS is insufficient for full-posterior confidence. SPIDER ships per-event ESS but
does **not** implement split-Rhat or rank plots. For production inference:

1. Run multiple chains from overdispersed starts (`spider sample-multi`).
2. Compute split-Rhat and rank plots with an external tool (e.g. ArviZ) from the merged HDF5,
   using the `_chain_slices` / `_sample_chain_idx` metadata returned by `read_all_samples`.
3. Compare posterior summaries across chains.
4. Use posterior predictive checks when possible.
