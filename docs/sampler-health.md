# Sampler Health Checks

This page summarizes practical checks for pSGLD/SGHMC behavior in Phase 4.

## Core diagnostics

- `t_eff_var_over_target` (temperature calibration proxy)
- `grad_noise_to_langevin_dr` / `grad_noise_to_langevin_dt` (gradient-noise vs injected-noise balance for spatial vs `dt`)
- per-parameter/block ESS and split-Rhat (multi-chain)

## Suggested targets

- `t_eff_var_over_target`: around `0.8-1.2`
- `grad_noise_to_langevin_dr`: around `0.1-1.0` (order of magnitude)
- `grad_noise_to_langevin_dt`: around `0.1-1.0` (order of magnitude)

These are heuristics, not strict guarantees.

## If exploration looks too cold

- Reduce SGHMC friction (`sghmc_alpha`) if overly damped.
- Verify `inference.batching.standard.shuffle=true`.
- Revisit preconditioner freeze timing (`freeze_preconditioner_sampling`).
- Adjust `eps` and learning rate conservatively.

## Robust validation protocol

Single-chain ESS is insufficient for full-posterior confidence. For production inference:

1. Run multiple chains from overdispersed starts.
2. Check split-Rhat and rank plots per key parameter block.
3. Compare posterior summaries across chains.
4. Use posterior predictive checks when possible.
