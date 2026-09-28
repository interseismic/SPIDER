# Sampler Health Checks

This page summarizes practical checks for pSGLD/SGHMC behavior in Phases 2–4. Phase 2 is a
deterministic preconditioned drift (no noise), Phase 3 ramps the noise in (`ramp=` in the console
line), and Phase 4 is the sampling phase whose draws are written to `io.samples_outfile`. The same
metrics are logged in all three phases; in Phase 3 they are expected to be off-target while the
noise ramps. Metric definitions are in {doc}`diagnostics-catalog`.

## First check: is the sampler actually moving?

Before tuning anything, confirm no epoch reports skipped batches. A `skipped=k/M` field in the
epoch line (present only when k > 0) or the accompanying warning:

```text
[spider][INFO][RUN] phase4 | 42/60 | L=nan | ... | precond=rmsprop noise=on(scale=1,T=1) lr=0.01 | skipped=128/128 | sigma_p=0.0200 sigma_s=0.0300
[spider][WARN][RUN] epoch 42: skipped 128/128 batches (non-finite loss=0, non-finite grad=128); parameters unchanged for those batches. ALL batches skipped -> optimization is frozen. Typical cause: a NaN gradient from a source that has converged onto a receiver ...
```

`skipped_batches > 0` (metric) or a `loss` of `nan` means some or all minibatches were discarded
by the finiteness guards, so the chain is partly or entirely frozen — which looks identical to
"too cold" exploration but has a different cause (typically an event that has converged onto a
receiver, producing a NaN travel-time gradient). Fix that before touching `sghmc_alpha`, `eps` or
the learning rate; see {doc}`troubleshooting`.

Also read the sampler state at the end of the epoch line, e.g.
`precond=rmsprop noise=on(scale=1,T=1) lr=0.01`. `noise=off` in Phase 4 means no sampling is
happening (`inference.sampler.temperature` is 0, or an `overrides.core.temperature_mult` of 0);
`precond=none` means `inference.sampler.preconditioning.enabled` is false.

## Second check: is the prior what you think it is?

With the hierarchical event prior (`model.priors.event.hyper`, recommended) the sampling run must
start from the precision learned in Phase 1. Look for

```text
[spider][INFO][PRIORS] Adopted hierarchical event-prior precision from bundle: std≈[0.31,0.29,0.52,0.08] (largest component)
```

at the start of `spider sample`, and for the periodic
`Hierarchical event prior updated: std≈[...] corr_zt≈... [largest of K clusters, n_events=N]`
lines afterwards. If the adoption line is missing (or replaced by a `Bundle precision matrix
shape ... does not match` warning), Phases 2–4 run under the loose base `params.std` until the
first Wishart update and weakly constrained events wander; regenerate the bundle with
`spider locate-map`. If the reported widths keep shrinking epoch after epoch during Phase 4, the
prior is collapsing onto the current samples — raise `update.min_cluster_events` or `params.df`.

## Core diagnostics

- `grad_noise_to_langevin_dr` / `grad_noise_to_langevin_dt` — ratio of the update variance caused
  by minibatch gradient noise to the injected Langevin noise variance, for the spatial and origin-
  time coordinates (both backends).
- `t_eff_var_over_target` (and `_gm`) — effective temperature over target temperature; **SGHMC
  only** (pSGLD has no momentum and omits the key).
- Per-event ESS: offline via `compute_ess_summary` on the finished samples (see
  {doc}`postprocessing-and-analysis-tools`), or online via the `ess_online/*` metrics.
- `centroid/*` — per-component mean shifts and their z-scores against the gauge prior.

The first two groups are emitted to W&B only (group `sampler`, and only when W&B is enabled at
runtime); they do not appear on the console.

## Suggested targets

- `grad_noise_to_langevin_dr`, `grad_noise_to_langevin_dt`: well below 1 — roughly `0.1` or less.
  Around 1 the minibatch noise is as large as the deliberately injected noise and the chain is
  effectively running at an unknown, higher temperature; reduce the learning rate or increase
  the sampling minibatch size (`inference.batching.standard.sgld`).
- `t_eff_var_over_target` (SGHMC): around `0.8–1.2`.
- `centroid/z_max`: below 3 in steady state. Persistent values above 3 for a small component mean
  the gauge prior is holding that component against the data; check it in `_MAP.csv` and the
  post-Phase-1 event filter (see {doc}`troubleshooting`).
- Per-event ESS: no hard rule, but the *distribution* matters more than the median — see the next
  section.

These are heuristics, not strict guarantees.

## Known limitation: the RMSprop preconditioner floor

The pSGLD (and the SGHMC diagonal) preconditioner is

```text
v   <- beta * v + (1 - beta) * grad^2
G   =  1 / (eps + sqrt(v))
step = lr * G * grad + sqrt(2 * lr * T) * sqrt(G) * N(0, 1)
```

with `eps = inference.sampler.eps` and `beta = inference.sampler.beta`. The intent is that
`sqrt(v)` (the typical gradient magnitude of each coordinate) sets a per-event step size: tightly
constrained events with steep posteriors get small steps, weakly constrained events with flat
posteriors get large ones, and all events mix at a similar rate.

In practice the gradients are small in SPIDER's units: in a measured production run `sqrt(v)` was
about `2e-7` to `7e-7` for most events, while `eps` was `1e-6` (the example config uses `1e-5`,
the template `1e-8`). Whenever `eps` is larger than `sqrt(v)` the floor dominates, `G ≈ 1/eps` for
every event, and the sampler degenerates to a plain SGLD with one common step size. A random walk
with a fixed step needs a number of steps proportional to `(posterior width / step)^2` to
decorrelate, so a weakly constrained event whose posterior is `r` times wider than a well
constrained one mixes about `r^2` times slower. In the same run, events with fewer than 50
differential times reached an ESS of roughly 100 out of 4800 stored samples, versus roughly 1850
for well constrained events.

How to detect it:

1. Compute per-event ESS from the finished samples (`compute_ess_summary(...)['ess_per_event_xyzt_min']`).
2. Compute per-event data coverage — the number of differential times each event participates in
   (its degree in the event-pair graph; count rows of the filtered dtimes table where `evid1` or
   `evid2` is the event).
3. Plot ESS against degree, or use `python -m spider.analysis.diagnose_sticky_events --params
   <params> --out-csv ess_by_degree.csv`, which writes exactly this join (`evid`, `degree`,
   `std_xyz_km`, `ess_xyzt_min`, ...) and prints `corr(degree, ess_xyzt_min)`. A strongly positive
   correlation, with the low-degree tail an order of magnitude below the median, is the signature
   of the floor.
4. Cross-check against the sampler state: if `precond=rmsprop` is shown but the low-degree events
   still mix much more slowly, the preconditioner is not doing its job.

What you can do with the current code:

- Lower `inference.sampler.eps` towards the measured `sqrt(v)` scale (the run above would need
  `1e-7` or less) so the preconditioner becomes active. Do this gradually: once `G` is no longer
  floored, the effective step of the *best* constrained events grows by up to `eps / sqrt(v)`,
  and the run may need a lower `lr` or `inference.sampler.grad_clip_norm` to stay stable (see
  "Sampler instability" in {doc}`troubleshooting`). `freeze_preconditioner_sampling: true`
  freezes `v` for Phases 2–4 so the kernel stays time-homogeneous.
- Run longer, or run more chains (`spider sample-multi`) and merge them: `compute_ess_summary`
  sums ESS across chains (`chain_mode = "multi_chain_sum"`) when the merged file carries chain
  metadata.
- Report the uncertainties of low-ESS events with caution, or filter them by
  `ess_per_event_xyzt_min` when publishing a catalog.

The structural fix is `inference.sampler.preconditioning.type: "block_gn"`: a per-event 4x4
Gauss-Newton metric (data Fisher block plus prior precision, inverted) that gives every event
and coordinate the same dimensionless step, so mixing no longer depends on degree. See the
"Block Gauss-Newton preconditioner" section of {doc}`sampler-options`. It does not accelerate
collective modes of weakly anchored *groups*; for those see the next section and multi-chain
runs.

## Frame-mode drift (coherent slow trends)

If trace plots show smooth, slowly relaxing trends in one coordinate with a flat loss, fit a
per-event linear trend over the post-burn-in chain and map it (a *trend-field map*; this is a
manual recipe, not a SPIDER function): use the per-event slope normalized by the residual
(detrended) standard deviation, with a well-mixed coordinate as the null control. Spatially
coherent same-sign patches are cluster-frame or long-wavelength modes: the differential data are
nearly blind to them, so the sampler mixes them at the `(width / step)^2` pace discussed above.

Remedies: check that the centroid (gauge) prior is enabled and tight (`model.priors.centroid`,
e.g. `std = 0.01`), since it pins each connected component's mean shift; watch `centroid/max_dr`
and `centroid/z_max`; run more epochs or parallel chains for sub-cluster modes; and treat
absolute frame coordinates as prior-dominated in any report.

## If exploration looks too cold

- Reduce SGHMC friction (`inference.sampler.sghmc_alpha`, must be `> 0`) if overly damped.
- Verify `inference.batching.standard.shuffle = true`.
- Revisit preconditioner freeze timing (`inference.sampler.freeze_preconditioner_sampling`):
  freezing before the RMSprop statistics have converged locks in Phase-2 step sizes.
- Adjust `eps` and the learning rate conservatively (see the floor discussion above).

## Robust validation protocol

Single-chain ESS is insufficient for full-posterior confidence. SPIDER ships per-event ESS but
does **not** implement split-Rhat or rank plots. For production inference:

1. Run multiple chains from the same bundle with different seeds (`spider sample-multi --devices
   0,1,2,3`); chains are merged into `io.samples_outfile` unless `--no-merge` is given.
2. Compute split-Rhat and rank plots with an external tool (e.g. ArviZ) from the merged HDF5,
   using the `_chain_slices` / `_sample_chain_idx` metadata returned by `read_all_samples`.
3. Compare posterior summaries across chains.
4. Use posterior predictive checks when possible (`spider analyze-resid --use-latest-checkpoint`
   evaluates the residual diagnostics at the last sampling checkpoint).
