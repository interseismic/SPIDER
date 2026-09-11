# Troubleshooting

## `loss=nan` or "ALL batches skipped -> optimization is frozen"

```text
[spider][WARN][RUN] epoch 7: skipped 64/64 batches (non-finite loss=0, non-finite grad=64); parameters unchanged for those batches. ALL batches skipped -> optimization is frozen. ...
```

Every minibatch was discarded by the finiteness guards, so parameters never updated and the epoch
loss is reported as `nan` (older versions misreported it as `0.0`). Watch the `skipped_batches`
metric. The most common cause, as the message says, is an event that has converged onto a
receiver: at zero horizontal offset the gradient of the travel-time model's distance term is
undefined. Current EikoNet builds include a 1 m distance floor inside the `sqrt()` that keeps the
gradient finite; if you see this with an older model, check `<catalog_outfile>_MAP.csv` for events
located at or within a few metres of a station. If only a few batches are skipped, the run is
still progressing but the affected observations are being ignored.

## Events pinned at the safety clamp

If `dr_max` in the Phase-1 log equals the norm of `inference.safety.max_abs_dX` epoch after epoch,
some events are being held at the clamp rather than at their data-preferred positions. Widen the
clamp (e.g. `[10, 10, 10, 3]`); it is a safety net, not a prior. Likewise, if the median shifts
(`dx/dy/dz/dT`) exceed `model.priors.event.params.std`, the event prior is fighting the data for
most of the catalog — loosen it and rely on the centroid prior to fix the gauge.

## Second MAP pass never runs

`inference.sampler.phase1_two_pass` is not supported under `torchrun`/DDP (the post-MAP filters
run on rank 0 only) and is skipped with a `[WARN][RUN] phase1_two_pass is not supported under
torchrun/DDP` message. Run `locate-map` single-process if you need it.

## Phase-2 bundle contains unfiltered dtimes

The post-Phase-1 filter block is best-effort: any failure is reported as
`[WARN][FILTER] Post-Phase1 filters failed: ...` and Phase 1 still finalizes, so the bundle can
carry the unfiltered dataset. Grep the Phase-1 log for `[WARN][FILTER]` and for the expected
`Residual pre-filter: dropping N / M dtimes` and `Applied linearization_error ... kept N/M`
lines.

## `Phase-2 bundle not found`

`sample`, `sample-multi` and `analyze-resid` read `<checkpoint_dir>/phase2_bundle.pth`. Run
`spider locate-map <params> --device <id>` first, or pass `--bundle PATH`.

## `locate-map can only resume/run Phase 1`

The latest checkpoint in `io.checkpoint_dir` is from Phases 2–4. Either delete the checkpoints or
set `inference.runtime.reset_batch_numbers: true` to start fresh.

## Config validation errors

Most failures come from legacy key locations: top-level `wandb`/`phases`/`priors` (now
`observability.wandb`, `inference.sampler.epochs_per_phase`/`lr`, `model.priors`), scalar `lr`,
`inference.sampler.dt_lr_mult` (now `reparameterization.dt_scale`), flat `shared_event_re` keys
(now `model`/`limits`/`fallback`/`numerics` sub-blocks), or `priors.noise`/`learn_noise_scale`
(removed; noise scales are fixed). Run `spider validate-config <params>` and follow the path in
the error. Note that `model.filters`, `model.likelihoods`, `model.eikonet` and
`observability.diagnostics` are not key-checked, so typos there are silently ignored.

## Shared-event PCG fallback warning

If logs show frequent fallback or "all groups fell back to diagonal":

- inspect the fallback reason counters (`rows_cap`, `nodes_cap`, `tau_zero`),
- raise `model.likelihoods.sample.shared_event_re.limits.max_rows` and/or `limits.max_nodes`
  (defaults 200000 / 512 are far below typical needs),
- verify `shared_event_re.model.tau_s` is positive,
- set `shared_event_re.fallback.abort_on_pcg_fallback: true` while debugging.

See {doc}`pcg-whitening-convergence` for the full checklist.

## Sampler instability at higher LR

If trajectories explode when freezing the preconditioner:

- ensure the adaptation phase is long enough before the freeze,
- raise `inference.sampler.eps`,
- use gradient clipping (`inference.sampler.grad_clip_norm`; when unset Phase 1 clips at 100 and
  Phases 2–4 do not clip),
- lower the learning rate and retune gradually.

## `torch.compile` warning on Python 3.13+

`inference.runtime.torch.compile_eikonet` defaults to `false`, so this only affects runs that
opted in — remove the key or set it to `false`. Related pass-through keys under
`inference.runtime.torch`: `allow_tf32`, `matmul_precision`, `compile_mode`, `compile_backend`,
`compile_dynamic`, `compile_fullgraph`.
