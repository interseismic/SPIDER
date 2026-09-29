# Weak-event Metropolis resampling

`inference.sampler.weak_event_mh` adds a periodic Metropolis move for weakly-observed events. For
every event with at most `max_dtimes` rows it fits the event's conditional posterior (given the
current positions of its partners) with a few Gauss-Newton iterations over the event's own rows,
proposes from a Student-t centred on that fit, and accepts with the exact independence-Metropolis
ratio. An accepted move is a near-independent draw from the event's conditional, so few-row events
mix in a handful of sweeps instead of the slow random walk they perform under the RMSprop metric
(on Kumamoto v1 their integrated autocorrelation time was 600-2500 SGLD steps against 30-80 for
well-observed events).

## Configuration

```json
"inference": { "sampler": { "weak_event_mh": {
    "enabled": true,
    "every_epochs": 1,
    "start_epoch": 0,
    "max_dtimes": 50,
    "proposal": "conditional",
    "gn_iters": 4,
    "t_dof": 5,
    "scale_inflation": 1.2
} } }
```

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | master switch; sweeps run in sampling (Phase 4) only |
| `every_epochs` | 1 | sweep cadence (a sweep costs a few residual/Jacobian passes over the weak rows only) |
| `start_epoch` | 0 | first eligible Phase-4 epoch |
| `max_dtimes` | 50 | events with at most this many rows are resampled |
| `proposal` | `conditional` | `conditional`: Student-t around the Gauss-Newton conditional fit; `prior`: the event prior N(0, P_k^-1) (previous behaviour, accepts only near-isolated events) |
| `gn_iters` | 4 | Gauss-Newton iterations of the conditional fit, from a fixed start (the shifts at the first sweep) |
| `t_dof` | 5 | Student-t degrees of freedom of the proposal (`null` = Gaussian); heavier tails keep the independence sampler robust where the Laplace fit is narrower than the target |
| `scale_inflation` | 1.2 | multiplies the proposal scale (square root of the covariance) |

## Mechanics

- Weak events sharing a row are never tested in the same pass (greedy graph colouring on the
  shares-a-row adjacency); within a colour all fits, proposals and ratios are batched.
- Conditional fit: minimise `0.5 sum_r w_r r_r(x)^2 + 0.5 x^T P_k x` over the event's rows with
  the partners fixed; `P_k` is the current (hierarchical) event-prior precision. The proposal is
  `t_dof(m, scale_inflation^2 (F + P_k)^-1)` with `F = sum_r w_r J_r J_r^T` at the mode. Because the
  fit starts from a fixed point and uses only the partners' current state, the proposal does not
  depend on the event's own current value and the move is an exact independence-Metropolis step.
- Target: the data term over the event's rows (independent Gaussian), the event prior, and the
  n_k-weighted per-component centroid prior when enabled.
- Data scales: `phase_unc`; with the shared-event random-effect likelihood the epoch runner passes
  `sqrt(sigma^2 + 2 tau^2)`, the batch-marginal variance of a weak event's row (neither its own nor
  its partner's effect is identifiable from about one row per batch), which is what the SGLD
  gradient effectively applies to these rows.
- Console (every `display_precond_every` epochs):
  `[SAMPLER] Weak-event MH sweep (conditional): accepted 5100/6522 (median jump 412 m) [1-10 dtimes: 1500/1600 | ...]`.

## Interpretation and limits

- Interaction with `inference.runtime.gauge_projection`: the move targets each event's own conditional and
  ignores the gauge; the post-step re-centering redistributes the (centimetre) mean shift of an accepted move
  over the component. This is consistent because the samplers project the preconditioned update, not the raw
  gradient: an earlier version subtracted the cluster-mean gradient before preconditioning, which made every
  event carry the same gradient at the fixed point and drove prior-held events kilometres deep (on Kumamoto v1
  three quarters of the 1-2-dtime events sat on the +10 km depth clamp, and with that version the Metropolis
  move only pulled them back part of the way each sweep).

- Acceptance of 40-80 % is typical for the conditional proposal; a bucket with low acceptance means
  the Gauss-Newton fit does not describe that population (multimodal conditionals, events pinned
  at the `max_abs_dX` box): inspect those events' residuals.
- The move resamples one event at a time given its partners. Events that are only weakly linked to
  the rest as a *group* (a few events connected mostly to each other) still move as a group through
  the SGLD steps; the sweep does not accelerate that collective motion.
- Per-event ESS should be checked by degree (`spider.analysis.diagnose_sticky_events`).
