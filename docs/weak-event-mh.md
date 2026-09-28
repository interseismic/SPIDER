# Weak-event independence-Metropolis resampling

`inference.sampler.weak_event_mh` adds a periodic Metropolis move for weakly-observed events:
propose a fresh position from the hierarchical event prior, accept/reject against the event's
own differential times (plus the per-component centroid-prior change). An accepted move is an
(approximately) independent draw from the event's marginal — so prior-dominated events
contribute properly smeared sample clouds (important when samples feed KDE/density imaging;
an under-mixed weak event otherwise adds a spuriously tight blob).

## Relation to the block Gauss-Newton preconditioner

This move patches the symptom for near-isolated events only (its acceptance is roughly the
posterior-to-prior volume ratio, so it rejects events that are even moderately constrained).
`inference.sampler.preconditioning.type: "block_gn"` (see {doc}`sampler-options`) removes the
cause for all events by giving each one a step scaled to its own conditional posterior; with
`block_gn` this move is normally unnecessary. It remains available and exact under any
preconditioner.

## Configuration

```json
"inference": { "sampler": { "weak_event_mh": {
    "enabled": true,
    "every_epochs": 10,
    "start_epoch": 0,
    "max_dtimes": 100
} } }
```

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | master switch; sweeps run in sampling (phase 4) only |
| `every_epochs` | 10 | sweep cadence |
| `start_epoch` | 0 | first eligible epoch |
| `max_dtimes` | 100 | events with at most this many dtime rows are eligible |

## Mechanics

- Proposal: `dX_i' ~ N(0, P_k^{-1})` from the current hierarchical per-cluster precision, so
  proposals track the learned prior. Prior and proposal cancel in the MH ratio, leaving the
  data-likelihood ratio over the event's rows (independent-Gaussian `phase_unc` term) times
  the exact n_k-weighted centroid-prior change.
- Weak events sharing a row are never tested in the same pass (greedy graph coloring on the
  shares-a-row adjacency); each color needs two batched residual evaluations over its rows.
- Cost per sweep ~ two forward passes over the weak rows only (a small fraction of the data).
- The per-sweep log line reports acceptance and the median accepted jump.

## Interpretation and limits (read this)

Acceptance tells you which events are *individually* prior-dominated. In practice many
"weak" events (tens of rows) are tightly constrained **relative to their partners** even
though their absolute marginals are wide — the width comes from slow *group* modes (a weakly
anchored neighborhood wandering together), not from a flat single-event likelihood. For such
events a single-event teleport is data-forbidden and will be rejected; only genuinely
isolated few-row events accept at high rates. Consequences:

- Expect low aggregate acceptance when `max_dtimes` is generous; the accepted moves
  concentrate in the lowest-degree events (which are exactly the ones that need it most).
- Set `max_dtimes` to the population you actually mean (10-20 targets near-isolated events).
- Wide marginals produced by correlated weak *groups* need group-level moves or the LRD
  preconditioner / longer chains; this move does not (and cannot) accelerate them.
