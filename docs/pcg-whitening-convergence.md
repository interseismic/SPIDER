# PCG Whitening: Convergence and Fallback Control

This page focuses on one objective: ensure shared-event whitening solves on the PCG path with stable convergence and minimal diagonal fallback.

## Preconditions

- `model.likelihoods.sample.type = "correlated_gaussian"`
- `model.likelihoods.sample.shared_event_re.enabled = true`
- `model.likelihoods.sample.shared_event_re.model.group_by = "station_phase"`
- `model.likelihoods.sample.shared_event_re.solver.kind = "pcg"`

## Critical knobs

## Correlation model

- `shared_event_re.model.tau_s`: `[P, S]` RE scales.
  - Must be strictly positive to avoid `tau_zero` fallback.
  - If too small, model behaves near diagonal and fallback risk increases.

## Group caps (most important for fallback)

- `shared_event_re.limits.max_nodes`
- `shared_event_re.limits.max_rows`

If groups exceed either cap, those groups can fall back to diagonal.

## PCG convergence controls

- `shared_event_re.solver.max_iters`
- `shared_event_re.solver.min_iters`
- `shared_event_re.solver.tol`
- `shared_event_re.numerics.jitter0`
- `shared_event_re.numerics.jitter_max`

Use these to stabilize or tighten solves after cap issues are resolved.

## Safety behavior

- `shared_event_re.fallback.to_diag`
- `shared_event_re.fallback.abort_on_pcg_fallback`

Recommended workflow:

- tuning phase: `to_diag=true`, `abort_on_pcg_fallback=false`
- strict validation phase: `abort_on_pcg_fallback=true` to force immediate failure and inspect cause

## Convergence-first baseline config

```json
"shared_event_re": {
  "enabled": true,
  "model": {
    "group_by": "station_phase",
    "tau_s": [0.03, 0.04],
    "cluster": { "mode": "none", "k": 1 }
  },
  "limits": {
    "max_nodes": 50000,
    "max_rows": 2000000
  },
  "fallback": {
    "to_diag": true,
    "abort_on_pcg_fallback": false
  },
  "numerics": {
    "jitter0": 1e-8,
    "jitter_max": 1e-3
  },
  "solver": {
    "kind": "pcg",
    "max_iters": 80,
    "min_iters": 2,
    "tol": 1e-3,
    "batched": true
  }
}
```

## What to monitor

Use periodic whitening stats logs (`shared_event_re.logging.stats_log_every_epochs > 0`) and check:

- `shared_event_re/groups_pcg_mean`
- `shared_event_re/groups_fallback_diag_mean`
- `shared_event_re/max_rows_max`
- `shared_event_re/max_nodes_max`
- Console fallback reason counters: `rows_cap`, `nodes_cap`, `tau_zero`

Healthy target:

- `groups_pcg_mean > 0`
- `groups_fallback_diag_mean` close to zero
- fallback reasons remain zero in steady state

## Fallback elimination playbook

If fallback is nonzero:

1. Check fallback reason summary first.
2. If `rows_cap > 0`, raise `limits.max_rows`.
3. If `nodes_cap > 0`, raise `limits.max_nodes`.
4. If `tau_zero > 0`, fix `tau_s` values and config plumbing.
5. If fallback persists after caps/tau are fixed:
   - raise `solver.max_iters`,
   - increase `jitter0` gradually,
   - tighten or relax `tol` depending on instability vs. stagnation.

If all groups still fall back:

- temporarily set `abort_on_pcg_fallback=true`,
- rerun and inspect the first failing context,
- verify group sizes are within caps and `tau_s` is valid.

## Notes on optional solver knobs

`node_bin_edges`, warm starts, cache sizing, sparse-bin merge, and precompute controls are operational knobs. They can improve robustness in difficult workloads, but fallback elimination should first be solved with:

- valid `tau_s`,
- sufficient `max_nodes`/`max_rows`,
- stable `max_iters`/`tol`/`jitter0`.
