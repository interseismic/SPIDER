# PCG Whitening: Convergence and Fallback Control

This page has one objective: make the shared-event whitening solves run on the PCG path with
stable convergence and no diagonal fallback. The shared-event random effect is the correlated
part of the `correlated_gaussian` likelihood; every station-phase group whose solve falls back to
the diagonal loses that correlation, silently narrowing the posterior. Background on the model is
in {doc}`probability-model`; all keys live under `model.likelihoods.sample.shared_event_re` and are
listed in {doc}`configuration-reference`.

## Preconditions

- `model.likelihoods.sample.type = "correlated_gaussian"`
- `shared_event_re.enabled = true`
- `shared_event_re.model.group_by = "station_phase"` — the only legal value; `"phase"` raises
  `shared_event_re.grouping='phase' is no longer supported` at run time.
- `shared_event_re.solver.kind = "pcg"` (default; `pcg_sparse` is an accepted alias). Any other value
  is rejected by `validate-config`: only the PCG kinds enable the correlated likelihood.

Defaults when keys are omitted: `limits.max_nodes = 512`, `limits.max_rows = 200000`,
`solver.max_iters = 50`, `solver.min_iters = 2`, `solver.tol = 1e-3`, `solver.batched = true`,
`numerics.jitter0 = 1e-8`, `numerics.jitter_max = 1e-3`, `fallback.to_diag = true`,
`fallback.abort_on_pcg_fallback = false`. The stock limits are roughly 50–100x smaller than the
baseline recommended below — leaving them at default is the most common cause of diagonal
fallback.

The whole `shared_event_re` block is key-checked: legacy flat keys (`tau_s`, `max_nodes_per_group`,
`max_rows_per_group`, `fallback_to_diag`, `abort_on_pcg_fallback`, `jitter0`, `jitter_max`,
`cluster_mode`, ... directly under `shared_event_re`) are rejected with a message naming the
sub-block they moved to.

## Critical knobs

### Correlation model (`shared_event_re.model`)

- `tau_s`: `[P, S]` random-effect scales in seconds (a scalar is broadcast to both phases).
  - Must be strictly positive for both phases; a zero entry sends every group of that phase to the
    `tau_zero` fallback.
  - If too small, the model is effectively diagonal (no benefit); if too large, the whitened
    system becomes ill-conditioned and PCG needs more iterations. `spider analyze-resid` with
    `observability.diagnostics.shared_event_re_tau.enabled: true` estimates `tau_s` from the
    Phase-1 residuals (see {doc}`diagnostics-catalog`).
- `cluster.{mode, k}`: `mode` is `none` (default), `component` (one random effect per connected
  component) or `dd_khop` (k-hop neighbourhoods, `k`).

### Group caps (`shared_event_re.limits`, most important for fallback)

- `max_nodes` — maximum number of events in one station-phase group.
- `max_rows` — maximum number of differential times in one group.

Groups exceeding either cap fall back to the diagonal likelihood (`nodes_cap` / `rows_cap`
counters). The `max_nodes_all` / `max_rows_all` values in the periodic stats line are the actual
maxima in your dataset: the caps have to exceed them.

`autotune.raise_nodes_cap` (default `true`) lifts `max_nodes` automatically when `nodes_cap`
fallbacks are observed, up to `autotune.nodes_cap_max` (default 65536), and prints
`shared_event_re auto-tune: max_nodes_per_group -> N`. There is no equivalent for `max_rows`, so
that cap must be set by hand.

### PCG convergence controls

- `solver.max_iters`, `solver.min_iters`, `solver.tol`
- `numerics.jitter0`, `numerics.jitter_max` — diagonal jitter added to the whitening system,
  starting at `jitter0` and growing towards `jitter_max` on failure.

Use these to stabilize or tighten solves *after* the cap issues are resolved.

### Safety behavior (`shared_event_re.fallback`)

- `to_diag` — fall back to the diagonal likelihood for groups that cannot be solved (default).
- `abort_on_pcg_fallback` — raise instead of falling back. The error message prints the full
  reason breakdown and a `max_seen: rows=... nodes=...` line, which is the quickest way to learn
  what the caps must be.

Recommended workflow:

- tuning phase: `to_diag = true`, `abort_on_pcg_fallback = false`;
- strict validation phase: `abort_on_pcg_fallback = true` to force immediate failure and inspect
  the cause;
- production: back to `to_diag = true`, with the caps sized so fallback never triggers.

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
  },
  "logging": { "stats_log_every_epochs": 10 }
}
```

## What to monitor

Console (`[LIKELIHOOD]` section, no W&B needed):

- Once per run: `shared_event_re stats groups=G pcg=P fallback=F max_rows=R max_nodes=N tau_s=[...]`
  and `shared_event_re whitening groups=... chol=... fallback=... max_rows=... max_nodes=...`.
  `pcg + fallback` should equal `groups`, with `fallback = 0`.
- Every `logging.stats_log_every_epochs` epochs (set it to `> 0`):
  `shared_event_re stats epoch=E grouping=station_phase groups=G pcg=P fallback=F rows_cap=a nodes_cap=b tau0=c max_rows=... max_nodes=... max_rows_all=... max_nodes_all=...`
- Whenever any group fell back in an epoch:
  `shared_event_re fallback reasons rows_cap=a nodes_cap=b tau_zero=c` (and the whitening-path
  variant `shared_event_re whitening fallback reasons ...`).
- `shared_event_re warning: all groups fell back to diagonal; increase shared_event_re.max_nodes_per_group or max_rows_per_group`
  — the correlated likelihood is completely off. The message still uses the legacy key names; the
  current keys are `limits.max_nodes` and `limits.max_rows`.
- `[WARN][RUN] epoch N: skipped k/M batches ...` and the `skipped_batches` metric: PCG
  instability that produces non-finite losses makes the epoch runner drop batches, and this
  warning is the only trace of it.

W&B (`observability.diagnostics.profile_shared_event_re: true`):

- `shared_event_re/groups_pcg_mean` — should equal `shared_event_re/groups_mean`;
- `shared_event_re/groups_fallback_diag_mean` — should be 0;
- `shared_event_re/max_rows_max`, `shared_event_re/max_nodes_max` — compare with the caps;
- `shared_event_re/whitening_solve_ms_sum`, `whitening_pcg_kernel_ms_sum` — where the time goes
  once fallback is solved.

Healthy target: `groups_pcg_mean > 0`, `groups_fallback_diag_mean` at zero, all fallback reason
counters zero in steady state, no skipped batches.

## Fallback elimination playbook

If fallback is nonzero:

1. Read the fallback reason summary first.
2. If `rows_cap > 0`, raise `limits.max_rows` above `max_rows_all`.
3. If `nodes_cap > 0`, raise `limits.max_nodes` above `max_nodes_all` (or let the auto-tune do it
   and copy the final value into the config so the next run starts there).
4. If `tau_zero > 0`, fix `model.tau_s` (both entries positive) and check that the value actually
   reached the runtime (`validate-config --print-resolved`, flat key `_shared_event_re_tau_s`).
5. If fallback persists after the caps and `tau_s` are fixed:
   - raise `solver.max_iters`,
   - increase `numerics.jitter0` gradually (one decade at a time),
   - tighten or relax `solver.tol` depending on whether you see instability or stagnation.

If all groups still fall back:

- temporarily set `fallback.abort_on_pcg_fallback = true` (it also trips on whitening-path and
  GPU-path fallbacks, and its error message prints the full reason breakdown including
  `max_seen: rows=... nodes=...`),
- rerun and inspect the first failing context,
- verify group sizes are within the caps and `tau_s` is valid.

## Notes on optional solver knobs

`solver.node_bin_edges` (default `[512, 1024, ..., 32768]`), `solver.warm_start`,
`solver.cache_max_entries` (8), `solver.merge_sparse_node_bins`, `solver.min_groups_per_node_bin`,
`solver.max_node_bins_per_node`, `solver.prefetch_grouping`, `solver.precompute.{enabled, device}`
(whitening structures precomputed once at the start of bundle-based sampling) and the
`autotune.*` bucketing keys are operational knobs. The node-bin auto-tuner only runs with
`solver.kind = "pcg"` and `solver.batched = true`. They can improve throughput and robustness in
difficult workloads, but fallback elimination should first be solved with:

- valid `tau_s`,
- sufficient `limits.max_nodes` / `limits.max_rows`,
- stable `solver.max_iters` / `solver.tol` / `numerics.jitter0`.
