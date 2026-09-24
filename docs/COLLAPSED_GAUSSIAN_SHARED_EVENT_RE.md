# Collapsed Gaussian likelihood (shared-event random effects)

This note documents the collapsed (marginalized) Gaussian random-effects likelihood that SPIDER
uses in the sampling phases (`model.likelihoods.sample`) to capture **shared-event correlation**
within each station x phase group, without carrying an explicit latent state `b` in the sampler.

It also lists what a fully consistent version (correlated log-determinant, richer covariance
structure) would require; those parts are **not implemented**.

## Background

Double-difference residuals show:

- Strong correlation for DD residuals that share an event **within the same station x phase**
  (they share that event's path anomaly and pick error)
- Near-zero correlation across stations for the same event pair
- Near-zero correlation as a function of event distance (event-space ACF ~ 0)

This suggests a "station x phase-only" correlation structure is a good first-order target.

## Model (per group)

For each station x phase group `g`, define an undirected event graph from the DD observations in
that group:

- Each observation `e` connects events `(i, j)` and has residual `r_e = dt_obs - dt_pred` with
  `dt_pred = T(evid2) - T(evid1)` (SPIDER's sign convention)
- Latent per-event random effect `b_i ~ N(0, tau^2)`
- Observation noise `eps_e ~ N(0, sigma^2)`
- Observation model: `r_e = (b_j - b_i) + eps_e`

After integrating out `b`, the residual vector `r` of the group has a correlated Gaussian:

```
r ~ N(0, Sigma),    Sigma = sigma^2 I + tau^2 A A^T
```

where `A` is the signed incidence matrix of the event graph, `sigma` is the per-phase
`phase_unc` and `tau` the per-phase `tau_s`.

## What is implemented

**Code paths**

- Schema shape checks: `spider/core/config_v2/validate.py` (`_validate_shared_event_re_shape`)
- Runtime materialization (all `_shared_event_re_*` flat keys and their defaults):
  `spider/core/config_v2/legacy_bridge.py` (`_materialize_block3`)
- Likelihood entry point: `spider/core/modeling.py`, `compute_likelihood_loss()`
- Whitening / batched PCG kernels: `spider/core/shared_event_re_whitening.py`,
  `spider/core/shared_event_re_gpu.py`
- Optional precompute of the whitening plan at sampling start:
  `spider/core/locate.py`, `_maybe_precompute_shared_event_re_whitening`
- Node-bin autotuning: `spider/core/locate.py`, `_maybe_autotune_whitening_bucket_nodes`

**Solver**

- `solver.kind: "pcg"` (default; `pcg_sparse` accepted; other values are rejected by validation) uses a node-space PCG solve to compute `u ~ Sigma^-1 r`
  without forming `Sigma`. `"pcg_sparse"` is accepted as a legacy alias.
- Groups are packed into node-count bins (`solver.node_bin_edges`) and solved batched on the GPU
  (`solver.batched`); `autotune` re-plans the bins from the observed group-size histogram during
  the first sampling epochs.
- A custom autograd function makes the gradient w.r.t. residuals **exactly** `u`, while **not
  differentiating through** the PCG solve (stable and fast).

**Objective**

The likelihood uses the correlated *quadratic term* only:

```
l(r) = 1/2 r^T Sigma^-1 r
```

plus a `mean(log(sigma))` term. With fixed `sigma` this is a constant shift.

**Fallbacks**

A group is evaluated with the plain diagonal Gaussian (`fallback.to_diag`, default true) when it
exceeds `limits.max_rows` or `limits.max_nodes`, when its `tau` is zero, or when PCG fails. The
per-epoch counts are reported as `shared_event_re/groups_fallback_diag_mean`,
`groups_pcg_mean`, `max_rows_max`, `max_nodes_max` (console and W&B). With
`fallback.abort_on_pcg_fallback: true` any fallback aborts the run with a message listing the
counts by reason (rows cap, nodes cap, tau zero, PCG failure).

**Important limitations**

- No correlated logdet term, so `sigma` and `tau` cannot be learned consistently. This is moot
  today: **noise-scale learning has been removed from SPIDER entirely.** `sigma_P`, `sigma_S` are
  fixed by `phase_unc`, the optimizer parameter list contains only `dX`, and a
  `model.priors.noise` block is a hard configuration error. There is no `learn_noise_scale` key;
  a stray one in a config is silently ignored.
- Joint P/S coupling and heteroscedastic per-observation variances are not implemented (there
  are no `joint_ps` / `sigma_extra_var` keys). The internal `_shared_event_re_hierarchical`,
  `_tau_event_s`, `_tau_cluster_s`, `_rho_ps` flags are always materialized as disabled.

**Grouping**

- `model.group_by` must be `"station_phase"`. `"phase"` grouping was removed and now raises a
  `ValueError` at run time; there is **no** silent fallback to phase-only grouping.
- Per-row station indices are supplied by the epoch runner; if they are missing the solve fails
  loudly rather than degrading.
- `model.cluster.mode` optionally splits each station x phase graph further: `none` (default),
  `component` (one graph per connected component of the DD graph) or `dd_khop` (greedy
  non-overlapping `k`-hop balls, `model.cluster.k`).

## How to enable (config)

Under `model.likelihoods.sample`:

```json
{
  "type": "correlated_gaussian",
  "phase_unc": [0.02, 0.03],
  "shared_event_re": {
    "enabled": true,
    "model": {
      "group_by": "station_phase",
      "tau_s": [0.033, 0.06],
      "cluster": { "mode": "none", "k": 1 }
    },
    "limits": { "max_nodes": 25000, "max_rows": 1500000 },
    "fallback": { "to_diag": true, "abort_on_pcg_fallback": false },
    "numerics": { "jitter0": 1e-8, "jitter_max": 1e-3 },
    "solver": { "kind": "pcg", "max_iters": 100, "min_iters": 2,
                "tol": 3e-4, "batched": true,
                "node_bin_edges": [4096, 16384, 25000],
                "warm_start": false, "cache_max_entries": 32,
                "precompute": { "enabled": true, "device": "gpu" } },
    "edge_weights": { "mode": "uniform" },
    "autotune": { "enabled": true },
    "logging": { "quiet": true, "stats_log_every_epochs": 0 }
  }
}
```

Notes:

- `model.likelihoods.sample.type` **must** be `correlated_gaussian` (or `correlated`); disable
  the correlation with `shared_event_re.enabled = false`, not by changing `type`.
- `tau_s` is the per-phase random-effect standard deviation in seconds, `[P, S]`. A single
  number is broadcast to both phases. `tau_s = 0` disables correlation and reduces to iid Gaussian
  (reported as the `tau_zero` fallback reason). Values of the order of `phase_unc` or somewhat
  larger are typical; `spider analyze-resid` (residual variograms) helps choose them.
- **Flat legacy keys are rejected.** `grouping`, `cluster_mode`, `cluster_k`, `tau_s`,
  `max_nodes_per_group`, `max_rows_per_group`, `fallback_to_diag`, `abort_on_pcg_fallback`,
  `jitter0`, `jitter_max`, and `station_phase_re` at the top level of `shared_event_re` all raise a
  `legacy_key` error, as do the old solver keys (`bucket_nodes`, `merge_sparse_edge_bins`,
  `min_groups_per_edge_bin`, `max_edge_bins_per_node`) and autotune keys (`max_bins`,
  `min_bin_groups`, `min_bucket_node`). Any other unknown key at the top level of
  `shared_event_re` is an `unknown_key` error; keys *inside* the sub-blocks are not checked.
- Defaults when omitted: `limits.max_nodes = 512`, `limits.max_rows = 200000`,
  `solver.max_iters = 50`, `solver.min_iters = 2`, `solver.tol = 1e-3`, `solver.batched = true`,
  `solver.node_bin_edges = [512, 1024, 2048, 4096, 8192, 16384, 32768]`,
  `solver.warm_start = false`, `solver.cache_max_entries = 8`, `solver.prefetch_grouping = false`,
  `solver.profile_micro_steps = false`, `solver.merge_sparse_node_bins = true`,
  `solver.min_groups_per_node_bin = 32`, `solver.max_node_bins_per_node = 4`,
  `solver.precompute.enabled = false` (`device = "gpu"`), `numerics.jitter0 = 1e-8`,
  `numerics.jitter_max = 1e-3`, `fallback.to_diag = true`, `fallback.abort_on_pcg_fallback = false`,
  `autotune.enabled = true` (`observe_epochs 1`, `latest_epoch 2`, `min_groups 128`,
  `max_node_bins 8`, `min_groups_per_node_bin 24`, `min_node_bin 512`, `min_gain 0.08`,
  `raise_nodes_cap true`, `nodes_cap_max 65536`), `logging.quiet = true`,
  `logging.stats_log_every_epochs = 0`. The stock `limits` are far below what dense catalogs
  need -- see `docs/pcg-whitening-convergence.md`.
- `edge_weights.mode`: `uniform` (default) or a distance-based weighting of each pair's edge
  (`distance_rbf`, `distance_linear`, `distance_power`) controlled by `ell_km` (1.0), `eps_km`
  (1e-3), `power` (1.0), `scale_km` (1.0), `global_scale` (1.0), `normalize` (false). The shipped
  example uses `distance_power` with `power 0.25`, `scale_km 40`, `normalize true`.
- `station_phase_term.{enabled, tau_s}`: an optional station x phase-level random effect (one
  shared offset per group) layered on the per-event one; disabled by default.

## Diagnostics

- Console: on first use SPIDER prints `[shared_event_re] enabled=True grouping=station_phase ...`
  with the solver settings, and `spider sample` prints `[shared_event_re] config present ->
  enabled=... grouping=...` at start-up so activation can be verified.
- W&B / console (`observability.diagnostics.profile_shared_event_re`): `shared_event_re/groups_mean`,
  `groups_pcg_mean`, `groups_fallback_diag_mean`, `max_rows_max`, `max_nodes_max`,
  `whitening_cache_hits/misses`, PCG timing breakdown (`whitening_pcg_*_ms_sum`),
  `station_phase_groups`, `station_phase_quad`.
- A healthy run has `groups_fallback_diag_mean` near zero; if not, raise `limits.max_nodes` /
  `max_rows` (or let `autotune.raise_nodes_cap` do it) and check `tau_s > 0`.

## Not implemented: logdet via SLQ

To make the collapsed likelihood fully consistent (and enable learning `sigma`, `tau`) one would
need to:

- Add the correlated logdet term `1/2 log det(Sigma)`.
- Implement stochastic Lanczos quadrature for `log det(I + c L)` using only matvecs with
  `(I + c L)`, amortized (evaluate every K epochs, cache Lanczos tridiagonals per bucket/group).
- Reintroduce noise-scale learning (removed from the current codebase) by providing proper
  gradients for `sigma` and optionally `tau`. This would require restoring both a learnable
  `phase_unc` path in `init_state.py` and a noise prior block in `priors_config.py`.

## Not implemented: richer structures

Only pursue if diagnostics warrant it:

- Cross-station coupling for the same event pair (block structure across stations)
- Joint P/S coupling (`joint_ps`, `rho_ps`) with a 2x2 per-node prior
- Heteroscedastic diagonal (per-observation `sigma_e^2`), requiring weighted operators
  `A^T D^-1 A`
