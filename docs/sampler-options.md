# Sampler Options and Mathematical Updates

This page documents the sampler backends available in SPIDER, the update equations they implement
(`spider/optim/sgld.py`, `spider/optim/sghmc.py`), and how the phase drivers in
`spider/core/locate.py` set their knobs. Every equation was checked against the optimizer code.

## Available sampler backends

Configured at:

- `inference.sampler.backend`

Supported values:

- `psgld` -- preconditioned SGLD (`spider.optim.sgld.pSGLD`)
- `sghmc` -- preconditioned SGHMC (`spider.optim.sghmc.SGHMC`)

Backend selection is handled in `spider.optim.backends.create_sampler_backend`. Phase 1 (MAP)
always uses `torch.optim.Adam` and is not affected by the backend choice.

## What each phase does

`inference.sampler.epochs_per_phase = [P1, P2, P3, P4]` and `lr = [lr1, lr2, lr3, lr4]`.

| Phase | Optimizer | Likelihood block | Learning rate | Injected noise | Preconditioner statistics | Gradient clip |
|---|---|---|---|---|---|---|
| 1 MAP (`phase1`, `phase1-pass2`) | Adam | `locate_map` | `lr[0]`, optional cosine schedule | none | -- | `grad_clip_norm`, else 100.0 |
| 2 preconditioner warm-up (`phase2`) | backend | `sample` | `lr[1] / N` | off (`noise=off`) | adapting | `grad_clip_norm`, else none |
| 3 noise ramp (`phase3`) | backend | `sample` | `lr[2] / N` | `noise_scale = (t+1)/P3 × noise_scale_mult` | frozen iff `freeze_preconditioner_sampling` | `grad_clip_norm`, else none |
| 4 sampling (`phase4`) | backend | `sample` | `lr[3] / N` | `noise_scale = noise_scale_mult` | frozen iff `freeze_preconditioner_sampling` | `grad_clip_norm`, else none |

`N` is the number of dtimes after all filters (including the post-Phase-1 filters). Every epoch
prints one line, for example

```
phase4 | 120/100000 | L=1.2345e+00 | dx=1.234e-02 | dy=… | dz=… | dT=… | dr_max=… | dr_90=… | t=3.2s | precond=rmsprop noise=on(scale=1,T=1) lr=1e-08 | sigma_p=0.0200 sigma_s=0.0300
```

where `dx/dy/dz/dT` are the median absolute shifts (km, s), `dr_max`/`dr_90` the maximum and 90th
percentile of the 3-D shift norm, `lr` the effective step (already divided by `N`), and
`noise=on(scale=…,T=…)` or `noise=off` the current noise state. Phase 3 appends `ramp=…`.

Posterior samples are collected only in Phase 4, every `io.save_every_n` optimizer steps.

## Common runtime conventions

For both backends:

- Drift uses the minibatch-mean gradient scaled by the total number of observations:
  $g_{\text{drift}} = N\,\bar g$. The minibatch loss is the per-observation objective
  $\mathcal J$ of {doc}`probability-model`, so $N\bar g$ estimates the gradient of the full
  negative log posterior and the samplers target the true posterior.
- The configured learning rate is per-observation: at sampler creation and again at every phase
  transition (and on resume) SPIDER sets $\lambda = \lambda_{\text{config}}[i] / N$, where `i`
  indexes `inference.sampler.lr` by phase (`lr[1]` → Phase 2, `lr[2]` → Phase 3, `lr[3]` → Phase 4).
  `lr[0]` is the Phase-1 Adam learning rate and is *not* divided by `N`.
- Preconditioner statistics (RMSprop `v`, LRD buffers) are updated from the minibatch-mean gradient
  $\bar g$, not from $N\bar g$.
- Injected noise carries a phase-dependent factor `noise_scale` (table above) and the temperature
  $T$ = `inference.sampler.temperature`: the noise standard deviation scales as
  $\text{noise\_scale}\sqrt{T}$. `temperature: 0` disables noise in every phase (SGHMC also skips
  the noise draw when `noise_scale` is 0).
- After every step the shifts are made finite (`nan_to_num`) and clamped to
  `inference.safety.max_abs_dX` when set.
- On resume (`inference.runtime.reset_batch_numbers: false`) the current JSON values of `lr`,
  `temperature`, and the preconditioning flags are re-applied to the loaded sampler state, so you
  can change them between restarts; `beta` and `eps` are only filled in if the checkpoint lacks
  them (use `overrides.core.eps` to force a new `eps`).

Full key set accepted under `inference.sampler` (unknown keys are rejected):

`backend`, `epochs_per_phase` (4 ints), `lr` (4 floats), `temperature`, `beta`, `eps`,
`freeze_preconditioner_sampling`, `sghmc_alpha` (required key; for `sghmc` a finite number `> 0`
that the SGHMC class additionally requires to be `<= 1`; may be `null` for `psgld`),
`noise_scale_mult`, `grad_clip_norm`, `preconditioning`, `reparameterization`, `overrides`,
`lr_schedule`, `phase1_two_pass`, `weak_event_mh` (independence-Metropolis resampling of weakly-observed events; see [weak-event-mh.md](weak-event-mh.md)). See {doc}`configuration-reference` for constraints and defaults.

## Gradient clipping

`inference.sampler.grad_clip_norm` clips the global L2 norm of the ΔX gradient before each step.
When set it applies to all phases (`0` disables clipping everywhere). When the key is omitted,
Phase 1 (Adam/MAP) clips at the legacy default of 100.0 and Phases 2-4 do not clip. Clipping in
the sampling phases biases the stationary distribution; treat it as a guardrail while tuning, not
as a production setting.

## Phase-1 learning-rate schedule

```json
"lr_schedule": { "phase1": { "type": "cosine", "eta_min": 1e-5, "T_max": null } }
```

`type` is `none` (default) or `cosine` (`torch.optim.lr_scheduler.CosineAnnealingLR`, stepped once
per epoch, from `lr[0]` down to `eta_min`). `T_max: null` means "the epoch count of the current MAP
pass", so an optional second pass gets its own cosine horizon. On resume the schedule is
fast-forwarded to the resumed epoch. Phase 1 logs the resolved schedule, e.g.
`Phase 1: MAP (optimizer=adam) | lr_schedule=cosine(T_max=1000, eta_min=1e-05) | precond=none noise=off lr=0.001`.

Use the cosine schedule to remove the end-of-MAP jitter that a constant Adam step leaves in the
locations; the example configs combine `lr[0] = 1e-3` with `eta_min = 1e-5`.

## Phase-1 two-pass MAP

```json
"phase1_two_pass": { "enabled": true, "epochs": null, "warm_start": true }
```

With `enabled=true`, Phase 1 runs MAP, applies the post-MAP filters at the relocated (MAP)
positions -- residual filter, pair/station ratio, linearization error, then the post-Phase-1 event
filter -- and then re-runs MAP on the filtered dtimes with a fresh Adam optimizer (and a fresh
cosine schedule when configured). `warm_start=true` (default) keeps the pass-1 MAP locations as the
starting point; `false` resets ΔX to zero. `epochs: null` reuses `epochs_per_phase[0]`.

Outputs: the pass-1 catalog is preserved as `<catalog_outfile>_MAP_pass1.csv` and
`<catalog_outfile>_MAP.csv` holds the final result. Second-pass epoch lines are labelled
`phase1-pass2`. The second pass is skipped when `epochs_per_phase[0] = 0`, and under torchrun/DDP
(with a warning).

## Per-group overrides

```json
"overrides": { "core": { "lr_mult": 1.0, "temperature_mult": 1.0, "eps": 1e-5, "freeze_preconditioner_sampling": false } }
```

The samplers hold a single parameter group named `core` (the event shifts ΔX); only that name is
honoured, other group names are ignored. `lr_mult` multiplies the phase learning rate after the
`/N` scaling, `temperature_mult` multiplies `temperature`, `eps` replaces the RMSprop damping for the
group, and `freeze_preconditioner_sampling` is OR-ed with the global flag (an override can freeze but
never unfreeze). The overrides are re-applied after every phase transition and on resume.

## Preconditioner options

Configured at:

- `inference.sampler.preconditioning.enabled`
- `inference.sampler.preconditioning.type`
- `inference.sampler.preconditioning.include_gamma`
- `inference.sampler.preconditioning.lrd.*`

Supported preconditioner types:

- `rmsprop` (diagonal)
- `lrd` (low-rank plus diagonal)
- `component_lrd` (component-wise low-rank plus diagonal; no cross-component coupling; aliases
  `cc_lrd`, `block_lrd`, `component-lrd`)
- `block_gn` (per-event 4x4 Gauss-Newton metric, held fixed between refreshes; see
  [Block Gauss-Newton preconditioner](#block-gauss-newton-preconditioner) below). **Under
  `block_gn` the sampler learning rates `lr[1..3]` are dimensionless natural steps** (typical
  0.05-0.5), not per-observation rates.

`preconditioning.enabled: false` forces type `none` regardless of `type` (identity metric). When
preconditioning is enabled, both drift and injected noise are scaled by the same metric. Phase 2
always lets the statistics adapt; Phases 3 and 4 freeze them when
`freeze_preconditioner_sampling` is `true`. The resolved setting is logged once at sampler
creation: `[SAMP] Sampler preconditioning resolved: enabled=True type=rmsprop | …`.

Phase 4 emits warnings for the settings that most often produce sampling artifacts:
`freeze_preconditioner_sampling=false` with preconditioning on (time-inhomogeneous kernel),
`inference.batching.standard.shuffle=false` (periodic gradient noise), and `epochs_per_phase[1] = 0`
with preconditioning on (unwarmed RMSprop statistics).

## Blocked reparameterization

Configured at:

- `inference.sampler.reparameterization.enabled`
- `inference.sampler.reparameterization.spatial_scale`
- `inference.sampler.reparameterization.dt_scale`

When enabled, the samplers operate in transformed coordinates $\theta = s \odot \phi$ with one
static scale for the spatial columns (`x/y/z`) and one for the temporal column (`dt`). In code the
gradient is multiplied by $s$ before the preconditioner statistics and the drift are formed, and
the whole update (drift plus noise) is multiplied by $s$ again before it is applied to $\theta$.
The net effect is a drift step $\propto s^2$ and a noise step $\propto s$, i.e. a genuine change of
variables rather than a per-column learning-rate multiplier. Use it when the spatial (km) and
temporal (s) units are imbalanced enough to destabilize a frozen metric; the example pSGLD config
uses `dt_scale: 0.3`. The `block_gn` preconditioner ignores the reparameterization (its metric
already carries the coordinate units) and logs a warning if both are enabled. Non-finite or non-positive scales are silently coerced to `1.0`, and scales of
exactly `1.0` in both blocks are a no-op. `dt_scale` replaces the removed `dt_lr_mult` key.

## Gauge projection (optional)

`inference.runtime.gauge_projection` (see {doc}`configuration-reference`) removes the mean over
events -- globally or per connected component -- of the selected columns from the gradient before
the preconditioner statistics are updated, from the injected noise and the SGHMC momentum when
`apply_noise` / `apply_momentum` are true, and from the parameters after every step (exact
re-centering, all phases). The post-step re-centering is required because the preconditioned
update of a zero-mean gradient is not zero-mean. With `mode: cluster` and all four dims this pins
each connected component's frame to the input catalog exactly, with no strength parameter; use it
in place of the centroid prior when the input frame is to be preserved.

## pSGLD backend

Class:

- `spider.optim.sgld.pSGLD`

With diagonal preconditioner $G$, the implemented step is:

$$
\theta_{t+1} = \theta_t - \Big(\lambda\, G_t\, g_{\text{drift}} + \lambda\,\Gamma_t\Big)
              + \text{noise\_scale}\,\sqrt{2\,\lambda\,T}\,\sqrt{G_t}\,\xi_t
$$

where:

- $\xi_t \sim \mathcal N(0, I)$ and `noise_scale` is the phase-dependent factor above (0 in Phase 2)
- $\Gamma_t = -(1-\beta)\,\bar g_t\,\sqrt{v_t}\,/\,(\epsilon+\sqrt{v_t})^2$ is the diagonal
  approximation to the pSGLD correction term, computed from the minibatch-mean gradient (enabled
  by `include_gamma`, default `false`; both example configs leave it off)
- $G_t = (\epsilon + \sqrt{v_t})^{-1}$ for `rmsprop` mode (**no** bias correction)

RMSprop second-moment update (skipped while the preconditioner is frozen):

$$
v_t = \beta v_{t-1} + (1-\beta)\,\bar g_t^{\,2}
$$

With `preconditioning.enabled: false` the step is plain SGLD ($G = I$, no $\Gamma$).

## SGHMC backend

Class:

- `spider.optim.sghmc.SGHMC`

With momentum $p$, friction $\alpha$ = `sghmc_alpha`, and diagonal $G$:

$$
p_{t+1} = (1-\alpha)\,p_t - \lambda\,G_t\,g_{\text{drift}}
          + \text{noise\_scale}\,\sqrt{2\alpha\,\lambda}\,\,\sqrt{T}\,\sqrt{G_t}\,\xi_t
$$
$$
\theta_{t+1} = \theta_t + p_{t+1}
$$

The diagonal `rmsprop` metric in SGHMC uses a bias-corrected second moment:

$$
v_t = \beta v_{t-1} + (1-\beta)\,\bar g_t^{\,2},\qquad
\hat v_t = \frac{v_t}{1-\beta^t},\qquad
G_t = (\epsilon + \sqrt{\hat v_t})^{-1}
$$

There is no $\Gamma$ term in SGHMC. `alpha` must lie in `[0, 1]` for the class (`(0, 1]` after
config validation); smaller values mean less damping and longer momentum memory.

## LRD preconditioner math

Implemented in both samplers via `_build_lrd_metric(...)` (global) and
`_build_component_lrd_metric(...)` (component-wise, one subspace per connected component).

Metric form:

$$
P_t = \operatorname{diag}(d_t) + U_t\,\operatorname{diag}(\lambda_t)\,U_t^\top
$$

The diagonal part is the bias-corrected RMSprop metric floored at `diag_floor`
(default `inference.sampler.eps`):

$$
d_t = \max\!\Big(\big(\epsilon + \sqrt{\hat v_t}\big)^{-1},\ d_{\min}\Big),
\qquad \hat v_t = \frac{v_t}{1-\beta^t}
$$

This applies in both samplers (unlike the pSGLD `rmsprop` path, which is not bias-corrected).

Drift preconditioning:

$$
P_t g = d_t \odot g + U_t\Big(\lambda_t \odot (U_t^\top g)\Big)
$$

Noise is drawn with covariance proportional to $P_t$:

- diagonal part via $\sqrt{d_t}\odot z_1$
- low-rank part via $U_t(\sqrt{\lambda_t}\odot z_2)$

with independent standard-normal $z_1, z_2$. The pSGLD $\Gamma$ term is not applied on the LRD
paths.

### LRD subspace update modes

Configured under:

- `inference.sampler.preconditioning.lrd.mode`

Modes:

- `svd`: maintain a buffer of the last `buffer_size` gradients and every `update_every` steps
  refresh $U,\lambda$ from its batched SVD.
- `oja`: online Oja-style subspace updates with learning rate `eta`.

LRD knobs (with defaults), under `inference.sampler.preconditioning.lrd`:

- `rank` (16; clipped to the parameter count)
- `mode` (`svd`; `oja` also supported; `randomized_svd`/`stochastic_svd` are aliases for `svd`,
  other values fall back to `svd`)
- `update_every` (20, svd mode), `buffer_size` (64, svd mode; at least `rank + 1`)
- `eta`/`oja_eta` (0.02, oja mode; non-positive values fall back to 0.02)
- `diag_floor` (defaults to `inference.sampler.eps`)
- `target` (`dX_src_only`)
- `freeze_subspace_after` (null = never): sampling epoch after which the LRD subspace
  (U, lambda) stops refreshing while the RMSProp diagonal KEEPS adapting. This is the
  middle option between a fully adaptive metric (subspace rotations make the phase-4
  kernel time-inhomogeneous and show up as slow per-event drift) and
  `freeze_preconditioner_sampling=true` (which also snapshots the diagonal and can
  destabilize coordinates whose v-estimate was transiently low at the freeze instant).
  A static subspace keeps the soft-mode mixing gain; a stale one is merely less helpful,
  never unstable.

Unlike the rest of the sampler block, keys under `preconditioning.lrd` are not validated against a
whitelist -- typos are silently ignored.

## Block Gauss-Newton preconditioner

`preconditioning.type: "block_gn"` (both backends). Implemented in `spider/optim/block_gn.py`.

### Why

The event-shift posterior is stiff by orders of magnitude: an event with thousands of
differential times has a conditional posterior a few metres wide, an event with a handful of rows
is bounded only by the event prior (hundreds of metres to kilometres), and the origin-time
coordinate has its own scale. A random walk with one common step needs a number of steps
proportional to (posterior width / step)$^2$ to decorrelate, so the weakest events mix
thousands of times more slowly than the best constrained ones. The RMSprop diagonal only
partly compensates: its entries scale like the posterior *width* rather than the variance, they
are built from minibatch statistics that for row-sparse events reflect gradient noise rather
than curvature, and the `eps` floor compresses them further (see {doc}`sampler-health`).

### Metric

For every event $i$ the data term's Gauss-Newton (Fisher) block is accumulated over all rows
touching the event,

$$
F_i = \sum_{r \ni i} w_r\, J_{r,i}^\top J_{r,i}, \qquad w_r = \sigma_{\text{phase}(r)}^{-2},
$$

with $J_{r,i}$ the $1\times 4$ Jacobian of the predicted differential time with respect to the
event's $(x, y, z, t)$ shift ($[-\nabla T, -1]$ for the first endpoint of the pair, $[+\nabla T,
+1]$ for the second). Adding the event-prior precision $P_{k(i)}$ gives the Laplace-approximation
*conditional* precision of the event given its neighbours; the metric is its inverse,

$$
M_i = \big(F_i + P_{k(i)}\big)^{-1}, \qquad L_i L_i^\top = M_i .
$$

$P_{k(i)}$ is taken from, in order of preference: the hierarchical per-cluster precision
(`priors.event.hyper`), the fixed Gaussian event prior (`priors.event.params.std`), or
`preconditioning.block_gn.fallback_std`. With no prior at all the blocks are only regularized
by the relative eigenvalue floor `jitter` and a warning is logged.

The samplers then take the preconditioned step (pSGLD; SGHMC uses the same $M_i$, $L_i$ in its
momentum update)

$$
\Delta X_i \leftarrow \Delta X_i - s\, M_i\, g_i + \sqrt{2 s T}\, L_i\, z_i, \qquad z_i \sim
\mathcal N(0, I_4),
$$

where $g_i$ is the sum-likelihood gradient ($N$ times the minibatch mean) and $s$ is a
*dimensionless* step. For a locally quadratic posterior $s$ is the fraction of the Newton step
taken per iteration, independent of how well each event is observed and of the coordinate
units, so every event mixes on the same time scale in every coordinate. Because the metric is
constant between refreshes no $\Gamma$ correction is needed and the kernel is exactly
time-homogeneous within a phase.

### Gauge projection is required

Differential-time data are (nearly) invariant under a rigid translation of a connected
component, so the component mean is held only by priors. The centroid prior
$\tfrac12 n_k m_k^\top P_c m_k$ (a 10 m std is typical) has a curvature far above any single
event's block but spread over all events of the component, and a block-diagonal Newton step is
unstable along that direction. `block_gn` therefore requires
`inference.runtime.gauge_projection.enabled: true` (validation error otherwise), and, when the
centroid prior is enabled, `gauge_projection.dims: [0, 1, 2, 3]`. The component mean becomes a
hard constraint, exactly as for the other preconditioners under gauge projection.

On the constrained manifold the sampler takes the *exact* Newton step of the block-diagonal
model rather than the projection of the unconstrained one (the latter lets the kilometre-scale
steps of weak events leak into metre-scale events through the mean and diverges as soon as the
blocks are heterogeneous). With $E$ selecting the projected dims and one $d\times d$ matrix
$S_k = \sum_{i\in k} E M_i E^\top$ per component,

$$
y_i = M_i\big(g_i + E^\top \lambda_k\big),\qquad
\lambda_k = -S_k^{-1} \sum_{i\in k} E M_i g_i,\qquad
\xi_i = z_i - M_i E^\top S_k^{-1} \sum_{j\in k} E z_j,\quad z_j \sim \mathcal N(0, M_j),
$$

so the update has zero component mean by construction and the noise is the Gaussian
conditioned on the constraint. On the constrained subspace the data Hessian is bounded by twice
the block diagonal, so the drift-only iteration contracts for every $s < 1$ regardless of how
heterogeneous the blocks are.

### Step size

The samplers receive `lr = lr_user / N` and multiply minibatch-mean gradients by `N`; under
`block_gn` the natural step is therefore `s = lr_user`, i.e. **`inference.sampler.lr[1..3]` is
`s` directly.** Typical values are 0.05-0.5; on a Gaussian target the stationary variance of the
discretized chain is $\sigma^2 / (1 - s/2)$ and the lag-1 autocorrelation is $1 - s$, so
$s = 0.2$ gives a 10 % variance inflation and an integrated autocorrelation time of about nine
steps. Values $\ge 2$ diverge and are rejected by validation. Larger $s$ also amplifies the
minibatch gradient noise for the best constrained events; watch `grad_noise_to_langevin_dr/dt`
(they use the diagonal of $M$ under `block_gn`) and lower $s$ or enlarge the sampling minibatch
if they approach 1. `overrides.core.lr_mult` still applies (it multiplies $s$).

### Refresh policy

The metric is (re)built lazily on the first sampler step after a rebuild is requested: at the
start of every sampler phase, every `refresh_every_epochs` epochs, and after each
hierarchical-prior update (`refresh_on_prior_update`, default `true`). The last two are skipped
while the preconditioner is frozen for the phase (`freeze_preconditioner_sampling`), so a frozen
Phase 4 runs one fixed metric. Each build is one Jacobian pass over the data (`batch_size` rows
per autograd call; minutes at most for tens of millions of rows) and logs the implied
conditional std percentiles:

```text
[PRECOND] Block Gauss-Newton metric built: implied conditional std p5/p50/p95 x 4/18/210 m | ... | prior=hierarchical | unobserved events=0 | eigen-floored=0
```

The metric is stored in the optimizer state, so checkpoints and resumes carry it.

### Options (`preconditioning.block_gn`, all optional)

| key | default | meaning |
|---|---|---|
| `fallback_std` | `null` | `[km, km, km, s]` prior std used only when no event prior is active |
| `min_std` | `null` | `[km, km, km, s]` floor on the implied per-coordinate std (adds variance to every block; rarely needed) |
| `refresh_every_epochs` | `null` | rebuild cadence within a phase (`null` = only at phase start / prior updates) |
| `refresh_on_prior_update` | `true` | rebuild after each hierarchical-prior update (unless frozen) |
| `batch_size` | 262144 | rows per Jacobian batch |
| `jitter` | 1e-6 | relative eigenvalue floor for $F_i + P_{k(i)}$ |

### Limits

- The metric is block-diagonal: it equalizes *per-event* stiffness. Collective slow modes
  (weakly anchored groups moving together) are not accelerated.
- The data weights are the independent-Gaussian `phase_unc` terms; the shared-event
  random-effect correlation is ignored in the metric. This only makes the metric slightly
  conservative.
- `include_gamma`, `beta`, and `eps` have no effect under `block_gn`.
- Requires gauge projection (see above); the component mean is not sampled.

## What is currently exposed in config

Production path (via backend factory) currently exposes:

- `psgld`
- `sghmc`
- `rmsprop`, `lrd`, `component_lrd`, or `block_gn` preconditioning

There is an additional optimizer class in code (`AdaptiveDriftSGLDAdam`), but it is not selectable
through `inference.sampler.backend`.

## Practical tuning interpretation

- Increase `eps` to reduce extreme preconditioner amplification of poorly constrained events.
- Increase `beta` for smoother/slower preconditioner adaptation (the shipped configs use 0.99 for
  pSGLD and 0.999 for SGHMC).
- Give Phase 2 enough epochs for the RMSprop statistics to settle before noise is injected;
  Phase 4 warns when `epochs_per_phase[1] = 0`.
- In SGHMC, lower `sghmc_alpha` to reduce damping; raise it to damp oscillations.
- Use `freeze_preconditioner_sampling=true` for a time-homogeneous kernel in Phases 3 and 4 once
  adaptation is mature.
- Use `grad_clip_norm` as a safety guardrail when exploring higher learning rates, then remove it.
- Use `lr_schedule.phase1` (cosine) to remove end-of-MAP jitter, and `phase1_two_pass` to re-fit
  after outlier rejection at the MAP locations.
- Watch `dr_max` in the epoch line: a value that stops changing usually means an event is pinned
  at `inference.safety.max_abs_dX`.

See also:

- {doc}`sampler-health`
- {doc}`pcg-whitening-convergence`
