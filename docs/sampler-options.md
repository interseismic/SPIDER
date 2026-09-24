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
`lr_schedule`, `phase1_two_pass`. See {doc}`configuration-reference` for constraints and defaults.

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
uses `dt_scale: 0.3`. Non-finite or non-positive scales are silently coerced to `1.0`, and scales of
exactly `1.0` in both blocks are a no-op. `dt_scale` replaces the removed `dt_lr_mult` key.

## Gauge projection (optional)

`inference.runtime.gauge_projection` (see {doc}`configuration-reference`) removes the mean over
events -- globally or per connected component -- of the selected columns from the gradient before
the preconditioner statistics are updated and, when `apply_noise` / `apply_momentum` are true, from
the injected noise and the SGHMC momentum. It is an alternative to the centroid prior for pinning
the translation gauge; the shipped configs leave it disabled.

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

Unlike the rest of the sampler block, keys under `preconditioning.lrd` are not validated against a
whitelist -- typos are silently ignored.

## What is currently exposed in config

Production path (via backend factory) currently exposes:

- `psgld`
- `sghmc`
- `rmsprop`, `lrd`, or `component_lrd` preconditioning

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
