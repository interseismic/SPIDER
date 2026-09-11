# Sampler Options and Mathematical Updates

This page documents sampler backends available in SPIDER and the update equations used in practice.

## Available sampler backends

Configured at:

- `inference.sampler.backend`

Supported values:

- `psgld`
- `sghmc`

Backend selection is handled in `spider.optim.backends.create_sampler_backend`.

## Common runtime conventions

For both backends in SPIDER:

- Drift uses minibatch mean gradient scaled by total observations:
  - $g_{\text{drift}} = N \,\bar g$
- The configured learning rate is per-observation: at sampler creation and again at every phase
  transition SPIDER sets $\lambda_{\text{eff}} = \lambda_{\text{config}}[i] / N$, where `i` indexes
  `inference.sampler.lr` by phase (`lr[1]` → Phase 2, `lr[2]` → Phase 3, `lr[3]` → Phase 4).
  `lr[0]` is the Phase-1 Adam learning rate and is *not* divided by N.
- Injected noise scale: off in Phase 2; in Phase 3 it ramps as
  `(epoch+1)/phase3_epochs × noise_scale_mult`; in Phase 4 it is the constant `noise_scale_mult`.
  The update equations below carry this factor as `noise_scale`.

Full key set accepted under `inference.sampler` (unknown keys are rejected):

`backend`, `epochs_per_phase` (4 ints), `lr` (4 floats), `temperature`, `beta`, `eps`,
`freeze_preconditioner_sampling`, `sghmc_alpha` (required key; must be a number in `(0, 1]` for
`sghmc`, may be `null` for `psgld`), `noise_scale_mult`, `grad_clip_norm`, `preconditioning`,
`reparameterization`, `overrides`, `lr_schedule`, `phase1_two_pass`.

## Gradient clipping

`inference.sampler.grad_clip_norm` applies to all phases when set (`0` disables clipping
everywhere). When the key is omitted, Phase 1 (Adam/MAP) clips at the legacy default of 100.0 and
Phases 2–4 do not clip.

## Phase-1 learning-rate schedule

```json
"lr_schedule": { "phase1": { "type": "cosine", "eta_min": 1e-5, "T_max": null } }
```

`type` is `none` (default) or `cosine`. `T_max: null` means "the epoch count of the current MAP
pass", so an optional second pass gets its own cosine horizon. On resume the schedule is
fast-forwarded to the resumed epoch. Phase 1 logs the resolved schedule, e.g.
`lr_schedule=cosine(T_max=1000, eta_min=1e-05)`.

## Phase-1 two-pass MAP

```json
"phase1_two_pass": { "enabled": true, "epochs": null, "warm_start": true }
```

With `enabled=true`, Phase 1 runs MAP, applies the post-MAP filters at the relocated (MAP)
positions — residual filter, pair/station ratio, linearization error — and then re-runs MAP on the
filtered dtimes with a fresh Adam optimizer. `warm_start=true` (default) keeps the pass-1 MAP
locations as the starting point; `false` resets ΔX to zero. `epochs: null` reuses
`epochs_per_phase[0]`.

Outputs: the pass-1 catalog is preserved as `<catalog_outfile>_MAP_pass1.csv` and
`<catalog_outfile>_MAP.csv` holds the final result. Second-pass epoch lines are labelled
`phase1-pass2`. Not supported under torchrun/DDP — the second pass is skipped with a warning.

## Per-group overrides

```json
"overrides": { "core": { "lr_mult": 1.0, "temperature_mult": 1.0, "eps": 1e-5, "freeze_preconditioner_sampling": false } }
```

Only the `core` parameter group is honoured; other group names are ignored.

## Preconditioner options

Configured at:

- `inference.sampler.preconditioning.enabled`
- `inference.sampler.preconditioning.type`

Supported preconditioner types:

- `rmsprop` (diagonal)
- `lrd` (low-rank plus diagonal)
- `component_lrd` (component-wise low-rank plus diagonal; no cross-component coupling; aliases
  `cc_lrd`, `block_lrd`, `component-lrd`)

`preconditioning.enabled: false` forces type `none` regardless of `type`. When preconditioning is
enabled, both drift and injected noise are scaled by the same metric. `include_gamma` (default
`true`) adds the pSGLD correction term described below.

## Blocked Reparameterization

Configured at:

- `inference.sampler.reparameterization.enabled`
- `inference.sampler.reparameterization.spatial_scale`
- `inference.sampler.reparameterization.dt_scale`

When enabled, samplers operate in a blocked transformed coordinate system:

- one static scale for spatial coordinates (`x/y/z`)
- one static scale for temporal coordinate (`dt`)

This is useful when spatial and temporal units are imbalanced and can destabilize frozen metrics.
The scale multiplies the gradient, the drift **and** the injected noise (a genuine change of
variables, not just a learning-rate trick). Non-finite or non-positive scales are silently coerced
to `1.0`. `dt_scale` replaces the removed `dt_lr_mult` key.

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
- $\Gamma_t \approx -(1-\beta)\,\bar g_t\,\sqrt{v_t}\,/\,(\epsilon+\sqrt{v_t})^2$ is the diagonal
  approximation to the pSGLD correction term (enabled by `include_gamma`, default `true`)
- $G_t = (\epsilon + \sqrt{v_t})^{-1}$ for RMSprop mode (no bias correction)

RMSprop second-moment update:

$$
v_t = \beta v_{t-1} + (1-\beta)\,\bar g_t^{\,2}
$$

## SGHMC backend

Class:

- `spider.optim.sghmc.SGHMC`

With momentum $p$, friction $\alpha$, and diagonal $G$:

$$
p_{t+1} = (1-\alpha)\,p_t - \lambda\,G_t\,g_{\text{drift}}
          + \text{noise\_scale}\,\sqrt{2\alpha\,\lambda}\,\,\sqrt{T}\,\sqrt{G_t}\,\xi_t
$$
$$
\theta_{t+1} = \theta_t + p_{t+1}
$$

The diagonal `rmsprop` metric in SGHMC uses a bias-corrected second moment (the `lrd` /
`component_lrd` paths use `_build_lrd_metric` without bias correction):

$$
v_t = \beta v_{t-1} + (1-\beta)\,\bar g_t^{\,2},\qquad
\hat v_t = \frac{v_t}{1-\beta^t},\qquad
G_t = (\epsilon + \sqrt{\hat v_t})^{-1}
$$

## LRD preconditioner math

Implemented in both samplers via `_build_lrd_metric(...)` (global) and `_build_component_lrd_metric(...)` (component-wise).

Metric form:

$$
P_t = \operatorname{diag}(d_t) + U_t\,\operatorname{diag}(\lambda_t)\,U_t^\top
$$

Drift preconditioning:

$$
P_t g = d_t \odot g + U_t\Big(\lambda_t \odot (U_t^\top g)\Big)
$$

Noise is drawn with covariance proportional to $P_t$:

- diagonal part via $\sqrt{d_t}\odot z_1$
- low-rank part via $U_t(\sqrt{\lambda_t}\odot z_2)$

with independent standard-normal $z_1, z_2$.

### LRD subspace update modes

Configured under:

- `inference.sampler.preconditioning.lrd.mode`

Modes:

- `svd`: maintain a gradient buffer and periodically update $U,\lambda$ from batched SVD.
- `oja`: online Oja-style subspace updates with learning rate `eta`.

LRD knobs (with defaults), under `inference.sampler.preconditioning.lrd`:

- `rank` (16)
- `mode` (`svd`; `oja` also supported; `randomized_svd`/`stochastic_svd` are aliases for `svd`,
  other values fall back to `svd`)
- `update_every` (20, svd mode), `buffer_size` (64, svd mode)
- `eta`/`oja_eta` (0.02, oja mode)
- `diag_floor` (defaults to `inference.sampler.eps`)
- `target` (`dX_src_only`)

Unlike the rest of the sampler block, keys under `preconditioning.lrd` are not validated against a
whitelist — typos are silently ignored.

## What is currently exposed in config

Production path (via backend factory) currently exposes:

- `psgld`
- `sghmc`
- `rmsprop`, `lrd`, or `component_lrd` preconditioning

There is an additional optimizer class in code (`AdaptiveDriftSGLDAdam`), but it is not currently selected by `inference.sampler.backend`.

## Practical tuning interpretation

- Increase `eps` to reduce extreme preconditioner amplification.
- Increase `beta` for smoother/slower preconditioner adaptation.
- In SGHMC, lower `sghmc_alpha` to reduce damping; raise it to damp oscillations.
- Use `freeze_preconditioner_sampling=true` for time-homogeneous Phase 4 kernels after adaptation is mature.
- Use `grad_clip_norm` as a safety guardrail when exploring higher learning rates.
- Use `lr_schedule.phase1` (cosine) to remove end-of-MAP jitter, and `phase1_two_pass` to re-fit
  after outlier rejection at the MAP locations.

See also:

- {doc}`sampler-health`
- {doc}`pcg-whitening-convergence`
