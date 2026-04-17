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
  - \( g_{\text{drift}} = N \,\bar g \)
- User-configured sampler learning rate is internally scaled per observation at backend creation:
  - \( \text{lr}_{\text{effective}} = \text{lr}_{\text{config}} / N \)
- Noise is off in Phase 2, ramped in Phase 3, and active in Phase 4.

Common controls:

- `temperature`
- `noise_scale_mult`
- `beta`, `eps`
- `freeze_preconditioner_sampling`
- `grad_clip_norm`

## Preconditioner options

Configured at:

- `inference.sampler.preconditioning.enabled`
- `inference.sampler.preconditioning.type`

Supported preconditioner types:

- `rmsprop` (diagonal)
- `lrd` (low-rank plus diagonal)

When preconditioning is enabled, both drift and injected noise are scaled by the same metric.

## pSGLD backend

Class:

- `spider.optim.sgld.pSGLD`

With diagonal preconditioner \(G\), the implemented step is:

\[
\theta_{t+1} = \theta_t - \Big(\text{lr}\, G_t\, g_{\text{drift}} + \text{lr}\,\Gamma_t\Big)
              + \sqrt{2\,\text{lr}\,T}\,\sqrt{G_t}\,\xi_t
\]

where:

- \( \xi_t \sim \mathcal N(0, I) \)
- \( \Gamma_t \) is an optional diagonal approximation to the pSGLD correction term (enabled by `include_gamma`)
- \( G_t = (\epsilon + \sqrt{v_t})^{-1} \) for RMSprop mode

RMSprop second-moment update:

\[
v_t = \beta v_{t-1} + (1-\beta)\,\bar g_t^{\,2}
\]

## SGHMC backend

Class:

- `spider.optim.sghmc.SGHMC`

With momentum \(p\), friction \(\alpha\), and diagonal \(G\):

\[
p_{t+1} = (1-\alpha)\,p_t - \text{lr}\,G_t\,g_{\text{drift}}
          + \sqrt{2\alpha\,\text{lr}}\,\text{noise\_scale}\,\sqrt{T}\,\sqrt{G_t}\,\xi_t
\]
\[
\theta_{t+1} = \theta_t + p_{t+1}
\]

RMSprop metric in SGHMC uses bias-corrected second moment:

\[
v_t = \beta v_{t-1} + (1-\beta)\,\bar g_t^{\,2},\qquad
\hat v_t = \frac{v_t}{1-\beta^t},\qquad
G_t = (\epsilon + \sqrt{\hat v_t})^{-1}
\]

## LRD preconditioner math

Implemented in both samplers via `_build_lrd_metric(...)`.

Metric form:

\[
P_t = \operatorname{diag}(d_t) + U_t\,\operatorname{diag}(\lambda_t)\,U_t^\top
\]

Drift preconditioning:

\[
P_t g = d_t \odot g + U_t\Big(\lambda_t \odot (U_t^\top g)\Big)
\]

Noise is drawn with covariance proportional to \(P_t\):

- diagonal part via \( \sqrt{d_t}\odot z_1 \)
- low-rank part via \( U_t(\sqrt{\lambda_t}\odot z_2) \)

with independent standard-normal \(z_1, z_2\).

### LRD subspace update modes

Configured under:

- `inference.sampler.preconditioning.lrd.mode`

Modes:

- `svd`: maintain a gradient buffer and periodically update \(U,\lambda\) from batched SVD.
- `oja`: online Oja-style subspace updates with learning rate `eta`.

Useful LRD knobs:

- `rank`
- `mode` (`svd` or `oja`)
- `update_every` (svd mode)
- `buffer_size` (svd mode)
- `eta`/`oja_eta` (oja mode)
- `diag_floor`

## What is currently exposed in config

Production path (via backend factory) currently exposes:

- `psgld`
- `sghmc`
- `rmsprop` or `lrd` preconditioning

There is an additional optimizer class in code (`AdaptiveDriftSGLDAdam`), but it is not currently selected by `inference.sampler.backend`.

## Practical tuning interpretation

- Increase `eps` to reduce extreme preconditioner amplification.
- Increase `beta` for smoother/slower preconditioner adaptation.
- In SGHMC, lower `sghmc_alpha` to reduce damping; raise it to damp oscillations.
- Use `freeze_preconditioner_sampling=true` for time-homogeneous Phase 4 kernels after adaptation is mature.
- Use `grad_clip_norm` as a safety guardrail when exploring higher learning rates.

See also:

- {doc}`sampler-health`
- {doc}`pcg-whitening-convergence`
