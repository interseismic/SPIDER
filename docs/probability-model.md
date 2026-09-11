# Probability Model and Symbol Mapping

This page defines the SPIDER probabilistic model in symbols, then maps each symbol to configuration keys.

Equations intentionally use symbols only (no config key names inside math).

## Quick equation summary

Use this block as a one-screen reference for the full model:

$$
r_n = d_n - \hat d_n,
\qquad
u_n = \frac{r_n}{\sigma_n}
$$

$$
\mathcal{L}_{\text{ind}} = \frac{1}{B}\sum_{n=1}^{B}\Big(\rho(u_n) + \log \sigma_n\Big)
$$

where $\rho(\cdot)$ is selected from Gaussian, Laplace, Student-$t$, or Huber.

$$
\mathcal{L}_{\text{corr}} =
\frac{1}{B}\left[\frac12\sum_g \mathbf{r}_g^\top \boldsymbol{\Sigma}_g^{-1}\mathbf{r}_g\right]
+
\frac{1}{B}\sum_{n=1}^{B}\log \sigma_n
$$

$$
\mathcal{J}
=
\mathcal{L}_{\text{data}}
+
\frac{1}{N}\big(-\log p(\Delta \mathbf{Z})\big),
\qquad
\mathcal{L}_{\text{data}}\in\{\mathcal{L}_{\text{ind}},\mathcal{L}_{\text{corr}}\}
$$

Operational phase policy:

- Phase 1 (MAP warmup) uses robust independent likelihoods to damp outlier influence and support outlier identification.
- Phases 2-4 (sampling) use the correlated Gaussian likelihood with shared-event structure.

Quick-summary notation:

- $d_n$: observed differential time for datum $n$
- $\hat d_n$: model-predicted differential time for datum $n$
- $r_n$: residual, $r_n=d_n-\hat d_n$
- $u_n$: standardized residual, $u_n=r_n/\sigma_n$
- $\sigma_n$: phase-dependent scale for datum $n$
- $\rho(\cdot)$: per-datum robust penalty (Gaussian, Laplace, Student-$t$, or Huber form)
- $B$: number of observations in the current likelihood batch
- $g$: station-phase group index in the correlated model
- $\mathbf{r}_g$: residual vector for group $g$
- $\boldsymbol{\Sigma}_g$: group covariance in the correlated model
- $N$: total number of observations in the full dataset
- $\Delta \mathbf{Z}$: stacked event perturbations across all events

## Notation conventions

The symbols below are used throughout the page:

| Symbol | Definition |
| --- | --- |
| $n$ | Observation index (differential-time row) |
| $i,j$ | Event indices |
| $M$ | Number of events |
| $B$ | Number of observations in a batch used by the likelihood term |
| $N$ | Total number of observations in the full dataset |
| $\mathbf{z}_i$ | Event state for event $i$ (space + origin-time component) |
| $\Delta \mathbf{z}_i$ | Perturbation for event $i$ |
| $\Delta \mathbf{Z}$ | Collection of all event perturbations $\{\Delta \mathbf{z}_i\}_{i=1}^M$ |
| $\mathbf{x}_i$ | Spatial part of event state for event $i$ |
| $t_i$ | Origin-time part of event state for event $i$ |
| $T(\mathbf{x}, s, \varphi)$ | Travel-time surrogate evaluated at event location $\mathbf{x}$, receiver $s$, phase $\varphi$ |
| $\sigma_P,\sigma_S$ | Phase-specific residual scales |
| $\tau_P,\tau_S$ | Phase-specific shared-event random-effect scales |
| $\mathbf{I}$ | Identity matrix of appropriate dimension |

## 1) Forward model and residuals

For each differential-time datum $n$, let:

- $(i_n, j_n)$ be the event pair
- $s_n$ be the receiver
- $\varphi_n \in \{P,S\}$ be the phase
- $d_n$ be the observed differential time

Write each event state as $\mathbf{z}_i=[\mathbf{x}_i^\top\; t_i]^\top$, where $\mathbf{x}_i$ is spatial location and $t_i$ is origin time.

Event state is represented as:

$$
\mathbf{z}_i = \mathbf{z}_i^{(0)} + \Delta \mathbf{z}_i,
\qquad
\Delta \mathbf{z}_i =
\begin{bmatrix}
\Delta x_i & \Delta y_i & \Delta z_i & \Delta t_i
\end{bmatrix}^{\!\top}
$$

Predicted differential time:

$$
\hat d_n =
\Big(T(\mathbf{x}_{j_n}, s_n, \varphi_n) + t_{j_n}\Big)
-
\Big(T(\mathbf{x}_{i_n}, s_n, \varphi_n) + t_{i_n}\Big)
$$

Residual:

$$
r_n = d_n - \hat d_n
$$

Phase-dependent scale:

$$
\sigma_n =
\begin{cases}
\sigma_P, & \varphi_n = P \\
\sigma_S, & \varphi_n = S
\end{cases}
$$

## 2) Independent residual likelihood family (Phase 1 robust path)

Define standardized residual $u_n = r_n / \sigma_n$.  
The per-observation negative log-likelihood is:

### Gaussian

$$
\ell_n^{\text{Gauss}} = \tfrac12 u_n^2 + \log \sigma_n
$$

### Laplace

$$
\ell_n^{\text{Lap}} = |u_n| + \log \sigma_n
$$

### Student-$t$ (fixed degrees of freedom $\nu$)

$$
\ell_n^{t} =
\tfrac{\nu+1}{2}\log\!\left(1+\frac{u_n^2}{\nu}\right)
+ C(\nu)
+ \log \sigma_n
$$

where $C(\nu)$ is the Student-$t$ normalization constant (depends only on $\nu$).

### Huber (threshold $\delta_H$)

$$
\ell_n^{\text{Huber}} =
h_{\delta_H}(u_n) + \log \sigma_n
$$

with

$$
h_{\delta_H}(u)=
\begin{cases}
\tfrac12 u^2, & |u| \le \delta_H \\
\delta_H\left(|u|-\tfrac12\delta_H\right), & |u|>\delta_H
\end{cases}
$$

Batch-average independent likelihood term:

$$
\mathcal{L}_{\text{ind}} = \frac{1}{B}\sum_{n=1}^{B}\ell_n
$$

## 3) Correlated shared-event likelihood (Phases 2-4 sampling path)

For each station-phase group $g$, let $\mathbf{r}_g$ be grouped residuals and $\mathbf{b}_g$ latent event effects:

$$
\mathbf{r}_g = \mathbf{B}_g \mathbf{b}_g + \boldsymbol{\varepsilon}_g
$$

Here $\mathbf{B}_g$ is the signed incidence operator mapping event-level latent terms to edge-level residual contributions within group $g$.

$$
\mathbf{b}_g \sim \mathcal{N}(\mathbf{0}, \tau_g^2 \mathbf{I}),
\qquad
\boldsymbol{\varepsilon}_g \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{I})
$$

With optional edge weighting, using weighted incidence $\widetilde{\mathbf{B}}_g$:

$$
\boldsymbol{\Sigma}_g
=
\sigma_g^2 \mathbf{I}
+
\tau_g^2 \widetilde{\mathbf{B}}_g \widetilde{\mathbf{B}}_g^{\!\top}
$$

Current collapsed quadratic term:

$$
\mathcal{Q}_{\text{corr}}
=
\frac12\sum_g
\mathbf{r}_g^{\!\top}\boldsymbol{\Sigma}_g^{-1}\mathbf{r}_g
$$

Implemented sampling loss contribution:

$$
\mathcal{L}_{\text{corr}}
=
\frac{1}{B}\mathcal{Q}_{\text{corr}}
+
\frac{1}{B}\sum_{n=1}^{B}\log \sigma_n
$$

Note: this path currently uses the correlated quadratic form (whitening/PCG solve) and does not include a correlated log-determinant term.

## 4) Priors

### Event perturbation prior (default diagonal Gaussian)

$$
\Delta \mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{S}),
\qquad
\mathbf{S}=\operatorname{diag}(s_x^2,s_y^2,s_z^2,s_t^2)
$$

### Optional centroid prior

$$
\bar{\Delta \mathbf{z}} = \frac{1}{M}\sum_{i=1}^{M}\Delta \mathbf{z}_i,
\qquad
\bar{\Delta \mathbf{z}} \sim \mathcal{N}(\mathbf{0}, \mathbf{C})
$$

with

$$
\mathbf{C}=\operatorname{diag}(c_x^2,c_y^2,c_z^2,c_t^2)
$$

### Optional hierarchical event precision prior

$$
\Delta \mathbf{z}_i \mid \mathbf{\Lambda}_{k(i)}
\sim \mathcal{N}\!\left(\mathbf{0}, \mathbf{\Lambda}_{k(i)}^{-1}\right)
$$

$$
\mathbf{\Lambda}_{k} \sim \operatorname{Wishart}(\nu_0,\mathbf{V}_0)
$$

where $k(i)$ maps event $i$ to its cluster index, $\mathbf{V}_0$ is constructed from scale hyperparameters, and $\nu_0$ is the Wishart degrees of freedom.

## 5) Training objective (negative log posterior)

SPIDER uses a per-observation normalized objective:

$$
\mathcal{J}
=
\mathcal{L}_{\text{data}}
+
\frac{1}{N}\big(-\log p(\Delta \mathbf{Z})\big)
$$

where:

- $\mathcal{L}_{\text{data}} = \mathcal{L}_{\text{ind}}$ for independent-likelihood runs
- $\mathcal{L}_{\text{data}} = \mathcal{L}_{\text{corr}}$ when collapsed shared-event correlation is enabled
- $N$ is the total number of observations in the full dataset

## 6) Symbol-to-config mapping

### Likelihood symbols

| Symbol | Meaning | Config key(s) |
| --- | --- | --- |
| $\sigma_P, \sigma_S$ | Phase-dependent residual scales. **Fixed** — never learned. | `model.likelihoods.locate_map.phase_unc`, `model.likelihoods.sample.phase_unc` |
| $\nu$ | Student-$t$ degrees of freedom (Phase 1 only) | `model.likelihoods.locate_map.student_t.nu` (default 4.0) |
| $\delta_H$ | Huber threshold (Phase 1 only) | `model.likelihoods.locate_map.huber_delta` (default 1.0) |
| Likelihood family selector | `huber` (default), `laplace`/`l1`/`mae`, `gaussian`/`l2`/`mse`, `student_t`. Unrecognized values fall through to Huber. | `model.likelihoods.locate_map.type` |
| Sampling likelihood selector | **Must** be `correlated_gaussian` (or `correlated`); any other value is a configuration error. | `model.likelihoods.sample.type` |

There is no `learn_noise_scale` key and no `model.priors.noise` block; both were removed.
$\sigma_P,\sigma_S$ come from `phase_unc` and are held fixed in all phases; the sampler's parameter
list contains only the event perturbations. `model.likelihoods.sample.student_t.nu` and
`model.likelihoods.sample.huber_delta` are parsed but have no effect, because the sampling path
is always the correlated Gaussian.

### Shared-event correlated symbols

| Symbol | Meaning | Config key(s) |
| --- | --- | --- |
| $\tau_P, \tau_S$ | Phase-specific latent RE scales | `model.likelihoods.sample.shared_event_re.model.tau_s` |
| Group definition | Grouping for the correlated solve. **Only `station_phase` is supported**; `phase` raises at run time. | `model.likelihoods.sample.shared_event_re.model.group_by` |
| Edge-weight model (inside $\widetilde{\mathbf{B}}_g$) | Distance-based weighting mode | `model.likelihoods.sample.shared_event_re.edge_weights.mode` |
| Weight length scale | RBF weight scale | `model.likelihoods.sample.shared_event_re.edge_weights.ell_km` |
| Weight power/scale | Power-law weight controls | `model.likelihoods.sample.shared_event_re.edge_weights.power`, `model.likelihoods.sample.shared_event_re.edge_weights.scale_km` |
| Global weight multiplier | Global edge-weight factor | `model.likelihoods.sample.shared_event_re.edge_weights.global_scale` |
| Numerical jitter | Stabilization added to the node system | `model.likelihoods.sample.shared_event_re.numerics.jitter0`, `model.likelihoods.sample.shared_event_re.numerics.jitter_max` |
| Solver tolerance / iterations | PCG stopping controls | `model.likelihoods.sample.shared_event_re.solver.tol`, `model.likelihoods.sample.shared_event_re.solver.max_iters`, `model.likelihoods.sample.shared_event_re.solver.min_iters` |

### Prior symbols

| Symbol | Meaning | Config key(s) |
| --- | --- | --- |
| $s_x,s_y,s_z,s_t$ | Event prior standard deviations | `model.priors.event.params.std` |
| $c_x,c_y,c_z,c_t$ | Centroid prior standard deviations | `model.priors.centroid.params.std` |
| $\nu_0$ | Wishart hyperprior degrees of freedom | `model.priors.event.hyper.params.df` |
| $\mathbf{V}_0$ scale controls | Wishart scale hyperparameters | `model.priors.event.hyper.params.scale_std` |
| Hyperprior update cadence | Epoch cadence for precision updates | `model.priors.event.hyper.update.every_epochs` |

Configuration constraints for the hierarchical prior (`model.priors.event.hyper`): `type` must be
`wishart_precision`; `params.df` must be **> 3** (Wishart dof constraint for the 4-D event prior);
`params.scale_std` (length 4) and `update.every_epochs` (>= 1) are required when `enabled = true`.
Per-phase scheduling was removed: `model.priors.event.schedule` and
`model.priors.event.hyper.update.active_phases` are rejected, and priors/hyper-updates are active
in all phases when enabled. The priors block must live under `model.priors`; a top-level `priors`
block is an error.

## 7) Which likelihood is active in each stage

- Phase 1 (MAP warmup / outlier-screening stage) uses the `locate_map` likelihood block.
  Robust independent families (Laplace, Huber, Student-$t$) reduce sensitivity to outliers and
  support residual-based outlier identification.
- Phase 1 may run **twice**: with `inference.sampler.phase1_two_pass.enabled = true`, MAP runs,
  the post-MAP filters (residual, pair/station ratio, linearization error) are applied at the
  relocated positions, and MAP runs again on the filtered data. The pass-1 catalog is kept as
  `<catalog_outfile>_MAP_pass1.csv`.
- The residual outlier filter defaults to `model.filters.residual.phase = "after_phase1"`, i.e.
  residuals are screened at the MAP locations rather than at the initial catalog.
- Phases 2-4 use the `sample` likelihood block, whose `type` must be `correlated_gaussian`, with
  shared-event whitening.
- Setting `model.likelihoods.sample.shared_event_re.enabled = false` keeps the required
  `correlated_gaussian` type but reduces the sampling path to the independent Gaussian form.
