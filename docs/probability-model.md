# Probability Model and Symbol Mapping

This page defines the SPIDER probabilistic model in symbols, then maps each symbol to configuration
keys. Every formula matches the implementation in `spider/core/modeling.py`
(`compute_travel_times`, `compute_likelihood_loss`, `compute_prior_loss`, `total_loss`) and the
hierarchical update in `spider/core/epoch_runner.py` / `spider/core/hierarchy.py`.

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

- Phase 1 (MAP) uses a robust independent likelihood to damp outlier influence and support outlier identification.
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
- $N$: total number of observations in the full (filtered) dataset
- $\Delta \mathbf{Z}$: stacked event perturbations across all events

## Notation conventions

The symbols below are used throughout the page:

| Symbol | Definition |
| --- | --- |
| $n$ | Observation index (differential-time row) |
| $i,j$ | Event indices |
| $M$ | Number of events |
| $k$ | Connected component of the event-pair graph; $n_k$ its number of events, $k(i)$ the component of event $i$ |
| $B$ | Number of observations in a batch used by the likelihood term |
| $N$ | Total number of observations in the full dataset |
| $\mathbf{z}_i$ | Event state for event $i$ (space + origin-time component) |
| $\Delta \mathbf{z}_i$ | Perturbation for event $i$ |
| $\Delta \mathbf{Z}$ | Collection of all event perturbations $\{\Delta \mathbf{z}_i\}_{i=1}^M$ |
| $\mathbf{x}_i$ | Spatial part of event state for event $i$ (km, local projection) |
| $t_i$ | Origin-time part of event state for event $i$ (s) |
| $T(\mathbf{x}, s, \varphi)$ | Travel-time surrogate evaluated at event location $\mathbf{x}$, receiver $s$, phase $\varphi$ |
| $\sigma_P,\sigma_S$ | Phase-specific residual scales |
| $\tau_P,\tau_S$ | Phase-specific shared-event random-effect scales |
| $\mathbf{I}$ | Identity matrix of appropriate dimension |

## 1) Forward model and residuals

For each differential-time datum $n$, let:

- $(i_n, j_n)$ be the event pair, in the order of the `evid1`, `evid2` columns of the dtime file
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

with $\mathbf{z}_i^{(0)}$ the starting catalog and $\Delta \mathbf{z}_i = \mathbf{0}$ initially. Only
the perturbations are sampled; $t_i^{(0)}$ enters through $d_n$, so $\Delta t_i$ is the origin-time
correction.

Predicted differential time (`evid2` minus `evid1`):

$$
\hat d_n =
\Big(T(\mathbf{x}_{j_n}, s_n, \varphi_n) + \Delta t_{j_n}\Big)
-
\Big(T(\mathbf{x}_{i_n}, s_n, \varphi_n) + \Delta t_{i_n}\Big)
$$

If your dtime file uses the opposite convention, set `model.filters.dtimes.flip_dt_sign: true`.

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

$\sigma_P,\sigma_S$ are fixed inputs (`phase_unc`); they are never estimated.

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
+ \log \sigma_n,
\qquad
C(\nu) = \tfrac12\log(\nu\pi) + \log\Gamma\!\left(\tfrac{\nu}{2}\right) - \log\Gamma\!\left(\tfrac{\nu+1}{2}\right)
$$

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

$\delta_H$ is expressed in units of the standardized residual, so `huber_delta = 1.0` switches to
the linear regime at one $\sigma$.

Batch-average independent likelihood term:

$$
\mathcal{L}_{\text{ind}} = \frac{1}{B}\sum_{n=1}^{B}\ell_n
$$

Because $\sigma_n$ is fixed, the $\log\sigma_n$ terms are constants; they are kept so that reported
loss values are comparable across `phase_unc` settings.

## 3) Correlated shared-event likelihood (Phases 2-4 sampling path)

For each station-phase group $g$ (all rows of a batch that share the same receiver and phase),
let $\mathbf{r}_g$ be the grouped residuals and $\mathbf{b}_g$ latent per-event effects:

$$
\mathbf{r}_g = \widetilde{\mathbf{B}}_g \mathbf{b}_g + \boldsymbol{\varepsilon}_g
$$

Here $\widetilde{\mathbf{B}}_g$ is the (optionally weighted) signed incidence operator: row $n$ has
$+\sqrt{w_n}$ at event $j_n$ and $-\sqrt{w_n}$ at event $i_n$, with $w_n = 1$ for uniform edge
weights and a distance-dependent weight otherwise. It maps event-level latent terms (for example a
station-specific travel-time bias that is common to all rays from nearby events to that receiver)
to edge-level residual contributions within group $g$.

$$
\mathbf{b}_g \sim \mathcal{N}(\mathbf{0}, \tau_g^2 \mathbf{I}),
\qquad
\boldsymbol{\varepsilon}_g \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{I})
$$

with $\tau_g \in \{\tau_P, \tau_S\}$ and $\sigma_g \in \{\sigma_P, \sigma_S\}$ according to the
group's phase. Marginalizing $\mathbf{b}_g$ gives

$$
\boldsymbol{\Sigma}_g
=
\sigma_g^2 \mathbf{I}
+
\tau_g^2 \widetilde{\mathbf{B}}_g \widetilde{\mathbf{B}}_g^{\!\top}
$$

Collapsed quadratic term:

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

Implementation notes (all in `compute_likelihood_loss` and `shared_event_re_whitening.py`):

- $\boldsymbol{\Sigma}_g^{-1}\mathbf{r}_g$ is obtained through the Woodbury identity: a solve on the
  node system $\big(\tau_g^{-2}\mathbf{I} + \sigma_g^{-2}\widetilde{\mathbf{B}}_g^{\!\top}\widetilde{\mathbf{B}}_g\big)$,
  whose matrix is the weighted graph Laplacian of the group's event pairs, solved with
  Jacobi-preconditioned CG, batched over groups of similar size (see
  {doc}`pcg-whitening-convergence`). The gradient is taken as $\partial\mathcal{Q}/\partial\mathbf{r}_g = \boldsymbol{\Sigma}_g^{-1}\mathbf{r}_g$
  without differentiating through the solver.
- The correlated log-determinant $\tfrac12\log|\boldsymbol{\Sigma}_g|$ is **not** included; since
  $\boldsymbol{\Sigma}_g$ depends on $\Delta\mathbf{Z}$ only through the optional distance-based
  edge weights, this is exact for uniform weights and an approximation otherwise.
- Groups with $\tau_g = 0$, more rows than `limits.max_rows`, or more events than
  `limits.max_nodes` fall back to the diagonal form $\tfrac12\mathbf{r}_g^\top\mathbf{r}_g/\sigma_g^2$
  (counted as `fallback` in the whitening statistics; `fallback.abort_on_pcg_fallback` turns this
  into an error).
- With `model.cluster.mode` set to `component` or `dd_khop`, rows whose two events lie in different
  clusters are removed from the correlated solve and contribute the diagonal form instead.
- The optional `station_phase_term` adds a second collapsed effect, a single scalar offset per
  station-phase group with scale $\tau^{\text{sp}}_g$, layered on the per-event effect. It is
  folded into the **same** covariance,
  $\boldsymbol{\Sigma}_g = \sigma_g^2 \mathbf{I} + \tau_g^2 \widetilde{\mathbf{B}}_g \widetilde{\mathbf{B}}_g^{\!\top} + (\tau^{\text{sp}}_g)^2 \mathbf{1}\mathbf{1}^{\!\top}$,
  and evaluated through the Sherman-Morrison identity on top of the shared-event solve:
  with $\boldsymbol{\Sigma}_{0,g}$ the covariance without the station term,
  $\mathbf{u}_1 = \boldsymbol{\Sigma}_{0,g}^{-1}\mathbf{1}$ (a second PCG solve with a ones
  right-hand side) and $s_g = \mathbf{u}_1^{\!\top}\mathbf{r}_g$,
  $$\tfrac12\mathbf{r}_g^{\!\top}\boldsymbol{\Sigma}_g^{-1}\mathbf{r}_g
  = \tfrac12\mathbf{r}_g^{\!\top}\boldsymbol{\Sigma}_{0,g}^{-1}\mathbf{r}_g
  - \tfrac12\,\frac{(\tau^{\text{sp}}_g)^2\, s_g^2}{1 + (\tau^{\text{sp}}_g)^2\,\mathbf{1}^{\!\top}\mathbf{u}_1}.$$
  Only the (negative) correction is added to $\mathcal{Q}_{\text{corr}}$; its gradient
  $-(\tau^{\text{sp}}_g)^2 s_g\,\mathbf{u}_1/(1 + (\tau^{\text{sp}}_g)^2\mathbf{1}^{\!\top}\mathbf{u}_1)$
  is exact and the loss is continuous at $\tau^{\text{sp}}_g \to 0$. Groups evaluated with the
  diagonal fallback use $\mathbf{u}_1 = \sigma_g^{-2}\mathbf{1}$, and rows removed from the
  correlated solve by `model.cluster.mode` keep the diagonal $\mathbf{u}_1$ but still belong to
  their group's offset. (Before this fix the term was implemented as the full quadratic
  $\tfrac12\mathbf{r}_g^\top(\sigma_g^2\mathbf{I} + (\tau^{\text{sp}}_g)^2\mathbf{1}\mathbf{1}^\top)^{-1}\mathbf{r}_g$
  **added** to the shared-event quadratic, which counted the diagonal $\sigma_g^2\mathbf{I}$
  part twice and jumped by $\tfrac12\mathbf{r}_g^\top\mathbf{r}_g/\sigma_g^2$ as soon as
  $\tau^{\text{sp}}_g > 0$; `tests/test_station_phase_term.py` checks the current form against a
  dense covariance.) The term is off by default and not used in the shipped configs.

## 4) Priors

All priors act on the perturbations $\Delta\mathbf{Z}$ and are active in every phase when enabled
(per-phase scheduling was removed).

### Event perturbation prior

Each event's shift is Gaussian with a component-specific precision matrix
$\boldsymbol{\Lambda}_{k(i)}$:

$$
\Delta \mathbf{z}_i \sim \mathcal{N}\!\left(\mathbf{0}, \boldsymbol{\Lambda}_{k(i)}^{-1}\right),
\qquad
-\log p(\Delta\mathbf{Z}\mid\boldsymbol\Lambda)
=
\frac12\sum_{i=1}^{M}\Delta\mathbf{z}_i^{\top}\boldsymbol{\Lambda}_{k(i)}\Delta\mathbf{z}_i
-\frac12\sum_{k} n_k\log|\boldsymbol{\Lambda}_k|
+\text{const}
$$

Without the hierarchical prior (or before its first update), every component uses the same fixed
diagonal precision built from the configured standard deviations:

$$
\boldsymbol{\Lambda}_k = \mathbf{S}^{-1},
\qquad
\mathbf{S}=\operatorname{diag}(s_x^2,s_y^2,s_z^2,s_t^2)
$$

so the prior reduces to the familiar independent Gaussian
$\tfrac12\sum_i\sum_c (\Delta z_{i,c}/s_c)^2 + \text{const}$.

### Centroid (gauge) prior, per connected component

Differential times constrain only relative positions and times within a connected component of
the pair graph; a rigid translation (in space and time) of a whole component leaves the likelihood
unchanged. The centroid prior pins this gauge. For each component $k$,

$$
\bar{\Delta \mathbf{z}}_k = \frac{1}{n_k}\sum_{i \in k}\Delta \mathbf{z}_i,
\qquad
-\log p_{\text{cent}}(\Delta\mathbf{Z})
=
\sum_k n_k\left[\frac12\,\bar{\Delta \mathbf{z}}_k^{\top}\mathbf{C}^{-1}\bar{\Delta \mathbf{z}}_k + \frac12\log|2\pi\mathbf{C}|\right]
$$

with

$$
\mathbf{C}=\operatorname{diag}(c_x^2,c_y^2,c_z^2,c_t^2)
$$

The log-density of each component mean is weighted by its event count $n_k$, so the effective
prior width on the mean of a component with $n_k$ events is $c/\sqrt{n_k}$, and a single
loosely-connected satellite event cannot drag the main cluster. Components that share no
differential times are not coupled; there is no global (all-events) form.

### Hierarchical event precision prior (recommended)

The component precisions are given a conjugate Wishart hyperprior and learned from the data:

$$
\boldsymbol{\Lambda}_{k} \sim \operatorname{Wishart}(\nu_0,\mathbf{V}_0),
\qquad
\mathbf{V}_0^{-1} = \nu_0\operatorname{diag}(v_x^2, v_y^2, v_z^2, v_t^2)
$$

so that the prior mean $\mathbb{E}[\boldsymbol{\Lambda}_k]=\nu_0\mathbf{V}_0 = \operatorname{diag}(v_c^{-2})$
corresponds to event-shift standard deviations $(v_x,v_y,v_z,v_t)$ = `scale_std`, and $\nu_0$ =
`df` is the pseudo-count of events that this prior guess is worth ($\nu_0 > 3$ is required for a
proper 4-D Wishart).

Given the current shifts of the $n_k$ events in component $k$, with scatter matrix
$\mathbf{S}_k=\sum_{i\in k}\Delta\mathbf{z}_i\Delta\mathbf{z}_i^{\top}$, the conditional posterior is
again Wishart and SPIDER draws from it (a Gibbs step):

$$
\boldsymbol{\Lambda}_{k}\mid\Delta\mathbf{Z}_k \sim
\operatorname{Wishart}\!\big(\nu_0 + n_k,\ (\mathbf{V}_0^{-1} + \mathbf{S}_k)^{-1}\big)
$$

Schedule and pooling:

- The draw happens at the end of every epoch with $e \ge$ `update.start_epoch` and
  $e \bmod$ `update.every_epochs` $= 0$, in **every** phase (each MAP pass, and Phases 2-4), for
  every component at once. Between updates $\boldsymbol{\Lambda}_k$ is held fixed, so the sampler
  sees a fixed Gaussian prior within each block of epochs.
- Before the first update, $\boldsymbol{\Lambda}_k = \mathbf{S}^{-1}$ from the fixed `params.std`.
  `start_epoch` must therefore be large enough that the MAP has moved the events off
  $\Delta\mathbf{Z}=\mathbf{0}$; otherwise the first draw collapses onto a very tight prior.
- Components with fewer than `update.min_cluster_events` events do not get their own draw; they
  use a draw from the posterior formed with **all** events ($\mathbf{S}=\sum_i\Delta\mathbf{z}_i\Delta\mathbf{z}_i^\top$,
  $n = M$). A handful of shifts cannot pin a 4×4 precision, and a small component that drifts would
  otherwise loosen its own prior and keep drifting.
- The learned $\{\boldsymbol{\Lambda}_k\}$ are saved in every checkpoint and in the Phase-2 bundle
  written by `spider locate-map`; `spider sample` adopts them so that sampling starts under the
  learned prior rather than the loose `params.std`. The console reports the implied standard
  deviations of the largest component, e.g.
  `[PRIORS] Hierarchical event prior updated: std≈[0.31,0.29,0.42,0.05] corr_zt≈-0.12 [largest of 7 clusters, n_events=5120]`.

This is the preferred formulation. A fixed $\mathbf{S}$ has to serve two incompatible roles: the
MAP stage wants it loose so the differential data, not the starting catalog, sets the fine
structure; the sampling stages want it realistic, because the injected noise otherwise walks weakly
constrained events away from the data (and, through the centroid prior, shifts every other event).
On real catalogs the learned shift scale is typically a few hundred metres, several times tighter
than a safe fixed prior and tighter than one would dare set by hand.

## 5) Training objective (negative log posterior)

SPIDER uses a per-observation normalized objective:

$$
\mathcal{J}
=
\mathcal{L}_{\text{data}}
+
\frac{1}{N}\Big(-\log p(\Delta\mathbf{Z}\mid\boldsymbol\Lambda) - \log p_{\text{cent}}(\Delta\mathbf{Z})\Big)
$$

where:

- $\mathcal{L}_{\text{data}} = \mathcal{L}_{\text{ind}}$ in Phase 1 (`locate_map` block)
- $\mathcal{L}_{\text{data}} = \mathcal{L}_{\text{corr}}$ in Phases 2-4 (`sample` block); with
  `shared_event_re.enabled = false` it reduces to $\mathcal{L}_{\text{ind}}$ with the Gaussian $\rho$
- $N$ is the total number of observations in the full dataset **after all filters**, including the
  post-Phase-1 filters; the prior terms are evaluated over all $M$ events in every minibatch

Multiplying by $N$ recovers the full negative log posterior (up to the batch-mean estimate of the
data sum); this is what the samplers do internally (drift $= N\bar g$, see {doc}`sampler-options`),
so Phase 4 samples the posterior $p(\Delta\mathbf{Z}\mid d)\propto\exp(-N\mathcal{J})$, not a
tempered version of it. Phase 1 minimizes $\mathcal{J}$ with Adam to obtain the MAP.

## 6) Symbol-to-config mapping

### Likelihood symbols

| Symbol | Meaning | Config key(s) |
| --- | --- | --- |
| $\sigma_P, \sigma_S$ | Phase-dependent residual scales (s). **Fixed** -- never learned. | `model.likelihoods.locate_map.phase_unc`, `model.likelihoods.sample.phase_unc` |
| $\nu$ | Student-$t$ degrees of freedom (Phase 1 only) | `model.likelihoods.locate_map.student_t.nu` (default 4.0) |
| $\delta_H$ | Huber threshold in units of $u_n$ (Phase 1 only) | `model.likelihoods.locate_map.huber_delta` (default 1.0) |
| $\rho$ selector | `huber` (default), `laplace`/`l1`/`mae`, `l2`/`gaussian`/`mse`, `student_t`/`student-t`. Unrecognized values fall through to Huber. | `model.likelihoods.locate_map.type` |
| Sampling likelihood selector | **Must** be `correlated_gaussian` (or `correlated`); any other value is a configuration error. | `model.likelihoods.sample.type` |

There is no `learn_noise_scale` key and no `model.priors.noise` block; both were removed.
$\sigma_P,\sigma_S$ come from `phase_unc` and are held fixed in all phases; the sampler's parameter
list contains only the event perturbations. `model.likelihoods.sample.student_t.nu` and
`model.likelihoods.sample.huber_delta` are parsed but have no effect, because the sampling path
is always the correlated Gaussian.

### Shared-event correlated symbols

| Symbol | Meaning | Config key(s) |
| --- | --- | --- |
| $\tau_P, \tau_S$ | Phase-specific latent RE scales (s); both must be $> 0$ for the correlated form to be used | `model.likelihoods.sample.shared_event_re.model.tau_s` |
| Group definition | Grouping for the correlated solve. **Only `station_phase` is supported**; `phase` raises at run time. | `model.likelihoods.sample.shared_event_re.model.group_by` |
| Cluster restriction | `none`, `component`, `dd_khop` (+ `k`): rows crossing clusters use the diagonal form | `model.likelihoods.sample.shared_event_re.model.cluster.mode`, `.cluster.k` |
| Edge-weight model (inside $\widetilde{\mathbf{B}}_g$) | `uniform`, `distance_rbf`, `distance_linear`, `distance_power` | `model.likelihoods.sample.shared_event_re.edge_weights.mode` |
| Weight length scale | RBF weight scale (km) | `model.likelihoods.sample.shared_event_re.edge_weights.ell_km` |
| Weight power/scale | Power-law weight controls | `model.likelihoods.sample.shared_event_re.edge_weights.power`, `model.likelihoods.sample.shared_event_re.edge_weights.scale_km` |
| Global weight multiplier | Global edge-weight factor | `model.likelihoods.sample.shared_event_re.edge_weights.global_scale` |
| $\tau^{\text{sp}}_P,\tau^{\text{sp}}_S$ | Optional station-phase offset, folded into $\boldsymbol{\Sigma}_g$ as a rank-one term | `model.likelihoods.sample.shared_event_re.station_phase_term.enabled`, `.tau_s` |
| Numerical jitter | Stabilization added to the node system | `model.likelihoods.sample.shared_event_re.numerics.jitter0`, `model.likelihoods.sample.shared_event_re.numerics.jitter_max` |
| Solver tolerance / iterations | PCG stopping controls | `model.likelihoods.sample.shared_event_re.solver.tol`, `model.likelihoods.sample.shared_event_re.solver.max_iters`, `model.likelihoods.sample.shared_event_re.solver.min_iters` |
| Fallback caps | Groups above these sizes use the diagonal form | `model.likelihoods.sample.shared_event_re.limits.max_nodes`, `.limits.max_rows` |

### Prior symbols

| Symbol | Meaning | Config key(s) |
| --- | --- | --- |
| $s_x,s_y,s_z,s_t$ | Fixed event prior standard deviations (km, km, km, s); also the initial $\boldsymbol{\Lambda}_k^{-1/2}$ of the hierarchical prior | `model.priors.event.params.std` |
| $c_x,c_y,c_z,c_t$ | Centroid prior standard deviations per component mean | `model.priors.centroid.params.std` |
| $\nu_0$ | Wishart hyperprior degrees of freedom (pseudo-count) | `model.priors.event.hyper.params.df` |
| $v_x,v_y,v_z,v_t$ | Wishart prior-mean shift scales ($\mathbf{V}_0^{-1}=\nu_0\operatorname{diag}(v_c^2)$) | `model.priors.event.hyper.params.scale_std` |
| Hyperprior update cadence | Epoch cadence for the Gibbs draws | `model.priors.event.hyper.update.every_epochs` |
| Hyperprior warm-up / pooling | First update epoch; min component size for its own draw | `model.priors.event.hyper.update.start_epoch`, `update.min_cluster_events` |

Recommended settings (used by `example/SPIDER_example.json` and
`spider/examples/params_template.json`): `df = 10`, `scale_std = [0.5, 0.5, 0.5, 0.2]`,
`every_epochs = 10`, `start_epoch = 200`, `min_cluster_events = 100`, with a loose base
`params.std` such as `[5, 5, 5, 2]` and a tight centroid prior `[0.01, 0.01, 0.01, 0.01]`.

Configuration constraints for the hierarchical prior (`model.priors.event.hyper`): `type` must be
`wishart_precision`; `params.df` must be **> 3** (Wishart dof constraint for the 4-D event prior);
`params.scale_std` (length 4) and `update.every_epochs` (>= 1) are required when `enabled = true`;
`update.start_epoch` (>= 0, default 0) delays the first update so the scale is not estimated from ΔX≈0.
`update.min_cluster_events` (>= 0, default 0) makes connected components smaller than this use the
precision drawn from all events rather than their own Wishart posterior (recommended ~100 when the
graph has small satellite components: their own estimate is prior-dominated and, once the component
drifts, loosens further).
Per-phase scheduling was removed: `model.priors.event.schedule` and
`model.priors.event.hyper.update.active_phases` are rejected, and priors/hyper-updates are active
in all phases when enabled. `model.priors.centroid.scope` is rejected: the centroid prior is
always per connected component. The priors block must live under `model.priors`; a top-level
`priors` block is an error.

## 7) Which likelihood is active in each stage

- Phase 1 (MAP / outlier-screening stage) uses the `locate_map` likelihood block.
  Robust independent families (Laplace, Huber, Student-$t$) reduce sensitivity to outliers and
  support residual-based outlier identification.
- Phase 1 may run **twice**: with `inference.sampler.phase1_two_pass.enabled = true`, MAP runs,
  the post-MAP filters (residual, pair/station ratio, linearization error, then the post-Phase-1
  event filter) are applied at the relocated positions, and MAP runs again on the filtered data.
  The pass-1 catalog is kept as `<catalog_outfile>_MAP_pass1.csv`.
- The residual outlier filter defaults to `model.filters.residual.phase = "after_phase1"`, i.e.
  residuals are screened at the MAP locations rather than at the initial catalog.
- Phases 2-4 use the `sample` likelihood block, whose `type` must be `correlated_gaussian`, with
  shared-event whitening. $N$ in the objective is the row count after the post-Phase-1 filters.
- Setting `model.likelihoods.sample.shared_event_re.enabled = false` keeps the required
  `correlated_gaussian` type but reduces the sampling path to the independent Gaussian form.
- The priors (event, centroid, hierarchical updates) are identical in all phases; only the data
  term changes.
