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

## 1) Forward model and residuals

For each differential-time datum $n$, let:

- $(i_n, j_n)$ be the event pair
- $s_n$ be the receiver
- $\varphi_n \in \{P,S\}$ be the phase
- $d_n$ be the observed differential time

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

## 2) Independent residual likelihood family

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

where $C(\nu)$ is a $\nu$-dependent constant.

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

## 3) Correlated shared-event likelihood (sampling path)

For each station-phase group $g$, let $\mathbf{r}_g$ be grouped residuals and $\mathbf{b}_g$ latent event effects:

$$
\mathbf{r}_g = \mathbf{B}_g \mathbf{b}_g + \boldsymbol{\varepsilon}_g
$$

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

where $\mathbf{V}_0$ is constructed from scale hyperparameters and $\nu_0$ is the Wishart degrees of freedom.

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
| $\sigma_P, \sigma_S$ | Phase-dependent residual scales | `model.likelihoods.locate_map.phase_unc`, `model.likelihoods.sample.phase_unc` |
| $\nu$ | Student-$t$ degrees of freedom | `model.likelihoods.locate_map.student_t.nu`, `model.likelihoods.sample.student_t.nu` |
| $\delta_H$ | Huber threshold | `model.likelihoods.locate_map.huber_delta`, `model.likelihoods.sample.huber_delta` |
| Likelihood family selector | Choice among Gaussian/Laplace/Student-$t$/Huber | `model.likelihoods.locate_map.type` |
| Correlated sampling likelihood selector | Enables correlated Gaussian sampling path | `model.likelihoods.sample.type` |

### Shared-event correlated symbols

| Symbol | Meaning | Config key(s) |
| --- | --- | --- |
| $\tau_P, \tau_S$ | Phase-specific latent RE scales | `model.likelihoods.sample.shared_event_re.model.tau_s` |
| Group definition | Grouping strategy for correlated solve | `model.likelihoods.sample.shared_event_re.model.group_by` |
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

## 7) Which likelihood is active in each stage

- MAP stage uses the `locate_map` likelihood block.
- Sampling stages use the `sample` likelihood block.
- If collapsed shared-event correlation is enabled in the sampling block, the correlated quadratic path is used; otherwise the independent residual form is used.
