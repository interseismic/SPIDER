# SPIDER Documentation

SPIDER is a toolkit for probabilistic differential earthquake relocation with neural travel-time
surrogates and scalable MCMC inference.

A run has two commands. `spider locate-map` runs Phase 1 (a MAP relocation, optionally in two
passes) and writes a MAP catalog plus a "Phase-2 bundle". `spider sample` reads that bundle and
runs Phases 2–4 (preconditioner warm-up, noise ramp, posterior sampling), writing posterior samples
to HDF5.

Where to start:

- New to SPIDER: {doc}`quickstart` (the shipped example runs from `example/` with no edits),
  then {doc}`data-formats` to prepare your CSV inputs.
- Writing a config: {doc}`config-walkthrough` for an annotated tour of the template and the
  values to change first, {doc}`configuration` for the layout and the settings that matter most,
  {doc}`configuration-reference` for every key.
- Recipes: {doc}`EXAMPLES` for copy-paste command sequences and {doc}`EIKONET_TRAINING` for
  training the travel-time model.
- Running and reading the logs: {doc}`cli-workflow` and {doc}`outputs-and-formats`.
- Understanding the model: {doc}`probability-model` and {doc}`sampler-options`.
- Something looks wrong: {doc}`troubleshooting`, {doc}`sampler-health`, {doc}`diagnostics-catalog`.

```{toctree}
:maxdepth: 2
:caption: User Guide

quickstart
data-formats
configuration
configuration-reference
probability-model
sampler-options
cli-workflow
postprocessing-and-analysis-tools
outputs-and-formats
pcg-whitening-convergence
sampler-health
diagnostics-catalog
troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Recipes

config-walkthrough
EXAMPLES
EIKONET_TRAINING
```

```{toctree}
:maxdepth: 1
:caption: Reference

api-reference
citation
legacy-materials
```
