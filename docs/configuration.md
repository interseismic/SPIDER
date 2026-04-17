# Configuration Overview

SPIDER uses strict nested JSON configuration validated through `config_v2`.

Core modules:

- `spider/core/config_v2/validate.py`
- `spider/core/config_v2/load.py`
- `spider/core/config_v2/legacy_bridge.py`

## Top-level blocks

## `io`

Input/output file paths and checkpoint controls.

## `model`

- EikoNet model path and loader settings.
- Domain scaling and bounds.
- Priors.
- Likelihoods for MAP and sampling phases.
- Data filters.

## `inference`

- Sampler backend and hyperparameters.
- Batch settings.
- Runtime and safety controls.

## `observability`

- W&B run controls.
- Diagnostics group toggles.

## High-impact settings to validate early

- `model.likelihoods.sample.type` should match your intended sampling model.
- `model.likelihoods.sample.shared_event_re.enabled` should be explicitly set.
- `inference.sampler.backend` should be either `psgld` or `sghmc`.
- `inference.batching.standard.shuffle` should typically be `true` during production sampling.
- `inference.sampler.freeze_preconditioner_sampling` should match your Phase 4 policy.

For shared-event PCG setup and convergence checks, see {doc}`pcg-whitening-convergence`.

For a field-by-field schema reference, see {doc}`configuration-reference`.

For a symbol-level probability-model description with config-to-symbol mapping, see {doc}`probability-model`.
