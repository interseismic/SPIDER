## SPIDER package

This is the core Python package for SPIDER. The main entrypoint is the CLI:

```bash
python -m spider --help
```

### Core workflow

```bash
# Phase 1 (MAP) -> writes <checkpoint_dir>/phase2_bundle.pth by default
python -m spider locate-map path/to/params.json --device 0

# Phase 2–4 (sampling) from the Phase‑2 bundle
python -m spider sample path/to/params.json --device 0

# Multi‑GPU independent chains (one process per device)
python -m spider sample-multi path/to/params.json --devices 0,1,2,3
```

### Configuration schema

SPIDER uses a **strict nested JSON** schema:

- Schema/validation: `spider/core/config_schema.py`
- Priors validation: `spider/core/priors_config.py`

Template:

- `spider/examples/params_template.json`

### Module map

- `spider/core/`: inference pipeline, likelihoods, priors, sampling
- `spider/io/`: reading/writing, samples, checkpoints
- `spider/analysis/`: diagnostics and calibration
- `spider/plotting/`: visualization helpers
- `spider/diagnostics/`: online ESS and other runtime diagnostics
- `spider/optim/`: samplers and optimizers

### Utilities

```bash
# Static import‑graph audit (conservative)
python -m spider.tools.audit_reachability

# Clean common local artifacts (dry‑run by default; add --yes to delete)
python -m spider.tools.clean_worktree
python -m spider.tools.clean_worktree --yes
```

See the root `README.md` for a full workflow description.

