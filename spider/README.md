## SPIDER (code-only repository)

This repository is intentionally **code-only**: the tracked surface area is the
`spider/` Python package (plus a minimal root `.gitignore`). Large datasets,
plots, W&B runs, notebooks, and ad-hoc scripts may exist locally but are not
part of the tracked project state.

### Supported entrypoint

Run the CLI from the repository root:

```bash
python -m spider --help
```

Core workflow:

```bash
# Phase 1 (MAP) -> writes <checkpoint_dir>/phase2_bundle.pth by default
python -m spider locate-map path/to/params.json --device 0

# Phase 2–4 (sampling) from the Phase-2 bundle
python -m spider sample path/to/params.json --device 0

# Multi-GPU independent chains (one process per device)
python -m spider sample-multi path/to/params.json --devices 0,1,2,3
```

### Configuration

SPIDER uses a **strict nested JSON** schema:

- Schema/validation: `spider/core/config_schema.py`
- Priors validation: `spider/core/priors_config.py`

Start from the template:

- `spider/examples/params_template.json`

### Library utilities (optional)

The CLI does not depend on these modules, but they are available for scripting:

- `spider.analysis` (post-processing)
- `spider.plotting` (plot helpers)

### Maintenance / “what’s stale?”

See:

- `spider/MAINTENANCE.md`

Tools:

```bash
# Static import-graph audit (conservative)
python -m spider.tools.audit_reachability

# Clean common local artifacts (defaults to dry-run; add --yes to delete)
python -m spider.tools.clean_worktree
python -m spider.tools.clean_worktree --yes
```

