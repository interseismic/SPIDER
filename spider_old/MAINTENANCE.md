## SPIDER maintenance notes (current supported surface area)

This repo has accumulated many experiments, datasets, and legacy workflows over time.
To keep the project maintainable, it helps to draw a bright line between what is
currently supported vs. what is “historical / local”.

### What is “supported” (actively used by the current code path)

- **Primary entrypoint**: `python -m spider ...`
  - Implemented in `spider/__main__.py` → `spider/cli.py`
- **Primary workflow**:
  - `spider locate-map <params>`: Phase 1 (MAP) and writes a Phase-2 bundle
  - `spider sample <params>`: Phase 2–4 sampling from a Phase-2 bundle
  - `spider sample-multi <params>`: multi-GPU *independent* chains (one process per device)
  - `spider locate-full <params>`: Phase 1 + Phase 2–4 one-shot (kept for convenience / legacy one-shot flows)
- **Configuration**:
  - Strict nested config schema in `spider/core/config_schema.py`
  - Priors validation/materialization in `spider/core/priors_config.py`

### Library-only modules (not required by the CLI)

These modules are **not imported by the CLI**, but may still be useful for ad-hoc analysis scripts:

- `spider/analysis/` (post-processing and summaries)
- `spider/plotting/` (plot helpers)

If you want to keep the repo strictly “runtime-only”, these can be moved out of the
package (or into an explicit `spider/extra/` namespace). If you keep them, prefer
minimal dependencies and clear docstrings since they’re not exercised by the main pipeline.

### Common “stale” items (usually local artifacts, not part of the supported runtime)

These are typically **outputs** or **build artifacts** that should not be committed:

- `wandb/` (run logs)
- `plots/` (generated figures)
- `checkpoints/` (model/sampler checkpoints)
- `spider.egg-info/`, `dist/`, `build/` (packaging artifacts)
- `__pycache__/`, `*.pyc`
- Random scratch directories (e.g. `None/`) or ad-hoc test scripts

### How to *find* stale code inside `spider/`

Static “what is reachable?” audit (conservative; import-graph only):

```bash
python -m spider.tools.audit_reachability
```

If it reports modules as unreachable, that’s a strong signal they’re candidates for:
- deletion
- moving under an explicit `spider/legacy/` namespace
- or documenting as “used only by notebooks / dynamic import paths”

### House rule recommendation

- Keep **runtime** code in `spider/` reachable from the CLI.
- Keep **examples** (configs/notebooks) in a clearly named directory, and avoid absolute paths in committed configs.
- Treat generated outputs as ephemeral and keep them out of git.

