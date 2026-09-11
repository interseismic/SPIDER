# Legacy Materials

The Sphinx/RTD pages are the maintained source of truth for configuration and workflow.
`docs/` additionally contains:

Archival / theory notes:

- `SPIDER_User_Guide.tex` / `.pdf` — LaTeX user guide. Updated to the current `config_v2` schema
  in September 2026, but the Markdown pages remain the primary reference. Build with the
  instructions in `BUILD_USER_GUIDE.md`.
- `COLLAPSED_GAUSSIAN_SHARED_EVENT_RE.md` — design note for the collapsed shared-event random
  effect. Its config examples use the current sub-block form; the older flat keys it once
  described (`tau_s`, `max_rows_per_group`, `abort_on_pcg_fallback` at the top of
  `shared_event_re`, ...) are now **rejected** by the validator.
- `nuisance_field.tex`, `receiver_centric_autocorr.tex` — theory notes for features that are not
  exposed by the current schema (each carries a status banner).

Current but excluded from the rendered site (see `exclude_patterns` in `conf.py`):

- `API_REFERENCE.md` — duplicate of {doc}`api-reference` kept for the repository root.
- `BUILD_USER_GUIDE.md` — how to build these docs.
- `EXAMPLES.md`, `EIKONET_TRAINING.md` — current content, but gitignored and therefore not
  available to the Read the Docs build.

Project configuration files outside `example/` (e.g. `cahuilla/`, `yifan_redo/`, `maunaloa/`,
`noto_test/`, `synthetic_yifan/`, `legacy_configs/`) predate the canonical schema and do not
validate; use `example/SPIDER_example.json` or `spider/examples/params_template.json` as a
starting point.
