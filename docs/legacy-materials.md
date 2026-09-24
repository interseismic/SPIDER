# Legacy Materials

The Sphinx/RTD pages are the maintained source of truth for configuration and workflow. The files
below live in `docs/` next to them but are either not rendered on the site or are secondary to it.

Tracked in git, not rendered (see `exclude_patterns` in `conf.py`):

- `SPIDER_User_Guide.tex` - LaTeX user guide. Kept in sync with the `config_v2` schema but
  secondary to the Markdown pages. Build with the instructions in `BUILD_USER_GUIDE.md`; the
  resulting `.pdf` and the LaTeX aux files (`.aux`, `.log`, `.out`, `.toc`) stay gitignored.
- `COLLAPSED_GAUSSIAN_SHARED_EVENT_RE.md` - design note for the collapsed shared-event random
  effect. Its config examples use the current sub-block form; the older flat keys it once
  described (`tau_s`, `max_rows_per_group`, `abort_on_pcg_fallback` at the top of
  `shared_event_re`, ...) are now **rejected** by the validator.
- `BUILD_USER_GUIDE.md` - how to build the Sphinx site and the LaTeX guide.

Tracked in git and rendered under "Recipes" in the sidebar: {doc}`EXAMPLES` (run recipes) and
{doc}`EIKONET_TRAINING` (EikoNet training notes), together with {doc}`config-walkthrough`.

Gitignored (exist only in working copies that already have them):

- `nuisance_field.tex`, `receiver_centric_autocorr.tex` - theory notes for features that are not
  exposed by the current schema (each carries a status banner).
- `SPIDER_User_Guide.pdf` - build output of the LaTeX guide.

Project configuration files outside `example/` (e.g. `cahuilla/`, `yifan_redo/`, `maunaloa/`,
`noto_test/`, `synthetic_yifan/`, `legacy_configs/`) predate the canonical schema and do not
validate; use `example/SPIDER_example.json` or `spider/examples/params_template.json` as a
starting point. Only the inputs the quickstart needs are tracked under `example/` (the two configs,
`model_state_dict.pt`, `station.csv`, `events_from_prior.csv`, `dtimes_with_noise_from_truth.csv`
and a few legacy comparison files); run outputs, checkpoints and W&B directories there are ignored.
