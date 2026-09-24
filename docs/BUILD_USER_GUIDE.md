# Building the Documentation

This repository has two documentation paths:

1. The Sphinx / Read the Docs site built from the Markdown pages in `docs/` (the maintained
   documentation; `docs/index.md` is the table of contents).
2. A LaTeX user guide (`docs/SPIDER_User_Guide.tex`). The source is tracked in git (the built
   `.pdf` and LaTeX aux files are not); the Markdown pages are the reference.

## Build the Sphinx site locally

There is no `docs/Makefile`; invoke `sphinx-build` directly from the repository root:

```bash
pip install -r docs/requirements.txt          # sphinx, myst-parser, sphinx-rtd-theme, sphinx-copybutton
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. Add `-W --keep-going` to turn warnings (broken `{doc}`
targets, pages missing from the toctree) into a non-zero exit code, and `-E` to force a full
rebuild after editing `conf.py`.

Authoring notes:

- Pages are MyST Markdown (`myst_parser`) with the `colon_fence`, `deflist`, `fieldlist`,
  `dollarmath` and `amsmath` extensions enabled, so `$...$` / `$$...$$` math and `:::` fences
  work. Cross-reference other pages with ``{doc}`page-name` `` (file name without `.md`); the
  target must be a page listed in a toctree in `docs/index.md`.
- Every new page must be added to a toctree in `docs/index.md` or Sphinx warns and the page is
  unreachable from the sidebar.
- `docs/api-reference.md` is hand-written; `sphinx.ext.autodoc` is deliberately **not** enabled
  (see "Read the Docs" below).

Pages intentionally excluded from the site (`exclude_patterns` in `docs/conf.py`):
`BUILD_USER_GUIDE.md` (this file), `COLLAPSED_GAUSSIAN_SHARED_EVENT_RE.md` (design note; tracked
in git but not rendered) and all LaTeX/PDF artifacts (`*.tex`, `*.pdf`, `*.aux`, `*.log`,
`*.out`, `*.toc`). `EXAMPLES.md`, `EIKONET_TRAINING.md` and `config-walkthrough.md` are tracked
and rendered under the "Recipes" toctree in `docs/index.md`. The older `API_REFERENCE.md` was
folded into `api-reference.md` and removed.

## Build on Read the Docs

- The project root contains `.readthedocs.yaml` (Ubuntu 24.04, Python 3.11).
- RTD uses `docs/conf.py` and installs **only** `docs/requirements.txt` — the `spider` package and
  its dependencies (torch, polars, h5py, eikonet) are not available during the build. Do not enable
  `sphinx.ext.autodoc` / `autosummary` without adding an `autodoc_mock_imports` configuration.
- The RTD build therefore only sees git-tracked files. Every page listed in a toctree must be
  tracked (check with `git ls-files docs`); a page that renders locally but is gitignored breaks
  the RTD build.

## LaTeX guide (local copies only)

From the repository root:

```bash
cd docs
pdflatex -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
pdflatex -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
```

or, with `latexmk` installed:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
```

This produces `docs/SPIDER_User_Guide.pdf`. The PDF and the build litter (`.aux`, `.log`,
`.out`, `.toc`) are gitignored and excluded from the Sphinx build; only the `.tex` source is
tracked. The guide uses only common LaTeX packages
(`geometry`, `hyperref`, `listings`, ...) and does not require `minted`/Pygments; on a minimal TeX
installation you may need extra packages (e.g. `texlive-latex-extra` on Ubuntu).
