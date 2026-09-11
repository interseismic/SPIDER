# Building Documentation

This repository has two documentation paths:

1. Sphinx/Read the Docs site (maintained docs, `docs/*.md`)
2. LaTeX user guide (`docs/SPIDER_User_Guide.tex`; kept in sync with the schema but secondary)

## Build Sphinx docs locally

There is no `docs/Makefile`; invoke `sphinx-build` directly from the repo root:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`.

Pages intentionally excluded from the site (see `exclude_patterns` in `docs/conf.py`):
`API_REFERENCE.md`, `BUILD_USER_GUIDE.md`, `COLLAPSED_GAUSSIAN_SHARED_EVENT_RE.md`,
`EIKONET_TRAINING.md`, `EXAMPLES.md` (the last two are also gitignored, so they are unavailable to
the Read the Docs build), and all LaTeX/PDF artifacts.

## Build on Read the Docs

- The project root includes `.readthedocs.yaml`.
- RTD uses `docs/conf.py` and `docs/requirements.txt`.
- RTD install is docs-only via `docs/requirements.txt` — the `spider` package and its
  dependencies (torch, polars, eikonet) are **not** available during the RTD build, so do not
  enable `sphinx.ext.autodoc`/`autosummary` without adding a mock-imports configuration.

## LaTeX guide

### Build (pdflatex)

From the repo root:

```bash
cd docs
pdflatex -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
pdflatex -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
```

This produces `docs/SPIDER_User_Guide.pdf`. Build litter (`.aux`, `.log`, `.out`, `.toc`) is
excluded from the Sphinx build.

### Build (latexmk, if installed)

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
```

## Notes

- The guide uses only common LaTeX packages (`geometry`, `hyperref`, `listings`, etc.) and does
  **not** require `minted`/Pygments.
- If your LaTeX install is minimal, you may need extra TeX packages (e.g. on Ubuntu:
  `texlive-latex-extra`).
