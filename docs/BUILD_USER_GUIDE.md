# Building Documentation

This repository now supports two documentation paths:

1. Sphinx/Read the Docs site (maintained docs)
2. Legacy LaTeX user guide (archival)

## Build Sphinx docs locally

From the repo root:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open:

- `docs/_build/html/index.html`

## Build on Read the Docs

- The project root includes `.readthedocs.yml`.
- RTD uses `docs/conf.py` and `docs/requirements.txt`.
- Python package install is configured with `pip install -e .`.

## Legacy LaTeX guide

The legacy guide lives at `docs/SPIDER_User_Guide.tex`.

## Build (pdflatex)

From the repo root:

```bash
cd docs
pdflatex -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
pdflatex -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
```

This should produce `docs/SPIDER_User_Guide.pdf`.

## Build (latexmk, if installed)

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error SPIDER_User_Guide.tex
```

## Notes

- The guide uses only common LaTeX packages (`geometry`, `hyperref`, `listings`, etc.) and does **not** require `minted`/Pygments.
- If your LaTeX install is minimal, you may need to install extra TeX packages (e.g. on Ubuntu: `texlive-latex-extra`).


