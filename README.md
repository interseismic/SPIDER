# SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docs](https://readthedocs.org/projects/spider-docs/badge/?version=latest)](https://spider-docs.readthedocs.io/en/latest/)

SPIDER is a Python toolkit for probabilistic earthquake relocation using differential travel times, neural travel‑time prediction, and scalable MCMC sampling. It combines a fast surrogate travel‑time model (EikoNet) with a multi‑phase inference pipeline to estimate event locations with uncertainty.

This is a brand new codebase. Please be patient with us as we work to making this usable by the broader scientific community.

## Install

```bash
pip install -e /path/to/eikonet   # travel-time surrogate (separate package, required)
pip install -e .                  # SPIDER; use '.[wandb]' to add Weights & Biases logging
```

This installs the `spider` command (`python -m spider` is equivalent).

## Quick usage

Every run needs a trained EikoNet checkpoint, three CSV files (event catalog, station table,
differential times) and a JSON config. The repository ships a self-contained synthetic example in
`example/` (configs, CSVs and a 64-wide EikoNet checkpoint for the example domain). Its paths are
relative to the `example/` directory, so run it from there; no edits are needed:

```bash
cd example
spider validate-config SPIDER_example.json --mode sample   # schema check only; no GPU needed
spider locate-map      SPIDER_example.json --device 0      # Phase 1 (MAP)      -> SPIDER_out.cat_MAP.csv + checkpoints/phase2_bundle.pth
spider sample          SPIDER_example.json --device 0      # Phases 2-4 (MCMC)  -> SPIDER_samples.h5
spider summarize       SPIDER_example.json --burn-in 1000 --thin 10   # per-event posterior catalog with uncertainties
```

`--device` is required on every run command: a CUDA device id, or `-1` for CPU. The shipped
epochs are production length; copy the config and shrink `inference.sampler.epochs_per_phase` for
a first smoke test. `example/SPIDER_example_sghmc.json` is the same setup with the SGHMC sampler.

For your own data start from `spider/examples/params_template.json`;
[docs/config-walkthrough.md](docs/config-walkthrough.md) goes through it key by key and lists the
values to change first. [docs/EIKONET_TRAINING.md](docs/EIKONET_TRAINING.md) covers training the
travel-time model and [docs/EXAMPLES.md](docs/EXAMPLES.md) collects command recipes.

Differential times must follow SPIDER's sign convention `dt = t(evid2) - t(evid1)`. HypoDD
`dt.cc` files carry the opposite sign; set `model.filters.dtimes.flip_dt_sign: true` for those
(see [docs/data-formats.md](docs/data-formats.md)).

Full documentation: [docs/index.md](docs/index.md) (rendered at https://spider-docs.readthedocs.io).

## Citation

If you use SPIDER in your research, please cite:

```bibtex
@article{ross2026spider,
  title={SPIDER: Scalable probabilistic inference for differential earthquake relocation},
  author={Ross, Zachary E and Wilding, John D and Azizzadenesheli, Kamyar and Kato, Aitaro},
  journal={Journal of Geophysical Research: Solid Earth},
  volume={131},
  number={3},
  pages={e2025JB032769},
  year={2026},
  publisher={Wiley Online Library}
}
```
