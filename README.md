# SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docs](https://readthedocs.org/projects/spider-docs/badge/?version=latest)](https://spider-docs.readthedocs.io/en/latest/)

SPIDER is a Python toolkit for probabilistic earthquake relocation using differential travel times, neural travel‑time prediction, and scalable MCMC sampling. It combines a fast surrogate travel‑time model (EikoNet) with a multi‑phase inference pipeline to estimate event locations with uncertainty.

This is a brand new codebase. Please be patient with us as we work to making this usable by the broader scientific community.

## Install

```bash
pip install -e /path/to/eikonet   # travel-time surrogate (separate package)
pip install -e .                  # SPIDER; add '.[wandb]' for Weights & Biases logging
```

## Quick usage

```bash
spider validate-config example/SPIDER_example.json --mode sample
spider locate-map example/SPIDER_example.json --device 0   # Phase 1 (MAP) -> <catalog_outfile>_MAP.csv + <checkpoint_dir>/phase2_bundle.pth
spider sample     example/SPIDER_example.json --device 0   # Phases 2-4 (sampling) from that bundle
```

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
