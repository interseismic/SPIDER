# Quickstart

## Installation

From the repository root:

```bash
pip install -e /path/to/eikonet
pip install -e .
```

Optional extras:

```bash
pip install -e '.[wandb]'
```

## Minimal workflow

Use the example config in `example/SPIDER_example.json`.

```bash
# Phase 1 (MAP)
spider locate-map example/SPIDER_example.json --device 0

# Phase 2-4 (sampling)
spider sample example/SPIDER_example.json --device 0
```

Multi-GPU independent chains:

```bash
spider sample-multi example/SPIDER_example.json --devices 0,1,2,3
```

## Validate configuration first

```bash
spider validate-config example/SPIDER_example.json --mode sample
```

## Outputs

Paths are controlled by the `io` block:

- `io.catalog_outfile`
- `io.samples_outfile`
- `io.checkpoint_dir`
