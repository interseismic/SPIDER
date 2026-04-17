# CLI Workflow

SPIDER CLI entrypoint:

```bash
spider --help
```

## Main commands

```bash
# Phase 1 only (MAP)
spider locate-map my_params.json --device 0

# Phase 2-4 only (sampling; expects phase2 bundle/checkpoints)
spider sample my_params.json --device 0

# Full pipeline (Phase 1 through Phase 4)
spider locate-full my_params.json --device 0

# Multi-GPU independent chains
spider sample-multi my_params.json --devices 0,1,2,3

# Schema validation
spider validate-config my_params.json --mode sample
```

## Typical run order

1. Validate config.
2. Run `locate-map`.
3. Inspect residual and diagnostics sanity.
4. Run `sample`.
5. Analyze posterior samples from `io.samples_outfile`.
