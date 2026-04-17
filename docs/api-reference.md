# API Reference (High Level)

SPIDER is primarily CLI-driven. The Python API is available for workflow integration.

## Core pipeline

- `spider.core.data.prepare_input_dfs(...)`
  - Prepares stations, dtimes, and origins.
- `spider.core.locate.locate_all(...)`
  - Runs the full relocation/sampling pipeline.

## Configuration API

- `spider.core.config_v2.load.load_config(...)`
- `spider.core.config_v2.load.load_and_resolve_config(...)`
- `spider.core.config_v2.legacy_bridge.to_legacy_runtime_params(...)`

## I/O and checkpoints

- `spider.io.samples.read_all_samples(...)`
- `spider.io.checkpoint.save_checkpoint(...)`
- `spider.io.checkpoint.load_checkpoint(...)`

## Analysis utilities

- `spider.analysis.results.EventSamplesSummary`
- plotting helpers under `spider.plotting`
