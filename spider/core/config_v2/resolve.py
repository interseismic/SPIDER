"""
Default application and runtime resolution for config_v2.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Dict, List

from .types import CanonicalConfig, InferenceConfig, IOConfig, ResolvedConfig, RuntimeConfig


def _apply_io_defaults(io_cfg: IOConfig, defaults_applied: List[str]) -> IOConfig:
    if io_cfg.sample_write_interval is None:
        defaults_applied.append("io.sample_write_interval <- io.checkpoint_interval")
        return replace(io_cfg, sample_write_interval=int(io_cfg.checkpoint_interval))
    return io_cfg


def _apply_runtime_defaults(runtime_cfg: RuntimeConfig, defaults_applied: List[str]) -> RuntimeConfig:
    updated = runtime_cfg
    if updated.seed is None:
        defaults_applied.append("inference.runtime.seed <- 0")
        updated = replace(updated, seed=0)
    if not isinstance(updated.torch, dict):
        defaults_applied.append("inference.runtime.torch <- {}")
        updated = replace(updated, torch={})
    return updated


def _apply_inference_defaults(inf_cfg: InferenceConfig, defaults_applied: List[str]) -> InferenceConfig:
    runtime = _apply_runtime_defaults(inf_cfg.runtime, defaults_applied)
    return replace(inf_cfg, runtime=runtime)


def apply_defaults(cfg: CanonicalConfig) -> tuple[CanonicalConfig, List[str]]:
    defaults_applied: List[str] = []
    io = _apply_io_defaults(cfg.io, defaults_applied)
    inference = _apply_inference_defaults(cfg.inference, defaults_applied)
    resolved = replace(cfg, io=io, inference=inference)
    return resolved, defaults_applied


def build_runtime_map(cfg: CanonicalConfig) -> Dict[str, Any]:
    runtime: Dict[str, Any] = asdict(cfg)

    # Derived helpers for runtime consumers.
    phase_names = ("phase1", "phase2", "phase3", "phase4")
    epochs = list(cfg.inference.sampler.epochs_per_phase)
    lrs = list(cfg.inference.sampler.lr)
    runtime["_derived"] = {
        "phase_epochs": {name: int(epochs[i]) for i, name in enumerate(phase_names)},
        "phase_lr": {name: float(lrs[i]) for i, name in enumerate(phase_names)},
    }
    return runtime


def resolve_config(cfg: CanonicalConfig, mode: str | None = None) -> ResolvedConfig:
    canonical, defaults_applied = apply_defaults(cfg)
    runtime = build_runtime_map(canonical)
    return ResolvedConfig(
        canonical=canonical,
        runtime=runtime,
        defaults_applied=defaults_applied,
        warnings=[],
        mode=mode,
    )

