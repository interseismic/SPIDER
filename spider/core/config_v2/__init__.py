"""
SPIDER config_v2 package.

This package provides strict canonical config loading with no legacy compatibility.
"""

from .errors import ConfigError, ConfigIssue
from .legacy_bridge import to_legacy_runtime_params
from .load import load_config, load_config_file
from .resolve import apply_defaults, build_runtime_map, resolve_config
from .types import (
    CanonicalConfig,
    ComputeConfig,
    DomainConfig,
    InferenceConfig,
    IOConfig,
    ModelConfig,
    ObservabilityConfig,
    ResolvedConfig,
    RuntimeConfig,
    SafetyConfig,
    SamplerConfig,
    WandbConfig,
)
from .validate import parse_canonical_config, validate_config, validate_cross_field_config

__all__ = [
    "CanonicalConfig",
    "ComputeConfig",
    "ConfigError",
    "ConfigIssue",
    "DomainConfig",
    "InferenceConfig",
    "IOConfig",
    "ModelConfig",
    "ObservabilityConfig",
    "ResolvedConfig",
    "RuntimeConfig",
    "SafetyConfig",
    "SamplerConfig",
    "WandbConfig",
    "apply_defaults",
    "build_runtime_map",
    "load_config",
    "load_config_file",
    "parse_canonical_config",
    "resolve_config",
    "to_legacy_runtime_params",
    "validate_config",
    "validate_cross_field_config",
]

