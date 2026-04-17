"""
Typed contracts for config_v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from .errors import ConfigIssue


DeviceSpecifier = Union[int, str]


@dataclass(frozen=True)
class IOConfig:
    dtime_file: str
    station_file: str
    catalog_infile: str
    catalog_outfile: str
    samples_outfile: str
    checkpoint_dir: str
    checkpoint_interval: int
    save_every_n: int
    write_samples: bool
    sample_write_interval: Optional[int] = None


@dataclass(frozen=True)
class DomainConfig:
    lon_min: float
    lat_min: float
    z_min: float
    z_max: float
    scale: float


@dataclass(frozen=True)
class ModelConfig:
    model_file: str
    domain: DomainConfig
    priors: Dict[str, Any] = field(default_factory=dict)
    likelihoods: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    eikonet: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeConfig:
    devices: List[DeviceSpecifier] = field(default_factory=list)


@dataclass(frozen=True)
class SamplerConfig:
    backend: str
    epochs_per_phase: Sequence[int]
    lr: Sequence[float]
    temperature: float
    beta: float
    eps: float
    freeze_preconditioner_sampling: bool
    sghmc_alpha: Optional[float] = None
    noise_scale_mult: Optional[float] = None
    dt_lr_mult: Optional[float] = None
    grad_clip_norm: Optional[float] = None
    preconditioning: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchingConfig:
    standard: Dict[str, Any] = field(default_factory=dict)
    event_batches: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    cuda_empty_cache_every: int
    reset_batch_numbers: bool
    clear_samples_on_reset: bool
    min_samples_to_save: int
    verbose: bool
    cluster_events: bool
    seed: Optional[int] = None
    gauge_projection: Dict[str, Any] = field(default_factory=dict)
    torch: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyConfig:
    max_abs_dX: Optional[Sequence[float]] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceConfig:
    sampler: SamplerConfig
    batching: BatchingConfig
    runtime: RuntimeConfig
    safety: SafetyConfig
    compute: ComputeConfig = field(default_factory=ComputeConfig)


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project_name: Optional[str]
    run_name: Optional[str]


@dataclass(frozen=True)
class ObservabilityConfig:
    wandb: WandbConfig
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalConfig:
    io: IOConfig
    model: ModelConfig
    inference: InferenceConfig
    observability: ObservabilityConfig


@dataclass(frozen=True)
class ResolvedConfig:
    canonical: CanonicalConfig
    runtime: Dict[str, Any]
    defaults_applied: List[str] = field(default_factory=list)
    warnings: List["ConfigIssue"] = field(default_factory=list)
    mode: Optional[str] = None

