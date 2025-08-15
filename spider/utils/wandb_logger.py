"""Weights & Biases (wandb) integration for SPIDER.

This module provides wandb logging functionality for tracking metrics
across all phases of the SPIDER location pipeline.
"""

import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional
import time


class WandbLogger:
    """Wandb logger for SPIDER metrics tracking."""
    
    def __init__(self, project_name: str, config: Dict[str, Any], run_name: Optional[str] = None):
        """
        Initialize wandb logger.
        
        Args:
            project_name: Name of the wandb project
            config: Configuration dictionary to log
            run_name: Optional name for this run
        """
        self.project_name = project_name
        self.config = config
        self.run_name = run_name
        self.wandb_run = None
        self.phase_start_time = None
        
    def init_run(self):
        """Initialize the wandb run."""
        if self.wandb_run is None:
            self.wandb_run = wandb.init(
                project=self.project_name,
                config=self.config,
                name=self.run_name,
                reinit=True
            )
            print(f"Initialized wandb run: {self.wandb_run.name}")
    
    def start_phase(self, phase_name: str):
        """Start timing a new phase."""
        self.phase_start_time = time.time()
        if self.wandb_run:
            self.wandb_run.log({"phase": phase_name}, step=0)
            print(f"Started wandb logging for phase: {phase_name}")
    
    def log_phase1_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for Phase 1 (MAP estimation with Adam)."""
        if self.wandb_run:
            log_dict = {
                "phase": "phase1",
                "epoch": epoch,
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def log_phase2_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for Phase 2 (deterministic preconditioned drift)."""
        if self.wandb_run:
            log_dict = {
                "phase": "phase2",
                "epoch": epoch,
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def log_phase3_metrics(self, step: int, metrics: Dict[str, float]):
        """Log metrics for Phase 3 (noise ramp)."""
        if self.wandb_run:
            log_dict = {
                "phase": "phase3",
                "step": step,
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def log_phase4_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for Phase 4 (full SGLD sampling)."""
        if self.wandb_run:
            log_dict = {
                "phase": "phase4",
                "epoch": epoch,
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def log_sample_metrics(self, sample_count: int, metrics: Dict[str, float]):
        """Log metrics related to sample collection."""
        if self.wandb_run:
            log_dict = {
                "sample_count": sample_count,
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def finish(self):
        """Finish the wandb run."""
        if self.wandb_run:
            self.wandb_run.finish()
            print("Finished wandb run")


def extract_metrics_from_stats_tensor(stats_tensor: torch.Tensor) -> Dict[str, float]:
    """Extract metrics from the stats tensor.
    
    The stats tensor contains:
    [0] = mean_dX, [1] = mean_dY, [2] = mean_dZ,
    [3] = median_abs_dX, [4] = median_abs_dY, [5] = median_abs_dZ,
    [6] = max_radius, [7] = quantile_90_radius
    """
    stats_cpu = stats_tensor.detach().cpu().numpy()
    return {
        "mean_dX": float(stats_cpu[0]),
        "mean_dY": float(stats_cpu[1]),
        "mean_dZ": float(stats_cpu[2]),
        "median_abs_dX": float(stats_cpu[3]),
        "median_abs_dY": float(stats_cpu[4]),
        "median_abs_dZ": float(stats_cpu[5]),
        "max_radius": float(stats_cpu[6]),
        "quantile_90_radius": float(stats_cpu[7])
    }


def create_wandb_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a wandb config from SPIDER parameters."""
    # Select key parameters to log
    config_keys = [
        "phase1_epochs", "phase2_epochs", "phase3_epochs", "phase4_epochs",
        "lr_warmup", "lr_sgld", "batch_size_warmup", "batch_size_sgld",
        "save_every_n", "checkpoint_interval", "phase_unc",
        "prior_event_std", "prior_centroid_std", "devices",
        "n_warmup_epochs", "n_sgld_epochs"  # Legacy keys for backward compatibility
    ]
    
    config = {}
    for key in config_keys:
        if key in params:
            config[key] = params[key]
    
    # Add some computed values (only if they exist)
    if "total_events" in params:
        config["total_events"] = params["total_events"]
    if "total_dtimes" in params:
        config["total_dtimes"] = params["total_dtimes"]
    
    return config


def init_wandb_if_enabled(params: Dict[str, Any]) -> Optional[WandbLogger]:
    """Initialize wandb logger if enabled in parameters."""
    if not params.get("use_wandb", False):
        return None
    
    project_name = params.get("wandb_project_name", "spider-location")
    run_name = params.get("wandb_run_name", None)
    
    config = create_wandb_config(params)
    logger = WandbLogger(project_name, config, run_name)
    logger.init_run()
    
    return logger
