"""Weights & Biases (wandb) integration for SPIDER.

This module provides wandb logging functionality for tracking metrics
across all phases of the SPIDER location pipeline.
"""

import torch
from typing import Dict, Any, Optional, Tuple
import time
import os
import sys
import importlib
from pathlib import Path

from spider.utils.console import info, warn as log_warn

def _safe_import_wandb() -> Tuple[Optional[object], Optional[str]]:
    """
    Import the real Weights & Biases SDK robustly.

    Common failure mode in this repo: a top-level `wandb/` directory (run logs) can shadow
    the `wandb` pip package when running from the repo root, yielding an object without
    `wandb.init`.

    Returns:
        (wandb_module_or_none, warning_message_or_none)
    """
    # First try the normal import path
    try:
        wb = importlib.import_module("wandb")
        if hasattr(wb, "init"):
            return wb, None
        # If it's missing init, it's almost certainly the local `wandb/` directory.
        shadow_path = getattr(wb, "__file__", None)
        msg = (
            "W&B import looks shadowed (module has no attribute `init`). "
            f"Imported wandb from {shadow_path!r}. "
            "This commonly happens when a repo-local `wandb/` directory (run logs) shadows "
            "the `wandb` pip package. Will retry by prioritizing site-packages."
        )
    except Exception as e:
        wb = None
        msg = f"Failed to import wandb normally: {type(e).__name__}: {e}. Will retry with sys.path sanitized."

    # Retry with a sanitized sys.path (move cwd/repo root out of the way temporarily)
    try:
        orig_path = list(sys.path)
        cwd = os.getcwd()
        repo_root = str(Path(__file__).resolve().parents[2])  # .../SPIDER

        def _is_bad(p: str) -> bool:
            if p is None:
                return False
            if p == "":
                return True
            try:
                ap = os.path.abspath(p)
            except Exception:
                ap = p
            return ap in {os.path.abspath(cwd), os.path.abspath(repo_root)}

        sys.path = [p for p in sys.path if not _is_bad(p)]
        wb2 = importlib.import_module("wandb")
        if hasattr(wb2, "init"):
            return wb2, msg
        shadow_path2 = getattr(wb2, "__file__", None)
        return None, (
            (msg + " ") if msg else ""
        ) + f"Retry import still did not provide `wandb.init` (imported from {shadow_path2!r})."
    except Exception as e2:
        return None, ((msg + " ") if msg else "") + f"Retry import failed: {type(e2).__name__}: {e2}"
    finally:
        try:
            sys.path = orig_path
        except Exception:
            pass


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
        self._wandb = None
        
    def init_run(self):
        """Initialize the wandb run."""
        if self.wandb_run is None:
            if self._wandb is None:
                wb, warn = _safe_import_wandb()
                if warn:
                    warn_msg = str(warn)
                    # keep W&B import warnings readable but consistent
                    log_warn(warn_msg, section="W&B")
                if wb is None:
                    raise RuntimeError(
                        "use_wandb=True but the W&B SDK could not be imported correctly. "
                        "If you have a repo-local `wandb/` directory, it may be shadowing the pip package. "
                        "Fix options: (1) `pip install wandb` (2) rename the repo `wandb/` logs directory "
                        "(3) set `use_wandb=false`."
                    )
                self._wandb = wb

            # Reduce W&B banner spam and avoid deprecated boolean reinit.
            # Best-effort: not all W&B versions have Settings/silent.
            try:
                os.environ.setdefault("WANDB_SILENT", "true")
            except Exception:
                pass
            init_kwargs = dict(
                project=self.project_name,
                config=self.config,
                name=self.run_name,
                reinit="finish_previous",
            )
            try:
                Settings = getattr(self._wandb, "Settings", None)
                if Settings is not None:
                    init_kwargs["settings"] = Settings(silent=True)
            except Exception:
                pass
            self.wandb_run = self._wandb.init(**init_kwargs)
            # Use an explicit, monotonic step metric controlled by SPIDER, so phase-local
            # epoch counters (which may reset) never trigger out-of-order W&B step warnings.
            try:
                self._wandb.define_metric("global_step")
                self._wandb.define_metric("*", step_metric="global_step")
            except Exception:
                # define_metric can fail in offline/disabled scenarios; safe to ignore
                pass
            info(f"Initialized run name={self.wandb_run.name}", section="W&B")
    
    def start_phase(self, phase_name: str, global_step: int = 0):
        """Start timing a new phase."""
        self.phase_start_time = time.time()
        if self.wandb_run:
            # Never force step=0 (can be out-of-order if resuming); rely on global_step metric.
            self.wandb_run.log({"phase": phase_name, "global_step": int(global_step)})
            info(f"Started logging phase={phase_name}", section="W&B")
    
    def log_phase1_metrics(self, epoch: int, metrics: Dict[str, float], global_step: int):
        """Log metrics for Phase 1 (MAP estimation with Adam)."""
        if self.wandb_run:
            log_dict = {
                "phase": "phase1",
                "phase_epoch": int(epoch),
                "global_step": int(global_step),
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def log_phase2_metrics(self, epoch: int, metrics: Dict[str, float], global_step: int):
        """Log metrics for Phase 2 (deterministic preconditioned drift)."""
        if self.wandb_run:
            log_dict = {
                "phase": "phase2",
                "phase_epoch": int(epoch),
                "global_step": int(global_step),
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def log_phase3_metrics(self, step: int, metrics: Dict[str, float], global_step: int):
        """Log metrics for Phase 3 (noise ramp)."""
        if self.wandb_run:
            log_dict = {
                "phase": "phase3",
                "phase_epoch": int(step),
                "global_step": int(global_step),
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def log_phase4_metrics(self, epoch: int, metrics: Dict[str, float], global_step: int):
        """Log metrics for Phase 4 (full SGLD sampling)."""
        if self.wandb_run:
            log_dict = {
                "phase": "phase4",
                "phase_epoch": int(epoch),
                "global_step": int(global_step),
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def log_sample_metrics(self, sample_count: int, metrics: Dict[str, float], global_step: int):
        """Log metrics related to sample collection."""
        if self.wandb_run:
            log_dict = {
                "sample_count": sample_count,
                "global_step": int(global_step),
                **metrics
            }
            self.wandb_run.log(log_dict)
    
    def finish(self):
        """Finish the wandb run."""
        if self.wandb_run:
            self.wandb_run.finish()
            info("Finished run", section="W&B")


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
        "lr_warmup", "lr_sampler", "sampler_lr_mode", "sampler_backend",
        "sampler_temperature", "sampler_preconditioning", "sampler_preconditioner",
        "dt_lr_mult",
        "batch_size_warmup", "batch_size_sgld",
        "save_every_n", "checkpoint_interval", "sample_write_interval", "phase_unc",
        "devices",
        # Nested priors schema (strict)
        "priors",
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
    # Strict: config materialization must populate legacy wandb runtime keys.
    if "use_wandb" not in params or "wandb_project_name" not in params or "wandb_run_name" not in params:
        raise KeyError("Missing wandb configuration (expected materialized keys: use_wandb, wandb_project_name, wandb_run_name).")
    if not params["use_wandb"]:
        return None
    
    project_name = params["wandb_project_name"]
    if project_name is None or str(project_name).strip() == "":
        raise KeyError("wandb.project_name must be provided (non-empty) when wandb.enabled=true")
    run_name = params["wandb_run_name"]
    
    config = create_wandb_config(params)
    logger = WandbLogger(project_name, config, run_name)
    try:
        logger.init_run()
    except Exception as e:
        # Do not crash the entire locate run because W&B import/init failed.
        # This is especially common when the repo's `wandb/` logs directory shadows the package.
        log_warn(f"Disabling wandb due to init failure: {type(e).__name__}: {e}", section="W&B")
        return None
    
    return logger
