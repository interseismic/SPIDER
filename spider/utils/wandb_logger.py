from __future__ import annotations

from typing import Optional

from .console import info, warn


class WandbLogger:
    def __init__(self, project: str, run_name: Optional[str], config: dict):
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(f"wandb import failed: {e}") from e
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict, *, step: Optional[int] = None) -> None:
        try:
            if step is not None:
                self._wandb.log(metrics, step=int(step))
            else:
                self._wandb.log(metrics)
        except Exception:
            return

    def finish(self) -> None:
        try:
            self._wandb.finish()
        except Exception:
            return


def init_wandb(params: dict) -> Optional[WandbLogger]:
    enabled = bool(params.get("use_wandb", False))
    if not enabled:
        return None
    project = str(params.get("wandb_project_name", "") or "").strip()
    if not project:
        warn("wandb.enabled=true but project_name missing; disabling W&B.", section="W&B")
        return None
    run_name = params.get("wandb_run_name", None)
    try:
        logger = WandbLogger(project=project, run_name=run_name, config=params)
        info("W&B logging enabled.", section="W&B")
        return logger
    except Exception as e:
        warn(f"W&B init failed: {e}", section="W&B")
        return None
