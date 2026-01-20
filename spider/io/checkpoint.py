from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import torch

from ..utils.console import info


def save_checkpoint(
    *,
    params: dict,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    N: int,
    dX_src: torch.Tensor,
    phase: str,
) -> str:
    checkpoint_dir = Path(params["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phase_s = str(phase).strip().lower().replace(" ", "")
    checkpoint_path = checkpoint_dir / f"checkpoint_{phase_s}_epoch_{epoch}.pth"
    checkpoint_data: Dict[str, Any] = {
        "phase": str(phase),
        "epoch": int(epoch),
        "N": int(N),
        "dX_src": dX_src.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_type": optimizer.__class__.__name__,
    }
    torch.save(checkpoint_data, checkpoint_path)
    info(f"Saved checkpoint phase={phase_s} epoch={int(epoch)} path={checkpoint_path}", section="CKPT")
    return str(checkpoint_path)


def load_checkpoint(params: dict, device: torch.device) -> Optional[dict]:
    checkpoint_dir = Path(params["checkpoint_dir"])
    if not checkpoint_dir.exists():
        return None
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*_epoch_*.pth"))
    if not checkpoint_files:
        return None
    latest_checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    data = torch.load(latest_checkpoint_path, map_location=torch.device(device), weights_only=True)
    dX_data = data.get("dX_src")
    if isinstance(dX_data, torch.Tensor):
        dX_src = dX_data.to(device=device, dtype=torch.float32)
    else:
        dX_src = torch.tensor(dX_data, dtype=torch.float32, device=device)
    dX_src.requires_grad_(True)
    return {
        "phase": str(data.get("phase", "phase4")),
        "epoch": int(data.get("epoch", 0)),
        "N": int(data.get("N", 0)),
        "dX_src": dX_src,
        "optimizer_state_dict": data.get("optimizer_state_dict", {}),
        "optimizer_type": data.get("optimizer_type", ""),
    }
