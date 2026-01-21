from pathlib import Path

import os
import torch
from typing import Tuple

from spider.utils.console import info, warn


# Standardized stdout helper
def _log(*parts, section: str = "CKPT", **_kwargs) -> None:
    msg = " ".join(str(p) for p in parts)
    low = msg.strip().lower()
    if low.startswith("warning") or low.startswith("error"):
        warn(msg, section=section)
    else:
        info(msg, section=section)

def save_checkpoint(
    params,
    optimizer,
    epoch,
    N,
    ΔX_src,
    samples,
    stats_tensor,
    phase: str,
    global_step_count: int = 0,
    noise_log_scale=None,
    event_precision_matrix=None,
):
    """Save a checkpoint including phase and step metadata.

    optimizer: can be Adam (phase1/MAP) or SGLD-like (phases 2–4).
    phase: one of {"phase1", "phase2", "phase3", "phase4"}.
    """
    checkpoint_dir = Path(params["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Include phase in filename to avoid collisions across phases.
    phase_s = str(phase).strip().lower().replace(" ", "")
    checkpoint_path = checkpoint_dir / f"checkpoint_{phase_s}_epoch_{epoch}.pth"

    checkpoint_data = {
        "phase": str(phase),
        "epoch": int(epoch),
        "N": int(N),
        "ΔX_src": ΔX_src.cpu().detach(),
        "samples": samples,
        "stats_tensor": stats_tensor.cpu().detach(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_type": optimizer.__class__.__name__,
        "global_step_count": int(global_step_count or 0),
    }
    if noise_log_scale is not None:
        try:
            checkpoint_data["noise_log_scale"] = noise_log_scale.detach().cpu()
        except Exception:
            checkpoint_data["noise_log_scale"] = torch.as_tensor(noise_log_scale).detach().cpu()
    # Optional Hierarchical Prior P0
    if event_precision_matrix is not None:
        try:
            checkpoint_data["event_precision_matrix"] = event_precision_matrix.detach().cpu()
        except Exception:
            try:
                checkpoint_data["event_precision_matrix"] = torch.as_tensor(event_precision_matrix).detach().cpu()
            except Exception:
                pass
    torch.save(checkpoint_data, checkpoint_path)
    info(f"Saved checkpoint phase={phase_s} epoch={int(epoch)} path={checkpoint_path}", section="CKPT")


def load_checkpoint(params, device):
    """Load the most recent checkpoint by modification time.

    Returns a dict with keys:
      'phase', 'epoch', 'N', 'ΔX_src', 'samples', 'stats_tensor',
      'optimizer_state_dict', 'optimizer_type', 'global_step_count',
      and optional: 'noise_log_scale'.

    Returns None if no checkpoint available.
    """
    checkpoint_dir = Path(params["checkpoint_dir"])
    if not checkpoint_dir.exists():
        info(f"Checkpoint directory not found path={params['checkpoint_dir']}", section="CKPT")
        return None

    # Current format: checkpoint_<phase>_epoch_<n>.pth
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*_epoch_*.pth"))
    if not checkpoint_files:
        info(f"No checkpoint files found dir={params['checkpoint_dir']}", section="CKPT")
        return None

    latest_checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

    info(f"Loading latest checkpoint path={latest_checkpoint_path}", section="CKPT")
    data = torch.load(latest_checkpoint_path, map_location=torch.device(device), weights_only=True)

    # Normalize tensors to proper device/dtype
    dX_data = data.get("ΔX_src")
    if isinstance(dX_data, torch.Tensor):
        ΔX_src = dX_data.to(device=device, dtype=torch.float32)
    else:
        ΔX_src = torch.tensor(dX_data, dtype=torch.float32, device=device)
    ΔX_src.requires_grad_(True)

    st_data = data.get("stats_tensor")
    if isinstance(st_data, torch.Tensor):
        stats_tensor = st_data.to(device=device, dtype=torch.float32)
    else:
        stats_tensor = torch.tensor(st_data, dtype=torch.float32, device=device)

    out = {
        "phase": str(data.get("phase", "phase4")),
        "epoch": int(data.get("epoch", 0)),
        "N": int(data.get("N", 0)),
        "ΔX_src": ΔX_src,
        "samples": data.get("samples", []),
        "stats_tensor": stats_tensor,
        "optimizer_state_dict": data.get("optimizer_state_dict", {}),
        "optimizer_type": data.get("optimizer_type", ""),
        "global_step_count": int(data.get("global_step_count", 0)),
    }
    # Optional noise log-scale
    if "noise_log_scale" in data:
        nls = data["noise_log_scale"]
        if not isinstance(nls, torch.Tensor):
            nls = torch.tensor(nls)
        out["noise_log_scale"] = nls.to(device=device, dtype=torch.float32)

    # Optional Hierarchical Prior P0 (can be (4, 4) or (K, 4, 4))
    if "event_precision_matrix" in data:
        epm = data["event_precision_matrix"]
        if not isinstance(epm, torch.Tensor):
            epm = torch.tensor(epm)
        out["event_precision_matrix"] = epm.to(device=device, dtype=torch.float32)
        
    return out


def clear_checkpoint_files(params):
    """Clear all checkpoint files to reset the run"""
    checkpoint_dir = params.get("checkpoint_dir", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return True

    try:
        # Remove all checkpoint files
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".pth"):
                filepath = os.path.join(checkpoint_dir, filename)
                os.remove(filepath)
                info(f"Removed checkpoint file path={filepath}", section="CKPT")

        # Optionally remove the checkpoint directory if it's empty
        if not os.listdir(checkpoint_dir):
            os.rmdir(checkpoint_dir)
            info(f"Removed empty checkpoint directory dir={checkpoint_dir}", section="CKPT")

        return True
    except Exception as e:
        warn(f"Could not clear checkpoint files: {e}", section="CKPT")
        return False


def prune_checkpoints_after_phase1(params) -> Tuple[int, int]:
    """
    Delete any checkpoints whose saved 'phase' is not 'phase1'.
    Returns (num_deleted, num_kept).
    """
    checkpoint_dir = Path(params.get("checkpoint_dir", "checkpoints"))
    if not checkpoint_dir.exists():
        return 0, 0
    deleted = 0
    kept = 0
    for p in checkpoint_dir.glob("checkpoint*.pth"):
        try:
            # Load minimal metadata (weights-only).
            data = torch.load(p, map_location=torch.device("cpu"), weights_only=True)
            phase_label = str(data.get("phase", "phase4")).strip().lower()
            is_phase1 = (phase_label == "phase1")
        except Exception:
            # If unreadable, keep it to be safe
            is_phase1 = True
        if is_phase1:
            kept += 1
            continue
        try:
            os.remove(str(p))
            _log(f"Pruned non-phase1 checkpoint: {p}")
            deleted += 1
        except Exception as e:
            _log(f"Warning: failed to prune checkpoint {p}: {e}")
    return deleted, kept
