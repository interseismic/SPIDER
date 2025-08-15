from pathlib import Path
import os
import torch


def save_checkpoint(params, optimizer, epoch, N, ΔX_src, samples, stats_tensor, phase: str, global_step_count: int = 0):
    """Save a checkpoint including phase and step metadata.

    optimizer: can be Adam (MAP) or SGLD-like (phases 2–4).
    phase: one of {"map", "phase2", "phase3", "phase4"}.
    """
    checkpoint_dir = Path(params["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

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
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(params, device):
    """Load the most recent checkpoint by modification time.

    Returns a dict with keys:
      'phase', 'epoch', 'N', 'ΔX_src', 'samples', 'stats_tensor',
      'optimizer_state_dict', 'optimizer_type', 'global_step_count'.

    Returns None if no checkpoint available.
    """
    checkpoint_dir = Path(params["checkpoint_dir"])
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {params['checkpoint_dir']}")
        return None

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        print(f"No checkpoint files found in {params['checkpoint_dir']}")
        return None

    latest_checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading latest checkpoint: {latest_checkpoint_path}")
    try:
        data = torch.load(latest_checkpoint_path, map_location=torch.device(device), weights_only=True)
    except Exception as e:
        if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
            print("Falling back to weights_only=False for checkpoint loading")
            data = torch.load(latest_checkpoint_path, map_location=torch.device(device), weights_only=False)
        else:
            raise e

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
                print(f"Cleared checkpoint file: {filepath}")

        # Optionally remove the checkpoint directory if it's empty
        if not os.listdir(checkpoint_dir):
            os.rmdir(checkpoint_dir)
            print(f"Removed empty checkpoint directory: {checkpoint_dir}")

        return True
    except Exception as e:
        print(f"Warning: Could not clear checkpoint files: {e}")
        return False
