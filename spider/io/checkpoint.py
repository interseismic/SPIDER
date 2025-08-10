from pathlib import Path
import os
import torch


def save_checkpoint(params, optimizer, epoch, N, ΔX_src, samples, stats_tensor):
    checkpoint_dir = Path(params["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

    checkpoint_data = {
        "epoch": epoch,
        "N": N,
        "ΔX_src": ΔX_src.cpu().detach(),  # Keep as tensor instead of numpy
        "samples": samples,
        "stats_tensor": stats_tensor.cpu().detach(),  # Keep as tensor instead of numpy
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(params, optimizer, device):
    checkpoint_dir = Path(params["checkpoint_dir"])
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {params['checkpoint_dir']}")
        return 0, 0, None, None, None

    # Get all checkpoint files and sort them numerically by epoch number
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        print(f"No checkpoint files found in {params['checkpoint_dir']}")
        return 0, 0, None, None, None

    # Extract epoch numbers and sort numerically
    checkpoint_epochs = []
    for checkpoint_file in checkpoint_files:
        try:
            # Extract epoch number from filename like "checkpoint_epoch_499.pth"
            filename = checkpoint_file.name
            epoch_str = filename.replace("checkpoint_epoch_", "").replace(".pth", "")
            epoch_num = int(epoch_str)
            checkpoint_epochs.append((epoch_num, checkpoint_file))
        except (ValueError, AttributeError):
            continue

    if not checkpoint_epochs:
        print(f"No valid checkpoint files found in {params['checkpoint_dir']}")
        return 0, 0, None, None, None

    # Sort by epoch number and get the latest
    checkpoint_epochs.sort(key=lambda x: x[0])  # Sort by epoch number
    latest_epoch, latest_checkpoint_path = checkpoint_epochs[-1]

    print(f"Found {len(checkpoint_epochs)} checkpoint files with epochs: {[epoch for epoch, _ in checkpoint_epochs]}")
    print(f"Loading latest checkpoint from epoch {latest_epoch}: {latest_checkpoint_path}")
    try:
        # Try loading with weights_only=True first (PyTorch 2.6+ default)
        checkpoint_data = torch.load(latest_checkpoint_path, map_location=torch.device(device), weights_only=True)
    except Exception as e:
        if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
            # Fall back to weights_only=False for backward compatibility
            print("Falling back to weights_only=False for checkpoint loading")
            checkpoint_data = torch.load(latest_checkpoint_path, map_location=torch.device(device), weights_only=False)
        else:
            raise e

    epoch = checkpoint_data["epoch"]
    N = checkpoint_data["N"]

    # Handle both tensor and numpy formats for backward compatibility
    ΔX_src_data = checkpoint_data["ΔX_src"]
    if isinstance(ΔX_src_data, torch.Tensor):
        ΔX_src = ΔX_src_data.to(device=device, dtype=torch.float32)
    else:
        ΔX_src = torch.tensor(ΔX_src_data, dtype=torch.float32, device=device)

    # Ensure the tensor requires gradients for autograd
    ΔX_src.requires_grad_(True)

    samples = checkpoint_data["samples"]

    # Handle both tensor and numpy formats for backward compatibility
    stats_tensor_data = checkpoint_data["stats_tensor"]
    if isinstance(stats_tensor_data, torch.Tensor):
        stats_tensor = stats_tensor_data.to(device=device, dtype=torch.float32)
    else:
        stats_tensor = torch.tensor(stats_tensor_data, dtype=torch.float32, device=device)

    # Load the optimizer state
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    # Fix: Improved parameter reference management to avoid memory issues
    old_param = optimizer.param_groups[0]['params'][0]

    # Update the optimizer's parameter reference to point to the loaded tensor
    optimizer.param_groups[0]['params'][0] = ΔX_src

    # Transfer the state from the old parameter to the new one more safely
    if old_param in optimizer.state:
        # Create a copy of the state to avoid reference issues
        old_state = optimizer.state[old_param]
        optimizer.state[ΔX_src] = old_state
        # Clear the old state entry
        del optimizer.state[old_param]
        # Clear the old parameter reference
        del old_param

    return epoch, N, ΔX_src, samples, stats_tensor


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
