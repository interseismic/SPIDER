import torch
import gc


def clear_memory():
    """Clear GPU and CPU memory to prevent memory leaks."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
