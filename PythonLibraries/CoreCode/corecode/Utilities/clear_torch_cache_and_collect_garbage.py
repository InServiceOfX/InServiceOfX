import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("torch is not installed. GPU cache clearing will be skipped.", 
                  ImportWarning, stacklevel=2)
    TORCH_AVAILABLE = False

def clear_torch_cache_and_collect_garbage(device: str = None):
    import gc

    gc.collect()
    
    if not TORCH_AVAILABLE:
        return
    
    # Set default device
    if device is None:
        device = "cuda"

    if isinstance(device, str):
        if device.startswith("cuda:"):
            try:
                int(device.split(":", 1)[1])
            except ValueError:
                raise ValueError(
                    f"Invalid device format: {device}. Expected 'cuda' or "
                    f"'cuda:X' where X is an integer. Skipping GPU cache "
                    f"clearing."
                )
        elif device != "cuda":
            raise ValueError(
                f"Invalid device format: {device}. Expected 'cuda' or "
                f"'cuda:X' where X is an integer. Skipping GPU cache clearing."
            )
    else:
        raise ValueError(
            f"Device must be a string, got {type(device)}. Skipping GPU cache "
            f"clearing."
        )

    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        warnings.warn(f"CUDA is not available. Skipping GPU cache clearing.")