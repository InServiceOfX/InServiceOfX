import torch

def clear_torch_cache_and_collect_garbage():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()