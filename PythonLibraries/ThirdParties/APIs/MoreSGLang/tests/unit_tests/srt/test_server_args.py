from corecode.Utilities import DataSubdirectories

from sglang.srt.server_args import ServerArgs

import pytest

data_sub_dirs = DataSubdirectories()

MODEL_DIR = data_sub_dirs.ModelsLLM / "deepseek-ai" / \
    "DeepSeek-R1-Distill-Qwen-1.5B"

"""
From def __post_init__(self) around line 438 in server_args.py,
The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors."
"""
@pytest.mark.parametrize("kwargs", [
    {"model_path": MODEL_DIR, "mem_fraction_static": 0.70}
])
def test_ServerArgs_init_Engine(kwargs):
    """
    see srt/entrypoint/engine.py. Engine __init__()'s with ServerArgs.
    """
    server_args = ServerArgs(**kwargs)

    assert server_args.model_path == MODEL_DIR
    # See def __post_init__(self) around line 170, if tokenizer_path is None,
    # it is set to model_path.
    assert server_args.tokenizer_path == MODEL_DIR
    assert server_args.mem_fraction_static == 0.70
    # In server_args.py if mem_fraction_static None, tp_size determines the
    # mem_fraction_static. See def __post_init__(self) around line 190.
    assert server_args.tp_size == 1
    assert server_args.dp_size == 1
    assert server_args.max_prefill_tokens == 16384
