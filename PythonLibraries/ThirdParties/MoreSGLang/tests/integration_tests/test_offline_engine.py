from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories

import asyncio
import atexit
import os
import sglang.api
import sglang
import pytest


from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.utils import stream_and_merge
from sglang.srt.entrypoints.engine import _launch_subprocesses
from typing import Dict

data_sub_dirs = DataSubdirectories()

MODEL_DIR = data_sub_dirs.ModelsLLM / "deepseek-ai" / \
    "DeepSeek-R1-Distill-Qwen-1.5B"

# Skip reason that will show in pytest output
skip_reason = f"Directory {MODEL_DIR} is empty or doesn't exist"

def shutdown_function_from_engine():
    kill_process_tree(os.getpid(), include_parent=False)

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_offline_engine():
    llm = sglang.api.Engine(model_path=MODEL_DIR, mem_fraction_static=0.70)
    prompt = "The capital of France is"
    # temperature controls randomness, high value for more randomness.
    # top_p considers range of possible next tokens, low value considers only
    # most likely next token.
    sampling_params = {"temperature": 0.1, "top_p": 0.4}

    # https://docs.sglang.ai/backend/offline_engine_api.html#Non-streaming-Synchronous-Generation
    response = llm.generate(prompt, sampling_params=sampling_params)
    # Uncomment to print response
    #print(response)

    assert "Paris" in response['text']
    assert 'meta_info' in response.keys()
    assert 'id' in response['meta_info'].keys()
    assert 'finish_reason' in response['meta_info'].keys()
    assert 'prompt_tokens' in response['meta_info'].keys()
    assert 'completion_tokens' in response['meta_info'].keys()
    assert 'cached_tokens' in response['meta_info'].keys()

    # https://docs.sglang.ai/backend/offline_engine_api.html#Streaming-Synchronous-Generation

    prompt = "Provide a concise factual statement about France's capital city. The name of thecapital of France is"
    merged_output = stream_and_merge(llm, prompt, sampling_params)
    assert isinstance(merged_output, str)
    assert "capital" in merged_output

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
@pytest.mark.parametrize("kwargs", [
    {"model_path": MODEL_DIR, "mem_fraction_static": 0.70}
])
def test_engine_launches_subprocesses(kwargs):
    server_args = ServerArgs(**kwargs)
    atexit.register(shutdown_function_from_engine)

    print("Launching subprocesses\n")
    tokenizer_manager, scheduler_info = _launch_subprocesses(
        server_args=server_args)
    print("Subprocesses launched\n")
    print(type(tokenizer_manager))
    print(type(scheduler_info))
    assert isinstance(
        tokenizer_manager,
        sglang.srt.managers.tokenizer_manager.TokenizerManager)
    assert isinstance(scheduler_info, Dict)

    # srt/entrypoints/engine.py around line 126 says for def generate(self, ..)
    # that "the arguments for this function is the same as
    # 'sglang/srt/managers/io_struct.py::GenerateReqInput`. Please refor to 
    # `GenerateReqInput` for the documentation."
    # GenerateReqInput is found in io_struct.py.
    # It seems like all arguments for generate(..) can be optional, but code
    # will complain later.

    loop = asyncio.get_event_loop()
