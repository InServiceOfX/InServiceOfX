from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories

import sglang.api
import pytest

from sglang.utils import stream_and_merge

data_sub_dirs = DataSubdirectories()

MODEL_DIR = data_sub_dirs.ModelsLLM / "deepseek-ai" / \
    "DeepSeek-R1-Distill-Qwen-1.5B"

# Skip reason that will show in pytest output
skip_reason = f"Directory {MODEL_DIR} is empty or doesn't exist"

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
    print(response)
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
    print(merged_output)
    assert "Paris" in merged_output
