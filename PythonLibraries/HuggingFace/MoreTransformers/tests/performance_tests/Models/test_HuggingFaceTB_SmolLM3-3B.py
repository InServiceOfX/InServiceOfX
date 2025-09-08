"""
USAGE:

You can specify both the pytest file to run *and* a substring to match for in
the test name of the tests you want to run, e.g.

pytest -s ./integration_tests/Wrappers/Models/LLMs/test_HuggingFaceTB_SmolLM3-3B.py -k "test_AutoModelFor"
"""
from corecode.Statistics import get_tokens_per_second_statistics
from corecode.Utilities import DataSubdirectories, is_model_there

from transformers import (
    PreTrainedTokenizerFast,
    SmolLM3ForCausalLM)

import pytest
import time
import torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/HuggingFaceTB/SmolLM3-3B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_works():
    """
    No particular reason to use this example from here:
    https://huggingface.co/LiquidAI/LFM2-1.2B
    other than it's a simple example.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = SmolLM3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    dict_of_tokenizer_outputs = tokenizer.apply_chat_template(
        # src/transformers/tokenization_utils_base.py
        # Union[list[dict[str, str]], list[list[dict[str, str]]]]
        conversation=[{"role": "user", "content": prompt}],
        # bool, optional
        # If set, a prompt with the token(s) that indicate the start of an
        # assistant message will be appended to the formatted output. This is
        # useful when you want to generate a response from the model.
        add_generation_prompt=True,
        return_tensors="pt",
        # bool, defaults to True
        # Whether to tokenize output. If Calse, output will be a string.
        tokenize=True,
        # bool, defaults to False
        # Whether to return a dictionary with named outputs. Has no effect if
        # tokenize is False.
        return_dict=True).to(model.device)

    input_token_count = dict_of_tokenizer_outputs["input_ids"].shape[1]

    start_time = time.time()

    output = model.generate(
        input_ids=dict_of_tokenizer_outputs["input_ids"],
        attention_mask=dict_of_tokenizer_outputs["attention_mask"],
        do_sample=True,
        # temperature, min_p, repetition_penalty suggested by
        # https://huggingface.co/LiquidAI/LFM2-1.2B
        temperature=0.9,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=65536
        )

    end_time = time.time()

    output_token_count = output.shape[1]

    stats = get_tokens_per_second_statistics(
        input_token_count,
        output_token_count,
        start_time,
        end_time)

    for key, value in stats.items():
        print(f"{key}: {value}")

# 2025-07-20 NVIDIA GeForce RTX 3060
# total_time_seconds: 68.84408593177795
# input_token_count: 255
# output_token_count: 1615
# generated_token_count: 1360
# input_tokens_per_second: 3.704021871286024
# generated_tokens_per_second: 19.75478331352546
# output_tokens_per_second: 23.458805184811485
