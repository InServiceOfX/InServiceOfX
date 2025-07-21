"""
USAGE:

You can specify both the pytest file to run *and* a substring to match for in
the test name of the tests you want to run, e.g.

pytest -s ./integration_tests/Wrappers/Models/LLMs/test_Qwen_Qwen3-0.6B.py -k "test_generate_with_attention"
"""
from corecode.Utilities import DataSubdirectories, is_model_there
from moretransformers.Utilities import get_tokens_per_second_statistics

from transformers import (
    Qwen3ForCausalLM,
    Qwen2Tokenizer)

import pytest
import time
import torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/Qwen/Qwen3-4B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_works():
    """
    I used examples from here:
    https://huggingface.co/LiquidAI/LFM2-1.2B

    There was no particular reason to use these examples.
    """
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        return_tensors='pt')

    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    dict_of_tokenizer_outputs = tokenizer.apply_chat_template(
        conversation = [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True).to(model.device)

    input_token_count = dict_of_tokenizer_outputs["input_ids"].shape[1]

    start_time = time.time()

    output = model.generate(
        input_ids=dict_of_tokenizer_outputs["input_ids"],
        attention_mask=dict_of_tokenizer_outputs["attention_mask"],
        do_sample=True,
        # temperature, min_p, repetition_penalty suggested by
        # https://huggingface.co/LiquidAI/LFM2-1.2B
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=65536,
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

# total_time_seconds: 61.67629075050354
# input_token_count: 15
# output_token_count: 960
# generated_token_count: 945
# input_tokens_per_second: 0.24320528711233394
# generated_tokens_per_second: 15.321933088077039
# output_tokens_per_second: 15.565138375189372

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_with_greater_new_tokens():
    """
    I used examples from here:
    https://huggingface.co/LiquidAI/LFM2-1.2B    

    There was no particular reason to use these examples.
    """
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True).to(model.device)

    input_token_count = input_ids.shape[1]

    start_time = time.time()

    output = model.generate(
        input_ids,
        do_sample=True,
        # temperature, min_p, repetition_penalty suggested by
        # https://huggingface.co/LiquidAI/LFM2-1.2B
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=65536
        )

    end_time = time.time()

    generated_token_count = output.shape[1] - input_token_count

    total_time = end_time - start_time

    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))

    # Calculate statistics
    stats = {
        'total_time_seconds': total_time,
        'input_token_count': input_token_count,
        'generated_token_count': generated_token_count,
        'total_token_count': output.shape[1],
        'input_tokens_per_second': \
            input_token_count / total_time if total_time > 0 else 0,
        'generated_tokens_per_second': \
            generated_token_count / total_time if total_time > 0 else 0,
        'total_tokens_per_second': \
            output.shape[1] / total_time if total_time > 0 else 0,
    }

    for key, value in stats.items():
        print(f"{key}: {value}")

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_with_attention_mask():
    """
    I used examples from here:
    https://huggingface.co/LiquidAI/LFM2-1.2B    

    There was no particular reason to use these examples.
    """
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        # TODO: tokenize=False clearly returns a str so is return_tensors
        # needed?
        return_tensors="pt",
        tokenize=False)

    encoded = tokenizer(prompt_str, return_tensors='pt', padding=True).to(
        model.device)

    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    input_token_count = encoded["input_ids"].shape[1]

    start_time = time.time()

    output = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        do_sample=True,
        # temperature, min_p, repetition_penalty suggested by
        # https://huggingface.co/LiquidAI/LFM2-1.2B
        temperature=0.9,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=65536
        )
    end_time = time.time()

    generated_token_count = output.shape[1] - input_token_count

    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))

    total_time = end_time - start_time

    # Calculate statistics
    stats = {
        'total_time_seconds': total_time,
        'input_token_count': input_token_count,
        'generated_token_count': generated_token_count,
        'total_token_count': output.shape[1],
        'input_tokens_per_second': \
            input_token_count / total_time if total_time > 0 else 0,
        'generated_tokens_per_second': \
            generated_token_count / total_time if total_time > 0 else 0,
        'total_tokens_per_second': \
            output.shape[1] / total_time if total_time > 0 else 0,
    }

    for key, value in stats.items():
        print(f"{key}: {value}")

