"""
USAGE:

You can specify both the pytest file to run *and* a substring to match for in
the test name of the tests you want to run, e.g.

pytest -s ./integration_tests/Wrappers/Models/LLMs/test_Qwen_Qwen3-0.6B.py -k "test_generate_with_attention"
"""
from corecode.Utilities import DataSubdirectories
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3ForCausalLM,
    Qwen2Tokenizer)

import pytest
import torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/Qwen/Qwen3-0.6B"

is_model_downloaded = False
model_path = None

for path in data_subdirectories.DataPaths:
    if (Path(path)/ relative_model_path).exists():
        is_model_downloaded = True
        model_path = path / relative_model_path
        print(f"Model {relative_model_path} found at {model_path}")
        break

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_AutoModelForCausalLM_from_pretrained_works():
    model = AutoModelForCausalLM.from_pretrained(model_path)
    assert model is not None
    assert isinstance(model, Qwen3ForCausalLM)

def test_Qwen3ForCausalLM_from_pretrained_works():
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    assert model is not None
    assert isinstance(model, Qwen3ForCausalLM)

def test_AutoTokenizer_from_pretrained_works():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert tokenizer is not None
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_PreTrainedTokenizerFast_from_pretrained_works():
    # From tokenizer_utils_base.py class PreTrainedTokenizerbase
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)
    assert tokenizer is not None
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    assert tokenizer.chat_template is not None
    assert isinstance(tokenizer.chat_template, str)

    print(tokenizer.chat_template)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_Qwen2Tokenizer_from_pretrained_works():
    # From tokenizer_utils_base.py class PreTrainedTokenizerbase
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)
    assert tokenizer is not None
    assert isinstance(tokenizer, Qwen2Tokenizer)

    assert tokenizer.chat_template is not None
    assert isinstance(tokenizer.chat_template, str)

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
        # E       RuntimeError: FlashAttention only supports Ampere GPUs or newer.
        #attn_implementation="flash_attention_2")
        )

    prompt = "What is C. elegans?"
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True).to(model.device)

    output = model.generate(
        input_ids=input_ids,
        do_sample=True,
        # temperature, min_p, repetition_penalty suggested by
        # https://huggingface.co/LiquidAI/LFM2-1.2B
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
        )

    assert len(output) == 1

    print(
        "With special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))

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
        # E       RuntimeError: FlashAttention only supports Ampere GPUs or newer.
        #attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True).to(model.device)

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

    assert len(output) == 1

    print(
        "With special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))

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
        # E       RuntimeError: FlashAttention only supports Ampere GPUs or newer.
        #attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        # TODO: tokenize=False clearly returns a str so is return_tensors
        # needed?
        return_tensors="pt",
        tokenize=False)

    assert isinstance(prompt_str, str)

    encoded = tokenizer(prompt_str, return_tensors='pt', padding=True).to(
        model.device)

# E       AssertionError: assert False
# E        +  where False = isinstance({'input_ids': tensor([[151644,    872,    198,   3838,    374,    356,     13,  17720,    596,\n             30, 151645...   198]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}, <class 'torch.Tensor'>)
# E        +    where <class 'torch.Tensor'> = torch.Tensor
    # TODO: Fix, error msg is above.
    #assert isinstance(encoded, Dict)

    encoded = {k: v.to(model.device) for k, v in encoded.items()}

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

    assert len(output) == 1

    print(
        "With special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))