from corecode.Utilities import DataSubdirectories

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Lfm2ForCausalLM,
    PreTrainedTokenizerFast)

import pytest
import torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/LiquidAI/LFM2-1.2B"

is_model_downloaded = False
model_path = None

for path in data_subdirectories.DataPaths:
    if (path / relative_model_path).exists():
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
    assert isinstance(model, Lfm2ForCausalLM)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_Lfm2ForCausalLM_from_pretrained_works():
    model = Lfm2ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    assert model is not None
    assert isinstance(model, Lfm2ForCausalLM)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
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
def test_generate_works():
    """
    https://huggingface.co/LiquidAI/LFM2-1.2B    
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = Lfm2ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")

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
        max_new_tokens=512
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
    https://huggingface.co/LiquidAI/LFM2-1.2B    
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = Lfm2ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")

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