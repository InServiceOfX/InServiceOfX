from corecode.Utilities import is_model_there, DataSubdirectories

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForCausalLM,
    GemmaTokenizer,
    PreTrainedTokenizerFast,
)

from pathlib import Path

import pytest
import torch


data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/google/gemma-3-1b-it"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_AutoModelForCausalLM_from_pretrained_works():
    model = AutoModelForCausalLM.from_pretrained(model_path)
    assert model is not None
    # <class 'transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM'>
    #print(type(model))
    assert isinstance(model, Gemma3ForCausalLM)

def test_AutoTokenizer_from_pretrained_works():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert tokenizer is not None
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_works():
    # E           ImportError: 
    # E           GemmaTokenizer requires the SentencePiece library but it was not found in your environment. Check out the instructions on the
    # E           installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
    # E           that match your environment. Please note that you may need to restart your runtime after installation.
    #tokenizer = GemmaTokenizer.from_pretrained(
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        return_tensors='pt')

    model = Gemma3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
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
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        # 32K tokens for the 1B and 270M sizes
        # https://huggingface.co/google/gemma-3-270m-it
        max_new_tokens=16384,
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = Gemma3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        )

    prompt = "What is C. elegans?"
    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=False)

    encoded = tokenizer(prompt_str, return_tensors='pt', padding=True).to(
        model.device)

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

    print(
        "With special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))
