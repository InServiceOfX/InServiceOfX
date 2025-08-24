from corecode.Utilities import DataSubdirectories, is_model_there
from moretransformers.Applications import ModelAndTokenizer
from transformers import (
    Qwen3ForCausalLM,
    Qwen2Tokenizer)

import pytest
import torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/Qwen/Qwen3-0.6B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_ModelAndTokenizer_loads_tokenizer():
    mat = ModelAndTokenizer(
        model_path,
        model_class=Qwen3ForCausalLM,
        tokenizer_class=Qwen2Tokenizer)

    assert mat._model == None
    assert mat._tokenizer == None

    mat.load_tokenizer()

    assert isinstance(mat._tokenizer, Qwen2Tokenizer)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_ModelAndTokenizer_loads_model():
    mat = ModelAndTokenizer(
        model_path,
        model_class=Qwen3ForCausalLM,
        tokenizer_class=Qwen2Tokenizer)

    mat._fpmc.device_map = "cuda:0"
    mat._fpmc.torch_dtype = torch.bfloat16

    mat.load_model()

    assert isinstance(mat._model, Qwen3ForCausalLM)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_ModelAndTokenizer_apply_chat_template_works():
    mat = ModelAndTokenizer(
        model_path,
        model_class=Qwen3ForCausalLM,
        tokenizer_class=Qwen2Tokenizer)

    mat._fpmc.device_map = "cuda:0"

    mat.load_tokenizer(return_tensors='pt')

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]
    input_ids = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True)

    assert input_ids is not None
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape[0] == 1
    assert input_ids.shape[1] == 15

    prompt_str = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        to_device=False)

    assert isinstance(prompt_str, str)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_ModelAndTokenizer_generate_works():
    mat = ModelAndTokenizer(
        model_path,
        model_class=Qwen3ForCausalLM,
        tokenizer_class=Qwen2Tokenizer)

    mat._fpmc.device_map = "cuda:0"
    mat._fpmc.torch_dtype = torch.bfloat16

    mat._generation_configuration.max_new_tokens = 65536
    mat._generation_configuration.do_sample = True
    mat._generation_configuration.temperature = 0.3
    mat._generation_configuration.min_p = 0.15
    mat._generation_configuration.repetition_penalty = 1.05

    mat.load_tokenizer()
    mat.load_model()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]
    input_ids = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True)

    output = mat.generate(input_ids)

    assert len(output) == 1

    print(
        "With special tokens: ",
        mat.decode_with_tokenizer(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        mat.decode_with_tokenizer(output[0], skip_special_tokens=True))

    print("\n With attention mask: \n")

    prompt_str = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        to_device=False)

    assert isinstance(prompt_str, str)

    encoded = mat.encode_by_calling_tokenizer(prompt_str)
    encoded = mat.move_encoded_to_device(encoded)

    output = mat.generate(
        encoded["input_ids"],
        attention_mask=encoded["attention_mask"])

    print(
        "With special tokens: ",
        mat.decode_with_tokenizer(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        mat.decode_with_tokenizer(output[0], skip_special_tokens=True))

def test_ModelAndTokenizer_generate_works_with_direct_attention_mask():
    """
    And yet a 3rd way to generate, with attention mask.
    """
    mat = ModelAndTokenizer(
        model_path,
        model_class=Qwen3ForCausalLM,
        tokenizer_class=Qwen2Tokenizer)

    mat._fpmc.device_map = "cuda:0"
    mat._fpmc.torch_dtype = torch.bfloat16

    mat._generation_configuration.max_new_tokens = 65536
    mat._generation_configuration.do_sample = True
    mat._generation_configuration.temperature = 0.9
    mat._generation_configuration.min_p = 0.15
    mat._generation_configuration.repetition_penalty = 1.05

    mat.load_tokenizer()
    mat.load_model()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]

    tokenizer_outputs = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True)

    output = mat.generate(
        tokenizer_outputs["input_ids"],
        attention_mask=tokenizer_outputs["attention_mask"])

    response = mat.decode_with_tokenizer(output[0], skip_special_tokens=True)
    assert isinstance(response, str)
    print(response)

relative_model_path = "Models/LLM/google/gemma-3-270m-it"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_ModelAndTokenizer_apply_chat_template_and_generate_works():
    mat = ModelAndTokenizer(model_path)

    mat._fpmc.device_map = "cuda:0"
    mat._fpmc.torch_dtype = torch.bfloat16

    mat._generation_configuration.max_new_tokens = 65536
    mat._generation_configuration.do_sample = True
    mat._generation_configuration.temperature = 0.9
    mat._generation_configuration.min_p = 0.15
    mat._generation_configuration.repetition_penalty = 1.05

    mat.load_tokenizer()
    mat.load_model()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]

    response = mat.apply_chat_template_and_generate(
        conversation,
        with_attention_mask=True)
    assert isinstance(response, str)
    print(response)

    response = mat.apply_chat_template_and_generate(
        conversation,
        with_attention_mask=False)
    assert isinstance(response, str)
    print(response)
