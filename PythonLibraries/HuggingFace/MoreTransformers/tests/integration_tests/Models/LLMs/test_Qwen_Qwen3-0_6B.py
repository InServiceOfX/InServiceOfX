"""
USAGE:

You can specify both the pytest file to run *and* a substring to match for in
the test name of the tests you want to run, e.g.

pytest -s ./integration_tests/Wrappers/Models/LLMs/test_Qwen_Qwen3-0.6B.py -k "test_generate_with_attention"
"""
from corecode.Utilities import DataSubdirectories, is_model_there
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

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

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

from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,)

from typing import Dict

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_apply_chat_template_for_tokenize_False():
    """
    https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
    """
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")

    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Qwen3_thinking()

    mat = ModelAndTokenizer(
        model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration= \
            from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)

    mat.load_tokenizer()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]

    text = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        to_device=False,
        enable_thinking=True)

    assert isinstance(text, str)

    model_inputs = mat.encode_by_calling_tokenizer(text, return_tensors="pt")

    # <class 'transformers.tokenization_utils_base.BatchEncoding'>
    print(type(model_inputs))
    # input_ids, attention_mask,
    print(model_inputs.keys())

    generated_ids = mat.generate(**model_inputs)

    print(type(generated_ids))

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_follow_Qwen_code_snippet_for_thinking():
    """
    https://huggingface.co/Qwen/Qwen3-0.6B#quickstart
    for code snippet, to exercise.
    For suggested sammpling parameters.
    https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
    """
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")

    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Qwen3_thinking()

    mat = ModelAndTokenizer(
        model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration= \
            from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)

    mat.load_model()
    mat.load_tokenizer()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]

    text = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        to_device=False,
        enable_thinking=True)

    assert isinstance(text, str)

    model_inputs = mat.encode_by_calling_tokenizer(text, return_tensors="pt")

    # <class 'transformers.tokenization_utils_base.BatchEncoding'>
    #print(type(model_inputs))
    # input_ids, attention_mask,
    #print(model_inputs.keys())

    generated_ids = mat.generate(**model_inputs)
    # <class 'torch.Tensor'>
    #print(type(generated_ids))

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    #print(type(output_ids))

    index = len(output_ids) - output_ids[::-1].index(151668)
    # <class 'int'>
    #print(type(index))
    assert isinstance(index, int)

    thinking_content = mat._tokenizer.decode(
        output_ids[:index],
        skip_special_tokens=True)

    # <class 'str'>
    #print(type(thinking_content))
    assert isinstance(thinking_content, str)

    content = mat._tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True)

    # <class 'str'>
    #print(type(content))
    assert isinstance(content, str)

    # print("thinking_content: ", thinking_content)
    # print("content: ", content)

def create_configurations_and_model_for_test():
    from_pretrained_tokenizer_configuration = \
        FromPretrainedTokenizerConfiguration(
            pretrained_model_name_or_path=model_path)
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")

    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Qwen3_thinking()

    mat = ModelAndTokenizer(
        model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration= \
            from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)

    return (
        mat,
        from_pretrained_tokenizer_configuration,
        from_pretrained_model_configuration,
        generation_configuration)

def parse_generate_output_into_thinking_and_content(
        mat,
        model_inputs,
        generated_ids):
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    index = len(output_ids) - output_ids[::-1].index(151668)

    thinking_content = mat._tokenizer.decode(
        output_ids[:index],
        skip_special_tokens=True)

    content = mat._tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True)

    return (thinking_content, content)


@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_enable_thinking_false_explicit_steps():
    mat, _, _, _ = create_configurations_and_model_for_test()

    mat.load_model()
    mat.load_tokenizer()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]

    text = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        to_device=False,
        enable_thinking=False)

    assert isinstance(text, str)

    model_inputs = mat.encode_by_calling_tokenizer(text, return_tensors="pt")

    generated_ids = mat.generate(**model_inputs)

    with pytest.raises(ValueError, match="151668 is not in list"):
        thinking_content, content = parse_generate_output_into_thinking_and_content(
            mat,
            model_inputs,
            generated_ids)

    response = mat.decode_with_tokenizer(
        generated_ids,
        skip_special_tokens=True)

    print("response: ", response)

def test_with_enable_thinking_and_tokenize():
    mat, _, _, _ = create_configurations_and_model_for_test()

    mat.load_model()
    mat.load_tokenizer()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]

    tokenizer_outputs = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        enable_thinking=True)

    generated_ids = mat.generate(
        tokenizer_outputs["input_ids"],
        attention_mask=tokenizer_outputs["attention_mask"])

    # <class 'torch.Tensor'>
    #print(type(generated_ids))

    thinking_content, content = parse_generate_output_into_thinking_and_content(
        mat,
        tokenizer_outputs,
        generated_ids)

    print("thinking_content: ", thinking_content)
    print("content: ", content)
