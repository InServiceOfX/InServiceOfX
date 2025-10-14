from corecode.Utilities import DataSubdirectories, is_model_there
from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,)
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
        tokenize=True,
        return_dict=True)

    output = mat.generate(
        input_ids=input_ids["input_ids"],
        attention_mask=input_ids["attention_mask"])

    # <class 'torch.Tensor'>
    # print(type(output))
    assert len(output) == 1

    response_no_special_tokens = mat.decode_with_tokenizer(
        output, skip_special_tokens=True)
    print(
        "-------- response_no_special_tokens --------\n",
        response_no_special_tokens)

    prompt_str = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        to_device=False)

    assert isinstance(prompt_str, str)

    encoded = mat.encode_by_calling_tokenizer(prompt_str)
    encoded = mat.move_encoded_to_device(encoded)

    output = mat.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"])

    print(
        "With special tokens: ",
        mat.decode_with_tokenizer(output, skip_special_tokens=False))
    print(
        "Without special tokens: ",
        mat.decode_with_tokenizer(output, skip_special_tokens=True))

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

    response = mat.decode_with_tokenizer(output, skip_special_tokens=True)
    assert isinstance(response, str)
    print("\n -------- response --------\n", response)

def test_ModelAndTokenizer_apply_chat_template_and_generate_works_with_parsing():
    """
    Test for multiple responses.
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

    prompt = (
        "I am sending this user message to create test cases for deployment.")
    conversation = [{"role": "user", "content": prompt}]

    response = mat.apply_chat_template_and_generate(
        conversation,
        with_attention_mask=True)

    assert isinstance(response, str)
    #print("\n -------- response --------\n", response)

    thinking_content, content = mat._parse_thinking_and_content_from_text(
        response)

    assert isinstance(thinking_content, str)
    assert isinstance(content, str)
    print("\n -------- thinking_content --------\n", thinking_content)
    print("\n -------- content --------\n", content)

    conversation.append(
        {
            "role": "assistant",
            "content": (thinking_content + "\n" + content)})

    prompt_1 = "Yes, I am deploying you as a model locally."

    conversation.append({"role": "user", "content": prompt_1})

    response_1 = mat.apply_chat_template_and_generate(
        conversation,
        with_attention_mask=True)

    thinking_content_1, content_1 = mat._parse_thinking_and_content_from_text(
        response_1)
    assert isinstance(response_1, str)
    #print("\n -------- response_1 --------\n", response_1)

    assert isinstance(thinking_content_1, str)
    assert isinstance(content_1, str)
    print("\n -------- thinking_content_1 --------\n", thinking_content_1)
    print("\n -------- content_1 --------\n", content_1)

def test_ModelAndTokenizer_generate_with_enable_thinking():
    mat = ModelAndTokenizer(
        model_path,)

    mat._fpmc.device_map = "cuda:0"
    mat._fpmc.torch_dtype = torch.bfloat16
    mat._fpmc.trust_remote_code = True

    mat._generation_configuration.max_new_tokens = 65536
    mat._generation_configuration.do_sample = True
    mat._generation_configuration.temperature = 0.9
    mat._generation_configuration.min_p = 0.15
    mat._generation_configuration.repetition_penalty = 1.05

    mat._fptc.local_files_only = True
    mat._fptc.trust_remote_code = True

    mat.load_tokenizer()
    mat.load_model()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]

    tokenizer_outputs = mat.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=False,
        to_device=False,
        enable_thinking=True)

    model_inputs = mat.encode_by_calling_tokenizer(tokenizer_outputs)
    model_inputs = mat.move_encoded_to_device(model_inputs)

    output = mat.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
    )

    response = mat.decode_with_tokenizer(output, skip_special_tokens=True)
    assert isinstance(response, str)
    print("\n -------- response --------\n", response)

def test_generate_with_enable_thinking_works_with_Qwen3_0_6B():
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path,
    )
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16)

    generation_configuration = CreateDefaultGenerationConfigurations.for_Qwen3_thinking()
    generation_configuration.max_new_tokens = 65536
    generation_configuration.do_sample = True

    mat = ModelAndTokenizer(
        model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration=\
            from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)

    mat.load_tokenizer()
    mat.load_model()

    prompt = "What is C. elegans?"
    conversation = [{"role": "user", "content": prompt}]

    thinking_response, content_response = \
        mat.apply_chat_template_and_generate_with_thinking_enabled(
            conversation)

    assert isinstance(thinking_response, str)
    assert isinstance(content_response, str)

    print("\n -------- thinking_response --------\n", thinking_response)
    print("\n -------- content_response --------\n", content_response)

relative_model_path_1 = "Models/LLM/google/gemma-3-270m-it"

is_model_downloaded_1, model_path_1 = is_model_there(
    relative_model_path_1,
    data_subdirectories)

model_is_not_downloaded_message_1 = \
    f"Model {relative_model_path_1} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded_1, reason=model_is_not_downloaded_message_1)
def test_ModelAndTokenizer_generate_works_with_gemma_3_270m_it():
    mat = ModelAndTokenizer(model_path_1)

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

relative_model_path_2 = "Models/LLM/tencent/Hunyuan-0.5B-Instruct"

is_model_downloaded_2, model_path_2 = is_model_there(
    relative_model_path_2,
    data_subdirectories)

model_is_not_downloaded_message_2 = \
    f"Model {relative_model_path_2} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded_2, reason=model_is_not_downloaded_message_2)
def test_ModelAndTokenizer_works_with_Hunyuan_0_5B_Instruct():
    mat = ModelAndTokenizer(model_path_2)

    mat._fpmc.device_map = "cuda:0"
    mat._fpmc.torch_dtype = torch.bfloat16

    # https://huggingface.co/tencent/Hunyuan-0.5B-Instruct
    mat._generation_configuration.max_new_tokens = 65536
    mat._generation_configuration.do_sample = True
    mat._generation_configuration.temperature = 0.7
    mat._generation_configuration.top_k = 20
    mat._generation_configuration.top_p = 0.8
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