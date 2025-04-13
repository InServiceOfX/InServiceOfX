from moretransformers.Configurations import Configuration, GenerationConfiguration
from moretransformers.Wrappers.LLMEngines import LocalLlama

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

from commonapi.Messages import create_system_message, create_user_message

from pathlib import Path

import pytest

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

configuration_llama3 = Configuration(
    test_data_directory / "configuration-llama3.yml")

generation_configuration_llama3 = GenerationConfiguration(
    test_data_directory / "generation_configuration-llama3.yml")

def test_LocalLlama_inits():
    agent = LocalLlama(
        configuration_llama3,
        generation_configuration_llama3)
    assert isinstance(agent, LocalLlama)
    assert isinstance(agent.model, LlamaForCausalLM)
    assert isinstance(agent.tokenizer, PreTrainedTokenizerFast)
    assert agent.model.config.pad_token_id == \
        generation_configuration_llama3.eos_token_id[0]

    assert agent.model.config.name_or_path == \
        "/Data/Models/LLM/meta-llama/Llama-3.2-1B-Instruct"

def test_LocalLlama_generate_for_llm_engine_fails_on_empty_messages():

    agent = LocalLlama(
        configuration_llama3,
        generation_configuration_llama3)

    messages = []

    with pytest.raises(IndexError) as err:
        agent.generate_for_llm_engine(messages)
    assert "list index out of range" in str(err.value)

def test_LocalLlama_generate_for_llm_engine_generates():
    agent = LocalLlama(
        configuration_llama3,
        generation_configuration_llama3)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Tokyo?"}
    ]

    response = agent.generate_for_llm_engine(messages)
    assert isinstance(response, str)
    assert len(response) > 0

    print("response:", response)

def test_LocalLlama_generates_with_messages():
    import copy
    import torch

    configuration = copy.deepcopy(configuration_llama3)
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.bfloat16
    generation_configuration_llama3.do_sample = True

    agent = LocalLlama(
        configuration,
        generation_configuration_llama3)

    assert agent.model.device.type == "cuda"
    assert agent.model.device == torch.device("cuda:0")

    messages = [
        create_system_message(
            (
                "You are an academician specializing in physics. "
                "Provide detailed and accurate explanations using "
                "technical terminology")),
        create_user_message(
            "Write 10 subject lines for emails promoting eco-friendly products")
    ]

    response = agent.generate_for_llm_engine(messages)

    assert torch.get_default_device() == torch.device("cuda:0")
    assert isinstance(response, str)
    assert len(response) > 0

    print("response:", response)
