from moretransformers.Configurations import Configuration, GenerationConfiguration
from moretransformers.Wrappers.LLMEngines import LocalLlamaAgent

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

from pathlib import Path

import pytest

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

configuration_llama3 = Configuration(
    test_data_directory / "configuration-llama3.yml")

generation_configuration_llama3 = GenerationConfiguration(
    test_data_directory / "generation_configuration-llama3.yml")

def test_LocalLlamaAgent_inits():
    agent = LocalLlamaAgent(
        configuration_llama3,
        generation_configuration_llama3)
    assert isinstance(agent, LocalLlamaAgent)
    assert isinstance(agent.model, LlamaForCausalLM)
    assert isinstance(agent.tokenizer, PreTrainedTokenizerFast)
    assert agent.model.config.pad_token_id == \
        generation_configuration_llama3.eos_token_id[0]

def test_LocalLlamaAgent_generate_for_llm_engine_fails_on_empty_messages():

    agent = LocalLlamaAgent(
        configuration_llama3,
        generation_configuration_llama3)

    messages = []

    with pytest.raises(IndexError) as err:
        agent.generate_for_llm_engine(messages)
    assert "list index out of range" in str(err.value)

def test_LocalLlamaAgent_generate_for_llm_engine_generates():
    agent = LocalLlamaAgent(
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
