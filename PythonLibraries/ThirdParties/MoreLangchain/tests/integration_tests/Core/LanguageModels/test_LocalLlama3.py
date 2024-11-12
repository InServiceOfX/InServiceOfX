from morelangchain.Core.LanguageModels import LocalLlama3

from moretransformers.Configurations import (
    Configuration,
    GenerationConfiguration)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder)

from langchain_core.runnables import RunnableSequence

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

def test_LocalLlama3(more_transformers_test_data_directory):
    configuration = Configuration(
        more_transformers_test_data_directory / \
            "configuration-llama3.yml")
    generation_configuration = GenerationConfiguration(
        more_transformers_test_data_directory / \
            "generation_configuration-llama3.yml")

    agent = LocalLlama3(
        configuration=configuration,
        generation_configuration=generation_configuration)
    
    assert isinstance(agent.model, LlamaForCausalLM)
    assert isinstance(agent.tokenizer, PreTrainedTokenizerFast)
    assert agent.model.config.pad_token_id == 128009

def test_LocalLlama3_chains(more_transformers_test_data_directory):
    configuration = Configuration(
        more_transformers_test_data_directory / \
            "configuration-llama3.yml")
    generation_configuration = GenerationConfiguration(
        more_transformers_test_data_directory / \
            "generation_configuration-llama3.yml")
    
    agent = LocalLlama3(
        configuration=configuration,
        generation_configuration=generation_configuration)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | agent

    assert isinstance(chain, RunnableSequence)