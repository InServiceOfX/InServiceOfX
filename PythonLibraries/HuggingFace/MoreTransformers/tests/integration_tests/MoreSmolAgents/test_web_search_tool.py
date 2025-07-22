import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool
from smolagents import TransformersModel

@tool
def visit_website(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown
    string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if
        the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Raise an exception for bad status codes
        response.raise_for_status()

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()
    
        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def test_visit_website():
    url = "https://en.wikipedia.org/wiki/Hugging_Face"
    result = visit_website(url)[:500]
    print(result)

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    WebSearchTool,
    LiteLLMModel,
)

from corecode.Utilities import DataSubdirectories, is_model_there

from moretransformers.Configurations import (
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration)
from transformers import (
    PreTrainedTokenizerFast,
    SmolLM3ForCausalLM,
    TextIteratorStreamer)

import pytest
import torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/HuggingFaceTB/SmolLM3-3B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_init_ToolCallingAgent_with_local_model():
    from_pretrained_configuration = FromPretrainedModelConfiguration(
        model_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = SmolLM3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=from_pretrained_configuration.model_path,
        **from_pretrained_configuration.to_kwargs())
    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")
    # See
    # src/smolagents/agents.py
    assert web_agent.model == model
    assert web_agent.agent_name == "ToolCallingAgent"
    assert web_agent.max_steps == 10
    assert web_agent.description == "Runs web searches for you."
    assert web_agent.state == {}
    print(web_agent.prompt_templates)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_init_ToolCallingAgent_with_TransformersModel():
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        **from_pretrained_tokenizer_configuration.to_kwargs())

    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = SmolLM3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())
    model.config.rope_scaling = {
        "factor": 2.0, #2x65536=131072
        "original_max_position_embeddings": 65536,
        "type": "yarn"
    }
    model.config.max_position_embeddings = 131072

    transformers_model = TransformersModel(model=model, tokenizer=tokenizer)

    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")
    # See
    # src/smolagents/agents.py
    assert web_agent.model == model
    assert web_agent.agent_name == "ToolCallingAgent"
    assert web_agent.max_steps == 10
    assert web_agent.description == "Runs web searches for you."
    assert web_agent.state == {}
    print(web_agent.prompt_templates)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_init_CodeAgent_with_local_model():
    from_pretrained_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = SmolLM3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())
    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"])

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_init_CodeAgent_with_local_model():
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = SmolLM3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())
    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"])

    # "local", "e2b", "docker", "wasm"
    assert manager_agent.executor_type == "local"
    assert manager_agent.executor_kwargs == {}
    assert manager_agent.python_executor is not None

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_run_system_with_local_model():
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = SmolLM3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())
    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"])

    answer = manager_agent.run((
        "If LLM training continues to scale up at the current rhythm until "
        "2030, what would be the electric power in GW required to power the "
        "biggest training runs by 2030? What would that correspond to, "
        "compared to some countries? Please provide a source for any numbers "
        "used."))
    print(answer)
    assert isinstance(answer, str)
    assert len(answer) > 0

    answer = manager_agent.run("What is the capital of France?")
    print(answer)
    assert isinstance(answer, str)
    assert len(answer) > 0
