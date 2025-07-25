import requests
from markdownify import markdownify
from moretransformers.MoreSmolAgents \
    import create_TransformersModel_from_configurations
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
    configure_RoPE_scaling,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,
    GenerationConfiguration)
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
        **from_pretrained_configuration.to_dict())
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
        **from_pretrained_tokenizer_configuration.to_dict())

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

    generation_configuration = GenerationConfiguration(
        max_new_tokens=131072,
        do_sample=True,
        temperature=0.6,
        top_p=0.95)

    transformers_model = TransformersModel(
        # This can be a path or model identifier. See
        # src/smolagents/models.py
        model_id=str(
            from_pretrained_model_configuration.pretrained_model_name_or_path),
        device_map=from_pretrained_model_configuration.device_map,
        torch_dtype=from_pretrained_model_configuration.torch_dtype,
        trust_remote_code=from_pretrained_model_configuration.trust_remote_code,
        **generation_configuration.to_dict())

    # See
    # src/smolagents/agents.py
    assert transformers_model.tokenizer != tokenizer
    transformers_model.tokenizer = tokenizer
    assert transformers_model.tokenizer == tokenizer

    transformers_model.streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True)

    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=transformers_model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")

    assert web_agent.model == transformers_model
    assert isinstance(web_agent.system_prompt, str)
    assert len(web_agent.system_prompt) == 3669
    print(web_agent.system_prompt)

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
def test_init_CodeAgent_with_TransformersModel():
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        **from_pretrained_tokenizer_configuration.to_dict())

    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = SmolLM3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())
    assert model.config.max_position_embeddings == 65536
    model.config.rope_scaling = {
        "factor": 2.0, #2x65536=131072
        "original_max_position_embeddings": 65536,
        "type": "yarn"
    }
    assert model.config.max_position_embeddings == 65536
    model.config.max_position_embeddings = 131072

    generation_configuration = GenerationConfiguration(
        max_new_tokens=131072,
        do_sample=True,
        temperature=0.6,
        top_p=0.95)

    transformers_model = TransformersModel(
        model_id=str(
            from_pretrained_model_configuration.pretrained_model_name_or_path),
        device_map=from_pretrained_model_configuration.device_map,
        torch_dtype=from_pretrained_model_configuration.torch_dtype,
        trust_remote_code=from_pretrained_model_configuration.trust_remote_code,
        **generation_configuration.to_dict())

    assert transformers_model.model != model
    transformers_model.model = model
    assert transformers_model.model == model
    transformers_model.tokenizer = tokenizer
    transformers_model.streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True)

    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=transformers_model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")
    manager_agent = CodeAgent(
        tools=[],
        model=transformers_model,
        managed_agents=[web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"])

    # "local", "e2b", "docker", "wasm"
    assert manager_agent.executor_type == "local"
    assert manager_agent.executor_kwargs == {}
    assert manager_agent.python_executor is not None

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_run_system_with_local_model():
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        **from_pretrained_tokenizer_configuration.to_dict())
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = SmolLM3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())
    configure_RoPE_scaling(model.config, factor=2.0, type="yarn")
    assert model.config.max_position_embeddings == 131072
    assert model.config.rope_scaling["factor"] == 2.0
    assert model.config.rope_scaling["type"] == "yarn"
    assert model.config.rope_scaling["original_max_position_embeddings"] == \
        65536

    generation_configuration = GenerationConfiguration(
        max_new_tokens=131072,
        do_sample=True,
        temperature=0.6,
        top_p=0.95)

    transformers_model = create_TransformersModel_from_configurations(
        from_pretrained_model_configuration,
        generation_configuration)
    assert transformers_model.model != model
    transformers_model.model = model
    assert transformers_model.model == model
    assert transformers_model.tokenizer != tokenizer
    transformers_model.tokenizer = tokenizer
    assert transformers_model.tokenizer == tokenizer
    transformers_model.streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True)

    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=transformers_model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")
    manager_agent = CodeAgent(
        tools=[],
        model=transformers_model,
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

    # Obtained CUDA out of memory.

    answer = manager_agent.run("What is the capital of France?")
    print(answer)
    assert isinstance(answer, str)
    assert len(answer) > 0

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_run_system_with_lower_new_tokens():
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        **from_pretrained_tokenizer_configuration.to_dict())
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = SmolLM3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())

    generation_configuration = GenerationConfiguration(
        # CUDA out of memory.
        #max_new_tokens=32768,
        max_new_tokens=16384,
        do_sample=True,
        temperature=0.6,
        top_p=0.95)

    transformers_model = create_TransformersModel_from_configurations(
        from_pretrained_model_configuration,
        generation_configuration)

    del transformers_model.model
    del transformers_model.tokenizer
    del transformers_model.streamer

    transformers_model.model = model
    transformers_model.tokenizer = tokenizer
    transformers_model.streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True)

    web_agent = ToolCallingAgent(
        tools=[WebSearchTool(), visit_website],
        model=transformers_model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches for you.")
    manager_agent = CodeAgent(
        tools=[],
        model=transformers_model,
        managed_agents=[web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"])

    answer = manager_agent.run((
        "If LLM training continues to scale up at the current rhythm until "
        "2030, what would be the electric power in GW required to power the "
        "biggest training runs by 2030? What would that correspond to, "
        "compared to some countries? Please provide a source for any numbers "
        "used."))
    print("\n First answer: \n", answer)
    assert isinstance(answer, str)
    assert len(answer) > 0

    answer = manager_agent.run("What is the capital of France?")
    print(answer)
    assert isinstance(answer, str)
    assert len(answer) > 0
