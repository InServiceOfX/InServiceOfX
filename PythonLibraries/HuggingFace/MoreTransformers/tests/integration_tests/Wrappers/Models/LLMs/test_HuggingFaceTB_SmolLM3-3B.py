"""
USAGE:

You can specify both the pytest file to run *and* a substring to match for in
the test name of the tests you want to run, e.g.

pytest -s ./integration_tests/Wrappers/Models/LLMs/test_HuggingFaceTB_SmolLM3-3B.py -k "test_AutoModelFor"
"""

from commonapi.Messages import UserMessage
from corecode.Utilities import DataSubdirectories, is_model_there

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    SmolLM3ForCausalLM,
    SmolLM3Config)

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
def test_AutoModelForCausalLM_from_pretrained_works():
    model = AutoModelForCausalLM.from_pretrained(model_path)
    assert model is not None
    assert isinstance(model, SmolLM3ForCausalLM)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_SmolLM3ForCausalLM_from_pretrained_works():
    model = SmolLM3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    assert model is not None
    assert isinstance(model, SmolLM3ForCausalLM)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_AutoTokenizer_from_pretrained_works():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert tokenizer is not None
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_PreTrainedTokenizerFast_from_pretrained_works():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)
    assert tokenizer is not None
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    assert tokenizer.chat_template is not None
    assert isinstance(tokenizer.chat_template, str)

    # Uncomment to print the chat template
    print(tokenizer.chat_template)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_works():
    """
    No particular reason to use this example from here:
    https://huggingface.co/LiquidAI/LFM2-1.2B
    other than it's a simple example.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = SmolLM3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    dict_of_tokenizer_outputs = tokenizer.apply_chat_template(
        # src/transformers/tokenization_utils_base.py
        # Union[list[dict[str, str]], list[list[dict[str, str]]]]
        conversation=[{"role": "user", "content": prompt}],
        # bool, optional
        # If set, a prompt with the token(s) that indicate the start of an
        # assistant message will be appended to the formatted output. This is
        # useful when you want to generate a response from the model.
        add_generation_prompt=True,
        return_tensors="pt",
        # bool, defaults to True
        # Whether to tokenize output. If Calse, output will be a string.
        tokenize=True,
        # bool, defaults to False
        # Whether to return a dictionary with named outputs. Has no effect if
        # tokenize is False.
        return_dict=True).to(model.device)

    assert "input_ids" in dict_of_tokenizer_outputs
    assert "attention_mask" in dict_of_tokenizer_outputs

    output = model.generate(
        input_ids=dict_of_tokenizer_outputs["input_ids"],
        attention_mask=dict_of_tokenizer_outputs["attention_mask"],
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

    # Follow the example given in the README.md from here:
    # https://huggingface.co/HuggingFaceTB/SmolLM3-3B

    prompt = "Give me a brief explanation of gravity in simple terms."
    user_message = UserMessage(content=prompt)

    tokenizer_outputs = tokenizer.apply_chat_template(
        conversation=[user_message.to_dict()],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True).to(model.device)

    output = model.generate(
        input_ids=tokenizer_outputs["input_ids"],
        attention_mask=tokenizer_outputs["attention_mask"],
        do_sample=True,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        temperature=0.6,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        top_p=0.95,
        # Optional
        min_p=0.15,
        # Optional
        repetition_penalty=1.05,
        max_new_tokens=32768
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_config_is_SmolLM3Config():
    model = SmolLM3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )

    assert isinstance(model.config, SmolLM3Config)
    assert model.config.rope_scaling is None

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_modify_config_for_longer_context_window():
    """
    From here:
    https://huggingface.co/HuggingFaceTB/SmolLM3-3B#long-context-processing
    
    Long context processing

The current config.json is set for context length up to 65,536 tokens. To handle longer inputs (128k or 256k), we utilize YaRN you can change the max_position_embeddings and rope_scaling` to:

{
  ...,
  "rope_scaling": {
    "factor": 2.0, #2x65536=131 072 
    "original_max_position_embeddings": 65536,
    "type": "yarn"
  }
}
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = SmolLM3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )

    # From here:
    # src/transformers/models/smollm3/configuration_smollm3.py
    # rope_scaling (Dict, optional)
    # Dictionary containing scaling configuration for the RoPE embeddings. NOTE:
    # if you apply new rope type and you expect the model to work on longer
    # 'max_position_embeddings', we recommend you to update this value
    # accordingly.
    # rope_type str, The sub-variant of RoPE to use. Can be one of ['default,
    # 'linear', 'dynamic', 'yarn', longrope', ...]
    # factor (float, optional)
    model.config.rope_scaling = {
        "factor": 2.0, #2x65536=131072
        "original_max_position_embeddings": 65536,
        "type": "yarn"
    }

    model.config.max_position_embeddings = 131072

    prompt = "What is C. elegans?"
    user_message = UserMessage(content=prompt)

    tokenizer_outputs = tokenizer.apply_chat_template(
        conversation=[user_message.to_dict()],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True).to(model.device)

    output = model.generate(
        input_ids=tokenizer_outputs["input_ids"],
        attention_mask=tokenizer_outputs["attention_mask"],
        do_sample=True,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        temperature=0.6,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        top_p=0.95,
        max_new_tokens=131072
        )

    assert len(output) == 1

    print(tokenizer.decode(output[0], skip_special_tokens=True))

    prompt = "Give me a brief explanation of gravity in simple terms."
    user_message = UserMessage(content=prompt)

    tokenizer_outputs = tokenizer.apply_chat_template(
        conversation=[user_message.to_dict()],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True).to(model.device)

    output = model.generate(
        input_ids=tokenizer_outputs["input_ids"],
        attention_mask=tokenizer_outputs["attention_mask"],
        do_sample=True,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        temperature=0.6,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        top_p=0.95,
        max_new_tokens=131072
        )

    assert len(output) == 1

    print(tokenizer.decode(output[0], skip_special_tokens=True))

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_xml_tool_use_works():
    """
    From here:
    https://huggingface.co/HuggingFaceTB/SmolLM3-3B#agentic-usage    
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = SmolLM3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )
    model.config.rope_scaling = {
        "factor": 2.0, #2x65536=131072
        "original_max_position_embeddings": 65536,
        "type": "yarn"
    }

    model.config.max_position_embeddings = 131072

    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get the weather for"}
                }
            }
        }
    ]

    prompt = "Hello! How is the weather today in Copenhagen?"
    user_message = UserMessage(content=prompt)

    tokenizer_outputs = tokenizer.apply_chat_template(
        conversation=[user_message.to_dict()],
        xml_tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True).to(model.device)

    output = model.generate(
        input_ids=tokenizer_outputs["input_ids"],
        attention_mask=tokenizer_outputs["attention_mask"],
        do_sample=True,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        temperature=0.6,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        top_p=0.95,
        max_new_tokens=131072
        )

    assert len(output) == 1

    print(tokenizer.decode(output[0], skip_special_tokens=True))

    prompt = "Give me a brief explanation of gravity in simple terms."
    user_message = UserMessage(content=prompt)

    tokenizer_outputs = tokenizer.apply_chat_template(
        conversation=[user_message.to_dict()],
        xml_tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True).to(model.device)

    output = model.generate(
        input_ids=tokenizer_outputs["input_ids"],
        attention_mask=tokenizer_outputs["attention_mask"],
        do_sample=True,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        temperature=0.6,
        # Recommended by https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        top_p=0.95,
        max_new_tokens=131072
        )

    assert len(output) == 1

    print(tokenizer.decode(output[0], skip_special_tokens=True))
