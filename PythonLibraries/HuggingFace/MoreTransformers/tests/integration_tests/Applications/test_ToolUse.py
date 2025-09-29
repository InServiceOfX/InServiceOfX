from corecode.Utilities import DataSubdirectories, is_model_there

data_subdirectories = DataSubdirectories()
relative_model_path = "Models/LLM/Qwen/Qwen3-0.6B"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)
model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,)

import json, pytest, re, torch

# https://huggingface.co/docs/transformers/en/chat_extras#passing-tools
# Although passing Python functions is very convenient (pass functions to the
# tools argument of apply_chat_template().), parser can only handle Google-style
# docstrings.
# https://huggingface.co/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template

def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    # A real function should probably actually get the temperature!
    return 22.

def get_current_wind_speed(location: str):
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the wind speed for, in the format "City, Country"
    """
    # A real function should probably actually get the wind speed!
    return 6.

tools = [get_current_temperature, get_current_wind_speed]

messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]

def parse_tool_call(s: str) -> dict:
    return json.loads(
        re.search(
            r'<tool_call>\s*(.*?)\s*</tool_call>', s, re.DOTALL).group(1))

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_use_Qwen3_following_tool_use_example():
    """
    https://huggingface.co/docs/transformers/en/chat_extras
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
    # Note: If you *don't* set do_sample = True, you'll get this warning:
    # The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'min_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
    generation_configuration.do_sample = True

    mat = ModelAndTokenizer(
        model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration= \
            from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)

    mat.load_model()
    mat.load_tokenizer()

    tools = [get_current_temperature, get_current_wind_speed]

    input_ids = mat.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        tools=tools,
        to_device=True)

    # data: {'input_ids': ...}
    #for attr, value in vars(input_ids).items():
    #    print(f"{attr}: {value}")

    # KeysView 'input_ids', 'attention_mask', _encodings
    #print(input_ids.keys())

    assert len(vars(input_ids)) == 3

    outputs = mat._model.generate(
        **input_ids,
        max_new_tokens=mat._generation_configuration.max_new_tokens,
        temperature=mat._generation_configuration.temperature,
        top_p=mat._generation_configuration.top_p,
        top_k=mat._generation_configuration.top_k,
        min_p=mat._generation_configuration.min_p,
        do_sample=mat._generation_configuration.do_sample)

    # We will provide a function for detecthing a tool call.
    # skip_special_tokens=False results in tags <|im_start|>, <|im_end|>,
    decoded_from_mat = mat.decode_with_tokenizer(
        outputs,
        skip_special_tokens=True)

    # It includes the thinking.
    #print(decoded_from_mat)

    # This parses out the tool calling exactly.
    decoded = mat._tokenizer.decode(
        outputs[0][len(input_ids["input_ids"][0]):],
        skip_special_tokens=True)

    #<tool_call>
    #{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
    #</tool_call>
    #print(decoded)

    tool_call = parse_tool_call(decoded)

    # {'name': 'get_current_temperature', 'arguments': {'location': 'Paris, France', 'unit': 'celsius'}}
    #print(tool_call)

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": tool_call
                }
            ]
        }
    )

    print(messages)
