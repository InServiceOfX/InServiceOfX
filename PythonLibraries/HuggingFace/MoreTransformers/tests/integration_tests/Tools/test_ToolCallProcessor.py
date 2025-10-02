from corecode.Utilities import DataSubdirectories, is_model_there

data_subdirectories = DataSubdirectories()
relative_model_path = "Models/LLM/Qwen/Qwen3-0.6B"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)
model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

from commonapi.Messages import ToolMessage

from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,)
from moretransformers.Tools import ToolCallProcessor

import pytest, torch

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
    tool_call_processor = ToolCallProcessor(
        available_functions={
            "get_current_temperature": get_current_temperature,
            "get_current_wind_speed": get_current_wind_speed
        })

    assert tools == tool_call_processor.get_tools_as_list()

    # Pass messages, and list of tools to apply_chat_template. Tokenize the
    # chat and then generate a response.

    input_ids = mat.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        tools=tools,
        to_device=True)

    outputs = mat._model.generate(
        **input_ids,
        **mat._generation_configuration.to_dict(),
        )

    # We will provide a function for detecting a tool call.
    # skip_special_tokens=False results in tags <|im_start|>, <|im_end|>,
    decoded_from_mat = mat.decode_with_tokenizer(
        outputs,
        skip_special_tokens=True)

    assert ToolCallProcessor.has_tool_call(decoded_from_mat)

    # The chat model should have called get_current_temperature tool with the
    # correct parameters from the docstring. It inferred France as the location
    # based on Parise, and it should use Celsius for the units of temperature.

    # This parses out the tool calling exactly.
    decoded = mat._tokenizer.decode(
        ToolCallProcessor.parse_generate_output_for_output_only(
            outputs,
            input_ids),
        skip_special_tokens=True)

    # This could possible be None as in no tool calls were found. But we are
    # checking above that there is a tool call.
    tool_calls = ToolCallProcessor._parse_tool_call(decoded)

    # Hold the call in the tool_calls key of an assistant message. This is the
    # recommended API, and should be supported by chat template of most
    # tool-using models.
    # Although tool_calls is similar to the OpenAI API, the OpenAI API uses a
    # JSON string as its tool_calls format. This may cause errors if used in
    # Transformers, which expects a dict.

    assistant_message_with_tool_calls = \
        ToolCallProcessor._convert_tool_calls_to_assistant_message(tool_calls)

    messages.append(assistant_message_with_tool_calls.to_dict())

    #print(messages)

    tool_call_responses = tool_call_processor.handle_possible_tool_calls(
        tool_calls)

    # [22.0]
    #print(tool_call_responses)
    assert tool_call_responses[0] == 22.0

    # Append the tool response to the chat history with the tool role.

    tool_response_message = ToolMessage(
        content=str(tool_call_responses[0]),
        role="tool")

    messages.append(tool_response_message.to_dict())

    # Finally, allow the model to read tool response and reply to user.

    input_ids = mat.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        tools=tools,
        to_device=True)

    outputs = mat._model.generate(
        **input_ids,
        **mat._generation_configuration.to_dict(),
        )

    decoded_from_mat = mat.decode_with_tokenizer(
        outputs,
        skip_special_tokens=True)

    assert isinstance(decoded_from_mat, str)
    # This also has the system message, Tools section
    #print("\n decoded_from_mat: ", decoded_from_mat)

    decoded = mat._tokenizer.decode(
        ToolCallProcessor.parse_generate_output_for_output_only(
            outputs,
            input_ids),
        skip_special_tokens=True)

    #print("\n decoded: ", decoded)

    thinking_content, content = mat._parse_generate_output_into_thinking_and_content(
        input_ids,
        outputs)

    print("\n thinking_content: ", thinking_content)
    print("\n content: ", content)