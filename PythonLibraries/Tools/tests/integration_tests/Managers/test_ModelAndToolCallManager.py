from commonapi.Messages import (
    AssistantMessage,
    ConversationSystemAndPermanent,
    UserMessage)
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
from moretransformers.Tools import ToolCallProcessor
from moretransformers.Tools.ToolCallChatTemplates import \
    AssistantMessageWithToolCalls

import pytest, torch

from TestSetup.ToolsForTest import ToolsForTest
from tools.Managers import ModelAndToolCallManager

def create_messages():
    messages = [
        (
            "You are a bot that responds to weather queries. You should reply "
            "with the unit used in the queried location."
        ),
        "Hey, what's the temperature in Paris right now?"
    ]
    return messages

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_process_messages_once_for_any_tool_calls_with_Qwen3_and_single_tool():
    messages = create_messages()

    csp = ConversationSystemAndPermanent()
    csp.add_system_message(messages[0])
    csp.append_message(UserMessage(content=messages[1]))

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "get_current_temperature": ToolsForTest.get_current_temperature,
            "get_current_wind_speed": ToolsForTest.get_current_wind_speed
        })

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

    matcm = ModelAndToolCallManager(
        mat,
        tool_call_processor)

    assert len(csp.get_conversation_as_list_of_dicts()) == 2

    new_messages = []
    process_once_results = matcm._process_messages_once_for_any_tool_calls(
        csp.get_conversation_as_list_of_dicts(),
        new_messages)

    assert len(csp.get_conversation_as_list_of_dicts()) == 2
    assert len(process_once_results) == 4
    assert process_once_results[0]
    assert len(process_once_results[1]) == 3
    assert process_once_results[1][2]["role"] == "assistant"
    assert len(process_once_results[2]) == 1
    # Single tool call was created.
    assert len(new_messages) == 1

    print(process_once_results[2])

def setup_test_for_ModelAndToolCallManager_with_Qwen3(messages):
    csp = ConversationSystemAndPermanent()
    csp.add_system_message(messages[0])
    csp.append_message(UserMessage(content=messages[1]))

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "get_current_temperature": ToolsForTest.get_current_temperature,
            "get_current_wind_speed": ToolsForTest.get_current_wind_speed
        })

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

    return (mat, tool_call_processor, csp)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_run_tool_calls_with_Qwen3_and_single_tool():
    messages = create_messages()

    mat, tool_call_processor, csp = \
        setup_test_for_ModelAndToolCallManager_with_Qwen3(messages)

    mat.load_model()
    mat.load_tokenizer()

    matcm = ModelAndToolCallManager(
        mat,
        tool_call_processor)

    new_messages = []
    process_once_results = matcm._process_messages_once_for_any_tool_calls(
        csp.get_conversation_as_list_of_dicts(),
        new_messages)

    assert process_once_results[0]
    assert len(new_messages) == 1

    messages = matcm._run_tool_calls_and_append_to_messages(
        process_once_results[1],
        process_once_results[2],
        new_messages)

    # Break down, i.e. explicitly run, the steps of the second call of
    # processing messages once for any tool calls.

    print(isinstance(messages, list))
    for index, message in enumerate(messages):
        print(f"index: {index}, message: {message}")
        print(isinstance(message, dict))

    # TODO: We have to add padding=True here but not in the function that wraps
    # this call in ModelAndToolCallManager.process_messages().
    input_ids = mat.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        tools=tool_call_processor.get_tools_as_list(),
        to_device=True,
        padding=True)

    outputs = mat._model.generate(
        **input_ids,
        **mat._generation_configuration.to_dict()
    )

    decoded = mat.decode_with_tokenizer(
        outputs,
        skip_special_tokens=True
    )
    # There are still tool calls from previous messages.
    assert ToolCallProcessor.has_nonempty_tool_call(decoded)

    decoded = mat._tokenizer.decode(
        ToolCallProcessor.parse_generate_output_for_output_only(
            outputs,
            input_ids),
        skip_special_tokens=True)

    tool_calls = ToolCallProcessor._parse_tool_call(decoded)

    assert tool_calls == None

    print("messages[-1]: ", messages[-1])
    print("decoded: ", decoded)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_process_messages_with_Qwen3_and_single_tool():
    messages = create_messages()

    mat, tool_call_processor, csp = \
        setup_test_for_ModelAndToolCallManager_with_Qwen3(messages)

    mat.load_model()
    mat.load_tokenizer()

    matcm = ModelAndToolCallManager(
        mat,
        tool_call_processor)

    process_messages_results = matcm.process_messages(
        csp.get_conversation_as_list_of_dicts())

    # Expected that function returns True in first element.
    assert process_messages_results[0]
    # length 4 for expected tool call completion.
    assert len(process_messages_results) == 4
    # Returning messages.
    assert len(process_messages_results[1]) == 4
    # Returning the successful tool calls.
    assert isinstance(process_messages_results[2], str)
    # These are the newly created messages; first, an
    # AssistantMessageWithToolCalls, and second, a ToolMessage
    assert len(process_messages_results[3]) == 2
    print("process_messages_results[2]: ", process_messages_results[2])

def create_more_user_queries():
    return [
        "Hey, what's the temperature in Paris right now?",
        (
            "What's the weather like in New York? I need both temperature and "
            "wind information."
        ),
        "How windy is it in Tokyo right now?",
        "Tell me about the weather in London."
    ]

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_process_messages_with_Qwen3_and_two_tools():
    more_user_queries = create_more_user_queries()
    messages = [
        ToolsForTest.create_wind_weather_system_message(),
        more_user_queries[0]
    ]

    mat, tool_call_processor, csp = \
        setup_test_for_ModelAndToolCallManager_with_Qwen3(messages)

    mat.load_model()
    mat.load_tokenizer()

    matcm = ModelAndToolCallManager(
        mat,
        tool_call_processor)

    process_messages_results = matcm.process_messages(
        csp.get_conversation_as_list_of_dicts())

    # True for successful tool call to completion.
    assert process_messages_results[0]
    # Length would not be 4 if tool call was not processed to completion.
    assert len(process_messages_results) == 4
    # Returned messages.
    assert len(process_messages_results[1]) == 4
    # These are the newly created messages
    assert len(process_messages_results[3]) == 2
    # Conversation was not mutated through the as_list_of_dicts() method.
    assert len(csp.get_conversation_as_list_of_dicts()) == 2

    assert process_messages_results[1][2] == \
        process_messages_results[3][0].to_dict()
    assert process_messages_results[1][3] == \
        process_messages_results[3][1].to_dict()

    assert not hasattr(process_messages_results[3][0], "content")
    assert isinstance(
        process_messages_results[3][0],
        AssistantMessageWithToolCalls)
    csp.append_message(process_messages_results[3][1])
    assistant_message = \
        matcm._add_assistant_message_to_conversation_system_and_permanent(
            csp,
            process_messages_results[2])
    assert isinstance(assistant_message, AssistantMessage)

    assert len(csp.get_conversation_as_list_of_dicts()) == 4

    csp.append_message(UserMessage(content=more_user_queries[1]))
    process_messages_results = matcm.process_messages(
        csp.get_conversation_as_list_of_dicts())
    assert process_messages_results[0]
    assert len(process_messages_results) == 4
    # To obtain this result, try running again since this appears stochastic for
    # now.
    assert len(process_messages_results[1]) == 8
    assert len(process_messages_results[3]) == 3
    assert len(csp.get_conversation_as_list_of_dicts()) == 5

    new_assistant_message = \
        matcm.update_conversation_system_and_permanent_from_process_messages(
            process_messages_results,
            csp)
    print("new_assistant_message: ", new_assistant_message)

    assert len(csp.get_conversation_as_list_of_dicts()) == 8

    csp.append_message(UserMessage(content=more_user_queries[2]))
    process_messages_results = matcm.process_messages(
        csp.get_conversation_as_list_of_dicts())
    assert process_messages_results[0]
    assert len(process_messages_results) == 4

    new_assistant_message = \
        matcm.update_conversation_system_and_permanent_from_process_messages(
            process_messages_results,
            csp)
    print("new_assistant_message: ", new_assistant_message)

    csp.append_message(UserMessage(content=more_user_queries[3]))
    process_messages_results = matcm.process_messages(
        csp.get_conversation_as_list_of_dicts())
    assert process_messages_results[0]
    assert len(process_messages_results) == 4

    new_assistant_message = \
        matcm.update_conversation_system_and_permanent_from_process_messages(
            process_messages_results,
            csp)
    print("new_assistant_message: ", new_assistant_message)

    for index, message in enumerate(csp.get_conversation_as_list_of_dicts()):
        print(f"index: {index}, message: {message}")
