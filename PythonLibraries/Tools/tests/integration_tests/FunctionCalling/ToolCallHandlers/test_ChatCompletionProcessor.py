from tools.FunctionCalling import ParseFunctionAsTool
from tools.FunctionCalling.FunctionDefinition import Tool
from tools.FunctionCalling.ToolCallHandlers import ChatCompletionProcessor
from tools.Messages.AssistantMessageWithToolCalls import (
    create_assistant_message_with_tool_calls_from_chat_completion_message,
    process_response_choices_into_assistant_messages
)

from commonapi.Clients.OpenAIxGroqClient import OpenAIxGroqClient
from commonapi.Clients.OpenAIxGrokClient import OpenAIxGrokClient
from commonapi.Messages import create_user_message

from corecode.Utilities import get_environment_variable, load_environment_file
load_environment_file()

from TestSetup.function_calling_test_setup import get_horoscope

def test_chat_completion_works_on_OpenAI_API_function_tool_example_with_Groq():
    api_key = get_environment_variable("GROQ_API_KEY")

    client = OpenAIxGroqClient(api_key)
    client.clear_chat_completion_configuration()
    client.configuration.model = "llama-3.3-70b-versatile"

    function_definition = ParseFunctionAsTool.parse_for_function_definition(
        get_horoscope)

    # E               openai.BadRequestError: Error code: 400 - {'error': {'message': 'code=400, message=tools[0].function.name is required, type=invalid_request_error', 'type': 'invalid_request_error'}}
    #tool_dict = Tool(function=function_definition).to_dict()

    tool_dict = Tool(function=function_definition).to_dict_for_function()

    client.configuration.tools = [tool_dict,]

    messages = [
        create_user_message("What is my horoscope? I am an Aquarius.")
    ]
    response = client.create_chat_completion(messages)

    # <class 'openai.types.chat.chat_completion.ChatCompletion'>
    #print("type(response): ", type(response))
    # 'choices', 'construct', 'copy', 'created', 'dict', 'from_orm', 'id', 'json', 'model', 'model_computed_fields', 'model_config', 'model_construct', 'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 'model_validate_json', 'model_validate_strings', 'object', 'parse_file', 'parse_obj', 'parse_raw', 'schema', 'schema_json', 'service_tier', 'system_fingerprint', 'to_dict', 'to_json', 'update_forward_refs', 'usage', 'validate'
    #print([attr for attr in dir(response)])
    # [<class 'openai.types.chat.chat_completion.Choice'>]
    #print([type(choice) for choice in response.choices])
    # 'construct', 'copy', 'dict', 'finish_reason', 'from_orm', 'index', 'json', 'logprobs', 'message', 'model_computed_fields', 'model_config', 'model_construct', 'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 'model_validate_json', 'model_validate_strings', 'parse_file', 'parse_obj', 'parse_raw', 'schema', 'schema_json', 'to_dict', 'to_json', 'update_forward_refs', 'validate'
    # print([attr for attr in dir(response.choices[0])])
    # <class 'openai.types.chat.chat_completion_message.ChatCompletionMessage'>
    #print(type(response.choices[0].message))
    # 'annotations', 'audio', 'construct', 'content', 'copy', 'dict', 'from_orm', 'function_call', 'json', 'model_computed_fields', 'model_config', 'model_construct', 'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 'model_validate_json', 'model_validate_strings', 'parse_file', 'parse_obj', 'parse_raw', 'refusal', 'role', 'schema', 'schema_json', 'to_dict', 'to_json', 'tool_calls', 'update_forward_refs', 'validate'
    #print([attr for attr in dir(response.choices[0].message)])

    # Notice content, function_call (?), role, and tool_calls.

    assert response.choices[0].message.role == "assistant"

    # None
    #print(response.choices[0].message.content)
    assert response.choices[0].message.content == None

    # None
    #print(response.choices[0].message.function_call)

    assert response.choices[0].message.function_call == None

    # [ChatCompletionMessageFunctionToolCall(id='rcn22ja0d', function=Function(arguments='{"sign":"Aquarius"}', name='get_horoscope'), type='function')]
    #print(response.choices[0].message.tool_calls)
    # <class 'list'>
    #print(type(response.choices[0].message.tool_calls))

    assert len(response.choices[0].message.tool_calls) == 1

    # <class 'openai.types.chat.chat_completion_message_function_tool_call.ChatCompletionMessageFunctionToolCall'>
    #print(type(response.choices[0].message.tool_calls[0]))

    # Notice function, id, type.
    # 'construct', 'copy', 'dict', 'from_orm', 'function', 'id', 'json', 'model_computed_fields', 'model_config', 'model_construct', 'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 'model_validate_json', 'model_validate_strings', 'parse_file', 'parse_obj', 'parse_raw', 'schema', 'schema_json', 'to_dict', 'to_json', 'type', 'update_forward_refs', 'validate'
    # print([attr for attr in dir(response.choices[0].message.tool_calls[0])])
    # Example (actual) id:
    # 7d7q63jzq
    #print(response.choices[0].message.tool_calls[0].id)
    # function
    #print(response.choices[0].message.tool_calls[0].type)
    # <class 'openai.types.chat.chat_completion_message_function_tool_call.Function'>
    #print((type(response.choices[0].message.tool_calls[0].function)))

    # Notice arguments, name
    # 'arguments', 'construct', 'copy', 'dict', 'from_orm', 'json', 'model_computed_fields', 'model_config', 'model_construct', 'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 'model_validate_json', 'model_validate_strings', 'name', 'parse_file', 'parse_obj', 'parse_raw', 'schema', 'schema_json', 'to_dict', 'to_json', 'update_forward_refs', 'validate'
    # print([attr for attr in dir(response.choices[0].message.tool_calls[0].function)])

    #print(response.choices[0].message.tool_calls[0].function.arguments)

    assistant_messages = []
    for choice in response.choices:

        assistant_message = \
            create_assistant_message_with_tool_calls_from_chat_completion_message(
                choice.message)
        assistant_messages.append(assistant_message)

    for assistant_message in assistant_messages:
        messages.append(assistant_message.to_dict())

    processor = ChatCompletionProcessor(
        process_function_result=ChatCompletionProcessor.default_result_to_string
    )

    processor.add_function("get_horoscope", get_horoscope)

    all_tool_call_messages = []

    for choice in response.choices:
        tool_call_messages = processor.handle_possible_tool_calls(
            choice.message)
        all_tool_call_messages += tool_call_messages

    for tool_call_message in all_tool_call_messages:
        messages.append(tool_call_message)

    for index, message in enumerate(messages):
        print(f"index: {index}, message: {message}")

    # TypeError: Completions.create() got an unexpected keyword argument 'instructions'
    # client.configuration.instructions = \
    #     "Respond only with a horoscope generated by a tool."

    response = client.create_chat_completion(messages)

    assert not processor.choices_has_tool_calls(response.choices)

    # I don't have have any more information on your horoscope. If you want your horoscope for a different date, let me know.
    print(response.choices[0].message.content)

def test_chat_completion_works_on_OpenAI_API_function_tool_example_with_Grok():
    api_key = get_environment_variable("XAI_API_KEY")

    client = OpenAIxGrokClient(api_key)
    client.clear_chat_completion_configuration()
    client.configuration.model = "grok-4"

    function_definition = ParseFunctionAsTool.parse_for_function_definition(
        get_horoscope)

    tool_dict = Tool(function=function_definition).to_dict_for_function()

    client.configuration.tools = [tool_dict,]

    messages = [
        create_user_message("What is my horoscope? I am an Aquarius.")
    ]
    response = client.create_chat_completion(messages)

    # Add to messages the messages calling for, requesting, a tool call.
    assistant_messages = process_response_choices_into_assistant_messages(
        response.choices)
    for assistant_message in assistant_messages:
        messages.append(assistant_message.to_dict())

    processor = ChatCompletionProcessor(
        process_function_result=ChatCompletionProcessor.default_result_to_string
    )
    processor.add_function("get_horoscope", get_horoscope)

    all_tool_call_messages = processor.process_all_tool_call_requests(
        response.choices)

    for tool_call_message in all_tool_call_messages:
        messages.append(tool_call_message)

    response = client.create_chat_completion(messages)

    assert not processor.choices_has_tool_calls(response.choices)

    # I don't have have any more information on your horoscope. If you want your horoscope for a different date, let me know.
    print(response.choices[0].message.content)
