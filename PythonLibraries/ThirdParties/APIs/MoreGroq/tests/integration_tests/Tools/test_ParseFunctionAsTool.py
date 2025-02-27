from corecode.Utilities import (get_environment_variable, load_environment_file)
from moregroq.Tools import ParseFunctionAsTool, ToolCallProcessor
from moregroq.Wrappers import GroqAPIWrapper
from moregroq.Wrappers.ChatCompletionConfiguration import Tool

from TestUtilities.TestSetup import calculate

from commonapi.Messages import (
    create_system_message,
    create_user_message
)

load_environment_file()

def test_receive_and_handle_tool_results_with_ParseFunctionAsTool():
    """
    https://console.groq.com/docs/tool-use
    """
    user_prompt = "What is 25 * 10 + 10?"
    messages = [
        create_system_message(
            "You are a calculator assistant. Use the calculate function to perform mathematical operations and provide the results."),
        create_user_message(user_prompt)
    ]

    function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(calculate)

    assert function_definition.name == 'calculate'
    assert function_definition.description == calculate.__doc__
    assert len(function_definition.parameters.properties) == 1
    assert function_definition.parameters.properties[0].name == 'expression'
    assert function_definition.parameters.properties[0].type == 'string'
    assert function_definition.parameters.properties[0].description == ''
    function_definition.parameters.properties[0].description = \
        "The mathematical expression to evaluate"

    tools = [Tool(function=function_definition),]

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.tools = tools
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    response = groq_api_wrapper.create_chat_completion(messages)

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "calculate": calculate
        },
        messages=messages
    )

    process_result = tool_call_processor.process_response(
        response.choices[0].message)

    assert process_result == 1

    response = groq_api_wrapper.create_chat_completion(messages)
    response_message = response.choices[0].message
    print(response_message)
    print(response_message.tool_calls)
    print(response_message.content)
    assert "260" in response_message.content
