from corecode.Utilities import (get_environment_variable, load_environment_file)
from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from moregroq.Tools.ParseFunctionAsTool import ParseFunctionAsTool

from moregroq.Wrappers.ChatCompletionConfiguration import (
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty,
    Tool)

from moregroq.Wrappers import GroqAPIWrapper
from TestUtilities.TestSetup import (
    calculate,
    get_bakery_prices,
    reverse_string)

from moregroq.Tools import ToolCallProcessor

load_environment_file()

def test_ToolCallProcessor_works_on_parallel_tool_use():
    """
    https://github.com/groq/groq-api-cookbook/blob/main/tutorials/parallel-tool-use/parallel-tool-use.ipynb
    """
    messages = [
        create_system_message("You are a helpful assistant."),
        create_user_message("What is the price of a cappuccino and croissant?")
    ]
    
    tools = [
        Tool(
            function=FunctionDefinition(
                name="get_bakery_prices",
                description="Get the price of a bakery item",
                parameters=FunctionParameters(
                    properties=[
                        ParameterProperty(
                            name="bakery_item",
                            description="The name of the bakery item",
                            type="string",
                            required=True
                        )
                    ]
                )
            )
        )
    ]

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tools = tools
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_tokens = 4096

    response = groq_api_wrapper.create_chat_completion(messages)

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "get_bakery_prices": get_bakery_prices
        },)

    if GroqAPIWrapper.has_message_in_response(response):
        response_message = response.choices[0].message
        print(response_message.tool_calls)
        handle_possible_tool_calls_result = \
            tool_call_processor.handle_possible_tool_calls(
                response.choices[0].message)

        messages.append(response_message)
        if handle_possible_tool_calls_result is not None:
            for tool_call in handle_possible_tool_calls_result:
                messages.append(tool_call)

        response = groq_api_wrapper.create_chat_completion(messages)
        response_message = response.choices[0].message
        print("response_message: ", response_message)
        print(response_message.tool_calls)
        print(response_message.content)
        assert "4.25" in response_message.content
        assert "4.75" in response_message.content

    else:
        raise Exception("No message in response")


def setup_calculate_tool():
    function_definition = ParseFunctionAsTool.parse_for_function_definition(
        calculate)

    tool = Tool(function=function_definition)
    system_message = (
        "You are a calculator assistant. Use the calculate function to "
        "perform mathematical operations and provide the results.")
    user_prompt = "What is 25 * 10 + 10?"

    messages = [
        create_system_message(system_message),
        create_user_message(user_prompt)
    ]

    return tool, messages

def test_ToolCallProcessor_handle_possible_tool_calls_on_calculate():
    tool, messages = setup_calculate_tool()

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tools = [tool]
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_tokens = 4096

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "calculate": calculate
        },)

    response = groq_api_wrapper.create_chat_completion(messages)

    if GroqAPIWrapper.has_message_in_response(response):
        response_message = response.choices[0].message
        handle_possible_tool_calls_result = \
            tool_call_processor.handle_possible_tool_calls(
                response_message)

        messages.append(response_message)

        print("response_message: ", response_message)
        if handle_possible_tool_calls_result is not None:
            for tool_call in handle_possible_tool_calls_result:
                print("tool_call: ", tool_call)
                messages.append(tool_call)

            second_response = groq_api_wrapper.create_chat_completion(messages)

            print("second_response: ", second_response)
        else:
            print("No tool calls returned.")
 

def test_ToolCallProcessor_handles_one_tool_call():
    tool, messages = setup_calculate_tool()

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tools = [tool]
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_tokens = 4096

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "calculate": calculate,
            "reverse_string": reverse_string
        },)

    response = groq_api_wrapper.create_chat_completion(messages)

    if response is not None and hasattr(response, "choices") and \
        len(response.choices) > 0 and \
        hasattr(response.choices[0], "message"):

        messages.append(response.choices[0].message)
        handle_possible_tool_calls_result = \
            tool_call_processor.handle_possible_tool_calls(
                response.choices[0].message)
        if handle_possible_tool_calls_result is None:
            print(
                "No tool_calls attribute in result: ",
                handle_possible_tool_calls_result)
        else:
            print(
                "len(handle_possible_tool_calls_result): ",
                len(handle_possible_tool_calls_result))
            for tool_call in handle_possible_tool_calls_result:
                print("type(tool_call): ", type(tool_call))
                print("tool_call: ", tool_call)

                messages.append(tool_call)

            second_response = groq_api_wrapper.create_chat_completion(messages)

            if second_response is not None and \
                hasattr(second_response, "choices") and \
                len(second_response.choices) > 0 and \
                hasattr(second_response.choices[0], "message"):
                print(
                    "second_response.choices[0].message: ",
                    second_response.choices[0].message)
            else:
                print("Second response is not as expected:", second_response)
    else:
        print("No response message returned with response:", response)

    for message in messages:
        print("message: ", message)

def test_ToolCallProcessor_handles_two_tools_one_call_at_a_time():
    tool, _ = setup_calculate_tool()

    function_definition = ParseFunctionAsTool.parse_for_function_definition(
        reverse_string)
    reverse_string_tool = Tool(function=function_definition)

    system_message = (
        "You are either a calculator assistant or you help reverse a given "
        "string. Use the appropriate tool for the task; either use the "
        "calculate function to perform mathematical operations and provide the "
        "results, or use the reverse_string function to reverse a given string "
        "if asked and provide the results."
    )

    user_prompt = "What is 27 * 14 + 15?"

    messages = [
        create_system_message(system_message),
        create_user_message(user_prompt)
    ]

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tools = [tool, reverse_string_tool]
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_tokens = 4096

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "calculate": calculate,
            "reverse_string": reverse_string
        },)

    response = groq_api_wrapper.create_chat_completion(messages)

    handle_possible_tool_calls_result = None

    if GroqAPIWrapper.has_message_in_response(response):
        print("response.choices[0].message: ", response.choices[0].message)
        handle_possible_tool_calls_result = \
            tool_call_processor.handle_possible_tool_calls(
                response.choices[0].message)
        if handle_possible_tool_calls_result is None:
            print(
                "No tool_calls attribute in result: ",
                handle_possible_tool_calls_result)
        else:
            print(
                "len(handle_possible_tool_calls_result): ",
                len(handle_possible_tool_calls_result))
            # We need to append the original tool call message to messages here
            # before appending all the tool calls, instead of waiting upon the
            # next cycle.
            messages.append(response.choices[0].message)
            for tool_call in handle_possible_tool_calls_result:
                print("type(tool_call): ", type(tool_call))
                print("tool_call: ", tool_call)
                messages.append(tool_call)
    else:
        print("No response message returned with response:", response)

    if GroqAPIWrapper.has_message_in_response(response) and \
        handle_possible_tool_calls_result is not None and \
        len(handle_possible_tool_calls_result) > 0:
        response = groq_api_wrapper.create_chat_completion(messages)

        if GroqAPIWrapper.has_message_in_response(response):
            print("response.choices[0].message: ", response.choices[0].message)
            messages.append(response.choices[0].message)
            handle_possible_tool_calls_result = \
                tool_call_processor.handle_possible_tool_calls(
                    response.choices[0].message)
            if handle_possible_tool_calls_result is None:
                print(
                    "No tool_calls attribute in result: ",
                    handle_possible_tool_calls_result)
            else:
                print(
                    "len(handle_possible_tool_calls_result): ",
                    len(handle_possible_tool_calls_result))
                messages.append(response.choices[0].message)
                for tool_call in handle_possible_tool_calls_result:
                    print("type(tool_call): ", type(tool_call))
                    print("tool_call: ", tool_call)
                    messages.append(tool_call)
        else:
            print("No response message returned with response:", response)
    elif GroqAPIWrapper.has_message_in_response(response):
        print("response.choices[0].message: ", response.choices[0].message)
        messages.append(response.choices[0].message)
    else:
        print("No response message returned with response:", response)

    print("handle_possible_tool_calls_result: ", handle_possible_tool_calls_result)

    for message in messages:
        print("message: ", message)

    user_prompt = "Reverse the string 'Hello, world!'"
    print("user_prompt: ", user_prompt)

    messages.append(create_user_message(user_prompt))

    response = groq_api_wrapper.create_chat_completion(messages)

    print("response: ", response)

    if GroqAPIWrapper.has_message_in_response(response) and \
        handle_possible_tool_calls_result is None:
        print("response.choices[0].message: ", response.choices[0].message)
        handle_possible_tool_calls_result = \
            tool_call_processor.handle_possible_tool_calls(
                response.choices[0].message)

    print("handle_possible_tool_calls_result: ", handle_possible_tool_calls_result)

