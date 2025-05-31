from corecode.Utilities import (get_environment_variable, load_environment_file)

from moregroq.Tools import GroqAPIAndToolCall
from moregroq.Wrappers import GroqAPIWrapper

from TestUtilities.TestSetup import calculate, reverse_string

load_environment_file()

def test_replicate_run_conversation():
    """
    Reference
    https://console.groq.com/docs/tool-use

    This replicates the testing in test_tool_use.py for run_conversation or
    test_receive_and_handle_tool_results.
    """
    system_message = (
        "You are a calculator assistant. Use the calculate function to "
        "perform mathematical operations and provide the results.")
    user_prompt = "What is 25 * 10 + 10?"

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    groq_api_and_tool_call = GroqAPIAndToolCall(groq_api_wrapper)
    groq_api_and_tool_call.set_tool_choice()

    assert groq_api_and_tool_call.groq_api_wrapper.configuration.model == \
        "llama-3.3-70b-versatile"
    assert groq_api_and_tool_call.groq_api_wrapper.configuration.max_completion_tokens == \
        4096
    assert groq_api_and_tool_call.groq_api_wrapper.configuration.tool_choice == \
        "auto"

    groq_api_and_tool_call.add_system_message(system_message)
    groq_api_and_tool_call.add_tool(calculate)

    groq_api_tools = groq_api_and_tool_call.groq_api_wrapper.configuration.tools
    assert len(groq_api_tools) == 1
    assert groq_api_tools[0].function.name == "calculate"
    assert groq_api_tools[0].function.description == \
        "Evaluate a mathematical expression."
    assert groq_api_tools[0].function.parameters.properties[0].name == "expression"
    assert groq_api_tools[0].function.parameters.properties[0].type == "string"
    assert groq_api_tools[0].function.parameters.properties[0].description == \
        "The mathematical expression to evaluate."

    # Uncomment to print the tool
    # print("\n groq_api_tools: ", groq_api_tools)

    tool_call_available_functions = \
        groq_api_and_tool_call.tool_call_processor.available_functions
    assert len(tool_call_available_functions) == 1
    assert tool_call_available_functions["calculate"] == calculate

    conversation_history = \
        groq_api_and_tool_call.conversation_and_system_messages.get_conversation_as_list_of_dicts()
    assert len(conversation_history) == 1
    assert conversation_history[0]["role"] == "system"
    assert conversation_history[0]["content"] == system_message

    ready_to_call_new_user_prompt = True

    assert groq_api_and_tool_call.create_chat_completion(user_prompt)

    ready_to_call_new_user_prompt = False

    assert groq_api_and_tool_call._current_response is not None
    assert groq_api_and_tool_call._handle_possible_tool_calls_result is None

    print(
        "\n\t 0: groq_api_and_tool_call._current_response: \n",
        groq_api_and_tool_call._current_response)

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()

    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()

    conversation_history = \
        groq_api_and_tool_call.conversation_and_system_messages.get_conversation_as_list_of_dicts()
    assert len(conversation_history) == 3

    assert groq_api_and_tool_call._current_response is None
    assert groq_api_and_tool_call._handle_possible_tool_calls_result is not None

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()

    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()

    assert groq_api_and_tool_call.create_chat_completion()

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()

    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()

    ready_to_call_new_user_prompt = True

    assert ready_to_call_new_user_prompt

    for index, message in enumerate(
        groq_api_and_tool_call.conversation_and_system_messages.get_conversation_as_list_of_dicts()):
        print(
            "\n\t index: ", index,
            "\n\t message: \n", message)

def test_add_more_than_one_tool():
    system_message = (
        "You are either a calculator assistant or you help reverse a given "
        "string. Use the appropriate tool for the task; either use the "
        "calculate function to perform mathematical operations and provide the "
        "results, or use the reverse_string function to reverse a given string "
        "if asked and provide the results."
    )

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    groq_api_and_tool_call = GroqAPIAndToolCall(groq_api_wrapper)
    groq_api_and_tool_call.set_tool_choice()

    groq_api_and_tool_call.add_system_message(system_message)
    groq_api_and_tool_call.add_tool(calculate)
    groq_api_and_tool_call.add_tool(reverse_string)

    groq_api_tools = groq_api_and_tool_call.groq_api_wrapper.configuration.tools
    assert len(groq_api_tools) == 2
    assert groq_api_tools[0].function.name == "calculate"
    assert groq_api_tools[0].function.description == \
        "Evaluate a mathematical expression."

    assert groq_api_tools[1].function.name == "reverse_string"
    assert groq_api_tools[1].function.description == "Reverse a string."

    tool_call_available_functions = \
        groq_api_and_tool_call.tool_call_processor.available_functions
    assert len(tool_call_available_functions) == 2
    assert tool_call_available_functions["calculate"] == calculate
    assert tool_call_available_functions["reverse_string"] == reverse_string

    user_prompt = "What is 25 * 11 + 12?"

    ready_to_call_new_user_prompt = True
    iterations_after_create_chat_completion = 0

    assert groq_api_and_tool_call.create_chat_completion(user_prompt)
    ready_to_call_new_user_prompt = False

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion > 1

    assert groq_api_and_tool_call.create_chat_completion()
    iterations_after_create_chat_completion = 0

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion == 1
    ready_to_call_new_user_prompt = True

    assert ready_to_call_new_user_prompt

    user_prompt = "Reverse the string 'Hello, world!'"

    assert groq_api_and_tool_call.create_chat_completion(user_prompt)
    iterations_after_create_chat_completion = 0
    ready_to_call_new_user_prompt = False

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion > 1

    assert groq_api_and_tool_call.create_chat_completion()
    iterations_after_create_chat_completion = 0

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion == 1
    ready_to_call_new_user_prompt = True

    assert ready_to_call_new_user_prompt

    user_prompt = "What is 26 * 13 + 14?"

    assert groq_api_and_tool_call.create_chat_completion(user_prompt)
    iterations_after_create_chat_completion = 0
    ready_to_call_new_user_prompt = False

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion > 1

    assert groq_api_and_tool_call.create_chat_completion()
    iterations_after_create_chat_completion = 0

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion == 1

    ready_to_call_new_user_prompt = True

    assert ready_to_call_new_user_prompt

    user_prompt = "What is 26 * 13 + 14?"

    assert groq_api_and_tool_call.create_chat_completion(user_prompt)
    iterations_after_create_chat_completion = 0
    ready_to_call_new_user_prompt = False

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion > 1

    assert groq_api_and_tool_call.create_chat_completion()
    iterations_after_create_chat_completion = 0

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion == 1

    ready_to_call_new_user_prompt = True

    assert ready_to_call_new_user_prompt

    user_prompt = "What is the capital of Germany?"
    assert groq_api_and_tool_call.create_chat_completion(user_prompt)
    iterations_after_create_chat_completion = 0
    ready_to_call_new_user_prompt = False

    assert not groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    groq_api_and_tool_call.iteratively_handle_responses_and_tool_calls()
    iterations_after_create_chat_completion += 1

    assert groq_api_and_tool_call._is_no_response_and_no_tool_calls()
    assert iterations_after_create_chat_completion == 1

    ready_to_call_new_user_prompt = True

    assert ready_to_call_new_user_prompt
