from corecode.Utilities import (get_environment_variable, load_environment_file)

from moregroq.Tools import GroqAPIAndToolCall, GroqAPIAndToolCallRunner
from moregroq.Wrappers import GroqAPIWrapper

from moregroq.ToolApplications.StateMachine.FiniteStateMachine import (
    FiniteStateMachine,
    FiniteStateMachineRunner,
    create_fsm_functions,
    create_fsm_tools_with_strings,
    get_user_input
)

from TestUtilities.TestSetup import PizzaCustomerSupportFiniteStateMachine

import pytest

load_environment_file()

def customer_support_fsm_components():
    states={
        "q_start",
        "q_identify_intent",
        "q_fetch_data",
        "q_process_request",
        "q_resolve",
        "q_error"
    }
    alphabet={
        "query_received",
        "intent_identified",
        "data_fetched",
        "request_processed",
        "error",
        "retry"
    }
    initial_state = "q_start"
    transition_function={
        ("q_start", "query_received"): "q_identify_intent",
        ("q_identify_intent", "intent_identified"): "q_fetch_data",
        ("q_identify_intent", "error"): "q_error",
        ("q_fetch_data", "data_fetched"): "q_process_request",
        ("q_fetch_data", "error"): "q_error",
        ("q_process_request", "request_processed"): "q_resolve",
        ("q_process_request", "error"): "q_error",
        ("q_error", "retry"): "q_start",
        ("q_error", "error"): "q_error",
        ("q_resolve", "query_received"): "q_identify_intent",
        ("q_resolve", "error"): "q_error",
        ("q_start", "error"): "q_error",
        ("q_start", "intent_identified"): "q_error",
        ("q_start", "data_fetched"): "q_error",
        ("q_start", "request_processed"): "q_error",
        ("q_identify_intent", "query_received"): "q_error",
        ("q_identify_intent", "data_fetched"): "q_error",
        ("q_identify_intent", "request_processed"): "q_error",
        ("q_fetch_data", "query_received"): "q_error",
        ("q_fetch_data", "intent_identified"): "q_error",
        ("q_fetch_data", "request_processed"): "q_error",
        ("q_process_request", "query_received"): "q_error",
        ("q_process_request", "intent_identified"): "q_error",
        ("q_process_request", "data_fetched"): "q_error",
        ("q_resolve", "intent_identified"): "q_error",
        ("q_resolve", "data_fetched"): "q_error",
        ("q_resolve", "request_processed"): "q_error",
        ("q_error", "query_received"): "q_error",
        ("q_error", "intent_identified"): "q_error",
        ("q_error", "data_fetched"): "q_error",
        ("q_error", "request_processed"): "q_error",
    },
    final_states={"q_resolve", "q_error"}

    return {
        "states": states,
        "alphabet": alphabet,
        "initial_state": initial_state,
        "transition_function": transition_function,
        "final_states": final_states
    }

@pytest.fixture
def customer_support_fsm():
    return FiniteStateMachine(
        states=customer_support_fsm_components()["states"],
        alphabet=customer_support_fsm_components()["alphabet"],
        initial_state=customer_support_fsm_components()["initial_state"],
        transition_function=customer_support_fsm_components()[
            "transition_function"],
        final_states=customer_support_fsm_components()["final_states"]
    )

@pytest.fixture
def customer_support_fsm_runner(customer_support_fsm):
    runner = FiniteStateMachineRunner(customer_support_fsm)
    runner.reset()
    return runner

customer_support_alphabet_string = "{" + \
    ", ".join(customer_support_fsm_components()["alphabet"]) + \
        "}"

system_message_0 = (
    "You are a customer support assistant managing a state machine. Based "
    "on the current state and user input, decide the next action (symbol) "
    "from the alphabet: {customer_support_alphabet_string}. Return the symbol "
    "as a JSON object: {'symbol': 'symbol_name'}."
)


def test_customer_support_fsm_setup_with_groq_api(customer_support_fsm_runner):
    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    #groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.model = "qwen-qwq-32b"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    groq_api_and_tool_call = GroqAPIAndToolCall(groq_api_wrapper)
    groq_api_and_tool_call.set_tool_choice()

    groq_api_and_tool_call.add_system_message(system_message_0)

    input_get_current_state, input_run_step = create_fsm_functions(
        customer_support_fsm_runner)

    get_current_state, run_step = \
        create_fsm_tools_with_strings(input_get_current_state, input_run_step)

    groq_api_and_tool_call.add_tool(get_current_state)
    groq_api_and_tool_call.add_tool(run_step)

    assert len(customer_support_fsm_runner.transition_history) == 0
    assert customer_support_fsm_runner.transition_history == []

    assert get_current_state() == "q_start"
    assert customer_support_fsm_runner.current_state == "q_start"

def test_customer_support_fsm_steps(customer_support_fsm_runner, monkeypatch):
    try:
        groq_api_wrapper = GroqAPIWrapper(
            get_environment_variable("GROQ_API_KEY"))
        # groq.BadRequestError: Error code: 400 - {'error': {'message': "'messages.2' : for 'role:assistant' the following must be satisfied[('messages.2' : property 'reasoning' is unsupported, did you mean 'role'?)]", 'type': 'invalid_request_error'}}
        #groq_api_wrapper.configuration.model = "qwen-qwq-32b"
        groq_api_wrapper.configuration.model = "gemma2-9b-it"
        groq_api_wrapper.configuration.max_completion_tokens = 4096

        groq_api_and_tool_call = GroqAPIAndToolCall(groq_api_wrapper)
        groq_api_and_tool_call.set_tool_choice()

        groq_api_and_tool_call.add_system_message(system_message_0)

        input_get_current_state, input_run_step = create_fsm_functions(
            customer_support_fsm_runner)

        get_current_state, run_step = \
            create_fsm_tools_with_strings(
                input_get_current_state,
                input_run_step)

        groq_api_and_tool_call.add_tool(get_current_state)
        groq_api_and_tool_call.add_tool(run_step)

        runner = GroqAPIAndToolCallRunner(groq_api_and_tool_call)

        user_prompt = "Check my order status for order 12345"
        monkeypatch.setattr('builtins.input', lambda _: user_prompt)

        print("\n\t get_current_state():", get_current_state())
        print(
            "\n\t customer_support_fsm_runner.current_state:",
            customer_support_fsm_runner.current_state)

        runner.run_iteratively()

        i = 0
        ready_to_call_next_user_prompt = runner._ready_to_call_new_user_prompt
        assert not ready_to_call_next_user_prompt
        while not ready_to_call_next_user_prompt:
            runner.run_iteratively()
            ready_to_call_next_user_prompt = runner._ready_to_call_new_user_prompt
            i += 1
            print("\n\t i:", i)
            print("\n\t get_current_state():", get_current_state())
            print(
                "\n\t customer_support_fsm_runner.current_state:",
                customer_support_fsm_runner.current_state)

        for index, message in enumerate(
            groq_api_and_tool_call.conversation_and_system_messages.get_conversation_as_list_of_dicts()):
            print(
                "\n\t index: ", index,
                "\n\t message: \n", message)

    except ValueError as err:
        assert "is not in the alphabet" in str(err)
        assert True


def test_pizza_customer_support_fsm_steps(monkeypatch):
    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "gemma2-9b-it"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    groq_api_and_tool_call = GroqAPIAndToolCall(groq_api_wrapper)
    groq_api_and_tool_call.set_tool_choice()

    groq_api_and_tool_call.add_system_message(
        PizzaCustomerSupportFiniteStateMachine.SYSTEM_PROMPT)

    fsm = FiniteStateMachine(
        states=PizzaCustomerSupportFiniteStateMachine.STATES,
        alphabet=PizzaCustomerSupportFiniteStateMachine.ALPHABET,
        initial_state=PizzaCustomerSupportFiniteStateMachine.INITIAL_STATE,
        transition_function=PizzaCustomerSupportFiniteStateMachine.get_transition_function(),
        final_states=PizzaCustomerSupportFiniteStateMachine.FINAL_STATES
    )

    fsm_runner = FiniteStateMachineRunner(fsm)
    fsm_runner.reset()

    input_get_current_state, input_run_step = create_fsm_functions(fsm_runner)

    get_current_state, run_step = \
        create_fsm_tools_with_strings(input_get_current_state, input_run_step)

    groq_api_and_tool_call.add_tool(get_current_state)
    groq_api_and_tool_call.add_tool(run_step)

    runner = GroqAPIAndToolCallRunner(groq_api_and_tool_call)
    user_prompt = "I want to order a pizza"

    monkeypatch.setattr('builtins.input', lambda _: user_prompt)

    print("\n\t get_current_state():", get_current_state())
    print("\n\t fsm_runner.current_state:", fsm_runner.current_state)

    runner.run_iteratively()

    i = 0
    ready_to_call_next_user_prompt = runner._ready_to_call_new_user_prompt
    assert not ready_to_call_next_user_prompt
    while not ready_to_call_next_user_prompt:
        runner.run_iteratively()
        ready_to_call_next_user_prompt = runner._ready_to_call_new_user_prompt
        i += 1
        print("\n\t i:", i)
        print("\n\t get_current_state():", get_current_state())
        print("\n\t fsm_runner.current_state:", fsm_runner.current_state)

    print("last i: ", i)

    # print("\n\t len(tool_call_result):", len(tool_call_result))
    # #print("\n\t tool_call_result:", tool_call_result)

    # for index in range(len(tool_call_result)):
    #     print(
    #         "\n\t tool_call_result[", index, "]:",
    #         tool_call_result[index])

    # print("\n\t get_current_state():", get_current_state())
    # print(
    #     "\n\t fsm_runner.current_state:",
    #     fsm_runner.current_state)
    # print(
    #     "\n\t fsm_runner.transition_history:",
    #     fsm_runner.transition_history)

    # if len(tool_call_result) > 1:
    #     try:
    #         print("\n\t len(tool_call_result[1]): ", len(tool_call_result[1]))

    #         for index in range(len(tool_call_result[1])):
    #             print(
    #                 "\n\t tool_call_result[1][", index, "]:",
    #                 tool_call_result[1][index])
    #     except:
    #         print("\n\t tool_call_result[1] is not a list")

    # user_prompt = "I want to order a medium pizza. Add pepperoni and mushrooms toppings."
    # tool_call_result = \
    #     groq_api_and_tool_call.create_chat_completion_with_user_message_until_tool_call_ends(
    #         user_prompt)

    # print("\n\t len(tool_call_result):", len(tool_call_result))

    # print("\n\t get_current_state():", get_current_state())
    # print(
    #     "\n\t fsm_runner.current_state:",
    #     fsm_runner.current_state)
    # print(
    #     "\n\t fsm_runner.transition_history:",
    #     fsm_runner.transition_history)

    # if len(tool_call_result) > 1:
    #     try:
    #         print("\n\t len(tool_call_result[1]): ", len(tool_call_result[1]))

    #         for index in range(len(tool_call_result[1])):
    #             print(
    #                 "\n\t tool_call_result[1][", index, "]:",
    #                 tool_call_result[1][index])
    #     except:
    #         print("\n\t tool_call_result[1] is not a list")

def test_pizza_customer_support_fsm_with_user_input(monkeypatch):
    # List of prompts to cycle through
    prompts = [
        "I want to order a pizza",
        "Medium size please",
        "Add pepperoni and mushrooms toppings.",
        "Yes, that's correct",
        "That's all, thank you"
    ]
    
    # Counter to keep track of input calls
    input_counter = {'count': 0}
    
    def mock_input(_):
        current_count = input_counter['count']
        input_counter['count'] += 1
        # If we've gone through all prompts, return the last one
        return prompts[min(current_count, len(prompts) - 1)]
    
    monkeypatch.setattr('builtins.input', mock_input)

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "gemma2-9b-it"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    groq_api_and_tool_call = GroqAPIAndToolCall(groq_api_wrapper)
    groq_api_and_tool_call.set_tool_choice()

    groq_api_and_tool_call.add_system_message(
        PizzaCustomerSupportFiniteStateMachine.SYSTEM_PROMPT_2)

    fsm = FiniteStateMachine(
        states=PizzaCustomerSupportFiniteStateMachine.STATES,
        alphabet=PizzaCustomerSupportFiniteStateMachine.ALPHABET,
        initial_state=PizzaCustomerSupportFiniteStateMachine.INITIAL_STATE,
        transition_function=PizzaCustomerSupportFiniteStateMachine.get_transition_function(),
        final_states=PizzaCustomerSupportFiniteStateMachine.FINAL_STATES
    )

    fsm_runner = FiniteStateMachineRunner(fsm)
    fsm_runner.reset()

    input_get_current_state, input_run_step = create_fsm_functions(fsm_runner)

    get_current_state, run_step = \
        create_fsm_tools_with_strings(input_get_current_state, input_run_step)

    groq_api_and_tool_call.add_tool(get_current_state)
    groq_api_and_tool_call.add_tool(run_step)
    groq_api_and_tool_call.add_tool(get_user_input)

    runner = GroqAPIAndToolCallRunner(groq_api_and_tool_call)

    print("\n\t get_current_state():", get_current_state())
    print("\n\t fsm_runner.current_state:", fsm_runner.current_state)

    runner.run_iteratively()

    i = 0
    ready_to_call_next_user_prompt = runner._ready_to_call_new_user_prompt
    while not ready_to_call_next_user_prompt:
        runner.run_iteratively()
        ready_to_call_next_user_prompt = runner._ready_to_call_new_user_prompt
        i += 1
        print("\n\t i:", i)
        print("\n\t get_current_state():", get_current_state())
        print("\n\t fsm_runner.current_state:", fsm_runner.current_state)

    print("\n\t get_current_state():", get_current_state())
    print("\n\t fsm_runner.current_state:", fsm_runner.current_state)
    print("\n\t input_counter['count']: ", input_counter['count'])
    print("\n\t i:", i)


