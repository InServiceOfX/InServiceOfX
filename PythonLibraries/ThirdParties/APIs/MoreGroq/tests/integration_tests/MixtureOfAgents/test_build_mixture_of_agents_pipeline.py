from corecode.Utilities import (get_environment_variable, load_environment_file)

from commonapi.Messages import ConversationHistory

from moregroq.Wrappers import GroqAPIWrapper, AsyncGroqAPIWrapper
from commonapi.Messages import (
    create_user_message,
    create_system_message)

from moregroq.MixtureOfAgents import PromptData

from itertools import chain
from typing import Dict, Optional

import pytest

load_environment_file()

# https://github.com/groq/groq-api-cookbook/blob/main/tutorials/mixture-of-agents/mixture_of_agents.ipynb

def create_agent(
    groq_api_wrapper,
    system_prompt: str = PromptData.HELPFUL_SYSTEM_TEMPLATE):
    def run_created_agent(helper_response: str, input: str, messages):
        new_prompt = list(chain([
            create_system_message(
                system_prompt.format(helper_response=helper_response)),],
            messages))

        new_prompt.append(create_user_message(input))

        return groq_api_wrapper.create_chat_completion(new_prompt)
    
    return run_created_agent

def concatenate_messages(
    inputs: Dict[str, str],
    reference_system_prompt: Optional[str] = None) -> str:
    """Concatenate and format layer agent responses"""

    reference_system_prompt = reference_system_prompt or PromptData.REFERENCE_SYSTEM_PROMPT_TEMPLATE

    assert "{responses}" in reference_system_prompt, \
        "{responses} prompt varialbe not found in prompt. Please add it"

    responses = ""
    for i, out in enumerate(inputs.values()):
        responses += f"{i}, {out}\n"

    formatted_prompt = reference_system_prompt.format(responses=responses)
    return formatted_prompt

def test_create_agent_on_default_system_prompt():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    agent = create_agent(groq_api_wrapper)
    helper_response = ""
    input = "tell me about opentelemetry"
    messages = []

    response = agent(helper_response, input, messages)
    print(response.choices[0].message.content)

def test_create_agent_on_other_system_prompt():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.model = "llama3-8b-8192"

    agent = create_agent(groq_api_wrapper, PromptData.EXPERT_PLANNER_TEMPLATE)
    helper_response = ""
    input = "tell me about opentelemetry"
    messages = []

    response = agent(helper_response, input, messages)
    print("expert planner response: ", response.choices[0].message.content)

    # groq.BadRequestError: Error code: 400 - {'error': {'message':
    # 'The model `mixtral-8x7b-32768` has been decommissioned and is no longer supported.
    groq_api_wrapper.configuration.model = "llama-3.1-8b-instant"

    agent = create_agent(
        groq_api_wrapper,
        PromptData.THOUGHT_AND_QUESTION_RESPONSE_TEMPLATE)
    input = "tell me about opentelemetry"

    response = agent(helper_response, input, messages)
    print("thought and question response: ", response.choices[0].message.content)

    groq_api_wrapper.configuration.model = "gemma2-9b-it"

    agent = create_agent(groq_api_wrapper, PromptData.CHAIN_OF_THOUGHT_TEMPLATE)
    input = "tell me about opentelemetry"

    response = agent(helper_response, input, messages)
    print("chain of thought response: ", response.choices[0].message.content)    

@pytest.mark.asyncio
async def test_concatenate_responses():
    async_groq_api_wrapper_0 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_0.configuration.model = "llama3-8b-8192"

    async_groq_api_wrapper_1 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    # groq.BadRequestError: Error code: 400 - {'error': {'message':
    # 'The model `mixtral-8x7b-32768` has been decommissioned and is no longer supported.
    async_groq_api_wrapper_1.configuration.model = "llama-3.1-8b-instant"

    async_groq_api_wrapper_2 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_2.configuration.model = "gemma2-9b-it"

    agent0 = create_agent(
        async_groq_api_wrapper_0,
        PromptData.EXPERT_PLANNER_TEMPLATE)
    agent1 = create_agent(
        async_groq_api_wrapper_1,
        PromptData.THOUGHT_AND_QUESTION_RESPONSE_TEMPLATE)
    agent2 = create_agent(
        async_groq_api_wrapper_2,
        PromptData.CHAIN_OF_THOUGHT_TEMPLATE)

    helper_response = ""
    input = "tell me about opentelemetry"
    messages = []
    agent0_response = await agent0(
        helper_response,
        input,
        messages)

    agent1_response = await agent1(
        helper_response,
        input,
        messages)

    agent2_response = await agent2(
        helper_response,
        input,
        messages)

    concatenated_response = concatenate_messages(
        {
            "0": agent0_response.choices[0].message.content,
            "1": agent1_response.choices[0].message.content,
            "2": agent2_response.choices[0].message.content
        })
    
    print("concatenated response: ", concatenated_response)

async def async_layer_agents(runnable_agents, helper_response, input, messages):
    responses = {}
    for index, agent in enumerate(runnable_agents):
        response = await agent(helper_response, input, messages)
        responses[index] = response.choices[0].message.content

    return concatenate_messages(responses)

@pytest.mark.asyncio
async def test_async_layer_agents():
    async_groq_api_wrapper_0 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_0.configuration.model = "llama3-8b-8192"

    async_groq_api_wrapper_1 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    # groq.BadRequestError: Error code: 400 - {'error': {'message':
    # 'The model `mixtral-8x7b-32768` has been decommissioned and is no longer supported.
    async_groq_api_wrapper_1.configuration.model = "llama-3.1-8b-instant"

    async_groq_api_wrapper_2 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_2.configuration.model = "gemma2-9b-it"

    agent0 = create_agent(
        async_groq_api_wrapper_0,
        PromptData.EXPERT_PLANNER_TEMPLATE)
    agent1 = create_agent(
        async_groq_api_wrapper_1,
        PromptData.THOUGHT_AND_QUESTION_RESPONSE_TEMPLATE)
    agent2 = create_agent(
        async_groq_api_wrapper_2,
        PromptData.CHAIN_OF_THOUGHT_TEMPLATE)

    runnable_agents = [agent0, agent1, agent2]
    helper_response = ""
    input = "tell me about opentelemetry"
    messages = []

    concatenated_response = await async_layer_agents(
        runnable_agents, helper_response, input, messages)

    print("concatenated response: ", concatenated_response)

async def chat_cycles(
    cycle: int,
    query: str,
    runnable_agents,
    initial_helper_response: str,
    messages):
    input = query
    helper_response = initial_helper_response

    for _ in range(cycle):
        helper_response = await async_layer_agents(
            runnable_agents,
            helper_response,
            input,
            messages)
    return helper_response

@pytest.mark.asyncio
async def test_chat_cycles():
    async_groq_api_wrapper_0 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_0.configuration.model = "llama3-8b-8192"

    async_groq_api_wrapper_1 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    # groq.BadRequestError: Error code: 400 - {'error': {'message':
    # 'The model `mixtral-8x7b-32768` has been decommissioned and is no longer supported.
    async_groq_api_wrapper_1.configuration.model = "llama-3.1-8b-instant"

    async_groq_api_wrapper_2 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_2.configuration.model = "gemma2-9b-it"

    agent0 = create_agent(
        async_groq_api_wrapper_0,
        PromptData.EXPERT_PLANNER_TEMPLATE)
    agent1 = create_agent(
        async_groq_api_wrapper_1,
        PromptData.THOUGHT_AND_QUESTION_RESPONSE_TEMPLATE)
    agent2 = create_agent(
        async_groq_api_wrapper_2, PromptData.CHAIN_OF_THOUGHT_TEMPLATE)

    runnable_agents = [agent0, agent1, agent2]
    initial_helper_response = ""
    query = "tell me about opentelemetry"
    messages = []

    helper_response = await chat_cycles(
        3,
        query,
        runnable_agents,
        initial_helper_response,
        messages)

    print("helper response: ", helper_response)

async def chat_to_response(
    cycle: int,
    query: str,
    runnable_agents,
    runnable_main_agent,
    conversation_history):

    helper_response = ""

    helper_response = await chat_cycles(
        cycle,
        query,
        runnable_agents,
        helper_response,
        conversation_history.messages)

    main_agent_response = runnable_main_agent(
        helper_response,
        query,
        conversation_history.messages)

    conversation_history.append_message(create_user_message(query))
    try:
        conversation_history.append_message(
            main_agent_response.choices[0].message)
    except AttributeError as err:
        print("error: ", err)
        print("type(main_agent_response): ", type(main_agent_response))
        print("main_agent_response: ", main_agent_response)

    return main_agent_response

@pytest.mark.asyncio
async def test_main_agent_response():
    async_groq_api_wrapper = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper.configuration.model = "llama3-70b-8192"
    main_agent = create_agent(
        async_groq_api_wrapper,
        PromptData.HELPFUL_BOB_TEMPLATE)
    conversation_history = ConversationHistory()

    input = "tell me about opentelemetry"
    main_agent_response = await main_agent(
        "",
        input,
        conversation_history.messages)

    print(
        "main agent response: ",
        main_agent_response.choices[0].message.content)

@pytest.mark.asyncio
async def test_chat_to_response():
    async_groq_api_wrapper_0 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_0.configuration.model = "llama3-8b-8192"

    async_groq_api_wrapper_1 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    # groq.BadRequestError: Error code: 400 - {'error': {'message':
    # 'The model `mixtral-8x7b-32768` has been decommissioned and is no longer supported.
    async_groq_api_wrapper_1.configuration.model = "llama-3.1-8b-instant"

    async_groq_api_wrapper_2 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_2.configuration.model = "gemma2-9b-it"

    async_groq_api_wrapper_3 = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    async_groq_api_wrapper_3.configuration.model = "llama3-70b-8192"
    async_groq_api_wrapper_3.configuration.temperature = 0.1

    agent0 = create_agent(
        async_groq_api_wrapper_0,
        PromptData.EXPERT_PLANNER_TEMPLATE)
    agent1 = create_agent(
        async_groq_api_wrapper_1,
        PromptData.THOUGHT_AND_QUESTION_RESPONSE_TEMPLATE)
    agent2 = create_agent(
        async_groq_api_wrapper_2, PromptData.CHAIN_OF_THOUGHT_TEMPLATE)
    agent3 = create_agent(
        async_groq_api_wrapper_3, PromptData.HELPFUL_BOB_TEMPLATE)

    runnable_agents = [agent0, agent1, agent2]
    conversation_history = ConversationHistory()
    assert conversation_history.messages == []

    main_agent_response = await chat_to_response(
        3,
        "tell me about opentelemetry",
        runnable_agents,
        agent3,
        conversation_history)

    print("main agent response: ", main_agent_response.choices[0].message.content)
    assert conversation_history.messages != []
    assert conversation_history.messages[-1].content == \
        main_agent_response.choices[0].message.content
    assert conversation_history.messages[0].content == \
        "tell me about opentelemetry"
