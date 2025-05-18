from corecode.Utilities import (get_environment_variable, load_environment_file)
from commonapi.Messages import (
    create_system_message,
    create_user_message
)
from moregroq.Wrappers import GroqAPIWrapper
from moregroq.Wrappers.ChatCompletionConfiguration import (
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty,
    Tool)

from TestUtilities.TestSetup import calculate

import requests
import json

# Load environment variables at module level
load_environment_file()

def test_groq_tool_use_with_curl_request():
    """
    https://console.groq.com/docs/tool-use
    """
    api_key = get_environment_variable("GROQ_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in Boston today?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "required"
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    result = response.json()
    print(
        "test_groq_tool_use_with_curl_request response 1:",
        json.dumps(result, indent=2))

    assert response.status_code == 200

    # Verify response structure
    assert "choices" in result
    assert len(result["choices"]) > 0
    assert "message" in result["choices"][0]
    print(result["choices"][0])
    print(result["choices"][0]["message"])

    # Tool Call Streucture
    # Groq API tool calls are structured to be OpenAI-compatible.

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": "You are a weather assistant. Use the get_weather function to retrieve weather information for a given location."
            },
            {
                "role": "user",
                "content": "What's the weather like in New York today?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The unit of temperature to use. Defaults to fahrenheit."
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_completion_tokens": 4096
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    result = response.json()
    print(
        "test_groq_tool_use_with_curl_request response 2:",
        json.dumps(result, indent=2))

    # This is an example tool call response:
    # "model": "llama-3.3-70b-versatile",
    # "choices": [{
    #     "index": 0,
    #     "message": {
    #         "role": "assistant",
    #         "tool_calls": [{
    #         "id": "call_d5wg",
    #         "type": "function",
    #         "function": {
    #             "name": "get_weather",
    #             "arguments": "{\"location\": \"New York, NY\"}"
    #         }
    #         }]
    #     },
    #     "logprobs": null,
    #     "finish_reason": "tool_calls"
    # }],
    assert response.status_code == 200

    assert "choices" in result
    assert len(result["choices"]) > 0
    assert "message" in result["choices"][0]

    assert result["choices"][0]["message"]["tool_calls"] is not None
    assert len(result["choices"][0]["message"]["tool_calls"]) > 0

    print("result[choices][0]:", result["choices"][0])
    print("result[choices][0][message]:", result["choices"][0]["message"])

    # https://console.groq.com/docs/tool-use
    # When a model decides to use a tool, it returns a response with a tool_calls
    # object:
    # id: unique identifier for tool call
    assert "id" in result["choices"][0]["message"]["tool_calls"][0]
    # type: type of tool call, i.e. function
    assert "type" in result["choices"][0]["message"]["tool_calls"][0]
    # TODO: point out change in documentation.
    # name: name of tool being used.
    # assert "name" in result["choices"][0]["message"]["tool_calls"][0]
    # TODO: point out change in documentation.
    # parameters: object containing input being passed to the tool.
    #assert "parameters" in result["choices"][0]["message"]["tool_calls"][0]
    assert "name" in result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert "arguments" in result["choices"][0]["message"]["tool_calls"][0]["function"]

def run_conversation(user_prompt):
    """
    https://console.groq.com/docs/tool-use

    See Tools/test_ToolCallProcessing.py for how we wrap and handle these tool
    calls.
    """
    messages = [
        create_system_message(
            (
                "You are a calculator assistant. Use the calculate function to "
                "perform mathematical operations and provide the results.")),
        create_user_message(user_prompt)
    ]

    # Define the available tools (i.e. functions) for our model to use
    # Originally,
    # tools = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "calculate",
    #             "description": "Evaluate a mathematical expression",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "expression": {
    #                         "type": "string",
    #                         "description": "The mathematical expression to evaluate",
    #                     }
    #                 },
    #                 "required": ["expression"],
    #             },
    #         },
    #     }
    # ]
    tools = [
        # Tool, FunctionDefinition, etc. are our own custom wrappers.
        Tool(function=FunctionDefinition(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters=FunctionParameters(
                properties=[
                    ParameterProperty(
                        name="expression",
                        type="string",
                        description="The mathematical expression to evaluate",
                        required=True
                    )
                ]
            ),
        ))
    ]

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))

    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"

    groq_api_wrapper.configuration.tools = tools
    # Let our LLM decide when to use tools.
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    response = groq_api_wrapper.create_chat_completion(messages)
    # Extract the response and any tool call responses
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Define the available tools that can be called by the LLM
        available_functions = {
            # calculate is the actual Python function.
            "calculate": calculate
        }
        # Add the LLM's response to the conversation
        messages.append(response_message)

        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            # Call the tool and get the response
            function_response = function_to_call(
                expression=function_args.get("expression")
            )
            # Add the tool response to the conversation
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    # indicates this message is from tool use
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                }
            )
        # Make a second API call with the updated conversation
        second_response = groq_api_wrapper.create_chat_completion(messages)
        print("second_response:", second_response)
        return second_response.choices[0].message.content
    print("Did not make second API call.")
    return response_message.content

def test_receive_and_handle_tool_results():
    """
    https://console.groq.com/docs/tool-use
    """
    user_prompt = "What is 25 * 10 + 10?"
    result = run_conversation(user_prompt)
    print("result:", result)

    assert "260" in result

# Define models
ROUTING_MODEL = "llama3-70b-8192"
TOOL_USE_MODEL = "llama-3.3-70b-versatile"
GENERAL_MODEL = "llama3-70b-8192"

def route_query(query, groq_api_wrapper):
    """
    Routing logic to let LLM decide if tools are needed.
    """
    routing_prompt = f"""
        Given the following user query, determine if any tools are needed to
        answer it.
        If a calculation tool is needed, respond with 'TOOL: CALCULATE'.
        If no tools are needed, respond with 'NO TOOL'.
        
        User query: {query}
        
        Response:
        """
    groq_api_wrapper.clear_chat_completion_configuration()
    groq_api_wrapper.configuration.model = ROUTING_MODEL
    # We only need a short response.
    groq_api_wrapper.configuration.max_completion_tokens = 2048
    messages = [
        create_system_message(
            (
                "You are a routing assistant. Determine if tools are needed "
                "based on the user query.")),
        create_user_message(query)
    ]

    response = groq_api_wrapper.create_chat_completion(messages)
    routing_decision = response.choices[0].message.content.strip()

    if "TOOL: CALCULATE" in routing_decision:
        return "calculate tool needed"
    else:
        return "no tool needed"

def run_with_tool(query, groq_api_wrapper):
    """
    Use the tool use model to perform the calculation.
    """
    messages = [
        create_system_message(
            "You are a calculator assistant. Use the calculate function to perform mathematical operations and provide the results."
        ),
        create_user_message(query)
    ]
    tools = [
        Tool(type="function",
             function=FunctionDefinition(
                 name="calculate",
                 description="Evaluate a mathematical expression",
                 parameters={
                     "type": "object",
                     "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                     },
                     "required": ["expression"],
                 },
             )
        )
    ]

    groq_api_wrapper.clear_chat_completion_configuration()
    groq_api_wrapper.configuration.model = TOOL_USE_MODEL
    groq_api_wrapper.configuration.tools = tools
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    response = groq_api_wrapper.create_chat_completion(messages)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        messages.append(response_message)
        for tool_call in tool_calls:
            function_args = json.loads(tool_call.function.arguments)
            function_response = calculate(function_args.get("expression"))
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "calculate",
                    "content": function_response
                }
            )
        groq_api_wrapper.clear_chat_completion_configuration()
        groq_api_wrapper.configuration.model = TOOL_USE_MODEL
        second_response = groq_api_wrapper.create_chat_completion(messages)
        return second_response.choices[0].message.content
    return response_message.content

def run_general_model(query, groq_api_wrapper):
    """
    Use the general model to answer the query since no tool is needed.
    """
    groq_api_wrapper.clear_chat_completion_configuration()
    groq_api_wrapper.configuration.model = GENERAL_MODEL

    messages = [
        create_system_message("You are a helpful assistant."),
        create_user_message(query)
    ]
    response = groq_api_wrapper.create_chat_completion(messages)
    return response.choices[0].message.content

def process_query(query, groq_api_wrapper):
    """
    Process the query and route it to the appropriate model..
    """
    routing_decision = route_query(query, groq_api_wrapper)
    if routing_decision == "calculate":
        response = run_with_tool(query, groq_api_wrapper)
    else:
        response = run_general_model(query, groq_api_wrapper)
    return {
        "query": query,
        "route": routing_decision,
        "response": response
    }

def test_route_query():
    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    
    queries = [
        "What is the capital of the Netherlands?",
        "Calculate 25 * 4 + 10"        
    ]
    results = []
    for query in queries:
        result = process_query(query, groq_api_wrapper)
        results.append(result)
        print(f"Query: {result['query']}\n")
        print(f"Route: {result['route']}\n")
        print(f"Response: {result['response']}\n")

    assert results[0]["route"] == "no tool needed"
    assert "Amsterdam" in results[0]["response"]
    assert "110" in results[1]["response"]

# Define weather tools
def get_temperature(location: str):
    # This is a mock tool/function. In a real scenario, you would call a weather
    # API.    
    temperatures = {"New York": 22, "London": 18, "Tokyo": 26, "Sydney": 28}
    return temperatures.get(location, "Temperature data not available")

def get_weather_condition(location: str):
    # This is a mock tool/function. In a real scenario, you would call a weather
    # API.    
    conditions = {"New York": "Sunny", "London": "Rainy", "Tokyo": "Cloudy", "Sydney": "Clear"}
    return conditions.get(location, "Weather condition data not available")


def test_parallel_tool_use():
    """
    https://console.groq.com/docs/tool-use
    """
    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))

    messages = [
        create_system_message("You are a helpful weather assistant."),
        create_user_message("What's the weather like in New York and London?")
    ]

    tools = [
        Tool(
            type="function",
            function=FunctionDefinition(
                name="get_temperature",
                description="Get the temperature for a given location",
                parameters=FunctionParameters(
                    properties=[
                        ParameterProperty(
                            name="location",
                            type="string",
                            description="The name of the city",
                            required=True
                        )
                    ],
                ),
            ),
        ),
        Tool(
            type="function",
            function=FunctionDefinition(
                name="get_weather_condition",
                description="Get the current weather condition in a given location",
                parameters=FunctionParameters(
                    properties=[
                        ParameterProperty(
                            name="location",
                            type="string",
                            description="The name of the city",
                            required=True
                        )
                    ],
                ),
            ),
        ),
    ]

    # Make the initial request.
    groq_api_wrapper.clear_chat_completion_configuration()
    model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.model = model
    groq_api_wrapper.configuration.tools = tools
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_completion_tokens = 4096

    response = groq_api_wrapper.create_chat_completion(messages)
    print("initial request response:", response)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    print("initial request tool_calls:", tool_calls)
    # Process each tool calls
    messages.append(response_message)

    available_functions = {
        "get_temperature": get_temperature,
        "get_weather_condition": get_weather_condition
    }

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(**function_args)
        messages.append(
            {
                "role": "tool",
                "content": str(function_response),
                "tool_call_id": tool_call.id
            }
        )

    # Make the final request with tool call results.
    groq_api_wrapper.clear_chat_completion_configuration()
    groq_api_wrapper.configuration.model = model
    groq_api_wrapper.configuration.tools = tools
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_completion_tokens = 4096
    final_response = groq_api_wrapper.create_chat_completion(messages)
    print("final response:", final_response)
    print("final response content:", final_response.choices[0].message.content)
    assert "sunny" in final_response.choices[0].message.content or \
        "Sunny" in final_response.choices[0].message.content
    assert "rainy" in final_response.choices[0].message.content or \
        "Rainy" in final_response.choices[0].message.content

# Define the tool schema

tool_schema = {
    "name": "get_weather_info",
    "description": "Get the weather information for any location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location for which we want to get the weather information (e.g., New York)"
            }
        },
        "required": ["location"]
    }
}

from pydantic import BaseModel, Field
import instructor

class ToolCall(BaseModel):
    input_text: str = Field(description="The user's input text")
    tool_name: str = Field(description="The name of the tool to call")
    tool_parameters: str = Field(description="JSON string of tool parameters")

class ResponseModel(BaseModel):
    tool_calls: list[ToolCall]

def run_conversation2(user_prompt, groq_api_wrapper):
    # Prepare the messages
    messages = [
        create_system_message(
            f"You are an assistant that can use tools. You have access to the following tool: {tool_schema}"),
        create_user_message(user_prompt)
    ]

    # Make the initial request
    groq_api_wrapper.clear_chat_completion_configuration()
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.response_model = ResponseModel
    groq_api_wrapper.configuration.temperature = 0.7
    groq_api_wrapper.configuration.max_completion_tokens = 1000

    groq_api_wrapper.client = instructor.from_groq(
        groq_api_wrapper.client,
        mode=instructor.Mode.JSON)

    response = groq_api_wrapper.create_chat_completion(messages)
    
    return response.tool_calls

def test_tool_use_with_structured_outputs():
    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    user_prompt = "What's the weather like in San Francisco?"
    tool_calls = run_conversation2(user_prompt, groq_api_wrapper)
    for call in tool_calls:
        print(f"Input: {call.input_text}")
        print(f"Tool: {call.tool_name}")
        print(f"Parameters: {call.tool_parameters}")
        print()
