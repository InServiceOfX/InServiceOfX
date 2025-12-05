"""
https://github.com/groq/groq-api-cookbook/blob/main/tutorials/parallel-tool-use/parallel-tool-use.ipynb
"""
import json

from corecode.Utilities import (get_environment_variable, load_environment_file)
from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from moregroq.Wrappers import GroqAPIWrapper

from tools.FunctionCalling.FunctionDefinition import (
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty,
    Tool)

load_environment_file()

def get_bakery_prices(bakery_item: str):
    """
    Define a tool, or function, that the LLM can invoke to fetch pricing for
    bakery items.
    """
    if bakery_item == "croissant":
        return 4.25
    elif bakery_item == "brownie":
        return 2.50
    elif bakery_item == "cappuccino":
        return 4.75
    else:
        return "We're currently sold out!"

def test_single_bakery_example():
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
    # Groq must update documentation:
    # groq.BadRequestError: Error code: 400 -
    # {'error': {'message':
    # 'The model `llama3-groq-70b-8192-tool-use-preview` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.',
    # 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
    #model = "llama3-groq-70b-8192-tool-use-preview"
    model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.model = model
    groq_api_wrapper.configuration.tools = tools
    # We've set the tool_choice parameter to auto to allow our model to choose
    # between generating a text response or using the given tools, or functions,
    # to provide a response. This is the default when tools are available.

    # We could also set tool_choice to none so our model does not invoke any
    # tools (default when no tools are provided) or to required, which would
    # force our model to use the provided tools for its responses.

    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_tokens = 4096
    response = groq_api_wrapper.create_chat_completion(messages)
    response_message = response.choices[0].message
    print(response_message)
    print(response_message.tool_calls)
    print(response_message.content)

    # Processing the tool calls
    tool_calls = response_message.tool_calls

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    "type": tool_call.type,
                }
                for tool_call in tool_calls
            ],
        }
    )

    available_functions = {
        "get_bakery_prices": get_bakery_prices
    }

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(**function_args)

        # Note how we create a separate tool call message for each tool call.
        # The model is able to discern the tool call result through
        # 'tool_call_id'
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(function_response),
                "tool_call_id": tool_call.id,
            }
        )

    print(json.dumps(messages, indent=2))

    
    # Note: it's best practice to pass the tool definitions again to help the
    # model understand the assistant message with the tool call and to interpret
    # the tool results.
    
    groq_api_wrapper.clear_chat_completion_configuration()
    groq_api_wrapper.configuration.tools = tools
    groq_api_wrapper.configuration.tool_choice = "auto"
    groq_api_wrapper.configuration.max_tokens = 4096
    groq_api_wrapper.configuration.model = model
    response = groq_api_wrapper.create_chat_completion(messages)

    response_message = response.choices[0].message
    print(response_message)
    print(response_message.content)
    
    