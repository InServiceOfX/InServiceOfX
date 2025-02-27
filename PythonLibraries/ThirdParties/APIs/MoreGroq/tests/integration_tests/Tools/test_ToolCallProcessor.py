from corecode.Utilities import (get_environment_variable, load_environment_file)
from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from moregroq.Wrappers.ChatCompletionConfiguration import (
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty,
    Tool)

from moregroq.Wrappers import GroqAPIWrapper
from TestUtilities.TestSetup import get_bakery_prices
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

    print(response.choices[0].message.tool_calls)

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "get_bakery_prices": get_bakery_prices
        },
        messages=messages
    )

    process_result = tool_call_processor.process_response(
        response.choices[0].message)

    assert process_result == 2

    # ToolCallProcessor.process_response(..) mutates the original messages
    # input.

    assert tool_call_processor.messages == messages

    response = groq_api_wrapper.create_chat_completion(messages)
    response_message = response.choices[0].message
    print(response_message)
    print(response_message.tool_calls)
    print(response_message.content)
    assert "4.25" in response_message.content
    assert "4.75" in response_message.content
