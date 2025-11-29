from xai_sdk import Client
from xai_sdk.chat import user, image

from commonapi.Clients.OpenAIxGrokClient import OpenAIxGrokClient
from commonapi.Messages import (
    create_system_message,
    create_user_message,
)

from corecode.Utilities import (get_environment_variable, load_environment_file)
load_environment_file()

def test_Grok_analyzes_image():

    client = Client(
        api_key=get_environment_variable("XAI_API_KEY"),
        # Override default timeout with longer timeout for reasoning models.
        timeout=3600,
    )

    chat = client.chat.create(model="grok-4")
    chat.append(
        user(
            "What's in this image?",
            image("https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png")
        )
    )

    response = chat.sample()
    print(type(response))
    print(response.content)
    assert(isinstance(response.content, str))

# Guides
# Chat with Reasoning
# https://docs.x.ai/docs/guides/reasoning
def test_Grok_chat_with_reasoning_with_OpenAIxGrokClient():
    client = OpenAIxGrokClient(get_environment_variable("XAI_API_KEY"))
    
    client.clear_chat_completion_configuration()
    client.configuration.model = "grok-3-mini"
    client.configuration.reasoning_effort = "high"
    messages = [
        create_system_message("You are a highly intelligent AI assistant."),
        create_user_message("What is 101*3?")
    ]

    response = client.create_chat_completion(messages)
    # Example (actual) response:
    #     101 multiplied by 3 is 303.

    # \boxed{303}
    print(response.choices[0].message.content)
    print(response.usage)
    print("Number of completion tokens:")
    # 14
    print(response.usage.completion_tokens)
    print("Number of reasoning tokens:")
    # 298
    print(response.usage.completion_tokens_details.reasoning_tokens)