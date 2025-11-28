from commonapi.Clients.OpenAIxGrokClient import (
    OpenAIxGrokClient
)

from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from corecode.Utilities import (get_environment_variable, load_environment_file)

load_environment_file()

def test_OpenAIxGrokClient_generates_prose():
    """
    https://platform.openai.com/docs/guides/text-generation#quickstart
    """
    client = OpenAIxGrokClient(get_environment_variable("XAI_API_KEY"))
    
    client.clear_chat_completion_configuration()
    client.configuration.model = "grok-4"

    messages = [
        create_system_message("You are a helpful assistant."),
        create_user_message("Write a haiku about recursion in programming.")
    ]

    response = client.create_chat_completion(messages)

    print(response.choices[0].message.content)

    messages = [
        create_system_message(
            "You are Grok, a highly intelligent, helpful AI assistant."
        ),
        create_user_message(
            "What is the meaning of life, the universe, and everything?")
    ]
    response = client.create_chat_completion(messages)
    print(response.choices[0].message.content)
