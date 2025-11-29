from commonapi.Clients.OpenAIxGroqClient import OpenAIxGroqClient

from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from corecode.Utilities import (get_environment_variable, load_environment_file)

load_environment_file()

def test_OpenAIxGroqClient_generates_prose():
    """
    https://platform.openai.com/docs/guides/text-generation#quickstart
    """
    client = OpenAIxGroqClient(get_environment_variable("GROQ_API_KEY"))
    
    client.clear_chat_completion_configuration()
    client.configuration.model = "llama-3.3-70b-versatile"

    messages = [
        create_system_message("You are a helpful assistant."),
        create_user_message("Write a haiku about recursion in programming.")
    ]

    response = client.create_chat_completion(messages)

    print(response.choices[0].message.content)

