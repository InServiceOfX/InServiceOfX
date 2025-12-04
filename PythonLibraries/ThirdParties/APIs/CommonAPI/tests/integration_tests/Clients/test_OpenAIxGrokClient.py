from commonapi.Clients.OpenAIxGrokClient import (
    OpenAIxGrokClient
)

from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from corecode.Utilities import (get_environment_variable, load_environment_file)
load_environment_file()

from warnings import warn

def test_OpenAIxGrokClient_generates_prose():
    """
    https://platform.openai.com/docs/guides/text-generation#quickstart
    """
    api_key = get_environment_variable("XAI_API_KEY")

    if api_key is None or api_key == "":
        warn("XAI_API_KEY is not set")
        return

    client = OpenAIxGrokClient(api_key)

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

def test_OpenAIxGrokClient_does_basic_chat_completion():
    """
    https://docs.x.ai/docs/guides/chat
    """
    api_key = get_environment_variable("XAI_API_KEY")

    if api_key is None or api_key == "":
        warn("XAI_API_KEY is not set")
        return

    client = OpenAIxGrokClient(api_key)    
    client.clear_chat_completion_configuration()
    client.configuration.model = "grok-4"

    messages = [
        create_system_message("You are a PhD-level mathematician."),
        create_user_message("What is 2 + 2?")
    ]

    response = client.create_chat_completion(messages)
    # <class 'openai.types.chat.chat_completion.ChatCompletion'>
    print(type(response))
    # Ah, the timeless question that has puzzled philosophers and mathematicians alike! In the standard arithmetic of the natural numbers (or integers, or reals, for that matter), under the usual addition operation in base 10, 2 + 2 equals 4.

    # If you're asking in a different context—say, modular arithmetic, binary, or perhaps a more abstract algebraic structure—feel free to provide more details, and I'll dive deeper! 😊
    print(response.choices[0].message.content)
    # <class 'str'>
    print(type(response.choices[0].message.content))
    # 1
    print(len(response.choices))
    # <class 'list'>
    print(type(response.choices))
    # assistant
    print(response.choices[0].message.role)
    # <class 'str'>
    print(type(response.choices[0].message.role))
    # ChatCompletionMessage(content="Ah, the timeless question that has puzzled philosophers and mathematicians alike! In the standard arithmetic of the natural numbers (or integers, or reals, for that matter), under the usual addition operation in base 10, 2 + 2 equals 4.\n\nIf you're asking in a different context—say, modular arithmetic, binary, or perhaps a more abstract algebraic structure—feel free to provide more details, and I'll dive deeper! 😊", refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None)
    print(response.choices[0].message)
    # None
    print(response.choices[0].message.tool_calls)
    # <class 'NoneType'>
    print(type(response.choices[0].message.tool_calls))
    # None
    print(response.choices[0].message.function_call)
    # <class 'NoneType'>
    print(type(response.choices[0].message.function_call))

