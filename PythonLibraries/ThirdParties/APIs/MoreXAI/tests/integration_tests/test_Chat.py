import os
from xai_sdk import Client
from xai_sdk.chat import user, system

from corecode.Utilities import (get_environment_variable, load_environment_file)
load_environment_file()

from warnings import warn

def test_Grok_basic_chat_completion():
    """
    https://docs.x.ai/docs/guides/chat
    """
    api_key = get_environment_variable("XAI_API_KEY")
    if api_key is None or api_key == "":
        warn("XAI_API_KEY is not set")
        return

    client = Client(
        api_key=api_key,
        # Override default timeout with longer timeout for reasoning models
        timeout=3600,
    )
    chat = client.chat.create(model="grok-4")
    chat.append(system("You are a PhD-level mathematician."))
    chat.append(user("What is 2 + 2?"))
    response = chat.sample()
    # Example (actual) response:
    # As a PhD-level mathematician, I can confirm that in the standard arithmetic of the natural numbers, 2 + 2 equals 4. If you're inquiring about this in a different context—such as modular arithmetic, abstract algebra, or perhaps a philosophical debate on the foundations of mathematics—feel free to provide more details for a deeper exploration!
    print(response.content)
    # <class 'xai_sdk.chat.Response'>
    print(type(response))
    # <class 'str'>
    print(type(response.content))
    # <class 'list'>
    print(type(response.tool_calls))
    # []
    print(response.tool_calls)
    # <class 'xai.api.v1.usage_pb2.SamplingUsage'>
    print(type(response.usage))
    # completion_tokens: 69
    # prompt_tokens: 701
    # total_tokens: 893
    # prompt_text_tokens: 701
    # reasoning_tokens: 123
    # cached_prompt_text_tokens: 679
    print(response.usage)

