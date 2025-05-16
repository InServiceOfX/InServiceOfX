from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories
from commonapi.Configurations import OpenAIChatCompletionConfiguration

from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from moresglang.Configurations import (
    ServerConfiguration)
from moresglang.Wrappers.EntryPoints import HTTPServer
from moresglang.Wrappers.OpenAI import Client

from pathlib import Path

import pytest

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"
data_sub_dirs = DataSubdirectories()

MODEL_DIR = data_sub_dirs.ModelsLLM / "deepseek-ai" / \
    "DeepSeek-R1-Distill-Qwen-1.5B"

if not Path(MODEL_DIR).exists():
    MODEL_DIR = Path(
        "/Data1/Models/LLM/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Skip reason that will show in pytest output
skip_reason = f"Directory {MODEL_DIR} is empty or doesn't exist"

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_Client_inits():
    config = ServerConfiguration.from_yaml(
        test_data_directory / "server_configuration.yml")
    config.model_path = MODEL_DIR

    open_ai_configuration = OpenAIChatCompletionConfiguration.from_yaml(
        test_data_directory / "openai_chat_completion_configuration.yml")

    server = HTTPServer(config)
    server.start()

    try:
        client = Client(config, open_ai_configuration)

        assert client is not None
        assert client.server_configuration == config
        assert client.chat_completion_configuration == open_ai_configuration
        assert client.client is not None
        assert client.current_chat_completion is None

    finally:
        server.shutdown()


@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_Client_create_chat_completion_works():
    """
    For this model, DeepSeek R1 Distill Qwen 1.5B, max_completion_tokens doesn't
    seem to affect the completion_tokens taken. Instead, max_tokens does.

    https://platform.openai.com/docs/api-reference/chat/object
    completion_tokens integer
    Number of tokens in the generated completion.
    """
    config = ServerConfiguration.from_yaml(
        test_data_directory / "server_configuration.yml")
    config.model_path = MODEL_DIR

    open_ai_configuration = OpenAIChatCompletionConfiguration.from_yaml(
        test_data_directory / "openai_chat_completion_configuration.yml")

    server = HTTPServer(config)
    server.start()

    try:
        client = Client(config, open_ai_configuration)

        messages = [
            create_system_message("You are a helpful assistant."),
            create_user_message("What is the capital of France?")]

        response = client.create_chat_completion(messages)

        assert response is not None
        print("response: ", response.choices[0].message.content)
        print("finish reason: ", response.choices[0].finish_reason)
        print("completion tokens: ", response.usage.completion_tokens)
        print("prompt tokens: ", response.usage.prompt_tokens)
        if response.usage.completion_tokens_details is not None:
            print(
                "reasoning tokens: ",
                response.usage.completion_tokens_details.reasoning_tokens)
        else:
            print("reasoning tokens: None")
        assert "Paris" in response.choices[0].message.content
        assert response.choices[0].message.role == "assistant"

        messages = [
            create_system_message(
                "You are a helpful and resourceful assistant."),
            create_user_message("What is the capital of Germany?")]

        response = client.create_chat_completion(messages)

        assert response is not None
        print("response: ", response.choices[0].message.content)
        print("finish reason: ", response.choices[0].finish_reason)
        print("completion tokens: ", response.usage.completion_tokens)
        print("prompt tokens: ", response.usage.prompt_tokens)
        if response.usage.completion_tokens_details is not None:
            print(
                "reasoning tokens: ",
                response.usage.completion_tokens_details.reasoning_tokens)
        else:
            print("reasoning tokens: None")

        assert "Berlin" in response.choices[0].message.content
        assert response.choices[0].message.role == "assistant"

    finally:
        server.shutdown()

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_Client_get_finish_reason_and_token_usage_works():
    config = ServerConfiguration.from_yaml(
        test_data_directory / "server_configuration.yml")
    config.model_path = MODEL_DIR

    open_ai_configuration = OpenAIChatCompletionConfiguration.from_yaml(
        test_data_directory / "openai_chat_completion_configuration.yml")

    server = HTTPServer(config)
    server.start()

    try:
        client = Client(config, open_ai_configuration)

        messages = [
            create_system_message("You are a helpful and precise assistant."),
            create_user_message("What is 1 + 1?")]

        response = client.create_chat_completion(messages)

        statistics = Client.get_finish_reason_and_token_usage(
            client.current_chat_completion)

        print(response.choices[0].message.content)
        print(statistics["finish_reason"])
        print(statistics["completion_tokens"])
        print(statistics["prompt_tokens"])
        assert statistics["finish_reason"] == response.choices[0].finish_reason
        assert statistics["completion_tokens"] > 0
        assert statistics["prompt_tokens"] == 19
        assert statistics["reasoning_tokens"] == None

    finally:
        server.shutdown()

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_Client_get_parsed_completion_works():
    config = ServerConfiguration.from_yaml(
        test_data_directory / "server_configuration.yml")
    config.model_path = MODEL_DIR

    open_ai_configuration = OpenAIChatCompletionConfiguration.from_yaml(
        test_data_directory / "openai_chat_completion_configuration.yml")

    server = HTTPServer(config)
    server.start()

    try:
        client = Client(config, open_ai_configuration)

        messages = [
            create_system_message("You are a helpful and precise assistant."),
            create_user_message("What is 1 + 2?")]

        response = client.create_chat_completion(messages)

        parsed_completion = Client.get_parsed_completion(response)

        print(parsed_completion)
        assert parsed_completion["role"] == "assistant"
        assert "3" in parsed_completion["content"]

    finally:
        server.shutdown()
