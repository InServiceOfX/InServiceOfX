from corecode.Utilities import (get_environment_variable, load_environment_file)

from groq import Groq
from moregroq.Prompting.PromptTemplates import (
    create_user_message,
    create_system_message)
from moregroq.Wrappers import AsyncGroqAPIWrapper, GroqAPIWrapper

import pytest

load_environment_file()

def test_GroqAPIWrapper_inits():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    assert isinstance(groq_api_wrapper.client, Groq)

def test_GroqAPIWrapper_chat_completion_works():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    assert groq_api_wrapper.check_model_is_available(
        get_environment_variable("GROQ_API_KEY"))

    messages = [
        create_system_message("you are a helpful assistant."),
        create_user_message("Explain the importance of fast language models")]

    groq_api_wrapper.configuration.temperature = 0.5
    groq_api_wrapper.configuration.max_tokens = 1024
    # A stop sequence is a predefined or user-specified text string that signals
    # an AI to stop generating content, ensuring its responses remain focused
    # and concise. Examples include punctuation marks and markers like "[end]".
    groq_api_wrapper.configuration.stop=None
    groq_api_wrapper.configuration.stream=False

    result = groq_api_wrapper.create_chat_completion(messages)

    # https://console.groq.com/docs/api-reference#chat
    expected_keys = [
        "choices",
        "id",
        "model",
        "object",
        "created",
        "usage",
        "system_fingerprint",
        "x_groq"]
    for key in expected_keys:
        assert hasattr(result, key)
    assert len(result.choices) == 1
    expected_choice_keys = ["index", "message", "logprobs", "finish_reason"]
    for key in expected_choice_keys:
        assert hasattr(result.choices[0], key)

    assert result.choices[0].message.content is not None
    expected_message_keys = ["content", "role"]
    for key in expected_message_keys:
        assert hasattr(result.choices[0].message, key)
    assert result.choices[0].message.role == "assistant"

def test_GroqAPIWrapper_chat_completion_works_with_no_max_tokens():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    messages = [
        create_system_message(
            "You are an expert at coding or in other words programming."),
        create_user_message("Explain the pros and cons of Rust.")]

    groq_api_wrapper.configuration.temperature = 1.0
    groq_api_wrapper.configuration.max_tokens = None
    groq_api_wrapper.configuration.stop=None
    groq_api_wrapper.configuration.stream=False

    result = groq_api_wrapper.create_chat_completion(messages)

    expected_keys = [
        "choices",
        "id",
        "model",
        "object",
        "created",
        "usage",
        "system_fingerprint",
        "x_groq"]
    for key in expected_keys:
        assert hasattr(result, key)
    assert len(result.choices) == 1
    expected_choice_keys = ["index", "message", "logprobs", "finish_reason"]
    for key in expected_choice_keys:
        assert hasattr(result.choices[0], key)

    assert result.choices[0].message.content is not None
    expected_message_keys = ["content", "role"]
    for key in expected_message_keys:
        assert hasattr(result.choices[0].message, key)
    assert result.choices[0].message.role == "assistant"

    assert isinstance(result.choices[0].message.content, str)
    assert len(result.choices[0].message.content) > 0

"""
Make a thread safe dynamically allocated string.
"""

from threading import Lock
from io import StringIO

class ThreadSafeStringBuilder:
    def __init__(self):
        self._lock = Lock()
        self._buffer = StringIO()

    def append(self, text: str) -> None:
        with self._lock:
            self._buffer.write(text)

    def get_value(self) -> str:
        with self._lock:
            return self._buffer.getvalue()


def test_GroqAPIWrapper_chat_completion_works_with_stream():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    messages = [
        create_system_message("you are a helpful assistant."),
        create_user_message("Explain the importance of fast language models")]

    groq_api_wrapper.configuration.temperature = 0.5
    groq_api_wrapper.configuration.max_tokens = 1024
    # A stop sequence is a predefined or user-specified text string that signals
    # an AI to stop generating content, ensuring its responses remain focused
    # and concise. Examples include punctuation marks and markers like "[end]".
    groq_api_wrapper.configuration.stop=None
    groq_api_wrapper.configuration.stream=True

    result = groq_api_wrapper.create_chat_completion(messages)

    expected_choice_keys = [
        "delta",
        "finish_reason",
        "index",
        "logprobs"]

    expected_delta_keys = ["content", "function_call", "role", "tool_calls"]

    thread_safe_string_builder = ThreadSafeStringBuilder()

    for chunk in result:
        assert len(chunk.choices) == 1

        for key in expected_choice_keys:
            assert hasattr(chunk.choices[0], key)

        for key in expected_delta_keys:
            assert hasattr(chunk.choices[0].delta, key)

        if chunk.choices[0].finish_reason is not None:
            break

        thread_safe_string_builder.append(chunk.choices[0].delta.content)

    # Uncomment to see the result.
    #print(thread_safe_string_builder.get_value())

def test_GroqAPIWrapper_chat_completion_stops_at_stop_sequence():
    # https://console.groq.com/docs/text-chat#streaming-a-chat-completion
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    messages = [
        create_system_message("you are a helpful assistant."),
        create_user_message(
            "Count to 10.  Your response must begin with \"1, \".  example: 1, 2, 3, ...")]

    groq_api_wrapper.configuration.temperature = 0.5
    groq_api_wrapper.configuration.max_tokens = 1024
    # A stop sequence is a predefined or user-specified text string that signals
    # an AI to stop generating content, ensuring its responses remain focused
    # and concise. Examples include punctuation marks and markers like "[end]".
    # For this example, we will use ", 6" so that the llm stops counting at 5.
    # If multiple stop values are needed, an array of string may be passed,
    # stop=[", 6", ", six", ", Six"]
    groq_api_wrapper.configuration.stop=", 6"
    groq_api_wrapper.configuration.stream=False

    result = groq_api_wrapper.create_chat_completion(messages)

    assert "1, " in result.choices[0].message.content
    assert "2" in result.choices[0].message.content
    assert "3" in result.choices[0].message.content
    assert "4" in result.choices[0].message.content
    assert "5" in result.choices[0].message.content
    # TODO: This works randomly.
    #assert "6" not in result.choices[0].message.content


def test_GroqAPIWrapper_chat_completion_stops_at_stop_sequence_with_streaming():
    # https://console.groq.com/docs/text-chat#streaming-a-chat-completion
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    messages = [
        create_system_message("you are a helpful assistant."),
        create_user_message(
            "Count to 10.  Your response must begin with \"1, \".  example: 1, 2, 3, ...")]

    groq_api_wrapper.configuration.temperature = 0.5
    groq_api_wrapper.configuration.max_tokens = 1024
    # A stop sequence is a predefined or user-specified text string that signals
    # an AI to stop generating content, ensuring its responses remain focused
    # and concise. Examples include punctuation marks and markers like "[end]".
    # For this example, we will use ", 6" so that the llm stops counting at 5.
    # If multiple stop values are needed, an array of string may be passed,
    # stop=[", 6", ", six", ", Six"]
    groq_api_wrapper.configuration.stop="6"
    groq_api_wrapper.configuration.stream=True

    result = groq_api_wrapper.create_chat_completion(messages)
    thread_safe_string_builder = ThreadSafeStringBuilder()

    for chunk in result:
        if chunk.choices[0].delta.content is not None:
            thread_safe_string_builder.append(chunk.choices[0].delta.content)

    # Uncomment to see the result.
    #print(thread_safe_string_builder.get_value())

    assert "1, " in thread_safe_string_builder.get_value()
    assert "2" in thread_safe_string_builder.get_value()
    assert "3" in thread_safe_string_builder.get_value()
    assert "4" in thread_safe_string_builder.get_value()
    assert "5" in thread_safe_string_builder.get_value()
    # TODO: This works randomly.
    #assert "6" not in thread_safe_string_builder.get_value()

@pytest.mark.asyncio
async def test_AsyncGroqAPIWrapper_chat_completion_works():
    async_groq_api_wrapper = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    messages = [
        create_system_message("you are a helpful assistant."),
        create_user_message("Explain the importance of fast language models")]

    async_groq_api_wrapper.configuration.temperature = 0.5
    async_groq_api_wrapper.configuration.max_tokens = 1024
    async_groq_api_wrapper.configuration.stop = None
    async_groq_api_wrapper.configuration.stream = False

    result = await async_groq_api_wrapper.create_chat_completion(messages)

    assert len(result.choices) == 1
    assert result.choices[0].message.content is not None
    # Uncomment to see the result.
    #print(result.choices[0].message.content)

@pytest.mark.asyncio
async def test_AsyncGroqAPIWrapper_chat_completion_works_with_streaming():
    async_groq_api_wrapper = AsyncGroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    messages = [
        create_system_message("you are a helpful assistant."),
        create_user_message("Explain the importance of fast language models")]

    async_groq_api_wrapper.configuration.temperature = 0.5
    async_groq_api_wrapper.configuration.max_tokens = 1024
    async_groq_api_wrapper.configuration.stop = None
    async_groq_api_wrapper.configuration.stream = True

    result = await async_groq_api_wrapper.create_chat_completion(messages)

    thread_safe_string_builder = ThreadSafeStringBuilder()

    # Without async keyword, TypeError: 'AsyncStream' object is not iterable.
    async for chunk in result:
        if chunk.choices[0].delta.content is not None:
            thread_safe_string_builder.append(chunk.choices[0].delta.content)

    # Uncomment to see the result.
    #print(thread_safe_string_builder.get_value())

"""
See
https://console.groq.com/docs/text-chat#streaming-a-chat-completion
for JSON Mode example.
"""

from pydantic import BaseModel
from typing import List, Optional
import json

class Ingredient(BaseModel):
    name: str
    quantity: str
    quantity_unit: Optional[str]

class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[Ingredient]
    directions: List[str]

def test_GroqAPIWrapper_chat_completion_works_with_json_mode():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    system_prompt = "You are a recipe database that outputs recipes in JSON.\n"
    # Pass the json schema to the model. Pretty printing improves results.
    json_schema = json.dumps(Recipe.model_json_schema(), indent=2)
    system_prompt += f" The JSON object must use the schema: {json_schema}"

    recipe_name = "apple pie"

    groq_api_wrapper.configuration.temperature = 0.0
    # Streaming is not supported in JSON mode.
    groq_api_wrapper.configuration.stream = False

    messages = [
        create_system_message(system_prompt),
        create_user_message("Fetch a recipe for {recipe_name}.")] 

    recipe_response = groq_api_wrapper.get_json_response(
        messages).choices[0].message.content

    assert "recipe_name" in recipe_response
    assert "ingredients" in recipe_response
    assert "directions" in recipe_response

    # Uncomment to see the result.
    #print(recipe_response)