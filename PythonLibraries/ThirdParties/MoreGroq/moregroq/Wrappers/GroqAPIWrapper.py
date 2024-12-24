# ABC stands for Abstract Base Class.
from abc import ABC, abstractmethod
from groq import AsyncGroq, Groq
from moregroq.Wrappers import GetAllActiveModels
from pydantic import BaseModel, Field
from typing import Optional

class ChatCompletionConfiguration(BaseModel):
    """
    See
    https://console.groq.com/docs/api-reference#chat-create
    """
    # integer or null Optional
    # The maximum number of tokens that can be generated in the chat
    # completion. Total length of input tokens and generated tokens is
    # limited by model's context length.
    max_tokens: Optional[int] = Field(default=None)
    # array Required
    # A list of messages comprising conversation so far.
    messages: list[dict] = Field(default=[])
    model: str = Field(default="llama-3.3-70b-versatile")
    # integer or null Optional Defaults to 1
    # How many chat completion choices to generate for each input message.
    # Note that current moment, only n=1 is supported. Other values will
    # result in a 400 response.
    n: int = Field(default=1)
    # response_format object or null Optional
    # An object specifying format that model must output.
    # Setting to {"type": "json_object"} enables JSON mode, which guarantees the
    # message the model generates is a valid JSON.
    # Important: when using JSON mode, you *must* also instruct model to produce
    # JSON yourself via a system or user message..
    response_format: Optional[dict] = Field(default=None)
    # string / array or null Optional
    # Up to 4 sequences where API will stop generating further tokens. The
    # returned text will not contain the stop sequence.
    stop: str | list[str] | None = Field(default=None)
    # boolean or null Optional Defaults to false.
    # If set, partial message deltas will be sent. Tokens will be sent as
    # data-only server-sent events as they become available, with the
    # stream terminated by a data: [DONE] message.
    stream: bool = Field(default=False)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)

class BaseGroqWrapper(ABC):
    def __init__(self, api_key: str):
        self.configuration = ChatCompletionConfiguration()
        self.client = self._create_client(api_key)

    @abstractmethod
    def _create_client(self, api_key: str):
        """Create and return appropriate Groq client."""
        pass

    def check_model_is_available(self, api_key: str) -> bool:
        get_all_active_models = GetAllActiveModels(api_key=api_key)
        get_all_active_models()
        result = get_all_active_models.get_all_available_models_names()
        return self.configuration.model in result

    @abstractmethod
    def create_chat_completion(self, messages: list[dict]):
        """Create chat completion with current configuration."""
        pass

class GroqAPIWrapper(BaseGroqWrapper):
    def _create_client(self, api_key: str) -> Groq:
        return Groq(api_key=api_key)
        
    def create_chat_completion(self, messages: list[dict]):
        return self.client.chat.completions.create(
            model=self.configuration.model,
            messages=messages,
            max_tokens=self.configuration.max_tokens,
            n=self.configuration.n,
            stream=self.configuration.stream,
            temperature=self.configuration.temperature)

    def get_json_response(self, messages: list[dict]):
        self.configuration.response_format = {"type": "json_object"}

        return self.client.chat.completions.create(
            model=self.configuration.model,
            messages=messages,
            max_tokens=self.configuration.max_tokens,
            n=self.configuration.n,
            response_format=self.configuration.response_format,
            stream=self.configuration.stream,
            temperature=self.configuration.temperature)


class AsyncGroqAPIWrapper(BaseGroqWrapper):
    def _create_client(self, api_key: str) -> AsyncGroq:
        return AsyncGroq(api_key=api_key)
        
    async def create_chat_completion(self, messages: list[dict]):
        return await self.client.chat.completions.create(
            model=self.configuration.model,
            messages=messages,
            max_tokens=self.configuration.max_tokens,
            n=self.configuration.n,
            stream=self.configuration.stream,
            temperature=self.configuration.temperature)

    async def get_json_response(self, messages: list[dict]):
        self.configuration.response_format = {"type": "json_object"}

        return await self.client.chat.completions.create(
            model=self.configuration.model,
            messages=messages,
            max_tokens=self.configuration.max_tokens,
            n=self.configuration.n,
            response_format=self.configuration.response_format,
            stream=self.configuration.stream,
            temperature=self.configuration.temperature)
