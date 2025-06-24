# ABC stands for Abstract Base Class.
from abc import ABC, abstractmethod
from groq import AsyncGroq, Groq
from moregroq.Wrappers import GetAllActiveModels
from moregroq.Wrappers.ChatCompletionConfiguration import ChatCompletionConfiguration

class BaseGroqWrapper(ABC):
    def __init__(self, api_key: str, groq_client_configuration = None):
        self.configuration = ChatCompletionConfiguration()
        self.client = self._create_client(api_key)

        if groq_client_configuration is not None:
            client_configuration = groq_client_configuration.to_dict()
            for key in client_configuration:
                if hasattr(self.configuration, key):
                    setattr(self.configuration, key, client_configuration[key])

    def clear_chat_completion_configuration(self):
        self.configuration = ChatCompletionConfiguration()

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

    @staticmethod
    def has_message_in_response(response) -> bool:
        if response is not None and \
            hasattr(response, "choices") and \
            len(response.choices) > 0 and \
            hasattr(response.choices[0], "message"):
            return True
        return False

class GroqAPIWrapper(BaseGroqWrapper):
    def _create_client(self, api_key: str) -> Groq:
        return Groq(api_key=api_key)
        
    def create_chat_completion(self, messages: list[dict]):
        config_dict = self.configuration.to_dict()
        return self.client.chat.completions.create(
            messages=messages,
            **config_dict)

    def get_json_response(self, messages: list[dict]):
        self.configuration.response_format = {"type": "json_object"}

        config_dict = self.configuration.to_dict_for_json_response()

        return self.client.chat.completions.create(
            messages=messages,
            **config_dict)

class AsyncGroqAPIWrapper(BaseGroqWrapper):
    def _create_client(self, api_key: str) -> AsyncGroq:
        return AsyncGroq(api_key=api_key)
        
    async def create_chat_completion(self, messages: list[dict]):
        config_dict = self.configuration.to_dict()
        return await self.client.chat.completions.create(
            messages=messages,
            **config_dict)

    async def get_json_response(self, messages: list[dict]):
        self.configuration.response_format = {"type": "json_object"}

        config_dict = self.configuration.to_dict_for_json_response()

        return await self.client.chat.completions.create(
            messages=messages,
            **config_dict)
