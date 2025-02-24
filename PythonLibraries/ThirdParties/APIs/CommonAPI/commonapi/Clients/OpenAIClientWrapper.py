from commonapi.Configurations import OpenAIChatCompletionConfiguration
from openai import OpenAI, AsyncOpenAI, ChatCompletion
from abc import ABC, abstractmethod

class BaseOpenAIClientWrapper(ABC):
    def __init__(
            self,
            api_key: str,
            base_url: str,
            configuration: OpenAIChatCompletionConfiguration = None):
        """
        See
        https://github.com/openai/openai-python/blob/main/src/openai/_client.py
        for the def __init__(..) of the OpenAI client.
        Note that OpenAI = client, i.e. openai.OpenAI = openai.client, as the
        openai code explicitly states.
        """
        if configuration is None:
            self.configuration = OpenAIChatCompletionConfiguration()
        else:
            self.configuration = configuration

        self.client = self._create_client(api_key, base_url)

    def clear_chat_completion_configuration(self):
        self.configuration = OpenAIChatCompletionConfiguration()

    @abstractmethod
    def _create_client(self, api_key: str, base_url: str):
        pass

    @abstractmethod
    def create_chat_completion(self, messages: list[dict]):
        pass

    @staticmethod
    def get_finish_reason_and_token_usage(response: ChatCompletion):
        statistics = {
            "finish_reason": response.choices[0].finish_reason,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens
        }

        if response.usage.completion_tokens_details is not None:
            statistics["reasoning_tokens"] = \
                response.usage.completion_tokens_details.reasoning_tokens
        else:
            statistics["reasoning_tokens"] = None

        return statistics

    @staticmethod
    def get_parsed_completion(response: ChatCompletion):
        completion_message = {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content
        }

        if len(response.choices) == 1:
            return completion_message
        else:
            completion_messages = []
            completion_messages.append(completion_message)
            for choice in response.choices[1:]:
                completion_messages.append({
                    "role": choice.message.role,
                    "content": choice.message.content
                })

            return completion_messages

class OpenAIClientWrapper(BaseOpenAIClientWrapper):
    def _create_client(self, api_key: str, base_url: str):
        """
        https://github.com/openai/openai-python/blob/main/src/openai/_client.py
        Client = OpenAI explicitly here
        https://github.com/openai/openai-python/blob/main/src/openai/_client.py#L563
        """
        return OpenAI(
            api_key=api_key,
            base_url=base_url)

    def create_chat_completion(self, messages: list[dict]):
        config_dict = self.configuration.to_dict()
        return self.client.chat.completions.create(
            messages=messages,
            **config_dict)

class AsyncOpenAIClientWrapper(BaseOpenAIClientWrapper):
    def _create_client(self, api_key: str, base_url: str):
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url)

    async def create_chat_completion(self, messages: list[dict]):
        config_dict = self.configuration.to_dict()
        return await self.client.chat.completions.create(
            messages=messages,
            **config_dict)
