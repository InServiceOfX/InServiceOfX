from moresglang.Configurations import (
    OpenAIChatCompletionConfiguration,
    ServerConfiguration
)

import openai
from typing import Union, List, Optional, Dict

class Client:
    def __init__(
        self, 
        server_configuration: ServerConfiguration,
        chat_completion_configuration: OpenAIChatCompletionConfiguration
    ):
        self.server_configuration = server_configuration
        self.chat_completion_configuration = chat_completion_configuration
        self.client = openai.Client(
            base_url=f"http://localhost:{self.server_configuration.port}/v1",
            api_key="None"
        )

        self.current_chat_completion = None

    def create_chat_completion(self, messages, response_format=None,tools=None):
        """
        https://platform.openai.com/docs/api-reference/chat/create

        tools array Optional
        List of tools model may call. Currently, only functions are supported as
        a tool. Use this to provide a list of functions model may generate JSON
        inputs for.
        """

        # Get all field names from the dataclass
        valid_keys = self.chat_completion_configuration.__dataclass_fields__.keys()
        
        # Create kwargs dict from configuration, filtering out None values
        kwargs = {
            k: getattr(self.chat_completion_configuration, k)
            for k in valid_keys 
            if getattr(self.chat_completion_configuration, k) is not None
        }
        
        kwargs["model"] = str(self.server_configuration.model_path)

        kwargs["messages"] = messages

        if response_format is not None:
            kwargs["response_format"] = response_format

        if tools is not None:
            kwargs["tools"] = tools

        self.current_chat_completion = self.client.chat.completions.create(
            **kwargs)
        return self.current_chat_completion

    @staticmethod
    def create_developer_message(
        content: Union[str, List[str]],
        name: Optional[str] = None) -> Dict[str, str]:
        """
        https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages

        Developer-provided instructions that model should follow, regardless of
        messages sent by user. With o1 models and newer, use developer messages
        for this purpose instead.
        """
        message = {
            "role": "developer",
            "content": content
        }
        if name is not None:
            message["name"] = name
        return message

    @staticmethod
    def create_system_message(
        content: Union[str, List[str]],
        name: Optional[str] = None) -> Dict[str, str]:
        """
        https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages

        Developer-provided instructions that model should follow, regardless of
        messages sent by the user. With o1 models and newer, use developer
        messages for this purpose instead.
        """
        message = {
            "role": "system",
            "content": content
        }
        if name is not None:
            message["name"] = name
        return message

    @staticmethod
    def create_user_message(
        content: Union[str, List[str]],
        name: Optional[str] = None) -> Dict[str, str]:
        """
        https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages

        Messages sent by an end user.
        """
        message = {
            "role": "user",
            "content": content
        }
        if name is not None:
            message["name"] = name
        return message

    @staticmethod
    def get_finish_reason_and_token_usage(response: openai.ChatCompletion):
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
    def get_parsed_completion(response: openai.ChatCompletion):
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
