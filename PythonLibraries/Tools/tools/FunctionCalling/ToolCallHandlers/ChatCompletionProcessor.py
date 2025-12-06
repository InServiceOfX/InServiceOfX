from .BaseToolCallProcessor import BaseToolCallProcessor
from commonapi.Messages import create_tool_message
from typing import Any, Callable

import json

class ChatCompletionProcessor(BaseToolCallProcessor):

    default_result_to_string = BaseToolCallProcessor.default_result_to_string

    def __init__(
        self,
        process_function_result: Callable = default_result_to_string):
        super().__init__(process_function_result)

    def handle_possible_tool_calls(self, response_message: Any):
        tool_calls = getattr(response_message, 'tool_calls', None)
        if not tool_calls:
            return None

        tool_call_messages = []

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = self._available_functions.get(function_name)

            if function_to_call:
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                
                tool_call_messages.append(
                    create_tool_message(
                        content=self._process_function_result(function_response),
                        name=function_name,
                        tool_call_id=tool_call.id))
        return tool_call_messages

    # Helpful utility functions

    @staticmethod
    def message_has_tool_calls(message) -> bool:
        """
        Args:
            message: This is typically the result of response.choices[0].message
            where response is what chat completion returns from the LLM model.
        """
        if not hasattr(message, "tool_calls"):
            return False

        if message.tool_calls == None or len(message.tool_calls) == 0:
            return False

        return True

    @staticmethod
    def choices_has_tool_calls(choices) -> bool:
        """
        Args:
            choices: This is typically response.choices, where response was the
            returned object from a chat completion by the LLM model. It is
            typically a list of
            <class 'openai.types.chat.chat_completion.Choice'>
        """
        has_tool_calls = True

        for choice in choices:
            if ChatCompletionProcessor.message_has_tool_calls(choice.message):
                return True

        return False

    def process_all_tool_call_requests(self,choices):
        """

        Args:
            choices: This is typically response.choices, where response was the
            returned object from a chat completion by the LLM model. It is 
            typically a list of Choice objects, i.e.
            <class 'openai.types.chat.chat_completion.Choice'>
        """
        all_tool_call_messages = []
        for choice in choices:
            tool_call_messages = self.handle_possible_tool_calls(choice.message)
            all_tool_call_messages += tool_call_messages
        return all_tool_call_messages