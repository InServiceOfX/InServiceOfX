from .BaseToolCallProcessor import BaseToolCallProcessor
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
                        content=str(function_response),
                        name=function_name,
                        tool_call_id=tool_call.id))
        return tool_call_messages