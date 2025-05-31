from dataclasses import dataclass
from typing import Dict, Any, Callable, List
import json

from commonapi.Messages import create_tool_message

@dataclass
class ToolCallProcessor:
    """
    Handles processing of tool calls and message management for multi-step LLM
    interactions
    """
    available_functions: Dict[str, Callable] = None

    def handle_possible_tool_calls(self, response_message: Any):
        """
        Given a response "message" (not the response object itself, but we
        assume it has an attribute/field called "message"), then if it doesn't
        have a "tool_calls" attribute, do nothing.
        Otherwise, for each tool call, get the function response or i.e. the
        output of the function call, and create a new tool message out of it,
        and return this list of new tool call messages.
        """
        tool_calls = getattr(response_message, 'tool_calls', None)
        if not tool_calls:
            return None

        tool_call_messages = []
    
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = self.available_functions.get(function_name)

            if function_to_call:
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                
                tool_call_messages.append(
                    create_tool_message(
                        content=str(function_response),
                        name=function_name,
                        tool_call_id=tool_call.id))
        return tool_call_messages

    def add_function(self, function_name: str, function: Callable):
        if self.available_functions is None:
            self.available_functions = {}

        self.available_functions[function_name] = function
