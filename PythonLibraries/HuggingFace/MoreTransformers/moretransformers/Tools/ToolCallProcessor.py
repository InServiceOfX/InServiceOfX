from commonapi.Messages import ToolMessage
from dataclasses import dataclass
from typing import Dict, Callable, List, Any, Union

from .ToolCallChatTemplates import (
    AssistantMessageWithToolCalls,
    FunctionDefinition,
    ToolCall,
)

import json, re

@dataclass
class ToolCallProcessor:
    available_functions: Dict[str, Callable] = None

    @staticmethod
    def has_tool_call(response_message: Any) -> bool:
        return "<tool_call>" in response_message and \
            "</tool_call>" in response_message

    @staticmethod
    def _parse_tool_call(tool_call: str) -> Dict[str, Any]:
        """
        Parse multiple tool calls from text and return a list of tool call
        dictionaries.
        
        Args:
            text: String containing one or more <tool_call>...</tool_call>
            blocks
            
        Returns:
            List of parsed tool call dictionaries
            
        Raises:
            ValueError: If no valid tool calls are found or JSON parsing fails
        """
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, tool_call, re.DOTALL)

        if not matches:
            return None

        tool_calls = []

        for i, json_content in enumerate(matches):
            try:
                # Parse the JSON content
                tool_call = json.loads(json_content)
                tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse tool call {i+1}: {e}\nContent: {json_content}")

        return tool_calls

    @staticmethod
    def parse_generate_output_for_output_only(generate_outputs, input_ids):
        return generate_outputs[0][len(input_ids["input_ids"][0]):]

    @staticmethod
    def _convert_tool_calls_to_messages(tool_calls: List[Dict[str, Any]]) \
        -> List[ToolCall]:
        tool_calls_messages = []

        for tool_call in tool_calls:
            tool_calls_messages.append(
                ToolCall(
                    type="function",
                    function=FunctionDefinition(
                        name=tool_call.get('name'),
                        arguments=tool_call.get('arguments'))))

        return tool_calls_messages

    @staticmethod
    def _convert_tool_calls_to_assistant_message(tool_calls: List[Dict[str, Any]]) \
        -> AssistantMessageWithToolCalls:
        tool_calls_for_message = \
            ToolCallProcessor._convert_tool_calls_to_messages(tool_calls)

        return AssistantMessageWithToolCalls(
            role="assistant",
            tool_calls=tool_calls_for_message)

    def add_function(self, function_name: str, function: Callable):
        if self.available_functions is None:
            self.available_functions = {}

        self.available_functions[function_name] = function

    def handle_possible_tool_calls(
            self,
            possible_tool_calls: Union[str, List[Any]]):
        """
        Args:
            possible_tool_calls: This is typically the output of either
            parse_generate_output_for_output_only or _parse_tool_call.
        """
        if isinstance(possible_tool_calls, str):
            possible_tool_calls = self._parse_tool_call(possible_tool_calls)

        if possible_tool_calls is None:
            return None

        tool_call_responses = []

        for tool_call in possible_tool_calls:
            function_name = tool_call.get('name')
            function_to_call = self.available_functions.get(function_name)

            if function_to_call:
                function_args = tool_call.get('arguments')
                function_response = function_to_call(**function_args)
    
                tool_call_responses.append(function_response)

        return tool_call_responses

    def get_tools_as_list(self):
        return list(self.available_functions.values())