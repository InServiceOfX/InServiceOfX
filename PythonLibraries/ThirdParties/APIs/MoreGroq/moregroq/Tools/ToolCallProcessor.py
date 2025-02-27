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
    available_functions: Dict[str, Callable]
    messages: List[Dict[str, Any]] = None

    current_tool_calls: List[Any] = None

    def process_response(self, response_message: Any):
        """
        Process response message and handle any tool calls

        response_message - this is typically the result of chat completion and
        then getting .choices[0].message.        
        Returns:
            bool: True if tool calls were processed, False otherwise
        """
        tool_calls = getattr(response_message, 'tool_calls', None)
        if not tool_calls:
            return None

        self.current_tool_calls = tool_calls

        # Append assistant's response with tool calls
        self.messages.append(response_message)
        
        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = self.available_functions.get(function_name)
            
            if not function_to_call:
                raise ValueError(
                    f"Function {function_name} not found in available functions")
                
            # Parse and execute function
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            
            # Add tool response to messages
            self.messages.append(
                create_tool_message(
                    content=str(function_response),
                    name=function_name,
                    tool_call_id=tool_call.id))
            
        return len(tool_calls)

    def call_with_tool_calls(
            self,
            messages: List[Dict[str, Any]],
            groq_api_wrapper):
        response = groq_api_wrapper.create_chat_completion(messages)
        self.messages = messages
        if (getattr(response, 'choices', None) and \
            len(response.choices) > 0 and \
            getattr(response.choices[0], 'message', None)):
            process_result = self.process_response(response.choices[0].message)
        else:
            return None

        response = groq_api_wrapper.create_chat_completion(self.messages)

        return process_result, response
