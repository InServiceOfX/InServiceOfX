from dataclasses import dataclass
from typing import Dict, Any, Callable, List
import json

@dataclass
class ToolCallProcessor:
    """
    Handles processing of tool calls and message management for multi-step LLM
    interactions
    """
    available_functions: Dict[str, Callable]
    messages: List[Dict[str, Any]]

    def process_response(self, response_message: Any):
        """
        Process response message and handle any tool calls
        
        Returns:
            bool: True if tool calls were processed, False otherwise
        """
        tool_calls = getattr(response_message, 'tool_calls', None)
        if not tool_calls:
            return None

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
            self.messages.append({
                "role": "tool",
                "content": str(function_response),
                "tool_call_id": tool_call.id,
                "name": function_name
            })
            
        return len(tool_calls)