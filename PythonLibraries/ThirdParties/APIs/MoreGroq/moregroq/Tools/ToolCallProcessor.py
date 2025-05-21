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

    def handle_possible_tool_calls(self, response_message: Any):
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

    def call_with_tool_calls(
            self,
            messages: List[Dict[str, Any]],
            groq_api_wrapper):
        """
        Returns:
            response - if in first .create_chat_completion(..) call, there was
            no 'choices' or no 'choices[0]' or no 'choices[0].message'
            None, response - if there were no tool_calls, but there was a
            response message.
            process_result, response - if there were tool_calls, and the
            process_result is the number of tool_calls.
        """
        response = groq_api_wrapper.create_chat_completion(messages)
        self.messages = messages
        if (getattr(response, 'choices', None) and \
            len(response.choices) > 0 and \
            getattr(response.choices[0], 'message', None)):
            process_result = self.process_response(response.choices[0].message)

            if process_result is None:
                return process_result, response
        else:
            return response

        second_response = groq_api_wrapper.create_chat_completion(self.messages)

        return process_result, response, second_response

    # But what if the second create_chat_completion(..) call returns yet another
    # tool call?

    def call_with_tool_calls_until_end(
        self,
        messages: List[Dict[str, Any]],
        groq_api_wrapper,
        call_limit = None):
        """
        Returns:
            response - if in first .create_chat_completion(..) call, there was
            no 'choices' or no 'choices[0]' or no 'choices[0].message'
        """
        DEFAULT_CALL_LIMIT = 32
        if call_limit is None or call_limit < 1:
            call_limit = DEFAULT_CALL_LIMIT

        responses = []

        self.messages = messages

        index = 0
        while index < call_limit:

            response = groq_api_wrapper.create_chat_completion(self.messages)
            responses.append(response)
            if hasattr(response, 'choices') and \
                len(response.choices) > 0 and \
                hasattr(response.choices[0], 'message'):
                process_result = self.process_response(
                    response.choices[0].message)

                if process_result is None:
                    return process_result, responses
            else:
                return responses

            index += 1

        try:
            return process_result, responses
        except:
            return None, responses
