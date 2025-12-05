from typing import Dict, Any, Callable, List

from tools.Messages import FunctionCallOutputMessage

import json

class OpenAIAPIToolCallProcessor:

    @staticmethod
    def default_result_to_string(result: Any) -> str:
        if isinstance(result, dict):
            return json.dumps(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)

    def __init__(
        self,
        process_function_result: Callable = default_result_to_string):
        self._available_functions: Dict[str, Callable] = None
        self._process_function_result = process_function_result

    def add_function(self, function_name: str, function: Callable):
        if self._available_functions is None:
            self._available_functions = {}

        self._available_functions[function_name] = function

    def change_process_function_result(self, process_function_result: Callable):
        self._process_function_result = process_function_result

    def handle_possible_tool_calls(self, response_message: Any):
        """
        For OpenAI API:
        https://platform.openai.com/docs/guides/function-calling#function-tool-example
        
        Execute function calls and append results

        for tool_call in response.output:
            if tool_call.type != "function_call":
                continue

            name = tool_call.name
            args = json.loads(tool_call.arguments)

            result = call_function(name, args)
            input_messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result)
            })
        """
        tool_call_messages = []

        for tool_call in response_message.tool_calls:
            if tool_call.type != "function_call":
                continue

            function_name = tool_call.name
            function_to_call = self._available_functions.get(function_name)

            if function_to_call is None:
                continue

            function_args = json.loads(tool_call.arguments)

            function_result = function_to_call(**function_args)

            tool_call_messages.append(FunctionCallOutputMessage(
                call_id=tool_call.call_id,
                output=self._process_function_result(function_result)
            ))

        return tool_call_messages

    def handle_possible_tool_calls_as_response(self, response: Any):
        """
        For OpenAI API:
        https://platform.openai.com/docs/guides/function-calling#function-tool-example
        
        it appears Open AI API has an object called Response as opposed to
        ChatCompletion.

        for tool_call in response.output:
            if tool_call.type != "function_call":
                continue

            name = tool_call.name
            args = json.loads(tool_call.arguments)

            result = call_function(name, args)
            input_messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result)
            })
        """
        tool_call_messages = []

        # Example (actual) response.output:
        # output=[ResponseFunctionToolCall(
        #     arguments='{"sign":"Aquarius"}',
        #     call_id='call_93754041',
        #     name='get_horoscope',
        #     type='function_call',
        #     id='fc_d6a807db-284e-2c20-1095-e029f2c08004_0',
        #     status='completed'
        # )]

        for tool_call in response.output:
            if tool_call.type != "function_call":
                continue

            function_name = tool_call.name
            function_to_call = self._available_functions.get(function_name)

            if function_to_call is None:
                continue

            function_args = json.loads(tool_call.arguments)
            function_result = function_to_call(**function_args)

            function_call_results = FunctionCallOutputMessage(
                call_id=tool_call.call_id,
                output=self._process_function_result(function_result)
            )

            tool_call_messages.append(function_call_results.to_dict())

        return tool_call_messages
