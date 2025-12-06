from .BaseToolCallProcessor import BaseToolCallProcessor
from typing import Any, Callable
from tools.Messages import FunctionCallOutputMessage

import json

class ResponseProcessor(BaseToolCallProcessor):

    default_result_to_string = BaseToolCallProcessor.default_result_to_string

    def __init__(
        self,
        process_function_result: Callable = default_result_to_string):
        super().__init__(process_function_result)

    @staticmethod
    def is_function_call(response: Any) -> bool:
        """Useful utility function to check if a response (i.e. of OpenAI API's
        Response type) has a function call. Can be used in a "predicate" or
        "conditional" statement (i.e. if is_function_call(response), then ...).
        """
        if hasattr(response, "output") and len(response.output) > 0:
            for item in response.output:
                if item.type == "function_call":
                    return True
        return False

    @staticmethod
    def is_text_response(response: Any) -> bool:
        """Useful utility function to check if a response (i.e. of OpenAI API's
        Response type) has a text response. Can be used in a "predicate" or
        "conditional" statement (i.e. if is_text_response(response), then ...).
        """
        for item in response.output:
            if item.type != "message":
                return False
        return True

    def handle_possible_tool_calls(self, response: Any):

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
