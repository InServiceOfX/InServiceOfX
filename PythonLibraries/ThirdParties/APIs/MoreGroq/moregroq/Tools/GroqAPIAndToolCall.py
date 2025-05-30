from commonapi.Messages import ConversationAndSystemMessages

from commonapi.Messages.Messages import UserMessage

# Because the following modules are in the same directory, we have to import
# them as such since __init__.py is not used.
from moregroq.Tools.ParseFunctionAsTool import ParseFunctionAsTool
from moregroq.Tools.ToolCallProcessor import ToolCallProcessor

from moregroq.Wrappers.GroqAPIWrapper import BaseGroqWrapper
from moregroq.Wrappers.ChatCompletionConfiguration import Tool

from typing import Callable

class GroqAPIAndToolCall:
    def __init__(self, groq_api_wrapper, tool_call_processor=None):
        self.groq_api_wrapper = groq_api_wrapper
        if tool_call_processor is None:
            self.tool_call_processor = ToolCallProcessor()
        else:
            self.tool_call_processor = tool_call_processor
        self.conversation_and_system_messages = ConversationAndSystemMessages()

        if hasattr(self.groq_api_wrapper.configuration, "tools") and \
            self.groq_api_wrapper.configuration.tools is not None:
            self.tools = self.groq_api_wrapper.configuration.tools
        else:
            self.tools = []

        self._current_response = None

    @staticmethod
    def has_message_response(response) -> bool:
        return BaseGroqWrapper.has_message_response(response)

    def set_tool_choice(self, tool_choice: str = "auto"):
        """
        "auto" means to let our LLM decide when to use tools.
        """
        self.groq_api_wrapper.configuration.tool_choice = tool_choice

    def add_tool(self, input_function: Callable):
        function_definition = ParseFunctionAsTool.parse_for_function_definition(
            input_function)
        tool = Tool(function=function_definition)
        self.tools.append(tool)
        self.groq_api_wrapper.configuration.tools = self.tools

        self.tool_call_processor.add_function(
            function_definition.name,
            input_function)

    def add_system_message(self, message: str):
        self.conversation_and_system_messages.add_system_message(message)

    def create_chat_completion_with_user_message(
        self,
        user_message: str = None):
        """This function will
        * Add user message to conversation history directly.
        * Have tool_call_processor call Groq API once with the messages. 
        - If tool call result length is 1, then no response message was returned.
        - If tool call result length is 2, then no request for a tool_call was
        made. Groq API create_chat_completion(..) only ran once.
        """
        if user_message is None or user_message == "":
            print("No user message provided.")
            return

        self.conversation_and_system_messages.append_message(
            UserMessage(content=user_message))

        tool_call_result = self.tool_call_processor.call_with_tool_calls(
            messages=\
                self.conversation_and_system_messages.get_conversation_as_list_of_dicts(),
            groq_api_wrapper=self.groq_api_wrapper
        )

        if tool_call_result is None or len(tool_call_result) < 1:
            return

        if len(tool_call_result) == 1:
            return tool_call_result[0]

        if len(tool_call_result) == 2:
            process_result, response = tool_call_result
            if response is not None and hasattr(response, "choices") and \
                len(response.choices) > 0 and \
                hasattr(response.choices[0], "message"):
                # This function call also handles if the given message, of some
                # type, at least has the "role" attribute and the role is
                # assistant.
                self.conversation_and_system_messages.append_general_message(
                    response.choices[0].message)
            else:
                print("No response message returned with response:", response)
            return process_result, response

        process_result, response, second_response = tool_call_result

        if response is not None and hasattr(response, "choices") and \
            len(response.choices) > 0 and \
            hasattr(response.choices[0], "message"):
            self.conversation_and_system_messages.append_general_message(
                response.choices[0].message)
        else:
            print("No response message returned with response:", response)

        if second_response is not None and hasattr(second_response, "choices") and \
            len(second_response.choices) > 0 and \
            hasattr(second_response.choices[0], "message"):
            self.conversation_and_system_messages.append_general_message(
                second_response.choices[0].message)
        else:
            print("No response message returned with second response:",
                second_response)

        return process_result, response, second_response

    def continuous_chat_completion_with_user_message(
        self,
        user_message: str = None):
        if self._current_response is not None:
            if hasattr(self._current_response, "choices") and \
                len(self._current_response.choices) > 0 and \
                hasattr(self._current_response.choices[0], "message"):
                self.conversation_and_system_messages.append_general_message(
                    self._current_response.choices[0].message)
                handle_possible_tool_calls_result = \
                    self.tool_call_processor.handle_possible_tool_calls(
                        self._current_response.choices[0].message)

                if handle_possible_tool_calls_result is not None:
                    for tool_call_message in handle_possible_tool_calls_result:
                        self.conversation_and_system_messages.append_general_message(
                            tool_call_message)
            else:
                self._current_response = None

        if user_message != None and user_message != "":

            self.conversation_and_system_messages.append_message(
                UserMessage(content=user_message))

            messages = \
                self.conversation_and_system_messages.get_conversation_as_list_of_dicts()
            tool_call_result = self.tool_call_processor.call_with_tool_calls(
                messages=messages,
                groq_api_wrapper=self.groq_api_wrapper)

            if tool_call_result is None or len(tool_call_result) < 1:
                return

            if len(tool_call_result) == 1:
                self._current_response = tool_call_result[0]
                return self._current_response

            if len(tool_call_result) == 2:
                process_result, response = tool_call_result
                if response is not None and hasattr(response, "choices") and \
                    len(response.choices) > 0 and \
                    hasattr(response.choices[0], "message"):
                    self.conversation_and_system_messages.append_general_message(
                        response.choices[0].message)
                    self._current_response = response
                    return self._current_response
                else:
                    print("No response message returned with response:", response)
                return process_result, response

            process_result, response, second_response = tool_call_result

            if response is not None and hasattr(response, "choices") and \
                len(response.choices) > 0 and \
                hasattr(response.choices[0], "message"):
                self.conversation_and_system_messages.append_general_message(
                    response.choices[0].message)

            if second_response is not None and hasattr(second_response, "choices") and \
                len(second_response.choices) > 0 and \
                hasattr(second_response.choices[0], "message"):
                self.conversation_and_system_messages.append_general_message(
                    second_response.choices[0].message)
                self._current_response = second_response
            else:
                print("No response message returned with second response:",
                    second_response)

            return process_result, response, second_response

    def create_chat_completion_with_user_message_until_tool_call_ends(
        self,
        user_message: str = None,
        call_limit = None):
        """This function will   
        * Add user message to conversation history directly.
        * Have tool_call_processor call Groq API once with the messages. 
        - If tool call result length is 1, then no response message was returned.
        - If tool call result length is 2, then no request for a tool_call was
        made at the very end. Responses length equal to N number of tool calls.
        """
        if user_message is None or user_message == "":
            print("No user message provided.")
            return

        self.conversation_and_system_messages.append_message(
            UserMessage(content=user_message))

        tool_call_result = \
            self.tool_call_processor.call_with_tool_calls_until_end(
                    messages=\
                        self.conversation_and_system_messages.get_conversation_as_list_of_dicts(),
                    groq_api_wrapper=self.groq_api_wrapper,
                    call_limit=call_limit)

        if tool_call_result is None or len(tool_call_result) < 1:
            return

        if len(tool_call_result) == 1:
            return tool_call_result

        if len(tool_call_result) == 2:
            process_result, responses = tool_call_result
            for response in responses:
                if response is not None and hasattr(response, "choices") and \
                    len(response.choices) > 0 and \
                    hasattr(response.choices[0], "message"):
                    # This function call also handles if the given message, of some
                    # type, at least has the "role" attribute and the role is
                    # assistant.
                    self.conversation_and_system_messages.append_general_message(
                        response.choices[0].message)
                else:
                    print("No response message returned with response:", response)
            return process_result, responses
        # We don't expect this to happen.
        else:
            try:
                process_result = tool_call_result[0]
                responses = tool_call_result[1]
                for response in responses:
                    if response is not None and hasattr(response, "choices") and \
                        len(response.choices) > 0 and \
                        hasattr(response.choices[0], "message"):
                        self.conversation_and_system_messages.append_general_message(
                            response.choices[0].message)
                    else:
                        print(
                            "No response message returned with response:",
                            response)
                return process_result, responses
            except Exception as e:
                print(
                    "Error in create_chat_completion_with_user_message_until_tool_call_ends:",
                    e)
                return None, []
