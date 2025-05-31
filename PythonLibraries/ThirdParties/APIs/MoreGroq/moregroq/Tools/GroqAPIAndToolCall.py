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

        # Needed to maintain state, state of if we have to process tool calls
        # or not.
        self._handle_possible_tool_calls_result = None

    @staticmethod
    def has_message_in_response(response) -> bool:
        """
        Recall what BaseGroqWrapper.has_message_response(..) does:
        * check if it has the attribute "choices", and
        * check if there is more than 0 choices, and
        * check if the 0th choice has attribute "message"
        """
        return BaseGroqWrapper.has_message_in_response(response)

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

    def _is_no_response_and_no_tool_calls(self) -> bool:
        return self._current_response is None and \
            self._handle_possible_tool_calls_result is None

    def iteratively_handle_responses_and_tool_calls(self):

        if self._current_response is not None and \
            self._handle_possible_tool_calls_result is None:
            if GroqAPIAndToolCall.has_message_in_response(self._current_response):
                self._handle_possible_tool_calls_result = \
                    self.tool_call_processor.handle_possible_tool_calls(
                        self._current_response.choices[0].message)
                self.conversation_and_system_messages.append_general_message(
                    self._current_response.choices[0].message)
            # Once we've tried to add the message, if any in the response, to
            # the conversation history, then we can set the current response to
            self._current_response = None
            return

        if self._current_response is None and \
            self._handle_possible_tool_calls_result is not None:
            for tool_call_message in self._handle_possible_tool_calls_result:
                self.conversation_and_system_messages.append_general_message(
                    tool_call_message)
            self._handle_possible_tool_calls_result = None
            return

        # This scenario isn't expected, but let's try to handle it.
        if self._current_response is not None and \
            self._handle_possible_tool_calls_result is not None:
            for tool_call_message in self._handle_possible_tool_calls_result:
                self.conversation_and_system_messages.append_general_message(
                    tool_call_message)
            self._handle_possible_tool_calls_result = None
            if GroqAPIAndToolCall.has_message_in_response(self._current_response):
                self._handle_possible_tool_calls_result = \
                    self.tool_call_processor.handle_possible_tool_calls(
                        self._current_response.choices[0].message)
                self.conversation_and_system_messages.append_general_message(
                    self._current_response.choices[0].message)
                self._current_response = None

    def create_chat_completion(self, user_message: str = None) -> bool:
        """
        Returns:
            True - if we actually created a chat completion..
            False - if we did not create a chat completion, but rather we should
            run iteratively_handle_responses_and_tool_calls() first.
        """
        if self._is_no_response_and_no_tool_calls():
            if user_message is not None and user_message != "":
                self.conversation_and_system_messages.append_message(
                    UserMessage(content=user_message))

            self._current_response = self.groq_api_wrapper.create_chat_completion(
                self.conversation_and_system_messages.get_conversation_as_list_of_dicts())
            return True
        else:
            return False

