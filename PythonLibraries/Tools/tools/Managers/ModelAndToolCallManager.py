from commonapi.Messages import AssistantMessage, ToolMessage
from typing import Dict
from warnings import warn

import json

class ModelAndToolCallManager:
    MAX_NUMBER_OF_ITERATIONS = 10
    def __init__(
            self,
            model_and_tokenizer,
            tool_call_processor,
            max_number_of_iterations = MAX_NUMBER_OF_ITERATIONS):
        self._model_and_tokenizer = model_and_tokenizer
        self._tool_call_processor = tool_call_processor
        self._max_number_of_iterations = max_number_of_iterations

    def process_messages(self, messages):
        """
        Returns:
            - tuple of 4: if first (index 0) element is True, then a list of
            Python dicts, messages, is returned in the second (index 1) element,
            and the result of decoding is returned in the third (index 2)
            element.
            If first (index 0) element is False, then we had surely exceeded the
            maximum number of iterations allowed while trying to handle tool
            calls. In this case, the second (index 1) element is the mutated
            messages.
            Last element is any new messages created.
        """
        new_messages = []
        iteration_count = 0

        while iteration_count < self._max_number_of_iterations:

            process_messages_once_results = \
                self._process_messages_once_for_any_tool_calls(
                    messages,
                    new_messages
                )
            # No tool call was obtained,
            if not process_messages_once_results[0]:
                return (
                    True,
                    process_messages_once_results[1],
                    process_messages_once_results[2],
                    new_messages)
            # Tool call was obtained
            else:
                messages, new_messages = \
                    self._run_tool_calls_and_append_to_messages(
                        process_messages_once_results[1],
                        process_messages_once_results[2],
                        new_messages)

        warn("Message processing iterations exceeded maximum due to tool calls")
        return False, messages, new_messages

    def _process_messages_once_for_any_tool_calls(
            self,
            messages: Dict[str, str],
            new_messages: list):
        """
        Returns:
            - tuple of 4; if first is True, then a tool call was detected and
            the tool_calls are returned as the 3rd (index 2) element.
            If first (index 0) is False, then no tool call was detected and the
            decoded response is returned as the 3rd (index 2) element.

            In both cases, the second (index 1) element is the mutated messages.
            The fourth (index 3) element is any new messages created.
        """
        tools = self._tool_call_processor.get_tools_as_list()

        input_ids = self._model_and_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            tools=tools,
            to_device=True)

        outputs = self._model_and_tokenizer._model.generate(
            **input_ids,
            **self._model_and_tokenizer._generation_configuration.to_dict(),
            )

        decoded = self._model_and_tokenizer.decode_with_tokenizer(
            outputs,
            skip_special_tokens=True)

        if self._tool_call_processor.has_nonempty_tool_call(decoded):
            decoded = self._model_and_tokenizer._tokenizer.decode(
                self._tool_call_processor.parse_generate_output_for_output_only(
                    outputs,
                    input_ids),
                skip_special_tokens=True)

            tool_calls = self._tool_call_processor._parse_tool_call(decoded)

            if tool_calls is None:
                return False, messages, decoded

            assistant_message_with_tool_calls = \
                self._tool_call_processor._convert_tool_calls_to_assistant_message(
                    tool_calls)

            messages.append(assistant_message_with_tool_calls.to_dict())
            new_messages.append(assistant_message_with_tool_calls)

            return True, messages, tool_calls, new_messages
        else:
            return False, messages, decoded, new_messages

    def _run_tool_calls_and_append_to_messages(
            self,
            messages: Dict[str, str],
            tool_calls,
            new_messages: list):
        tool_call_responses = self._tool_call_processor.handle_possible_tool_calls(
            tool_calls
        )

        # This should be checked beforehand if
        # _process_messages_once_for_any_tool_calls() is run before.
        if tool_call_responses is None:
            return messages, new_messages

        tool_response_messages = []
        for tool_call_response in tool_call_responses:
            tool_response_messages.append(ToolMessage(
                name=tool_call_response[0],
                content=json.dumps(tool_call_response[1]),
                role="tool"
            ))

        for tool_response_message in tool_response_messages:
            messages.append(tool_response_message.to_dict())
            new_messages.append(tool_response_message)
        return messages, new_messages

    @staticmethod
    def _create_assistant_message(decoded):
        assistant_message = AssistantMessage(content=str(decoded))
        return assistant_message

    @staticmethod
    def _add_assistant_message_to_conversation_system_and_permanent(
            conversation_system_and_permanent,
            decoded):
        conversation_system_and_permanent.append_message(
            ModelAndToolCallManager._create_assistant_message(decoded))

        return AssistantMessage(content=str(decoded))

    @staticmethod
    def update_conversation_system_and_permanent_from_process_messages(
        process_messages_results,
        conversation_system_and_permanent):
        assert len(process_messages_results) == 4
        new_messages = process_messages_results[3]
        for new_message in new_messages:
            if hasattr(new_message, "content"):
                conversation_system_and_permanent.append_message(
                    new_message)

        decoded = process_messages_results[2]
        conversation_system_and_permanent.append_message(
            ModelAndToolCallManager._create_assistant_message(
                decoded))

        return decoded