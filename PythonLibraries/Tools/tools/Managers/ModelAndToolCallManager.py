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

    def process_messages(self, messages,):

        is_finished_processing_messages = False
        iteration_count = 0

        while not is_finished_processing_messages and \
                iteration_count < self._max_number_of_iterations:

            input_ids = self._model_and_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                tools=tools,
                to_device=True)

        return messages 

    def _process_messages_once_for_any_tool_calls(self, messages):
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

        if self._tool_call_processor.has_tool_call(decoded):
            decoded = self._model_and_tokenizer._tokenizer.decode(
                self._tool_call_processor.parse_generate_output_for_output_only(
                    outputs,
                    input_ids),
                skip_special_tokens=True)

            tool_calls = self._tool_call_processor._parse_tool_call(decoded)

            assistant_message_with_tool_calls = \
                self._tool_call_processor._convert_tool_calls_to_assistant_message(
                    tool_calls)

            messages.append(assistant_message_with_tool_calls.to_dict())

            return True, messages, tool_calls
        else:
            return False, messages, decoded
