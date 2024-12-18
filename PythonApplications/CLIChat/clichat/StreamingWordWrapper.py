import shutil

from clichat.Utilities.Formatting import wrap_text

class StreamingWordWrapper:
    def __init__(self, runtime_configuration):
        self._streaming_finished = False
        self._runtime_configuration = runtime_configuration
        self._runtime_configuration.temp_chunk = ""

    def stream_outputs(
        self,
        streaming_event,
        chat_completion,
        is_openai=False):
        terminal_width = shutil.get_terminal_size().columns
        self._runtime_configuration.new_chat_response = ""

        def finish_outputs(
            wrap_words,
            chat_response,
            terminal_width=terminal_width):
            # reset config.tempChunk
            self._runtime_configuration.temp_chunk = ""
            # add chat response to messages
            if chat_response:
                self._runtime_configuration.new_chat_response = chat_response
            if self._runtime_configuration.current_messages is not None and \
                chat_response:
                self._runtime_configuration.current_messages.append(
                    {"role": "assistant", "content": chat_response})
            # auto pager feature
            if hasattr(self._runtime_configuration, "pagerView"):
                config.pagerContent += wrapText(chat_response, terminal_width) if config.wrapWords else chat_response
                #self.addPagerContent = False
                if config.pagerView:
                    config.launchPager(config.pagerContent)
            # finishing
            if hasattr(config, "conversationStarted"):
                config.conversationStarted = True
            self.streaming_finished = True

        chat_response = ""
        self.line_width = 0
        block_start = False
        first_event = True

        for event in chat_completion:
            if not streaming_event.is_set() and not self.streaming_finished:
                # Retrieve text from response
                if is_openai:
                    answer = event if isinstance(event, str) \
                        else event.choices[0].delta.content
                elif isinstance(event, dict):
                    if "message" in event:
                        # ollama chat
                        answer = event["message"].get("content", "")
                    else:
                        # llama.cpp chat
                        answer = event["choices"][0]["delta"].get("content", "")