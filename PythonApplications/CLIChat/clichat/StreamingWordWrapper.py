from threading import Event
import shutil
from typing import Any, Iterator
from prompt_toolkit.output import create_output
from clichat.Utilities.Formatting import get_string_width

class StreamingWordWrapper:
    def __init__(self, runtime_configuration):
        self._runtime_configuration = runtime_configuration
        self._terminal_width = shutil.get_terminal_size().columns
        self._current_line_length = 0
        self._output = create_output()

    def _write_char(self, char: str) -> None:
        """Write a single character with wrapping"""
        char_width = get_string_width(char)
        
        if char == '\n':
            self._output.write(char)
            self._current_line_length = 0
            return

        if self._current_line_length + char_width > self._terminal_width:
            self._output.write('\n')
            self._current_line_length = 0

        self._output.write(char)
        self._output.flush()
        self._current_line_length += char_width

    def stream_outputs(self, streaming_event: Event, chat_completion: Iterator[Any]) -> None:
        """Stream chat completion responses with character-by-character wrapping"""
        chat_response = ""
        
        # Start with a newline
        self._output.write('\n')
        self._current_line_length = 0
        
        for chunk in chat_completion:
            if streaming_event.is_set():
                break

            content = chunk.choices[0].delta.content
            if not content:
                continue

            chat_response += content
            for char in content:
                self._write_char(char)

        # Always ensure two newlines at the end:
        # One for ending the response, one for the next prompt
        self._output.write('\n\n')
        self._output.flush()
        
        self._runtime_configuration.new_chat_response = chat_response