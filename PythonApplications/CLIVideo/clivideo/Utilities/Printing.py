from clivideo.Utilities.Formatting import wrap_text
from prompt_toolkit import print_formatted_text, HTML
import shutil

class Printing:
    def __init__(self, configuration):
        self._configuration = configuration

    @staticmethod
    def print_info(message: str) -> None:
        print_formatted_text(
            HTML(
                "<ansigreen>"
                f">> {message}"
                "</ansigreen>"
            )
        )

    def print_error(self, message: str) -> None:
        print_formatted_text(
            HTML(
                f"<{self._configuration.terminal_ErrorColor}>"
                f"Error: {message}"
                f"</{self._configuration.terminal_ErrorColor}>"
            )
        )

    def print_wrapped_text(self, content, runtime_configuration):
        if runtime_configuration.wrap_words:
            # wrap words to fit terminal width
            terminal_width = shutil.get_terminal_size().columns
            print(wrap_text(content, terminal_width))
        else:
            print(content)

    def print_as_html_formatted_text(self, content):
        print_formatted_text(
            HTML(f"<{self._configuration.terminal_PromptIndicatorColor2}>{content}</{self._configuration.terminal_PromptIndicatorColor2}>"))

    def print_key_value(self, content):
        split_content = content.split(": ", 1)
        if len(split_content) == 2:
            key, value = split_content
            print_formatted_text(
                HTML(f"<{self._configuration.terminal_PromptIndicatorColor2}>{key}:</{self._configuration.terminal_PromptIndicatorColor2}> {value}"))
        else:
            self.print_as_html_formatted_text(split_content)
