from clichat.Utilities import Printing
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter

class Chatbot:
    def __init__(
        self,
        name: str="CLIChat",
        configuration=None):

        self.name = name
        if configuration is None:
            self.temperature = 1.0
            self.max_tokens = 8192
        else:
            self.temperature = configuration.temperature \
                if configuration.temperature is not None else 1.0
            self.max_tokens = configuration.max_tokens \
                if configuration.max_tokens is not None else 8192

        self.default_prompt = ""
        self.prompt_style = Style.from_dict({
            # User input (default text).
            "": configuration.terminal_CommandEntryColor2 \
                if configuration is not None and \
                    configuration.terminal_CommandEntryColor2 is not None \
                else "ansigreen",
            # Prompt.
            "indicator": configuration.terminal_PromptIndicatorColor2 \
                if configuration is not None and \
                    configuration.terminal_PromptIndicatorColor2 is not None \
                else "ansicyan",})

        self._configuration = configuration

    def _create_bottom_toolbar(self):
        return \
            f""" {str(self._configuration.hotkey_exit).replace("'", "")} {self._configuration.exit_entry}"""

    def _create_completer(self, runtime_configuration):
        completer_list = [
            ".new",
            ".api",
            ".model",
            ".systemmessage",
            ".temperature",
            ".maxtokens",
            ".togglewordwrap",
            self._configuration.exit_entry]

        return None if runtime_configuration.current_messages is not None else \
            FuzzyCompleter(WordCompleter(completer_list, ignore_case=True))

    def run(self, prompt: str = ""):
        printer = Printing(self._configuration)

        printer.print_as_html_formatted_text(f"\n{self.name} loaded!")
        printer.print_as_html_formatted_text(f"```system message```")
        printer.print_wrapped_text(self._configuration.system_message)
        printer.print_as_html_formatted_text("```")

        print(f"(To exit, enter '{self._configuration.exit_entry}')\n")
        bottom_toolbar = self.create_bottom_toolbar()






