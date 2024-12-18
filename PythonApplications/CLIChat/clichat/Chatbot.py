from clichat.Configuration import RuntimeConfiguration
from clichat import StreamingWordWrapper
from clichat.LLMModel import GroqAPIInterface
from clichat.Prompting.PromptTemplates import (
    create_user_message,
    create_system_message)
from clichat.Terminal import (
    CreateBottomToolbar,
    create_prompt_session,
    PromptWrapperInputs,
    SinglePrompt)
from clichat.Utilities.FileIO import get_existing_chat_history_path_or_fail
from clichat.Utilities import Printing
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter

import threading

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
        self._runtime_configuration = RuntimeConfiguration()

        self.messages = [create_system_message(
            self._runtime_configuration.system_message),]

        self.chat_history_path = get_existing_chat_history_path_or_fail(
            self._configuration)
    

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

    def run_iteration(self, prompt_session, bottom_toolbar, prompt: str=""):
        completer = self._create_completer(self._runtime_configuration)

        prompt_wrapper_input = PromptWrapperInputs(
            completer=completer,
            style=self.prompt_style)

        if not prompt:
            prompt = SinglePrompt.run(
                self._configuration,
                self._runtime_configuration,
                prompt_wrapper_inputs=prompt_wrapper_input,
                input_indicator="",
                prompt_session=prompt_session,
                bottom_toolbar=bottom_toolbar)
            user_message = create_user_message(prompt)
            self.messages.append(user_message)
            if prompt and \
                not prompt in (".new", self._configuration.exit_entry) and \
                self._runtime_configuration.current_messages is not None:
                self._runtime_configuration.current_messages.append(
                    user_message)
        else:
            prompt_wrapper_input.default = prompt
            prompt_wrapper_input.accept_default = True
            prompt = SinglePrompt.run(
                self._configuration,
                self._runtime_configuration,
                prompt_wrapper_inputs=prompt_wrapper_input,
                input_indicator="",
                prompt_session=prompt_session,
                bottom_toolbar=bottom_toolbar)
            user_message = create_user_message(prompt)
            self.messages.append(user_message)
        if prompt == self._configuration.exit_entry:
            return
        elif self._runtime_configuration.current_messages is None and \
            prompt.lower() == ".togglewordwrap":
            self._configuration.wrap_words = not self._configuration.wrap_words
            Printing.print_key_value(
                f"Word Wrap: {self._configuration.wrap_words}")
        elif prompt := prompt.strip():
            Printing(self._configuration).print_wrapped_text(prompt)

            streaming_word_wrapper = StreamingWordWrapper()

            groq_api_interface = GroqAPIInterface()
            chat_completion = groq_api_interface.chat_completion(
                messages=self.messages,
                model="llama3-8b-8192",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1,
                stream=True)

            streaming_event = threading.Event()

            self.streaming_thread = threading.Thread(
                target=streaming_word_wrapper.stream_outputs,
                args=(streaming_event, chat_completion, True))

            self.streaming_thread.start()


    def run(self, prompt: str = ""):
        if self.default_prompt:
            prompt, self.default_prompt = self.default_prompt, ""

        prompt_session = create_prompt_session(self.chat_history_path)

        printer = Printing(self._configuration)

        printer.print_as_html_formatted_text(f"\n{self.name} loaded!")
        printer.print_as_html_formatted_text(f"```system message```")
        printer.print_wrapped_text(self._runtime_configuration.system_message)
        printer.print_as_html_formatted_text("```")

        bottom_toolbar = CreateBottomToolbar(
            self._configuration).create_bottom_toolbar()
        print(f"(To exit, enter '{self._configuration.exit_entry}')\n")

        self.run_iteration(prompt_session, bottom_toolbar, prompt)





