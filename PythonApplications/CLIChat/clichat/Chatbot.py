from clichat.Configuration import RuntimeConfiguration
from clichat import StreamingWordWrapper
from clichat.Terminal import (
    CreateBottomToolbar,
    create_prompt_session,
    ModelSelector,
    PromptWrapperInputs,
    SinglePrompt)
from clichat.Utilities.FileIO import (
    get_existing_chat_history_path_or_fail,
    setup_chat_history_file)
from clichat.Utilities import Printing
from corecode.Utilities import get_environment_variable
from moregroq.Prompting.PromptTemplates import (
    create_user_message,
    create_system_message)
from moregroq.Wrappers import GroqAPIWrapper
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter

import threading, traceback

class Chatbot:
    def __init__(
        self,
        name: str="CLIChat",
        configuration=None):

        self.name = name
        if configuration is None:
            self.temperature = 1.0
        else:
            self.temperature = configuration.temperature \
                if configuration.temperature is not None else 1.0

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
        self._printer = Printing(self._configuration)

        self.messages = [create_system_message(
            self._runtime_configuration.system_message),]

        if self._configuration is not None:
            self.chat_history_path = get_existing_chat_history_path_or_fail(
                self._configuration)
        else:
            self.chat_history_path = setup_chat_history_file()
    

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

    def run_iteration(
        self,
        prompt_session,
        bottom_toolbar,
        prompt: str="") -> bool:
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
            return False
        elif self._runtime_configuration.current_messages is None and \
            prompt.lower() == ".togglewordwrap":
            self._runtime_configuration.wrap_words = \
                not self._runtime_configuration.wrap_words
            self._printer.print_key_value(
                f"Word Wrap: {self._runtime_configuration.wrap_words}")
        elif prompt.lower() == ".model":
            model_selector = ModelSelector(self._configuration)
            model, max_tokens = model_selector.select_model_and_tokens()
            self._runtime_configuration.model = model
            self._runtime_configuration.max_tokens = max_tokens
            self._printer.print_wrapped_text(
                f"Model updated to: {model}",
                self._runtime_configuration)
            if max_tokens:
                self._printer.print_wrapped_text(
                    f"Max tokens set to: {max_tokens}",
                    self._runtime_configuration)
            else:
                self._printer.print_wrapped_text(
                    "Max tokens: Using model default",
                    self._runtime_configuration)
        elif prompt := prompt.strip():
            self._printer.print_wrapped_text(
                prompt,
                self._runtime_configuration)

            streaming_word_wrapper = StreamingWordWrapper(
                self._runtime_configuration)

            try:
                groq_api_interface = GroqAPIWrapper(
                    api_key=get_environment_variable("GROQ_API_KEY"))
                groq_api_interface.configuration.temperature = self.temperature
                groq_api_interface.configuration.max_tokens = \
                    self._runtime_configuration.max_tokens
                groq_api_interface.configuration.model = \
                    self._runtime_configuration.model
                groq_api_interface.configuration.n = 1
                groq_api_interface.configuration.stream = True

                chat_completion = groq_api_interface.create_chat_completion(
                    messages=self.messages)

                streaming_event = threading.Event()

                self.streaming_thread = threading.Thread(
                    target=streaming_word_wrapper.stream_outputs,
                    args=(streaming_event, chat_completion))

                self.streaming_thread.start()

                # When streaming is done.
                self.streaming_thread.join()

            except Exception as err:
                self._printer.print_as_html_formatted_text(traceback.format_exc())
                print(f"Error: {err}")

        return True


    def run(self, prompt: str = ""):
        if self.default_prompt:
            prompt, self.default_prompt = self.default_prompt, ""

        prompt_session = create_prompt_session(self.chat_history_path)

        self._printer.print_as_html_formatted_text(f"\n{self.name} loaded!")
        self._printer.print_as_html_formatted_text(f"```system message```")
        self._printer.print_wrapped_text(
            self._runtime_configuration.system_message,
            self._runtime_configuration)
        self._printer.print_as_html_formatted_text("```")

        bottom_toolbar = CreateBottomToolbar(
            self._configuration).create_bottom_toolbar()
        print(f"(To exit, enter '{self._configuration.exit_entry}')\n")

        while True:
            if not self.run_iteration(prompt_session, bottom_toolbar, prompt):
                break





