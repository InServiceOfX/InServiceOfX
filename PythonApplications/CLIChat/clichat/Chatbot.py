from clichat.Configuration import RuntimeConfiguration
from clichat import StreamingWordWrapper
from clichat.Persistence import ChatLogger, SystemMessagesManager
from clichat.Terminal import (
    CreateBottomToolbar,
    create_prompt_session,
    GroqModelSelector,
    PromptWrapperInputs,
    SinglePrompt,
    show_system_message_dialog)
from clichat.Utilities.FileIO import (
    get_existing_chat_history_path_or_fail,
    setup_chat_history_file)
from clichat.Utilities import Printing
from clichat.Utilities import get_environment_variable
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

        self.system_messages_manager = SystemMessagesManager()
        self.system_messages_manager.handle_initialization(self._configuration)

        self.reset_messages()

        if self._configuration is not None:
            self.chat_history_path = get_existing_chat_history_path_or_fail(
                self._configuration)
            # TODO: Implement ChatLogger.
            #self.chat_log_path = ChatLogger(self._configuration.chat_log_path)
        else:
            self.chat_history_path = setup_chat_history_file()
            #self.chat_log_path = ChatLogger()

    def reset_messages(self):
        self.messages = [
            create_system_message(msg.content) \
                for msg in self.system_messages_manager.get_active_messages()]

    def _append_active_system_messages(self):
        current_system_contents = [msg["content"] for msg in self.messages \
            if msg["role"] == "system"]

        for active_msg in self.system_messages_manager.get_active_messages():
            if active_msg.content not in current_system_contents:
                self.messages.append(create_system_message(active_msg.content))

    def _create_completer(self, runtime_configuration):
        command_descriptions = {
            ".model": "Change the LLM model and max tokens for responses",
            ".active_system_messages": "Display all currently active system "
                "messages",
            ".add_system_message": "Add a new system message to the collection",
            ".configure_system_messages": "Configure which system messages "
                "are active",
            ".reset_messages": "Reset conversation to only include active "
                "system messages",
            ".temperature": "TODO: Adjust the temperature (creativity) of the "
                "model's responses",
            ".togglewordwrap": "Toggle word wrapping for long responses",
            ".toggle_prompt_history": "Toggle whether to keep previous user "
                "prompts in history",
            ".exit": "Exit the application and optionally save system messages"
        }
        
        completer_list = [(cmd, desc) for cmd, desc in command_descriptions.items()]
        
        return FuzzyCompleter(WordCompleter(
            words=dict(completer_list).keys(),
            meta_dict=command_descriptions,
            ignore_case=True))

    def run_iteration(
        self,
        prompt_session,
        bottom_toolbar) -> bool:
        completer = self._create_completer(self._runtime_configuration)

        prompt_wrapper_input = PromptWrapperInputs(
            completer=completer,
            style=self.prompt_style)

        prompt = SinglePrompt.run(
            self._configuration,
            self._runtime_configuration,
            prompt_wrapper_inputs=prompt_wrapper_input,
            input_indicator="",
            prompt_session=prompt_session,
            bottom_toolbar=bottom_toolbar)

        if prompt == self._configuration.exit_entry:
            self.system_messages_manager.handle_exit(self._configuration)
            return False
        elif prompt.lower() == ".active_system_messages":
            self.system_messages_manager.show_active_system_messages(
                self._configuration)
            return True
        elif prompt.lower() == ".add_system_message":
            self.system_messages_manager.add_system_message_dialog(
                self.prompt_style)
            return True
        elif prompt.lower() == ".configure_system_messages":
            action = show_system_message_dialog(
                self.system_messages_manager,
                self.prompt_style)
            
            if action == "reset":
                self.reset_messages()
            elif action == "append":
                self._append_active_system_messages()
            return True
        elif prompt.lower() == ".reset_messages":
            self.reset_messages()
            self._printer.print_info(
                "Messages reset to active system messages only")
            return True
        elif prompt.lower() == ".togglewordwrap":
            self._runtime_configuration.wrap_words = \
                not self._runtime_configuration.wrap_words
            self._printer.print_key_value(
                f"Word Wrap: {self._runtime_configuration.wrap_words}")
        elif prompt.lower() == ".model":
            model_selector = GroqModelSelector(self._configuration)
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
        elif prompt.lower() == ".toggle_prompt_history":
            self._runtime_configuration.is_user_prompt_history_active = \
                not self._runtime_configuration.is_user_prompt_history_active
            self._printer.print_key_value(
                f"User Prompt History: {'Active' if self._runtime_configuration.is_user_prompt_history_active else 'Inactive'}")
            return True
        # This is the "main" thing that happens: this is where the LLM gets fed
        # the prompt.
        elif prompt := prompt.strip():
            user_message = create_user_message(prompt)
            
            # Check if we need to remove the last user message
            if not self._runtime_configuration.is_user_prompt_history_active \
                and self.messages:
                if self.messages[-1]["role"] == "user":
                    self.messages.pop()
            
            self.messages.append(user_message)

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


    def run(self):

        prompt_session = create_prompt_session(self.chat_history_path)

        self._printer.print_as_html_formatted_text(f"\n{self.name} loaded!")
        self._printer.print_as_html_formatted_text(f"```system message(s)```")
        self.system_messages_manager.show_active_system_messages(
            self._configuration)
        self._printer.print_as_html_formatted_text("```")

        bottom_toolbar = CreateBottomToolbar(
            self._configuration,
            self._runtime_configuration).create_bottom_toolbar()
        print(f"(To exit, enter '{self._configuration.exit_entry}')\n")

        while True:
            if not self.run_iteration(prompt_session, bottom_toolbar):
                break





