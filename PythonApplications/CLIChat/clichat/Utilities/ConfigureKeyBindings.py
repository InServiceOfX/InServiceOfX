from clichat.Utilities import Printing
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import run_in_terminal

class ConfigureKeyBindings:
    def __init__(self, configuration):
        self._configuration = configuration
        self.key_bindings = KeyBindings()
        self._default_entry = None

    def configure_key_bindings(self):
        """Configure and return KeyBindings.
        
        Returns:
            KeyBindings object with configured bindings
        """
        @self.key_bindings.add(*self._configuration.hotkey_exit)
        def _(event):
            buffer = event.app.current_buffer
            buffer.text = self._configuration.exit_entry
            buffer.validate_and_handle()

        @self.key_bindings.add(*self._configuration.hotkey_cancel)
        def _(event):
            buffer = event.app.current_buffer
            buffer.reset()

        @self.key_bindings.add(*self._configuration.hotkey_insert_newline)
        def _(event):
            buffer = event.app.current_buffer
            buffer.newline()

        @self.key_bindings.add(*self._configuration.hotkey_toggle_word_wrap)
        def _(_):
            self._configuration.wrap_words = not self._configuration.wrap_words
            run_in_terminal(
                lambda: Printing.print_key_value(
                    f"Word Wrap: '{'enabled' if self._configuration.wrap_words else 'disabled'}'!"))

        @self.key_bindings.add(*self._configuration.hotkey_new)
        def _(event):
            buffer = event.app.current_buffer
            self._default_entry = buffer.text
            buffer.text = ".new"
            buffer.validate_and_handle()

        return self.key_bindings
