from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text

class CommandHandler:
    def __init__(self, app):
        self._app = app

        self.commands = {
            ".exit": self.handle_exit,
            ".help": self.handle_help,
            "._create_prompt_embeds": self._handle__create_prompt_embeds,
            "._delete_prompt_embeds": self._handle__delete_prompt_embeds,
        }

    def handle_command(self, command: str) -> tuple[bool, bool]:
        """
        Handle a command and return whether to continue running and if command
        was handled.

        Args:
            command: The command string (including the dot prefix)

        Returns:
            tuple: (continue_running, command_handled)
                - continue_running: True to continue running, False to exit
                - command_handled: True if command was handled, False if it
                should be treated as user input
        """
        command = command.strip().lower()

        if command in self.commands:
            return self.commands[command](), True
        else:
            return False

    def _handle__create_prompt_embeds(self) -> bool:

        if self._app._flux_nunchaku_and_loras.is_transformer_enabled():
            self._app._flux_nunchaku_and_loras.delete_transformer_and_pipeline()

        self._app._flux_nunchaku_and_loras.create_prompt_embeds()

        return True

    def _handle__delete_prompt_embeds(self) -> bool:
        self._app._flux_nunchaku_and_loras.delete_prompt_embeds()
        return True

    def handle_exit(self) -> bool:
        print_formatted_text(HTML("\n<ansigreen>Goodbye!</ansigreen>"))
        return False

    def handle_help(self) -> bool:
        help_text = """
        Available commands:
        .exit               - Exit the application
        .help               - Show this help message
        ._create_prompt_embeds - Create prompt embeds
        ._delete_prompt_embeds - Delete prompt embeds
        """
        print_formatted_text(HTML(f"<ansicyan>{help_text}</ansicyan>"))
        return True