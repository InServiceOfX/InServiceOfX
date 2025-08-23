from typing import Dict, Callable, Awaitable
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text

class CommandHandler:
    """Handles dot commands for CLITextToSpeech."""
    
    def __init__(self, app):
        """Initialize with a reference to the main application."""
        self._app = app
        # Dictionary mapping command strings to handler methods.
        # Add more commands as needed here.
        self.commands: Dict[str, Callable[[], Awaitable[bool]]] = {
            ".exit": self.handle_exit,
            ".help": self.handle_help,
            ".show_text_file_path": self.handle_show_text_file_path,
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
            # Not a recognized command, treat as regular user input
            return True, False

    def handle_exit(self) -> bool:
        print_formatted_text(HTML("\n<ansigreen>Goodbye!</ansigreen>"))
        return False
    
    def handle_help(self) -> bool:
        help_text = """
        Available commands:
        .exit               - Exit the application
        .help               - Show this help message
        .show_text_file_path - Show current text file path
        """
        print_formatted_text(HTML(f"<ansicyan>{help_text}</ansicyan>"))
        return True
    
    def handle_show_text_file_path(self) -> bool:
        message = self._app._cli_configuration.show_text_file_path()
        print_formatted_text(HTML(f"<ansicyan>{message}</ansicyan>"))
        return True