from typing import Dict, Callable, Awaitable
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text

class CommandHandler:
    """Handles dot commands for CLITextToSpeech."""
    
    def __init__(self, app):
        """Initialize with a reference to the main application."""
        self._app = app

        self._command_descriptions = {
            ".generate_with_vibe_voice": "Generate with VibeVoice",
            ".refresh_configurations": "Refresh configurations",
            ".exit": "Exit the application",
            ".help": "Show help message",
            ".show_text_file_path": "Show current text file path",
        }

        # Dictionary mapping command strings to handler methods.
        # Add more commands as needed here.
        self.commands: Dict[str, Callable[[], Awaitable[bool]]] = {
            ".generate_with_vibe_voice": self.handle_generate_with_vibe_voice,
            ".refresh_configurations": self.handle_refresh_configurations,
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
        self._app._terminal_ui.print_goodbye()
        return False
    
    def handle_help(self) -> bool:
        # Generate help text from command descriptions
        help_lines = ["Available commands:"]
        for command, description in self._command_descriptions.items():
            # Format: command (padded to 25 chars) - description
            help_lines.append(f"  {command:<25} - {description}")
        
        help_text = "\n".join(help_lines)
        self._app._terminal_ui.print_help(help_text)
        return True

    def handle_refresh_configurations(self) -> bool:
        self._app._terminal_ui.print_info("Refreshing configurations...")
        self._app._process_configurations.refresh_configurations()
        return True

    def handle_generate_with_vibe_voice(self) -> bool:
        self._app._terminal_ui.print_info("Generating with VibeVoice...")
        self._app._generate_with_vibe_voice.generate_with_vibe_voice()
        return True

    def handle_show_text_file_path(self) -> bool:
        message = \
            self._app._process_configurations.configurations[
                "cli_configuration"].show_text_file_path()
        self._app._terminal_ui.print_info(message)
        return True