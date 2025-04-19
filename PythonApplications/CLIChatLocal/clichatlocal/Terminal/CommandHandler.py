from typing import Dict, Callable, Awaitable, Any, Optional
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text
import asyncio

from clichatlocal.Messages import SystemMessagesDialogHandler

class CommandHandler:
    """Handles dot commands for CLIChatLocal."""
    
    def __init__(self, app):
        """Initialize with a reference to the main application."""
        self.app = app
        # Dictionary mapping command strings to handler methods
        self.commands: Dict[str, Callable[[], Awaitable[bool]]] = {
            ".exit": self.handle_exit,
            ".help": self.handle_help,
            ".clear": self.handle_clear,
            ".system_show_active": self.handle_show_active_system_messages,
            ".system_add": self.handle_add_system_message,
            # Add more commands as needed
        }

        self.system_dialog_handler = SystemMessagesDialogHandler(
            self.app.cli_configuration)

    async def handle_command(self, command: str) -> tuple[bool, bool]:
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
            return await self.commands[command](), True
        else:
            # Not a recognized command, treat as regular user input
            return True, False
    
    async def handle_exit(self) -> bool:
        """Handle the exit command."""
        self.app.terminal_ui.print_info("Exiting...")
        return False
    
    async def handle_help(self) -> bool:
        """Display help information."""
        help_text = """
        Available commands:
        .exit               - Exit the application
        .help               - Show this help message
        .clear              - Clear the screen
        .system_show_active - Show active system messages
        .system_add         - Add a system message
        """
        print_formatted_text(HTML(f"<{self.app.cli_configuration.info_color}>{help_text}</{self.app.cli_configuration.info_color}>"))
        return True
    
    async def handle_clear(self) -> bool:
        """Clear the screen."""
        self.app.terminal_ui.clear_screen()
        return True
    
    async def handle_show_active_system_messages(self) -> bool:

        self.system_dialog_handler.show_active_system_messages(
            self.app.llama3_engine.system_messages_manager)
        
        return True

    async def handle_add_system_message(self) -> bool:
        """Add a system message."""
        # Use the event loop's run_in_executor to run the blocking function
        loop = asyncio.get_event_loop()
        _ = await loop.run_in_executor(
            None,
            lambda: self.system_dialog_handler.add_system_message_dialog(
                self.app.terminal_ui.create_prompt_style(),
                self.app.llama3_engine)
        )
        return True