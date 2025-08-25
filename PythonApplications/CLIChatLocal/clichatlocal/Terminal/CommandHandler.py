from typing import Dict, Callable, Awaitable, Any
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text
import asyncio

class CommandHandler:
    """Handles dot commands for CLIChatLocal."""
    
    def __init__(self, app):
        """Initialize with a reference to the main application."""
        self._app = app

        # Dictionary mapping command strings to handler methods
        self.commands: Dict[str, Callable[[], Awaitable[bool]]] = {
            ".exit": self.handle_exit,
            ".help": self.handle_help,
            ".clear": self.handle_clear,
            ".clear_conversation_history": \
                self.handle_clear_conversation_history,
            ".system_show_active": self.handle_show_active_system_messages,
            ".system_add": self.handle_add_system_message,
            ".system_configure": self.handle_configure_system_messages,
            # Add more commands as needed
        }

        # Define available commands with descriptions
        self._command_descriptions = {
            ".help": "Show available commands; show this help message",
            ".exit": "Exit the application",
            ".clear": "Clear the screen",
            ".clear_conversation_history": "Clear conversation history",
            ".system_show_active": "Show active system messages",
            ".system_add": "Add a system message",
            ".system_configure": \
                "Configure system messages to be used or i.e. make active",
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

    async def handle_command_async(self, command: str) -> tuple[bool, bool]:
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

    def handle_exit(self) -> bool:

        self._app.terminal_ui.print_info("Exiting...Goodbye!")
        return False
    
    def handle_help(self) -> bool:
        """Display help information using command descriptions."""
        # Build the help text from command descriptions
        help_lines = []
        for command, description in self._command_descriptions.items():
            help_lines.append(f"{command:<25} - {description}")
        
        # Join all lines into a single string
        commands_text = "\n".join(help_lines)
        
        # Create the complete help message
        help_text = f"""
Available commands:
{commands_text}
"""
        
        print_formatted_text(
            HTML(
                f"<{self._app.cli_configuration.info_color}>{help_text}</{self._app.cli_configuration.info_color}>"))
        return True
    
    async def handle_clear(self) -> bool:
        """Clear the screen."""
        self._app.terminal_ui.clear_screen()
        return True

    async def handle_clear_conversation_history(self) -> bool:
        self._app._macm.clear_conversation_history()
        return True

    def handle_show_active_system_messages(self) -> bool:
        self._app._system_messages_dialog_handler.show_active_system_messages()

        self._app.terminal_ui.print_info(
            "Active system messages shown.")
        return True

    def handle_add_system_message(self) -> bool:
        self._app._system_messages_dialog_handler.add_system_message_dialog(
            self._app.terminal_ui.create_prompt_style())
        return True

    # async def handle_add_system_message_async(self) -> bool:
    #     """Add a system message."""
    #     # Use the event loop's run_in_executor to run the blocking function
    #     loop = asyncio.get_event_loop()
    #     result = await loop.run_in_executor(
    #         None,
    #         lambda: self.system_dialog_handler.add_system_message_dialog(
    #             self.app.terminal_ui.create_prompt_style(),
    #             self.app.llama3_engine)
    #     )

    #     if result:
    #         self.app.terminal_ui.print_info(
    #             "System message added and activated.")
    #         self.app.permanent_conversation_history.append_active_system_messages(
    #             self.app.llama3_engine.system_messages_manager.get_active_messages())
    #     else:
    #         self.app.terminal_ui.print_error(
    #             "System message not added.")

    #     return True

    def handle_configure_system_messages(self) -> bool:
        action = \
            self._app._system_messages_dialog_handler.configure_system_messages_dialog(
                self._app.terminal_ui.create_prompt_style())

        if action is not None:
            self._app._system_messages_dialog_handler.handle_configure_system_messages_dialog_choice(
                action)

        return True
