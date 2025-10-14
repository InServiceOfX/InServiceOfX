from .ConfirmationDialog import ConfirmationDialog
from typing import Dict, Callable, Awaitable, Any
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text
import asyncio
import re

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
            ".get_prompt_mode": self.handle_get_prompt_mode,
            ".toggle_prompt_mode": self.handle_toggle_prompt_mode,
            ".system_show_active": self.handle_show_active_system_messages,
            ".system_add": self.handle_add_system_message,
            ".system_configure": self.handle_configure_system_messages,
            "._show_conversation_history": \
                self._handle__show_conversation_history,
            "._show_permanent_conversation_history": \
                self._handle__show_permanent_conversation_history,
            "._show_permanent_conversation_message_chunks": \
                self._handle__show_permanent_conversation_message_chunks,
            # Add more commands as needed
        }

        # Define available commands with descriptions
        self._command_descriptions = {
            ".help": "Show available commands; show this help message",
            ".exit": "Exit the application",
            ".clear": "Clear the screen",
            ".clear_conversation_history": "Clear conversation history",
            ".get_prompt_mode": "Get prompt mode, Direct for no AI agents",
            ".toggle_prompt_mode": "Toggle prompt mode",
            ".system_show_active": "Show active system messages",
            ".system_add": "Add a system message",
            ".system_configure": \
                "Configure system messages to be used or i.e. make active",
            "._show_conversation_history": \
                (
                    "Show conversation history, with optional parameter for "
                    "number of most recent N messages"),
            "._show_permanent_conversation_history": \
                (
                    "Show permanent conversation history, with optional "
                    "parameter for number of most recent N messages"),
            "._show_permanent_conversation_message_chunks": \
                (
                    "Show permanent conversation message chunks, with optional "
                    "parameter for number of most recent N chunks"),
        }

        assert self._command_descriptions.keys() == self.commands.keys(), \
            f"Command descriptions and commands keys do not match: {self._command_descriptions.keys()} != {self.commands.keys()}"

        self._async_commands = [
            ".exit",
            "._show_permanent_conversation_message_chunks"]

        # This is intended to be used to handle parameter parsing of commands.
        # It should be a string str.
        self._last_command = None

        self._confirmation_dialog = ConfirmationDialog(
            self._app._terminal_ui.create_prompt_style(),
        )

    def _setup_confirmation_dialog(self, prompt_session_manager):
        self._confirmation_dialog.setup_with_prompt_session_manager(
            prompt_session_manager)

    def is_command_async(self, command: str) -> bool:
        command = command.strip().lower()
        return command in self._async_commands

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
        command = command.strip()
        # Store the last command for possible parameter parsing.
        self._last_command = command

        # Check for exact matches with no parameters.        
        if command.lower() in self.commands:
            return self.commands[command.lower()](), True

        command_base = command.lower().split()[0]

        if command_base in self.commands:
            return self.commands[command_base](), True

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
        self._last_command = command

        if command.lower() in self.commands:
            return await self.commands[command](), True

        command_base = command.lower().split()[0]
        if command_base in self.commands:
            return await self.commands[command_base](), True

        # Not a recognized command, treat as regular user input
        return True, False

    def _process_embed_conversation_results(self, result):
        if result == None:
            self._app._terminal_ui.print_error(
                "Embedding model had not been setup!")
            return
        message_chunks, _ = result
        if len(message_chunks) == 0:
            self._app._terminal_ui.print_info(
                "No message chunks to insert into database!")
        else:
            self._app._terminal_ui.print_info(
                f"Inserted {len(message_chunks)} message chunks into database!")

    async def handle_exit(self) -> bool:

        save_conversation = await self._confirmation_dialog.ask_confirmation(
            "Save current conversation to database?",
            default='yes'
        )
        
        if save_conversation:
            self._app._terminal_ui.print_info("💾 Saving conversation...")
            embed_results = \
                await self._app._pgsql_and_embedding.embed_conversation()
            self._process_embed_conversation_results(embed_results)
            self._app._terminal_ui.print_info("✅ Conversation saved!")
        else:
            self._app._terminal_ui.print_info("📝 Conversation not saved.")

        self._app._terminal_ui.print_info("Exiting...Goodbye!")
        return False
    
    def handle_help(self) -> bool:
        """Display help information using command descriptions."""
        # Build the help text from command descriptions
        help_lines = ["Available commands:"]
        for command, description in self._command_descriptions.items():
            help_lines.append(f"  {command:<25} - {description}")
        
        # Join all lines into a single string
        commands_text = "\n".join(help_lines)

        self._app._terminal_ui.print_help(commands_text)

        return True

    def handle_clear(self) -> bool:
        """Clear the screen."""
        self._app._terminal_ui.clear_screen()
        return True

    def handle_clear_conversation_history(self) -> bool:
        self._app._macm.clear_conversation_history()
        return True

    def handle_show_active_system_messages(self) -> bool:
        self._app._system_messages_dialog_handler.show_active_system_messages()

        self._app._terminal_ui.print_info(
            "Active system messages shown.")
        return True

    def handle_add_system_message(self) -> bool:
        self._app._system_messages_dialog_handler.add_system_message_dialog(
            self._app._terminal_ui.create_prompt_style())
        return True

    # async def handle_add_system_message_async(self) -> bool:
    #     """Add a system message."""
    #     # Use the event loop's run_in_executor to run the blocking function
    #     loop = asyncio.get_event_loop()
    #     result = await loop.run_in_executor(
    #         None,
    #         lambda: self.system_dialog_handler.add_system_message_dialog(
    #             self.app._terminal_ui.create_prompt_style(),
    #             self.app.llama3_engine)
    #     )

    #     if result:
    #         self.app._terminal_ui.print_info(
    #             "System message added and activated.")
    #         self.app.permanent_conversation_history.append_active_system_messages(
    #             self.app.llama3_engine.system_messages_manager.get_active_messages())
    #     else:
    #         self.app._terminal_ui.print_error(
    #             "System message not added.")

    #     return True

    def handle_configure_system_messages(self) -> bool:
        action = \
            self._app._system_messages_dialog_handler.configure_system_messages_dialog(
                self._app._terminal_ui.create_prompt_style())

        if action is not None:
            self._app._system_messages_dialog_handler.handle_configure_system_messages_dialog_choice(
                action)

        return True

    def _handle__show_conversation_history(self) -> bool:
        count = 10
        if self._last_command is not None:
            command_parts = self._last_command.split()
            if len(command_parts) > 1:
                try:
                    count = int(command_parts[1])
                    # Limit between 1-50
                    count = max(1, min(count, 50))
                except ValueError:
                    pass

        conversation_messages = \
            self._app._macm._csp.get_conversation_as_list_of_dicts()

        recent_messages = conversation_messages[-count:] \
            if len(conversation_messages) >= count else conversation_messages

        self._app._terminal_ui._print_conversation_history(recent_messages)

        return True

    def _handle__show_permanent_conversation_history(self) -> bool:
        count = 10
        if self._last_command is not None:
            command_parts = self._last_command.split()
            if len(command_parts) > 1:
                try:
                    count = int(command_parts[1])
                    # Limit between 1-50
                    count = max(1, min(count, 50))
                except ValueError:
                    pass

        conversation_messages = \
            self._app._macm._csp.get_permanent_conversation_messages()

        conversation_messages = [
            {"role": message.role, "content": message.content}
            for message in conversation_messages
        ]

        recent_messages = conversation_messages[-count:] \
            if len(conversation_messages) >= count else conversation_messages

        self._app._terminal_ui._print_conversation_history(recent_messages)

        return True

    async def _handle__show_permanent_conversation_message_chunks(self) -> bool:
        count = 10
        if self._last_command is not None:
            command_parts = self._last_command.split()
            if len(command_parts) > 1:
                try:
                    count = int(command_parts[1])
                    # Limit between 1-50
                    count = max(1, min(count, 50))
                except ValueError:
                    pass

        message_chunks = \
            await self._app._pgsql_and_embedding.get_latest_message_chunks(count)

        message_chunks = [
            {
                "role": message.role,
                "content": re.sub(r'\s+', ' ', message.content[:140]).strip()
            }
            for message in message_chunks
        ]

        self._app._terminal_ui._print_conversation_history(message_chunks)

        return True

    def handle_get_prompt_mode(self) -> bool:

        if self._app._pgsql_and_embedding is None:
            self._app._terminal_ui.print_info("Direct mode (no AI agents)")
            return True

        prompt_mode = self._app._pgsql_and_embedding.get_prompt_mode()
        self._app._terminal_ui.print_info(f"Prompt mode: {prompt_mode}")
        return True

    def handle_toggle_prompt_mode(self) -> bool:
        if self._app._pgsql_and_embedding is None:
            self._app._terminal_ui.print_info("Direct mode (no AI agents)")
            return True

        self._app._pgsql_and_embedding._toggle_prompt_mode()
        prompt_mode = self._app._pgsql_and_embedding.get_prompt_mode()
        self._app._terminal_ui.print_info(f"Prompt mode: {prompt_mode}")
        return True