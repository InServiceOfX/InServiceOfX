from typing import Optional, List
from pathlib import Path

from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import (
    confirm,
    checkboxlist_dialog,
    radiolist_dialog,
    yes_no_dialog)

class SystemMessagesDialogHandler:
    """Handles UI interactions for system messages."""
    
    def __init__(self, app, configuration):
        self._app = app
        self.configuration = configuration
    
    def show_active_system_messages(self):
        active_messages = self._app._macm._csp.casm.get_active_system_messages()
        if not active_messages:
            print_formatted_text(
                HTML(
                    f"<{self.configuration.info_color}>"
                    "No active system messages"
                    f"</{self.configuration.info_color}>"))
            return

        print_formatted_text(
            HTML(
                f"<{self.configuration.system_color}>"
                "Active system messages:\n"
                f"</{self.configuration.system_color}>"))
        
        for msg in active_messages:
            print_formatted_text(
                HTML(
                    f"<{self.configuration.system_color}>"
                    f"{msg.content}"
                    f"</{self.configuration.system_color}>"))
    
    def add_system_message_dialog(self, prompt_style) -> bool:
        """
        Dialog for adding a new system message.
        Returns True if a message was added, False otherwise.
        """
        # Use prompt_toolkit's prompt for input
        message_content = prompt(
            "Enter new system message:\n",
            style=prompt_style,)
            # This allow multiline input for system messages. We do not choose
            # to do so because we then have to depend on key binding for entry;
            # in testing, Alt-Enter entered the multiple lines.
            #multiline=True)
        
        if not message_content.strip():
            return False
            
        # Show preview and confirm
        print("\nPreview of system message:")
        print("-" * 40)
        print(message_content)
        print("-" * 40)

        if confirm("Add this system message and make it active?"):
            new_message = self._app._macm._csp.add_system_message(
                message_content)
            if new_message:
                print_formatted_text(
                    HTML(
                        f"<{self.configuration.info_color}>"
                        "System message added and activated."
                        f"</{self.configuration.info_color}>"))
                return True
            else:
                print_formatted_text(
                    HTML(
                        f"<{self.configuration.error_color}>"
                        f"This message was not added, {message_content}."
                        f"</{self.configuration.error_color}>"))
        
        return False
    
    def configure_system_messages_dialog(self, dialog_style):
        """
        Dialog for configuring which system messages are active.
        Returns action to take with conversation or None if canceled.

        Programming note: For async, add async to message declaration
        immediately above, i.e.
        async def configure_system_messages_dialog_async(..)
        and change .run() to .run_async().
        """
        messages = self._app._macm._csp.casm.get_all_system_messages()
        if not messages:
            print_formatted_text(
                HTML(
                    f"<{self.configuration.error_color}>"
                    "No system messages available"
                    f"</{self.configuration.error_color}>"))
            return None
        
        # Create values for checkbox dialog
        values = [(
            msg.hash,
            msg.content[:60] + "..." if len(msg.content) > 60 else msg.content)
            for msg in messages]

        default_values = [
            msg.hash 
            for msg in self._app._macm._csp.casm.get_active_system_messages()]

        selected_hashes = checkboxlist_dialog(
            title="System Messages",
            text="Select active system messages:",
            values=values,
            default_values=default_values,
            style=dialog_style
        ).run()
        
        if selected_hashes is None:
            return None
            
        # Update active states
        changes_made = False
        for msg in messages:
            should_be_active = msg.hash in selected_hashes
            if msg.is_active != should_be_active:
                self._app._macm._csp.casm.toggle_system_message(msg.hash)
                changes_made = True
        
        if changes_made:
            
            # Show options for conversation management
            return radiolist_dialog(
                title="System Messages Updated",
                text="What would you like to do with the conversation?",
                values=[
                    (
                        "reset",
                        "Reset conversation (keep only active system messages)"
                    ),
                    (
                        "append",
                        (
                            "Append active system messages to current "
                            "conversation, while removing any non-active "
                            "system messages.")
                    ),
                    ("nothing", "Do nothing")],
                style=dialog_style
            ).run()
        
        return None

    def handle_configure_system_messages_dialog_choice(self, choice: str):
        if choice == "reset":
            self._app._macm._csp.clear_conversation_history()
        elif choice == "append":
            self._app._macm._csp.casm.add_only_active_system_messages_to_conversation_history()
        elif choice == "nothing":
            pass

    async def delete_system_message_dialog(self) -> bool:
        """
        Dialog for deleting system messages.
        Returns True if a message was deleted, False otherwise.
        """
        messages = self.messages_manager.messages
        if not messages:
            print_formatted_text(
                HTML(
                    f"<{self.configuration.error_color}>"
                    "No system messages available to delete"
                    f"</{self.configuration.error_color}>"))
            return False
        
        # Create values for radiolist dialog
        values = [
            (msg.hash, f"{'[Active] ' if msg.is_active else ''}{msg.content[:60]}..." 
             if len(msg.content) > 60 else msg.content)
            for msg in messages
        ]
        
        # Add cancel option
        values.append(("cancel", "Cancel - Don't delete any message"))
        
        selected_hash = await radiolist_dialog(
            title="Delete System Message",
            text="Select a system message to delete:",
            values=values,
            style=self.configuration.prompt_style
        ).run_async()
        
        if selected_hash is None or selected_hash == "cancel":
            return False
            
        # Confirm deletion
        message = self.messages_manager.get_message_by_hash(selected_hash)
        if message:
            print("\nMessage to delete:")
            print("-" * 40)
            print(message.content)
            print("-" * 40)
            
            if confirm("Are you sure you want to delete this message?"):
                success = self.messages_manager.remove_message(selected_hash)
                if success:
                    print_formatted_text(
                        HTML(
                            f"<{self.configuration.info_color}>"
                            "System message deleted."
                            f"</{self.configuration.info_color}>"))
                    # Save changes immediately
                    self.messages_manager.save_messages()
                    return True
        
        return False

    async def handle_exit(self, llm_engine, system_messages_file_io):
        # Don't offer to save if no messages
        if not llm_engine.system_messages_manager.messages:  
            return

        print(
            (f"System messages file configured to be here: "
             f"{system_messages_file_io.file_path}"))

        if system_messages_file_io.is_file_path_valid():
            if await yes_no_dialog(
                title="Save System Messages",
                text=(
                    f"Would you like to save current system messages in the "
                    f"{system_messages_file_io.file_path} file?")
            ).run_async():
                try:
                    system_messages_file_io.save_messages(
                        llm_engine.system_messages_manager.messages)
                except json.JSONDecodeError:
                    pass
            return
        else:
            if await yes_no_dialog(
                title="Save System Messages",
                text=(
                    f"System messages path "
                    f"{system_messages_file_io.file_path} doesn't exist. "
                    f"Would you like to save current system messages in there?"
            )).run_async():
                try:
                    with open(system_messages_file_io.file_path, "w") as f:
                        pass
                    system_messages_file_io.save_messages(
                        llm_engine.system_messages_manager.messages)
                except json.JSONDecodeError:
                    pass
            return
