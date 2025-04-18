from typing import Optional, List
from pathlib import Path

from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import confirm, checkboxlist_dialog, radiolist_dialog
from prompt_toolkit.styles import Style

from clichatlocal.Messages.SystemMessagesManager import SystemMessagesManager, SystemMessage

class SystemMessagesDialogHandler:
    """Handles UI interactions for system messages."""
    
    def __init__(self, messages_manager: SystemMessagesManager, configuration):
        """Initialize with a SystemMessagesManager and configuration."""
        self.messages_manager = messages_manager
        self.configuration = configuration
    
    def show_active_system_messages(self):
        """Display all active system messages."""
        active_messages = self.messages_manager.get_active_messages()
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
                    f"{self.configuration.system_prefix} {msg.content}"
                    f"</{self.configuration.system_color}>"))
    
    async def add_system_message_dialog(self) -> bool:
        """
        Dialog for adding a new system message.
        Returns True if a message was added, False otherwise.
        """
        # Use prompt_toolkit's prompt for input
        message_content = prompt(
            "Enter new system message:\n",
            style=self.configuration.prompt_style,
            multiline=True)  # Allow multiline input for system messages
        
        if not message_content.strip():
            return False
            
        # Show preview and confirm
        print("\nPreview of system message:")
        print("-" * 40)
        print(message_content)
        print("-" * 40)

        if confirm("Add this system message and make it active?"):
            new_message = self.messages_manager.add_message(message_content, True)
            if new_message:
                print_formatted_text(
                    HTML(
                        f"<{self.configuration.info_color}>"
                        "System message added and activated."
                        f"</{self.configuration.info_color}>"))
                # Save changes immediately
                self.messages_manager.save_messages()
                return True
            else:
                print_formatted_text(
                    HTML(
                        f"<{self.configuration.error_color}>"
                        "This message already exists."
                        f"</{self.configuration.error_color}>"))
        
        return False
    
    async def configure_system_messages_dialog(self) -> Optional[str]:
        """
        Dialog for configuring which system messages are active.
        Returns action to take with conversation or None if canceled.
        """
        messages = self.messages_manager.messages
        if not messages:
            print_formatted_text(
                HTML(
                    f"<{self.configuration.error_color}>"
                    "No system messages available"
                    f"</{self.configuration.error_color}>"))
            return None
        
        # Create values for checkbox dialog
        values = [
            (msg.hash, msg.content[:60] + "..." if len(msg.content) > 60 else msg.content)
            for msg in messages
        ]
        
        default_values = [
            msg.hash 
            for msg in self.messages_manager.get_active_messages()
        ]

        selected_hashes = await checkboxlist_dialog(
            title="System Messages",
            text="Select active system messages:",
            values=values,
            default_values=default_values,
            style=self.configuration.prompt_style
        ).run_async()
        
        if selected_hashes is None:
            return None
            
        # Update active states
        changes_made = False
        for msg in self.messages_manager.messages:
            should_be_active = msg.hash in selected_hashes
            if msg.is_active != should_be_active:
                self.messages_manager.toggle_message(msg.hash)
                changes_made = True
        
        # Save changes immediately
        if changes_made:
            self.messages_manager.save_messages()
            
            # Show options for conversation management
            return await radiolist_dialog(
                title="System Messages Updated",
                text="What would you like to do with the conversation?",
                values=[
                    (
                        "reset",
                        "Reset conversation (keep only active system messages)"
                    ),
                    (
                        "append",
                        "Append active system messages to current conversation"
                    ),
                    ("nothing", "Do nothing")],
                style=self.configuration.prompt_style
            ).run_async()
        
        return None
    
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
