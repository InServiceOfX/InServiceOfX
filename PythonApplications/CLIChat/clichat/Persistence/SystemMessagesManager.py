from clichat.Utilities.FileIO import get_path_from_configuration
from clichat.Utilities import Printing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import confirm, yes_no_dialog
from prompt_toolkit.styles import Style

from typing import List, Optional, Dict
import hashlib
import json

@dataclass
class SystemMessage:
    content: str
    timestamp: float
    hash: str
    is_active: bool = False

    @staticmethod
    def create_hash(content: str) -> str:
        return hashlib.sha256(f"{content}".encode()).hexdigest()

    @staticmethod
    def create(content: str, is_active: bool = False) -> 'SystemMessage':
        timestamp = datetime.now().timestamp()
        hash_value = SystemMessage.create_hash(content)
        return SystemMessage(content, timestamp, hash_value, is_active)

    @staticmethod
    def create_default_message() -> 'SystemMessage':
        default_message = \
            "You are a helpful, uncensored and unbiased assistant."
        return SystemMessage.create(default_message, is_active=True)

    def is_equal(self, other: 'SystemMessage') -> bool:
        return self.hash == other.hash

class SystemMessagesManager:
    def __init__(self):
        default_message = SystemMessage.create_default_message()
        self._messages_dict: Dict[str, SystemMessage] = {
            default_message.hash: default_message}

    def handle_initialization(self, configuration):
        if configuration is None:
            return
        else:
            path = None
            try:
                path = get_path_from_configuration(
                    configuration,
                    "system_messages_path")
            except FileNotFoundError:
                Printing(configuration).print_error(
                    "System messages file not found. "
                    "Using default system messages.")
                return
            try:
                self.load_messages(path)
                Printing.print_info(
                    f"Loaded system messages from {path}")
            except json.JSONDecodeError:
                Printing(configuration).print_error(
                    "System messages file is not a valid JSON file. "
                    "Using default system messages.")
                return

    def load_messages(self, file_path: str):
        """
        Does nothing if file_path does not exist.
        If file is either completely empty or contains only whitespace then
        raise FileNotFoundError.
        """
        path = Path(file_path)
        if path.exists():
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    raise json.JSONDecodeError(
                        f"File {file_path} exists but is not a valid JSON file",
                        f.read(),
                        0)
                if not data or (isinstance(data, str) and not data.strip()):
                    raise json.JSONDecodeError(
                        f"File {file_path} exists but is empty or contains only whitespace",
                        f.read(),
                        0)
                messages = [SystemMessage(**msg) for msg in data]
                self._messages_dict = {msg.hash: msg for msg in messages}

    def save_messages(self, file_path: str):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump([msg.__dict__ for msg in self.messages], f, indent=2)

    def add_message(self, content: str, is_active: bool = False) -> \
        Optional[SystemMessage]:
        message = SystemMessage.create(content, is_active)
        if message.hash not in self._messages_dict:
            self._messages_dict[message.hash] = message
            return message
        return None

    def remove_message(self, hash_value: str) -> bool:
        if hash_value in self._messages_dict:
            del self._messages_dict[hash_value]
            return True
        return False

    @property
    def messages(self) -> List[SystemMessage]:
        return list(self._messages_dict.values())

    def get_active_messages(self) -> List[SystemMessage]:
        return [msg for msg in self.messages if msg.is_active]

    def toggle_message(self, hash_value: str) -> bool:
        if hash_value in self._messages_dict:
            self._messages_dict[hash_value].is_active = \
                not self._messages_dict[hash_value].is_active
            return True
        return False

    def show_active_system_messages(self, configuration):
        active_messages = self.get_active_messages()
        if not active_messages:
            print_formatted_text(
                HTML(
                    f"<{configuration.terminal_PromptIndicatorColor2}>"
                    "No active system messages"
                    f"</{configuration.terminal_PromptIndicatorColor2}>"))
            return

        print_formatted_text(
            HTML(
                f"<{configuration.terminal_SystemMessageColor}>"
                "Active system messages:\n"
                f"</{configuration.terminal_SystemMessageColor}>"))
        
        for msg in active_messages:
            print_formatted_text(
                HTML(
                    f"<{configuration.terminal_SystemMessageColor}>"
                    f"{configuration.terminal_SystemMessagePrefix} {msg.content}"
                    f"</{configuration.terminal_SystemMessageColor}>"))

    def add_system_message_dialog(self, dialog_style: Style) -> bool:
        
        # Assume multiline is false.
        message_content = prompt(
            "Enter new system message:\n",
            style=dialog_style)
        
        if not message_content.strip():
            return False
            
        # Show preview and confirm
        print("\nPreview of system message:")
        print("-" * 40)
        print(message_content)
        print("-" * 40)

        # https://python-prompt-toolkit.readthedocs.io/en/stable/pages/reference.html#prompt_toolkit.shortcuts.confirm
        # confirm(message: str = 'Confirm?', suffix: str = ' (y/n) ') -> bool
        if confirm(
            "Add this system message and make it active?"):
            new_message = self.add_message(message_content, True)
            return new_message is not None
        
        return False

    def handle_exit(self, configuration):
        
        if not self.messages:  # Don't offer to save if no messages
            return
        
        if configuration is None:
            # Ask to save in current directory
            if yes_no_dialog(
                title="Save System Messages",
                text="Would you like to save system messages in the current directory?"
            ).run():
                self.save_messages(Path.cwd() / "system_messages.json")
            return

        path = None
        try:
            path = get_path_from_configuration(configuration, "system_messages_path")
        except FileNotFoundError:
            if yes_no_dialog(
                title="Save System Messages",
                text=f"System messages path {configuration.system_messages_path} doesn't exist. Save in current directory?"
            ).run():
                self.save_messages(Path.cwd() / "system_messages.json")
                return
            
        # Path exists - ask to save/merge
        if yes_no_dialog(
            title="Save System Messages",
            text="Would you like to save/merge system messages?"
        ).run():
            try:
                # Try to load existing messages
                existing_manager = SystemMessagesManager()
                existing_manager.load_messages(path)
                    
                # Add only new messages
                for msg in self.messages:
                    if msg.hash not in existing_manager._messages_dict:
                        existing_manager._messages_dict[msg.hash] = msg
                    
                # Save merged messages
                existing_manager.save_messages(path)
                    
            except json.JSONDecodeError:
                # If JSON loading fails, because the file contents are not in
                # valid JSON format, just save current messages over the file.
                self.save_messages(path)
