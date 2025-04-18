from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

from clichatlocal.FileIO import JSONFile
@dataclass
class SystemMessage:
    """Represents a system message with metadata."""
    content: str
    timestamp: float
    hash: str
    is_active: bool = False

    @staticmethod
    def create_hash(content: str) -> str:
        """Create a hash from the content."""
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
        """Check if two messages are equal based on their hash."""
        return self.hash == other.hash


class SystemMessagesManager:
    """Manages system messages, including file IO."""
    
    def __init__(self, system_messages_file_path = None):
        """
        Typically, you'll find the system_messages_file_path defined as a field
        in an ApplicationPaths object.
        """
        self.system_messages_file_path = system_messages_file_path
        
        # Initialize with default message
        default_message = SystemMessage.create_default_message()
        self._messages_dict: Dict[str, SystemMessage] = {
            default_message.hash: default_message
        }

    def load_messages(self) -> bool:
        """Load system messages from file."""
        data = JSONFile.load_json(self.system_messages_file_path)
        if not data:
            return False
        
        try:
            messages = [SystemMessage(**msg) for msg in data]
            self._messages_dict = {msg.hash: msg for msg in messages}
            return True
        except (KeyError, TypeError):
            # Invalid data format
            return False
    
    def save_messages(self) -> bool:
        """Save system messages to file."""
        messages_data = [msg.__dict__ for msg in self.messages]
        return JSONFile.save_json(self.system_messages_file_path, messages_data)
    
    def add_message(self, content: str, is_active: bool = False) \
        -> Optional[SystemMessage]:
        message = SystemMessage.create(content, is_active)
        if message.hash not in self._messages_dict:
            self._messages_dict[message.hash] = message
            return message
        return None
    
    def remove_message(self, hash_value: str) -> bool:
        """Remove a system message by hash."""
        if hash_value in self._messages_dict:
            del self._messages_dict[hash_value]
            return True
        return False
    
    def toggle_message(self, hash_value: str) -> bool:
        """Toggle a message's active status."""
        if hash_value in self._messages_dict:
            self._messages_dict[hash_value].is_active = not self._messages_dict[hash_value].is_active
            self.save_messages()  # Auto-save when toggling a message
            return True
        return False
    
    @property
    def messages(self) -> List[SystemMessage]:
        """Get all system messages."""
        return list(self._messages_dict.values())
    
    def get_active_messages(self) -> List[SystemMessage]:
        """Get all active system messages."""
        return [msg for msg in self.messages if msg.is_active]
    
    def get_message_by_hash(self, hash_value: str) -> Optional[SystemMessage]:
        """Get a message by its hash."""
        return self._messages_dict.get(hash_value)
