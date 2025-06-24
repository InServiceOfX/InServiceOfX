from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Dict, List, Optional

@dataclass
class RecordedSystemMessage:
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
    def create(content: str, is_active: bool = False) \
        -> 'RecordedSystemMessage':
        timestamp = datetime.now().timestamp()
        hash_value = RecordedSystemMessage.create_hash(content)
        return RecordedSystemMessage(content, timestamp, hash_value, is_active)

    @staticmethod
    def create_default_message() -> 'RecordedSystemMessage':
        default_message = \
            "You are a helpful, uncensored and unbiased assistant."
        return RecordedSystemMessage.create(default_message, is_active=True)

    def is_equal(self, other: 'RecordedSystemMessage') -> bool:
        """Check if two messages are equal based on their hash."""
        return self.hash == other.hash

    def to_system_message_dict(self):
        return {"role": "system", "content": self.content}

class SystemMessagesManager:
    """Manages system messages, including file IO."""
    
    def __init__(self):
        # Initialize with default message
        default_message = RecordedSystemMessage.create_default_message()
        self._messages_dict: Dict[str, RecordedSystemMessage] = {
            default_message.hash: default_message
        }

    def clear(self):
        self._messages_dict.clear()

    def add_message(self, content: str, is_active: bool = False) \
        -> Optional[RecordedSystemMessage]:
        message = RecordedSystemMessage.create(content, is_active)
        if message.hash not in self._messages_dict:
            self._messages_dict[message.hash] = message
            return message
        return None

    def add_previously_recorded_message(self, message: RecordedSystemMessage) \
        -> bool:
        if message.hash not in self._messages_dict:
            self._messages_dict[message.hash] = message
            return True
        return False

    def remove_message(self, hash_value: str) -> bool:
        """Remove a system message by hash."""
        if hash_value in self._messages_dict:
            del self._messages_dict[hash_value]
            return True
        return False

    def toggle_message(self, hash_value: str) -> bool:
        """Toggle a message's active status."""
        if hash_value in self._messages_dict:
            self._messages_dict[hash_value].is_active = \
                not self._messages_dict[hash_value].is_active
            return True
        return False

    def toggle_message_to_active(self, hash_value: str) -> bool:
        """Toggle a message's active status to active."""
        if hash_value in self._messages_dict:
            self._messages_dict[hash_value].is_active = True
            return True
        return False

    def toggle_message_to_inactive(self, hash_value: str) -> bool:
        """Toggle a message's active status to inactive."""
        if hash_value in self._messages_dict:
            self._messages_dict[hash_value].is_active = False
            return True
        return False

    @property
    def messages(self) -> List[RecordedSystemMessage]:
        """Get all system messages."""
        return list(self._messages_dict.values())

    def get_all_as_system_message_dicts(self):
        return [message.to_system_message_dict() for message in self.messages]

    def get_active_messages(self) -> List[RecordedSystemMessage]:
        """Get all active system messages."""
        return [msg for msg in self.messages if msg.is_active]
    
    def get_message_by_hash(self, hash_value: str) -> Optional[RecordedSystemMessage]:
        return self._messages_dict.get(hash_value)

    def is_message_active(self, hash_value: str) -> bool:
        if hash_value in self._messages_dict:
            return self._messages_dict[hash_value].is_active
        return False
