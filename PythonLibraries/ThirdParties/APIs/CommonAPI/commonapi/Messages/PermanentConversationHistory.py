from dataclasses import dataclass
from typing import List, Dict
from commonapi.Messages.RecordedMessages import (
    RecordedMessage,
    RecordedUserMessage,
    RecordedAssistantMessage)

@dataclass
class PermanentConversationHistory:
    """Stores an ordered history of messages with optional content hashing with
    no deletion of messages."""
    recorded_messages: List[RecordedMessage] = None
    content_hashes: List[str] = None
    hash_to_index_reverse_map: Dict[str, int] = None

    def __post_init__(self):
        self.recorded_messages = self.recorded_messages or []
        self.content_hashes = self.content_hashes or []
        self.hash_to_index_reverse_map = self.hash_to_index_reverse_map or {}
    
    def append_recorded_message(self, message: RecordedMessage) -> None:
        self.recorded_messages.append(message)
        try:
            self.content_hashes.append(message.hash)
            self.hash_to_index_reverse_map[message.hash] = \
                len(self.content_hashes) - 1
        except AttributeError as err:
            print(f"Error hashing message content: {err}")
            print("type(message): ", type(message))
            print("message: ", message)
    
    def is_message_in_history_by_hash(self, hash: str) -> bool:
        return hash in self.hash_to_index_reverse_map

    def append_user_content(self, content: str) -> None:
        recorded_user_message = RecordedUserMessage.create_from_content(content)
        self.append_recorded_message(recorded_user_message)

    def append_assistant_content(self, content: str) -> None:
        recorded_assistant_message = \
            RecordedAssistantMessage.create_from_content(content)
        self.append_recorded_message(recorded_assistant_message)

    def append_active_system_messages(self, list_of_system_messages) -> None:
        for system_message in list_of_system_messages:
            if system_message.is_active:
                message = RecordedMessage(
                    system_message.content,
                    system_message.timestamp,
                    system_message.hash,
                    "system")
                self.append_recorded_message(message)
