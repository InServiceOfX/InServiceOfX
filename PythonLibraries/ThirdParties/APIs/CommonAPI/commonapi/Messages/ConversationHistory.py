from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import hashlib
from commonapi.Messages.Messages import Message, SystemMessage

@dataclass
class ConversationHistory:
    """Stores an ordered history of messages with optional content hashing"""
    messages: List[Message] = None
    content_hashes: List[str] = None
    hash_to_index_reverse_map: Dict[str, int] = None

    def __post_init__(self):
        self.messages = self.messages or []
        self.content_hashes = self.content_hashes or []
        self.hash_to_index_reverse_map = self.hash_to_index_reverse_map or {}

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate SHA256 hash of message content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def append_message(self, message: Message) -> None:
        self.messages.append(message)
        try:
            self.content_hashes.append(self._hash_content(message.content))
            self.hash_to_index_reverse_map[self.content_hashes[-1]] = \
                len(self.content_hashes) - 1
        except AttributeError as err:
            print(f"Error hashing message content: {err}")
            print("type(message): ", type(message))
            print("message: ", message)

    def delete_message_by_hash(self, hash: str) -> None:
        if hash in self.hash_to_index_reverse_map:
            index = self.hash_to_index_reverse_map[hash]
            del self.hash_to_index_reverse_map[hash]
            self.messages.pop(index)
            self.content_hashes.pop(index)
        else:
            print(f"Hash {hash} not found in hash_to_index_reverse_map")

    def pop_messages(self, count: int = 1, from_start: bool = False) \
        -> List[Message]:
        """Remove and return messages from either end of history"""
        if not self.messages or count <= 0:
            return []

        if from_start:
            removed_messages = self.messages[:count]
            self.messages = self.messages[count:]
            self.content_hashes = self.content_hashes[count:]
            for hash in removed_messages:
                del self.hash_to_index_reverse_map[hash]
        else:
            removed_messages = self.messages[-count:]
            self.messages = self.messages[:-count]
            self.content_hashes = self.content_hashes[:-count]
            for hash in removed_messages:
                del self.hash_to_index_reverse_map[hash]
            
        return removed_messages
    
    def get_conversation_text(self) -> Tuple[str, int, int]:
        """
        Get conversation as formatted string with stats
        Returns:
            Tuple of (conversation_text, char_count, word_count)
        """
        if not self.messages:
            return "", 0, 0
            
        conversation_parts = []
        for message in self.messages:
            conversation_parts.append(message.to_string(message))
            
        conversation_text = "\n".join(conversation_parts)
        char_count = len(conversation_text)
        word_count = len(conversation_text.split())
        
        return conversation_text, char_count, word_count

    def as_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [message.to_dict() for message in self.messages]

    def clear(self) -> None:
        self.messages.clear()
        self.content_hashes.clear()
        self.hash_to_index_reverse_map.clear()

    def get_all_system_messages(self) -> List[SystemMessage]:
        return [message for message in self.messages \
            if isinstance(message, SystemMessage)]

    def is_message_in_conversation_history_by_hash(self, hash: str) -> bool:
        return hash in self.hash_to_index_reverse_map
