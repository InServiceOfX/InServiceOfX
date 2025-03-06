from dataclasses import dataclass
from typing import List, Tuple
import hashlib
from commonapi.Messages import Message

@dataclass
class ConversationHistory:
    """Stores an ordered history of messages with optional content hashing"""
    messages: List[Message] = None
    content_hashes: List[str] = None
    
    def __post_init__(self):
        self.messages = self.messages or []
        self.content_hashes = self.content_hashes or []
    
    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of message content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def append_message(self, message: Message) -> None:
        self.messages.append(message)
        self.content_hashes.append(self._hash_content(message.content))
        
    def pop_messages(self, count: int = 1, from_start: bool = False) \
        -> List[Message]:
        """Remove and return messages from either end of history"""
        if not self.messages or count <= 0:
            return []

        if from_start:
            removed_messages = self.messages[:count]
            self.messages = self.messages[count:]
            self.content_hashes = self.content_hashes[count:]
        else:
            removed_messages = self.messages[-count:]
            self.messages = self.messages[:-count]
            self.content_hashes = self.content_hashes[:-count]
            
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
    
    def clear(self) -> None:
        self.messages.clear()
        self.content_hashes.clear()
