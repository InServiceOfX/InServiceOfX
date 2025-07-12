from commonapi.Messages.Messages import Message
from dataclasses import dataclass
from typing import Any, Optional

from .ConversationHistory import ConversationHistory

import time

@dataclass
class ConversationMessage:
    conversation_id: int
    content: str
    datetime: float
    hash: str
    role: str
    embedding: Optional[list[float]] = None

class PermanentConversation:
    messages: list[ConversationMessage] = None
    content_hashes: list[str] = None
    hash_to_index_reverse_map: dict[str, int] = None

    _counter = 0

    def __post_init__(self):
        self.messages = self.messages or []
        self.content_hashes = self.content_hashes or []
        self.hash_to_index_reverse_map = self.hash_to_index_reverse_map or {}

    def append_message(
            self,
            message: Message,
            hash: str,
            embedding: Optional[list[float]] = None):
        self.messages.append(ConversationMessage(
            conversation_id=self._counter,
            content=message.content,
            datetime=time.time(),
            hash=hash,
            role=message.role,
            embedding=embedding))
        self._counter += 1
        self.content_hashes.append(hash)
        self.hash_to_index_reverse_map[hash] = len(self.content_hashes) - 1

    def add_system_message(
            self,
            content: str,
            hash: str,
            embedding: Optional[list[float]] = None):
        self.messages.append(ConversationMessage(
            conversation_id=self._counter,
            content=content,
            datetime=time.time(),
            hash=hash,
            role="system",
            embedding=embedding))
        self.content_hashes.append(hash)
        self.hash_to_index_reverse_map[hash] = len(self.content_hashes) - 1

    def get_message_reference_by_hash(self, hash: str) \
        -> Optional[ConversationMessage]:
        if hash in self.hash_to_index_reverse_map:
            return self.messages[self.hash_to_index_reverse_map[hash]]
        return None
