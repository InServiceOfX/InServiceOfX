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

@dataclass
class ConversationMessagePair:
    """
    Why a message pair for user message and assistant message?

    https://devblogs.microsoft.com/surface-duo/android-openai-chatgpt-19/#:~:text=The%20code%20for%20steps%201,referenced%20in%20subsequent%20user%20queries

    'The query and response are concatenated together so that the embedding
    created will have the best chance of matching future user queries that might
    be related.'
    """
    conversation_pair_id: int
    content_0: str
    content_1: str
    datetime: float
    hash: str
    role_0: str
    role_1: str
    embedding: Optional[list[float]] = None

class PermanentConversation:
    def __init__(self):
        self.messages: list[ConversationMessage] = []
        self.message_pairs: list[ConversationMessagePair] = []
        self.content_hashes: list[str] = []
        self.hash_to_indices_reverse_map: dict[str, list[int]] = {}

        self._counter: int = 0
        self._message_pair_counter: int = 0

    def add_message(
            self,
            message: Message,
            embedding: Optional[list[float]] = None):
        self.messages.append(ConversationMessage(
            conversation_id=self._counter,
            content=message.content,
            datetime=time.time(),
            hash=Message._hash_content(message.content),
            role=message.role,
            embedding=embedding))
        self._counter += 1
        self.content_hashes.append(hash)
        if hash not in self.hash_to_indices_reverse_map:
            self.hash_to_indices_reverse_map[hash] = [
                len(self.content_hashes) - 1,]
        else:
            self.hash_to_indices_reverse_map[hash].append(
                len(self.content_hashes) - 1)

    def append_message_pair(
            self,
            message_0: Message,
            message_1: Message,
            embedding: Optional[list[float]] = None):
        self.message_pairs.append(ConversationMessagePair(
            conversation_pair_id=self._message_pair_counter,
            content_0=message_0.content,
            content_1=message_1.content,
            datetime=time.time(),
            hash=Message._hash_content(
                message_0.content + message_1.content),
            role_0=message_0.role,
            role_1=message_1.role,
            embedding=embedding))
        self._message_pair_counter += 1

    def add_message_as_content(
            self,
            content: str,
            role: str,
            embedding: Optional[list[float]] = None):
        self.messages.append(ConversationMessage(
            conversation_id=self._counter,
            content=content,
            datetime=time.time(),
            hash=Message._hash_content(content),
            role=role,
            embedding=embedding))
        self._counter += 1
        self.content_hashes.append(hash)
        if hash not in self.hash_to_indices_reverse_map:
            self.hash_to_indices_reverse_map[hash] = [
                len(self.content_hashes) - 1,]
        else:
            self.hash_to_indices_reverse_map[hash].append(
                len(self.content_hashes) - 1)

    def add_message_pair_as_content(
            self,
            content_0: str,
            content_1: str,
            role_0: str,
            role_1: str,
            embedding: Optional[list[float]] = None):
        self.message_pairs.append(ConversationMessagePair(
            conversation_pair_id=self._message_pair_counter,
            content_0=content_0,
            content_1=content_1,
            datetime=time.time(),
            hash=Message._hash_content(content_0 + content_1),
            role_0=role_0,
            role_1=role_1,
            embedding=embedding))
        self._message_pair_counter += 1

    def get_message_reference_by_hash(self, hash: str) \
        -> Optional[ConversationMessage]:
        if hash in self.hash_to_index_reverse_map:
            return self.messages[self.hash_to_index_reverse_map[hash]]
        return None
