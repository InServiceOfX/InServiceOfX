"""
\brief Messages reformatted with more metadata for storage.
"""
from commonapi.Messages.Messages import Message, UserMessage, AssistantMessage
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class RecordedMessage:
    content: str
    timestamp: float
    hash: str
    role: str

    def is_equal(self, other: 'RecordedMessage') -> bool:
        """Check if two messages are equal based on their hash."""
        return self.hash == other.hash

@dataclass
class RecordedUserMessage(RecordedMessage):
    content: str
    timestamp: float
    hash: str
    role: Literal["user"] = "user"

    @staticmethod
    def create_from_content(content: str) -> 'RecordedUserMessage':
        timestamp = datetime.now().timestamp()
        hash_value = Message._hash_content(content)
        return RecordedUserMessage(content, timestamp, hash_value, "user")

    @staticmethod
    def create_from_message(message: UserMessage) -> 'RecordedUserMessage':
        timestamp = datetime.now().timestamp()
        hash_value = Message._hash_content(message.content)
        return RecordedUserMessage(
            message.content,
            timestamp,
            hash_value,
            message.role)

    def to_message(self) -> UserMessage:
        return UserMessage(self.content, self.role)

@dataclass
class RecordedAssistantMessage(RecordedMessage):
    content: str
    timestamp: float
    hash: str
    role: Literal["assistant"] = "assistant"

    @staticmethod
    def create_from_content(content: str) -> 'RecordedAssistantMessage':
        timestamp = datetime.now().timestamp()
        hash_value = Message._hash_content(content)
        return RecordedAssistantMessage(
            content,
            timestamp,
            hash_value,
            "assistant")

    @staticmethod
    def create_from_message(message: AssistantMessage) \
        -> 'RecordedAssistantMessage':
        timestamp = datetime.now().timestamp()
        hash_value = Message._hash_content(message.content)
        return RecordedAssistantMessage(
            message.content,
            timestamp,
            hash_value,
            message.role)

    def to_message(self) -> AssistantMessage:
        return AssistantMessage(self.content, self.role)
