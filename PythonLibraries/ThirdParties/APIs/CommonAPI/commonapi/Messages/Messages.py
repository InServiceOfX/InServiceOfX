from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any, Tuple

@dataclass
class Message:
    """Base message class for LLM interactions"""
    content: str
    role: str

    @staticmethod
    def to_string_and_counts(message: 'Message', prefix: Optional[str] = None) \
        -> Tuple[str, int, int]:
        """Convert message to string with optional prefix override and return
        character and word counts of the content"""
        role = prefix if prefix else message.role.capitalize()
        formatted = f"{role}: {message.content}"
        char_count = len(message.content)
        word_count = len(message.content.split())
        return formatted, char_count, word_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API requests"""
        return asdict(self)

@dataclass
class SystemMessage(Message):
    """System message for setting AI behavior"""
    content: str
    role: Literal["system"] = "system"
    
    @staticmethod
    def to_string_and_counts(message: 'SystemMessage') -> Tuple[str, int, int]:
        return Message.to_string_and_counts(message, "System")

@dataclass
class UserMessage(Message):
    """User message for queries/prompts"""
    content: str
    role: Literal["user"] = "user"
    
    @staticmethod
    def to_string_and_counts(message: 'UserMessage') -> Tuple[str, int, int]:
        return Message.to_string_and_counts(message, "Human")

@dataclass
class AssistantMessage(Message):
    """Assistant message for AI responses"""
    content: str
    role: Literal["assistant"] = "assistant"
    
    @staticmethod
    def to_string_and_counts(message: 'AssistantMessage') -> \
        Tuple[str, int, int]:
        return Message.to_string_and_counts(message, "AI")

@dataclass
class ToolMessage(Message):
    """Tool message for function calls/responses"""
    content: str
    role: Literal["tool"] = "tool"
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    
    @staticmethod
    def to_string_and_counts(message: 'ToolMessage') -> Tuple[str, int, int]:
        base = Message.to_string_and_counts(message, "Tool")[0]
        if message.name:
            base += f" [{message.name}]"
        if message.tool_call_id:
            base += f" (call_id: {message.tool_call_id})"
        char_count = len(base)
        word_count = len(base.split())
        return base, char_count, word_count

@dataclass
class DeveloperMessage(Message):
    """Developer message for API debugging/testing"""
    role: Literal["developer"] = "developer"

def create_system_message(content: str) -> Dict[str, str]:
    """Create a system message dictionary"""
    return SystemMessage(content=content).to_dict()

def create_user_message(content: str) -> Dict[str, str]:
    """Create a user message dictionary"""
    return UserMessage(content=content).to_dict()

def create_assistant_message(content: str) -> Dict[str, str]:
    """Create an assistant message dictionary"""
    return AssistantMessage(content=content).to_dict()

def create_tool_message(
        content: str,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a tool message dictionary"""
    return ToolMessage(
        content=content,
        name=name,
        tool_call_id=tool_call_id).to_dict()

def create_developer_message(content: str) -> Dict[str, str]:
    """Create a developer message dictionary"""
    return DeveloperMessage(content=content).to_dict()
