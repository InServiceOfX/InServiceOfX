from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any

@dataclass
class Message:
    """Base message class for LLM interactions"""
    content: str
    role: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API requests"""
        return asdict(self)

@dataclass
class SystemMessage(Message):
    """System message for setting AI behavior"""
    content: str
    role: Literal["system"] = "system"

@dataclass
class UserMessage(Message):
    """User message for queries/prompts"""
    content: str
    role: Literal["user"] = "user"

@dataclass
class AssistantMessage(Message):
    """Assistant message for AI responses"""
    content: str
    role: Literal["assistant"] = "assistant"

@dataclass
class ToolMessage(Message):
    """Tool message for function calls/responses"""
    content: str
    role: Literal["tool"] = "tool"
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

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
