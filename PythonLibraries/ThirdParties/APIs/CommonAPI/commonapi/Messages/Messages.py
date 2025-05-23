from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any, Tuple, Union, List
import hashlib

@dataclass
class Message:
    """Base message class for LLM interactions"""
    content: Union[str, List[str]]
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

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate SHA256 hash of message content"""
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class SystemMessage(Message):
    """System message for setting AI behavior"""
    content: Union[str, List[str]]
    role: Literal["system"] = "system"
    
    @staticmethod
    def to_string_and_counts(message: 'SystemMessage') -> Tuple[str, int, int]:
        return Message.to_string_and_counts(message, "System")

@dataclass
class UserMessage(Message):
    """User message for queries/prompts"""
    content: Union[str, List[str]]
    role: Literal["user"] = "user"
    
    @staticmethod
    def to_string_and_counts(message: 'UserMessage') -> Tuple[str, int, int]:
        return Message.to_string_and_counts(message, "Human")

@dataclass
class AssistantMessage(Message):
    """Assistant message for AI responses"""
    content: Union[str, List[str]]
    role: Literal["assistant"] = "assistant"
    
    @staticmethod
    def to_string_and_counts(message: 'AssistantMessage') -> \
        Tuple[str, int, int]:
        return Message.to_string_and_counts(message, "AI")

@dataclass
class ToolMessage(Message):
    """Tool message for function calls/responses"""
    content: Union[str, List[str]]
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
    content: Union[str, List[str]]
    role: Literal["developer"] = "developer"

def create_system_message(content: str) -> Dict[str, str]:
    """Create a system message dictionary"""
    return SystemMessage(content=content).to_dict()

def create_system_message(
    content: Union[str, List[str]],
    name: Optional[str] = None) -> Dict[str, str]:
    """
    https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages

    Developer-provided instructions that model should follow, regardless of
    messages sent by the user. With o1 models and newer, use developer
    messages for this purpose instead.
    """
    message = SystemMessage(content=content).to_dict()
    if name is not None:
        message["name"] = name
    return message

def create_user_message(
    content: Union[str, List[str]],
    name: Optional[str] = None) -> Dict[str, str]:
    """
    https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages

    Messages sent by an end user.
    """
    message = UserMessage(content=content).to_dict()

    if name is not None:
        message["name"] = name
    return message

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

def create_developer_message(
        content: Union[str, List[str]],
        name: Optional[str] = None) -> Dict[str, str]:
    """Create a developer message dictionary
    
    https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages

    Developer-provided instructions that model should follow, regardless of
    messages sent by user. With o1 models and newer, use developer messages
    for this purpose instead.  
    """
    message = DeveloperMessage(content=content).to_dict()

    if name is not None:
        message["name"] = name
    return message

def parse_dict_into_specific_message(message):
    if "role" not in message:
        raise RuntimeError("Message must have a role")

    if message['role'] == 'system':
        return SystemMessage(content=message['content'])
    elif message['role'] == 'user':
        return UserMessage(content=message['content'])
    elif message['role'] == 'assistant':
        return AssistantMessage(content=message['content'])
    else:
        raise RuntimeError(f"Unknown message role: {message['role']}")