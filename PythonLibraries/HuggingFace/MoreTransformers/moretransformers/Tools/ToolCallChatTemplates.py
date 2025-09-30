from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class FunctionDefinition:
    name: str
    arguments: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments
        }

@dataclass
class ToolCall:
    type: str = "function"
    function: FunctionDefinition = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "function": self.function.to_dict()
        }

@dataclass
class AssistantMessageWithToolCalls:
    role: str = "assistant"
    tool_calls: List[ToolCall] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls]
        }