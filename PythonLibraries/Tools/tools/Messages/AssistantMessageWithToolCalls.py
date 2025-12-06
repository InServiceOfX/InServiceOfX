from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Function:
    """
    This follows the types and attributes of the type
    <class 'openai.types.chat.chat_completion_message_function_tool_call.Function'>
    """
    name: str
    # e.g. {"sign":"Aquarius"}
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments
        }

@dataclass
class ToolCall:
    """
    This follows the types and attributes of the type
    <class 'openai.types.chat.chat_completion_message_function_tool_call.ChatCompletionMessageFunctionToolCall'>

    For the OpenAI API, id is required. For huggingface's transformers, id is
    not a required field.
    """
    id: Optional[str] = None
    # function
    type: str = "function"
    function: Function = None

    def to_dict(self) -> Dict[str, Any]:
        if self.id is None:
            return {
                "type": self.type,
                "function": self.function.to_dict()
            }
        else:
            return {
                "id": self.id,
                "type": self.type,
                "function": self.function.to_dict()
            }

@dataclass
class AssistantMessageWithToolCalls:
    """
    This is to be used with the OpenAI API ChatCompletion objects, types.

    This follows the type and attributes of the type 
    <class 'openai.types.chat.chat_completion_message.ChatCompletionMessage'>

    This is to help "wrap" the returned ChatCompletionMessage in the case when
    the LLM model wants to make tool calls.

    "content" is deliberately excluded and would be None.
    """
    role: str = "assistant"
    tool_calls: List[ToolCall] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls]
        }

def create_assistant_message_with_tool_calls_from_chat_completion_message(
    message
    ):
    """
    Args:
        message: This is typically response.choices[0].message for response
        which is what chat completion returns back from the LLM model. message
        is typically of type
        <class 'openai.types.chat.chat_completion_message.ChatCompletionMessage'>
    """
    role = message.role

    tool_calls = []

    for tool_call in message.tool_calls:
        tool_call_type = tool_call.type
        function_name = tool_call.function.name
        function_arguments = tool_call.function.arguments
        function_object = Function(
            name=function_name,
            arguments=function_arguments)

        if not hasattr(tool_call, "id"):
            tool_call_id = None
        else:
            tool_call_id = tool_call.id

        tool_call_object = ToolCall(
            id=tool_call_id,
            type=tool_call_type,
            function=function_object)
        tool_calls.append(tool_call_object)
    return AssistantMessageWithToolCalls(
        role=role,
        tool_calls=tool_calls)

def process_response_choices_into_assistant_messages(choices):
    """
    Assume that in the choices, we have tool calls.
    """
    assistant_messages = []
    for choice in choices:
        assistant_message = \
            create_assistant_message_with_tool_calls_from_chat_completion_message(
                choice.message)
        assistant_messages.append(assistant_message)
    return assistant_messages