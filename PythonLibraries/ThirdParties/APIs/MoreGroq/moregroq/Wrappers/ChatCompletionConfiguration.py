from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ParameterProperty:
    name: str
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-compatible dictionary"""
        property_dict = {
            "type": self.type
        }
        
        if self.description is not None:
            property_dict["description"] = self.description
            
        if self.enum is not None:
            property_dict["enum"] = self.enum
            
        return property_dict

@dataclass
class FunctionParameters:
    properties: List[ParameterProperty]

    def to_dict(self) -> Dict[str, Any]:
        properties = {}
        required = []
        
        for param in self.properties:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: FunctionParameters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict()
        }

@dataclass
class Tool:
    type: str = "function"
    function: FunctionDefinition = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "function": self.function.to_dict()
        }

@dataclass
class ChatCompletionConfiguration:
    """
    See
    https://console.groq.com/docs/api-reference#chat-create
    """
    model: str = "llama-3.3-70b-versatile"
    # When smapling temperature to use, between 0 and 2.
    temperature: Optional[float] = 1.0
    # max_completion_tokens integer or null Optional
    # Deprecated in favor of max_completion_tokens. Max number of tokens that
    # can be generated in chat completion. Total length of input tokens and
    # generated tokens is limited by model's context length.
    max_completion_tokens: Optional[int] = None
    # integer or null Optional
    # The maximum number of tokens that can be generated in the chat
    # completion. Total length of input tokens and generated tokens is
    # limited by model's context length.
    max_tokens: Optional[int] = None
    # boolean or null Optional Defaults to false.
    # If set, partial message deltas will be sent. Tokens will be sent as
    # data-only server-sent events as they become available, with the
    # stream terminated by a data: [DONE] message.
    stream: bool = False
    # string / array or null Optional
    # Up to 4 sequences where API will stop generating further tokens. The
    # returned text will not contain the stop sequence.
    stop: Optional[List[str]] = None
    # integer or null Optional Defaults to 1
    # How many chat completion choices to generate for each input message.
    # Note that current moment, only n=1 is supported. Other values will
    # result in a 400 response.
    n: int = 1
    # response_format object or null Optional
    # An object specifying format that model must output.
    # Setting to {"type": "json_object"} enables JSON mode, which guarantees the
    # message the model generates is a valid JSON.
    # Important: when using JSON mode, you *must* also instruct model to produce
    # JSON yourself via a system or user message..
    response_format: Optional[Dict[str, str]] = None

    # Isn't documented in API reference, but shows up in Tool use Structured
    # outputs example.
    response_model: Optional[Any] = None

    # Tool use parameters
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None

    # https://console.groq.com/docs/api-reference#chat
    # top_p number or null Optional Defaults to 1.
    # "We generally recommend altering this or temperature but not both."

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to API-compatible dictionary"""
        config_dict = {
            "model": self.model,
            "n": self.n,
            "stream": self.stream
        }

        if self.temperature is not None:
            config_dict["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            config_dict["max_completion_tokens"] = self.max_completion_tokens
        if self.max_tokens is not None:
            config_dict["max_tokens"] = self.max_tokens
        if self.stop is not None:
            config_dict["stop"] = self.stop
        if self.response_format is not None:
            config_dict["response_format"] = self.response_format
        if self.response_model is not None:
            config_dict["response_model"] = self.response_model

        # Add tool use parameters if specified
        if self.tools is not None:
            config_dict["tools"] = [tool.to_dict() for tool in self.tools]
        if self.tool_choice is not None:
            config_dict["tool_choice"] = self.tool_choice
        if self.parallel_tool_calls is not None:
            config_dict["parallel_tool_calls"] = self.parallel_tool_calls
            
        return config_dict