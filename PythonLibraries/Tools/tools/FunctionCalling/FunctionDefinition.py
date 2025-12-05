from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

@dataclass
class ParameterProperty:
    """
    Groq API:
    https://console.groq.com/docs/tool-use/overview
    "properties": {
        "location": {
            "type": "string",
            "description": "City and state, e.g. San Francisco, CA"
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
        }

    xAI
    https://docs.x.ai/docs/guides/function-calling

    "location": {
        "type": "string",
        "description": "The city and state, e.g. San Francisco, CA",
    },
    "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "default": "fahrenheit",
    },

    OpenAI
    https://platform.openai.com/docs/guides/function-calling

    "location": {
        "type": "string",
        "description": "City and country e.g. Bogotá, Colombia"
    },
    "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Units the temperature will be returned in."
    }
    """
    name: str
    type: str
    actual_type: Optional[Type] = None
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    default: Optional[Any] = None
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

        if self.default is not None:
            property_dict["default"] = self.default

        return property_dict

@dataclass
class FunctionParameters:
    """
    Groq API:
    https://console.groq.com/docs/tool-use/overview
    "parameters": {
        // JSON Schema object
        "type": "object",
        "properties": {
        "location": {
            "type": "string",
            "description": "City and state, e.g. San Francisco, CA"
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
        }
        },
        "required": ["location"]
    }

    xAI
    https://docs.x.ai/docs/guides/function-calling
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "fahrenheit",
            },
        },
        "required": ["location"],
    },

    OpenAI
    https://platform.openai.com/docs/guides/function-calling
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Units the temperature will be returned in."
            }
        },
        "required": ["location", "units"],
        "additionalProperties": false
    },
    """
    properties: List[ParameterProperty]

    # OpenAI API has this field.
    # https://platform.openai.com/docs/guides/function-calling
    # TODO: Figure out what true does.
    additional_properties: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        properties = {}
        required = []
        
        for param in self.properties:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)

        function_parameters_dict = {}
        function_parameters_dict["type"] = "object"
        function_parameters_dict["properties"] = properties
        function_parameters_dict["required"] = required
    
        if self.additional_properties != None:
            function_parameters_dict["additionalProperties"] = \
                additional_properties

        return function_parameters_dict

@dataclass
class FunctionDefinition:
    """
    FunctionDefinition attempts to accommodate the following APIs for a function
    definition as a raw dictionary.

    Groq API:
    https://console.groq.com/docs/tool-use/overview

    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
        // JSON Schema object
        "type": "object",
        "properties": {
            "location": {
            "type": "string",
            "description": "City and state, e.g. San Francisco, CA"
            },
            "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
        }
    }

    whereas, for

    xAI
    https://docs.x.ai/docs/guides/function-calling

    name="get_current_temperature",
    description="Get the current temperature in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "fahrenheit",
            },
        },
        "required": ["location"],
    },

    OpenAI
    https://platform.openai.com/docs/guides/function-calling

    "type": "function",
    "name": "get_weather",
    "description": "Retrieves current weather for the given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Units the temperature will be returned in."
            }
        },
        "required": ["location", "units"],
        "additionalProperties": false
    },
    "strict": true    
    """
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
    """
    Groq API is peculiar in how the nested the function definition within an
    extra field called "function":

    Groq API:
    https://console.groq.com/docs/tool-use/overview
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          // JSON Schema object
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }

    whereas,

    OpenAI
    https://platform.openai.com/docs/guides/function-calling

    {
        "type": "function",
        "name": "get_horoscope",
        "description": "Get today's horoscope for an astrological sign.",
        "parameters": {
            "type": "object",
            "properties": {
                "sign": {
                    "type": "string",
                    "description": "An astrological sign like Taurus or Aquarius",
                },
            },
            "required": ["sign"],
        },
    },

    For xAI API, you can simply use FunctionDefinition.to_dict() directly.
    """
    type: str = "function"
    function: FunctionDefinition = None

    def to_dict_for_groq(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "function": self.function.to_dict()
        }

    def to_dict(self) -> Dict[str, Any]:
        tool_dict = {}

        if self.function is not None:
            tool_dict = self.function.to_dict()

        tool_dict["type"] = self.type

        return tool_dict