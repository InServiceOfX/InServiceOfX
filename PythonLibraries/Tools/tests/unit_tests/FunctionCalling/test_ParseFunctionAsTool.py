from tools.FunctionCalling import ParseFunctionAsTool
from tools.FunctionCalling.FunctionDefinition import FunctionDefinition

from typing import Any

# https://platform.openai.com/docs/guides/function-calling#function-tool-example
def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."

def test_parse_for_function_definition_on_bare_function():
    function_definition = ParseFunctionAsTool.parse_for_function_definition(
        get_horoscope)

    assert type(function_definition) == FunctionDefinition

    assert function_definition.name == 'get_horoscope'
    assert function_definition.description == ""
    assert function_definition.parameters.properties[0].name == 'sign'
    assert function_definition.parameters.properties[0].type == 'Any'
    assert function_definition.parameters.properties[0].description == ""
    assert function_definition.parameters.properties[0].required == True

    assert function_definition.to_dict() == {
        "name": "get_horoscope",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "sign": {
                    "type": 'Any',
                    "description": "",
                },
            },
            "required": ["sign"],
        }
    }

def get_horoscope_as_documented_function(sign: str):
    """Get today's horoscope for an astrological sign.
    
    Args:
        sign: An astrological sign like Taurus or Aquarius
    """
    return f"{sign}: Next Tuesday you will befriend a baby otter."


def test_parse_for_function_definition_replicates_function_tool_example():
    """
    OpenAI API:
    https://platform.openai.com/docs/guides/function-calling#function-tool-example

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
    """

    function_definition = ParseFunctionAsTool.parse_for_function_definition(
        get_horoscope_as_documented_function)

    assert function_definition.to_dict() == {
        "name": "get_horoscope_as_documented_function",
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
    }