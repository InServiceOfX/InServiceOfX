from moregroq.Wrappers.ChatCompletionConfiguration import (
    ChatCompletionConfiguration,
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty,
    Tool
)

location_parameter_description = (
    "The location for which we want to get the weather information "
    "(e.g., New York)")

def test_parameter_property_to_dict():
    parameter_property = ParameterProperty(
        name="location",
        description=location_parameter_description,
        type="string"
    )
    assert parameter_property.to_dict() == {
        "type": "string",
        "description": location_parameter_description,
    }

def test_function_parameters_to_dict():
    parameter_property = ParameterProperty(
        name="location",
        description=location_parameter_description,
        type="string"
    )

    function_parameters = FunctionParameters(
        parameters=[parameter_property]
    )
    assert function_parameters.to_dict() == {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": location_parameter_description,
                }
            },
            "required": ["location"]
        }

def test_function_parameters_can_have_multiple_parameters():
    parameter_property_1 = ParameterProperty(
        name="location",
        description="The city and state, e.g. San Francisco, CA",
        type="string"
    )
    parameter_property_2 = ParameterProperty(
        name="unit",
        enum=["celsius", "fahrenheit"],
        type="string",
        required=False
    )
    function_parameters = FunctionParameters(
        parameters=[parameter_property_1, parameter_property_2]
    )
    assert function_parameters.to_dict() == {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }        

def test_function_definition_to_dict():
    parameter_property = ParameterProperty(
        name="location",
        description="The name of the city",
        type="string"
    )

    function_parameters = FunctionParameters(
        parameters=[parameter_property]
    )

    function_definition = FunctionDefinition(
        name="get_weather_condition",
        description="Get the weather condition for a given location",
        parameters=function_parameters
    )
    print("function_definition.to_dict()", function_definition.to_dict())
    assert function_definition.to_dict() == {
            "name": "get_weather_condition",
            "description": "Get the weather condition for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city",
                    }
                },
                "required": ["location"],
            },
        }

def test_chat_completion_configuration_inits_with_default_values():
    configuration = ChatCompletionConfiguration()
    assert configuration.to_dict() == {
        "model": "llama-3.3-70b-versatile",
        "n": 1,
        "stream": False,
        "temperature": 1.0,
    }

def test_Tool_becomes_dict():
    tool = Tool(
        type="function",
        function=FunctionDefinition(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],            
            },
        ),
    )
    tools = [tool,]
    configuration = ChatCompletionConfiguration(
        tools=tools,
        tool_choice="auto"
    )

    assert configuration.to_dict() == {
        "model": "llama-3.3-70b-versatile",
        "n": 1,
        "stream": False,
        "temperature": 1.0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    }
                }
            }
        ],
        "tool_choice": "auto",
    }