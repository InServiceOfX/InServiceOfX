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
        properties=[parameter_property]
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

def test_tool_to_dict():
    tool = Tool(
        function=FunctionDefinition(
            name="get_bakery_prices",
            description="Returns the prices for a given bakery product.",
            parameters=FunctionParameters(
                properties=[
                    ParameterProperty(
                        name="bakery_item",
                        description="The name of the bakery item",
                        type="string",
                        required=True
                    )
                ]
            )
        )
    )

    assert tool.to_dict() == {
        "type": "function",
        "function": {
            "name": "get_bakery_prices",
            "description": "Returns the prices for a given bakery product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bakery_item": {
                        "type": "string",
                        "description": "The name of the bakery item",
                    }
                },
                "required": ["bakery_item"],
            },
        },
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
        properties=[parameter_property_1, parameter_property_2]
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
        properties=[parameter_property]
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

    assert configuration.model == "llama-3.3-70b-versatile"
    assert configuration.temperature == 1.0
    assert configuration.max_completion_tokens is None
    assert configuration.max_tokens is None
    assert configuration.stream == False
    assert configuration.stop is None
    assert configuration.n == 1
    assert configuration.response_format is None
    assert configuration.response_model is None
    assert configuration.tools is None
    assert configuration.tool_choice is None
    assert configuration.parallel_tool_calls is None

def test_Tool_becomes_dict():
    tool = Tool(
        type="function",
        function=FunctionDefinition(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters=FunctionParameters(
                properties=[
                    ParameterProperty(
                        name="expression",
                        type="string",
                        description="The mathematical expression to evaluate",
                        required=True
                    )
                ],
            ),
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