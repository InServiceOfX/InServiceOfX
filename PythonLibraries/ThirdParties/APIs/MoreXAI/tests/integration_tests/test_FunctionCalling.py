from xai_sdk import Client
from xai_sdk.chat import tool, tool_result, user

import json

from corecode.Utilities import (get_environment_variable, load_environment_file)
load_environment_file()

# Function definition using Pydantic

from pydantic import BaseModel, Field
from typing import Literal

class TemperatureRequest(BaseModel):
    location: str = Field(
        description="The city and state, e.g. San Francisco, CA")
    unit: Literal["celsius", "fahrenheit"] = Field(
        "fahrenheit", description="Temperature unit"
    )

class CeilingRequest(BaseModel):
    location: str = Field(
        description="The city and state, e.g. San Francisco, CA")

def get_current_temperature(request: TemperatureRequest):
    temperature = 59 if request.unit.lower() == "fahrenheit" else 15
    return {
        "location": request.location,
        "temperature": temperature,
        "unit": request.unit,
    }

def get_current_ceiling(request: CeilingRequest):
    return {
        "location": request.location,
        "ceiling": 15000,
        "ceiling_type": "broken",
        "unit": "ft",
    }

# Generate the JSON schema from the Pydantic models

# Originally:
#get_current_temperature_schema = TemperatureRequest.model_json_schema()
#get_current_ceiling_schema = CeilingRequest.model_json_schema()

def get_model_json_schema_for_temperature_as_pydantic_model():
    get_current_temperature_schema = TemperatureRequest.model_json_schema()
    get_current_ceiling_schema = CeilingRequest.model_json_schema()
    return get_current_temperature_schema, get_current_ceiling_schema

# Definition of parameters with Pydantic JSON schema

# Originally:
# tool_definitions = [
#     tool(
#         name="get_current_temperature",
#         description="Get the current temperature in a given location",
#         parameters=get_current_temperature_schema,
#     ),
#     tool(
#         name="get_current_ceiling",
#         description="Get the current cloud ceiling in a given location",
#         parameters=get_current_ceiling_schema,
#     ),
# ]

def get_tool_definitions_for_temperature_as_pydantic_model(
    get_current_temperature_schema,
    get_current_ceiling_schema):
    tool_definitions = [
        tool(
            name="get_current_temperature",
            description="Get the current temperature in a given location",
            parameters=get_current_temperature_schema,
        ),
        tool(
            name="get_current_ceiling",
            description="Get the current cloud ceiling in a given location",
            parameters=get_current_ceiling_schema,
        ),
    ]
    return tool_definitions

def test_function_calling_with_temperature_as_pydantic_model():
    client = Client(api_key=get_environment_variable("XAI_API_KEY"))
    chat = client.chat.create(model="grok-4")

    get_current_temperature_schema, get_current_ceiling_schema = \
        get_model_json_schema_for_temperature_as_pydantic_model()
    tool_definitions = \
        get_tool_definitions_for_temperature_as_pydantic_model(
            get_current_temperature_schema,
            get_current_ceiling_schema
        )

    # "Create a string -> function mapping, so we can call the function when
    # model sends it's name. e.g."
    tools_map = {
        "get_current_temperature": get_current_temperature,
        "get_current_ceiling": get_current_ceiling,
    }

    # Map function names to their Pydantic model classes
    models_map = {
        "get_current_temperature": TemperatureRequest,
        "get_current_ceiling": CeilingRequest,
    }

    chat = client.chat.create(
        model="grok-4",
        tools=tool_definitions,
        tool_choice="auto",
    )
    chat.append(user("What's the temperature like in San Francisco? Please give me the temperature in Fahrenheit"))
    response = chat.sample()

    # You can inspect the response tool calls which contains a tool call
    # <class 'xai_sdk.chat.Response'>
    # print(type(response))
    # <class 'list'>
    # print(type(response.tool_calls))
    # [id: "call_26551865"
    # type: TOOL_CALL_TYPE_CLIENT_SIDE_TOOL
    # status: TOOL_CALL_STATUS_COMPLETED
    # function {
    # name: "get_current_temperature"
    # arguments: "{\"location\":\"San Francisco, CA\"}"
    # }
    # print("response.tool_calls: ", response.tool_calls)
    
    #if (len(response.tool_calls) > 0):
        # <class 'xai.api.v1.chat_pb2.ToolCall'>
        # print("type(response.tool_calls[0]): ", type(response.tool_calls[0]))

    # <class 'str'>
    # print(type(response.content))

    # This looked empty.
    print("response.content: ", response.content)

    # Append assistant message including tool calls to messages
    chat.append(response)

    # Check if there is any tool calls in response body
    # You can also wrap this in a function to make the code cleaner
    if response.tool_calls:
        for tool_call in response.tool_calls:
            # Get the tool function name and arguments Grok wants to call
            function_name = tool_call.function.name
            
            # get_current_temperature
            #print("function_name: ", function_name)
            
            # <class 'xai.api.v1.chat_pb2.FunctionCall'>
            # print(type(tool_call.function))
            
            # <class 'str'>
            # print(type(tool_call.function.arguments))
            
            function_args = json.loads(tool_call.function.arguments)
            
            # <class 'dict'>
            # print("type(function_args):", type(function_args))

            # {'location': 'San Francisco, CA', 'unit': 'fahrenheit'}
            # print("function_args: ", function_args)

            # Convert dict to Pydantic model instance
            model_class = models_map[function_name]
            model_instance = model_class(**function_args)

            # Call one of the tool function defined earlier with arguments
            # TypeError: get_current_temperature() got an unexpected keyword argument 'location'
            #result = tools_map[function_name](**function_args)
            # Append the result from tool function call to the chat message history

            result = tools_map[function_name](model_instance)

            # <class 'dict'>
            #print("type(result): ", type(result))
            
            # dict_keys(['location', 'temperature', 'unit'])
            #print("result.keys(): ", result.keys())

            # {'location': 'San Francisco, CA', 'temperature': 59, 'unit': 'fahrenheit'}
            # print("result: ", result)

            # E       TypeError: Cannot set xai_api.Content.text to {'location': 'San Francisco, CA', 'temperature': 59, 'unit': 'fahrenheit'}: {'location': 'San Francisco, CA', 'temperature': 59, 'unit': 'fahrenheit'} has type <class 'dict'>, but expected one of: (<class 'bytes'>, <class 'str'>) for field Content.text
            # tool_result_on_result = tool_result(result)

            # Convert dict to JSON string for tool_result()
            result_json = json.dumps(result)

            tool_result_on_result = tool_result(result_json)

            # <class 'xai.api.v1.chat_pb2.Message'>
            # print("type(tool_result_on_result): ", type(tool_result_on_result))

            # content {
            #   text: "{\"location\": \"San Francisco, CA\", \"temperature\": 59, \"unit\": \"fahrenheit\"}"
            # }
            # print("tool_result_on_result: ", tool_result_on_result)

            chat.append(tool_result_on_result)

    response = chat.sample()

    # The current temperature in San Francisco is 59°F.
    # print(response.content)
    
    # []
    # print(response.tool_calls)

class FunctionDefinitionsUsingRawDictionary():

    @staticmethod
    def get_current_temperature(
        location: str,
        unit: Literal["celsius", "fahrenheit"] = "fahrenheit"):
        temperature = 59 if unit == "fahrenheit" else 15
        return {
            "location": location,
            "temperature": temperature,
            "unit": unit,
        }

    @staticmethod
    def get_current_ceiling(location: str):
        return {
            "location": location,
            "ceiling": 15000,
            "ceiling_type": "broken",
            "unit": "ft",
        }

def get_tool_definitions_using_raw_dictionary():
    tool_definitions = [
        tool(
            name="get_current_temperature",
            description="Get the current temperature in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": \
                            "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "fahrenheit",
                    },
                },
                "required": ["location"],
            },
        ),
        tool(
            name="get_current_ceiling",
            description="Get the current cloud ceiling in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": \
                            "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        ),
    ]
    return tool_definitions

def test_function_calling_with_temperature_using_raw_dictionary():

    tools_map = {
        "get_current_temperature": \
            FunctionDefinitionsUsingRawDictionary.get_current_temperature,
        "get_current_ceiling": \
            FunctionDefinitionsUsingRawDictionary.get_current_ceiling,
    }

    tool_definitions = get_tool_definitions_using_raw_dictionary()

    # <class 'xai.api.v1.chat_pb2.Tool'>
    # print("type(tool_definitions[0]): ", type(tool_definitions[0]))

    api_key = get_environment_variable("XAI_API_KEY")

    if api_key is None or api_key == "":
        warn("XAI_API_KEY is not set")
        return

    client = Client(api_key=api_key)
    chat = client.chat.create(
        model="grok-4",
        tools=tool_definitions,
        tool_choice="auto",
    )

    chat.append(user("What's the temperature like in San Francisco?"))
    response = chat.sample()

    # Example (actual) response.tool_calls:
    # [id: "call_10195684"
    # type: TOOL_CALL_TYPE_CLIENT_SIDE_TOOL
    # status: TOOL_CALL_STATUS_COMPLETED
    # function {
    #   name: "get_current_temperature"
    #   arguments: "{\"location\":\"San Francisco, CA\"}"
    # }
    # ]
    # print("response.tool_calls: ", response.tool_calls)

    chat.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:

            # Example (actual) tool_call
            # id: "call_10195684"
            # type: TOOL_CALL_TYPE_CLIENT_SIDE_TOOL
            # status: TOOL_CALL_STATUS_COMPLETED
            # function {
            #   name: "get_current_temperature"
            #   arguments: "{\"location\":\"San Francisco, CA\"}"
            # }
            #print("tool_call: ", tool_call)

            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            result = tools_map[function_name](**function_args)

            # Convert dict to JSON string for tool_result()
            result_json = json.dumps(result)

            tool_result_on_result = tool_result(result_json)
            chat.append(tool_result_on_result)

    response = chat.sample()
    print("response.content: ", response.content)
    print("response.tool_calls: ", response.tool_calls)