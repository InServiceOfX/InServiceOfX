from xai_sdk import Client
from xai_sdk.chat import tool, tool_result, user

import json
import os

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

    get_current_temperature_schema,
    get_current_ceiling_schema = \
        get_model_json_schema_for_temperature_as_pydantic_model()
    tool_definitions = \
        get_tool_definitions_for_temperature_as_pydantic_model(
            get_current_temperature_schema,
            get_current_ceiling_schema
        )


    chat = client.chat.create(
        model="grok-4",
        tools=tool_definitions,
        tool_choice="auto",
    )
    chat.append(user("What's the temperature like in San Francisco?"))
    response = chat.sample()

    # You can inspect the response tool calls which contains a tool call
    print(type(response))
    print(response.tool_calls)
    print(response.content)

    # Append assistant message including tool calls to messages
    chat.append(response)