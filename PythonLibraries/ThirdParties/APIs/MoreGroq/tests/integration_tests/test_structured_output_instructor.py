"""
https://github.com/groq/groq-api-cookbook/blob/main/tutorials/structured-output-instructor/structured_output_instructor.ipynb
"""
from corecode.Utilities import (get_environment_variable, load_environment_file)
from commonapi.Messages import (
    create_system_message,
    create_user_message
)
from pydantic import BaseModel, Field

from moregroq.Wrappers import GroqAPIWrapper

from tools.FunctionCalling.FunctionDefinition import (
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty,)

from pprint import pprint

import instructor

load_environment_file()

# Describe the desired output schema using pydantic models
class UserInfo(BaseModel):
    name: str
    age: int
    email: str

def test_extracting_structured_data():
    text = """
    John Doe, a 35-year-old software engineer from New York, has been working with large language models for several years.
    His email address is johndoe@example.com.
    """

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.client = instructor.from_groq(
        groq_api_wrapper.client,
        mode=instructor.Mode.JSON)

    # Call the API

    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    # Specify the response model
    groq_api_wrapper.configuration.response_model = UserInfo
    groq_api_wrapper.configuration.temperature = 0.65

    messages = [
        create_system_message(
            "Your job is to extract user information from the given text."),
        create_user_message(text)]

    user_info = groq_api_wrapper.create_chat_completion(messages)
    print(user_info)
    assert user_info.name == "John Doe"
    assert user_info.age == 35
    assert user_info.email == "johndoe@example.com"

class Example(BaseModel):
    input_text: str = Field(description="The example text")
    tool_name: str = Field(description="The tool name to call for this example")
    tool_parameters: str = Field(
        description=(
            "An object containing the key-value pairs for the parameters of this "
            "tool as a JSON serializbale STRING, make sure it is valid JSON and "
            "parameter values are of the correct type according to the tool schema"
        )
    )

class ResponseModel(BaseModel):
    examples: list[Example]

def test_generating_synthetic_data():
    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.clear_chat_completion_configuration()
    groq_api_wrapper.client = instructor.from_groq(
        groq_api_wrapper.client,
        mode=instructor.Mode.JSON)
        
    # The schema for get_weather_info tool.
    tool_schema = FunctionDefinition(
        name="get_weather_info",
        description="Get the weather information for a given location",
        parameters=FunctionParameters(
            properties=[
                ParameterProperty(
                    name="location",
                    description=(
                        "The location for which we want to get the weather "
                        "information (e.g. New York)"
                    ),
                    type="string",
                    required=True
                )
            ]
        )
    )
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.response_model = ResponseModel
    groq_api_wrapper.configuration.temperature = 0.65
    groq_api_wrapper.configuration.max_tokens = 8000
 
    prompt = """
    I am designing a weather agent. This agent can talk to the user and also fetch latest weather information.
    It has access to the `get_weather_info` tool with the following JSON schema:
    {json_schema}

    I want you to write some examples for `get_weather_info` and see if this functionality works correctly and can handle all the cases. 
    Now given the information so far and the JSON schema of the provided tool, write {num} examples.
    Make sure each example is varied enough to cover common ways of requesting for this functionality.
    Make sure you fill the function parameters with the correct types when generating the output examples. 
    Make sure your output is valid JSON.
    """

    messages = [
        create_system_message(
            prompt.format(json_schema=tool_schema.to_dict(), num=5)
        ),
    ]

    response = groq_api_wrapper.create_chat_completion(messages)
    print(type(response))
    assert len(response.examples) == 5
    
    pprint(response.examples)