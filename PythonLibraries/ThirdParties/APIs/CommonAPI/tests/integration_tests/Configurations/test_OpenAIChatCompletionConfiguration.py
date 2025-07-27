from commonapi.Clients.OpenAIxGroqClient import (
    OpenAIxGroqClient
)

from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from corecode.Utilities import (get_environment_variable, load_environment_file)

from pydantic import BaseModel
from datetime import date

import pytest

load_environment_file()

class CalendarEvent(BaseModel):
    title: str
    start_date: date
    end_date: date

def test_structured_outputs_on_pydantic_BaseModel_as_response_format():
    """
    https://platform.openai.com/docs/guides/text-generation#quickstart
    """
    client = OpenAIxGroqClient(
        get_environment_variable("GROQ_API_KEY"))
    
    client.clear_chat_completion_configuration()
    #client.configuration.model = "llama-3.3-70b-versatile"
    client.configuration.model = "deepseek-r1-distill-llama-70b"

    messages = [
        create_system_message("Extract the event information."),
        create_user_message(
            "Alice and Bob are going to a science fair on Friday.")
    ]

    client.configuration.response_format = CalendarEvent

    with pytest.raises(TypeError) as type_error:
        response = client.create_chat_completion(messages)
        print(response.choices[0].message)

    assert \
        "You tried to pass a `BaseModel` class to `chat.completions.create()`; You must use `chat.completions.parse()` instead" \
            in str(type_error.value)
