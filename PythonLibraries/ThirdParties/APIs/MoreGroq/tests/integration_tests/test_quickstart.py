# https://console.groq.com/docs/quickstart

from corecode.Utilities import (get_environment_variable, load_environment_file)

from groq import Groq
from moregroq.Wrappers import GetAllActiveModels

load_environment_file()

def test_Groq_inits():
    client = Groq(api_key=get_environment_variable("GROQ_API_KEY"))
    assert isinstance(client, Groq)

def test_chat_completions_create_works():
    client = Groq(api_key=get_environment_variable("GROQ_API_KEY"))

    keys = ["role", "content"]
    input_messages = [dict.fromkeys(keys)]

    input_messages[0]["role"] = "user"
    input_messages[0]["content"] = "Explain the importance of fast language models."

    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=input_messages)

    assert len(chat_completion.choices) == 1
    assert chat_completion.choices[0].message.content is not None
    # Uncomment out and run pytest with -s flag (for no capture) to see output.
    # print(chat_completion.choices[0].message.content)

def test_get_all_active_models_works():
    get_all_active_models = GetAllActiveModels(
        api_key=get_environment_variable("GROQ_API_KEY"))

    response = get_all_active_models()
    assert response is not None
    assert response.status_code == 200
    assert response.json() is not None
    assert len(response.json()) > 0
    assert set(response.json().keys()) == set(["object", "data"])
    # Uncomment out and run pytest with -s flag (for no capture) to see output.
    #print(response.json())
