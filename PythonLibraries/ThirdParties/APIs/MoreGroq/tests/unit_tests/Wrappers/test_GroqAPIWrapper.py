from corecode.Utilities import (get_environment_variable, load_environment_file)

from moregroq.Wrappers import GroqAPIWrapper

load_environment_file()

default_configuration_values = {
    "model": "llama-3.3-70b-versatile",
    "n": 1,
    "stream": False,
    "temperature": 1.0,
}

def test_groq_api_wrapper_inits_with_default_configuration_values():
    api_key = get_environment_variable("GROQ_API_KEY")
    groq_api_wrapper = GroqAPIWrapper(api_key)
    assert groq_api_wrapper.configuration.to_dict() == \
        default_configuration_values

def test_groq_api_wrapper_clears_chat_completion_configuration():
    api_key = get_environment_variable("GROQ_API_KEY")
    groq_api_wrapper = GroqAPIWrapper(api_key)
    groq_api_wrapper.clear_chat_completion_configuration()
    assert groq_api_wrapper.configuration.to_dict() == \
        default_configuration_values

    ROUTING_MODEL = "llama3-70b-8192"
    groq_api_wrapper.configuration.model = ROUTING_MODEL
    groq_api_wrapper.configuration.max_completion_tokens = 20
    print("groq_api_wrapper.configuration.to_dict()", groq_api_wrapper.configuration.to_dict())

    assert groq_api_wrapper.configuration.to_dict() == {
        "model": ROUTING_MODEL,
        "n": 1,
        "stream": False,
        "temperature": 1.0,
        "max_completion_tokens": 20,
    }

    groq_api_wrapper.clear_chat_completion_configuration()
    assert groq_api_wrapper.configuration.to_dict() == \
        default_configuration_values