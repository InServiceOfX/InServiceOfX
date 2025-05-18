from corecode.Utilities import (get_environment_variable, load_environment_file)

from groq import Groq
from moregroq.Wrappers import GetAllActiveModels

load_environment_file()

def test_get_parsed_response_works():
    get_all_active_models = GetAllActiveModels(
        api_key=get_environment_variable("GROQ_API_KEY"))

    get_all_active_models()

    result = get_all_active_models.get_parsed_response()
    assert result is not None
    assert len(result) > 0

    # Uncomment out and run pytest with -s flag (for no capture) to see output.
    #print(get_all_active_models.response.json())

def test_get_list_of_available_models_works():
    get_all_active_models = GetAllActiveModels(
        api_key=get_environment_variable("GROQ_API_KEY"))

    get_all_active_models()
    result = get_all_active_models.get_list_of_available_models()
    assert result is not None
    assert len(result) > 0
    assert set(result[0].keys()) == set(["id", "owned_by", "context_window"])

    TODO: Fix this.
    assert any(
        entry.get('id') == 'llama-3.3-70b-versatile' \
        and \
        # This used to be 32768.
        entry.get('context_window') == 131072 for entry in result)

    # Uncomment out and run pytest with -s flag (for no capture) to see output.
    for entry in result:
        print(entry)
