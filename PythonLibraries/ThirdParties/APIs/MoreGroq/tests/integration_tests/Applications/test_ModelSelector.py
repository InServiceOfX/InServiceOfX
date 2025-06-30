from corecode.Utilities import (get_environment_variable, load_environment_file)

from moregroq.Applications import ModelSelector

load_environment_file()

def test_model_selector_works():
    model_selector = ModelSelector(
        api_key=get_environment_variable("GROQ_API_KEY"))

    assert model_selector.get_all_available_models() is not None
    assert len(model_selector.get_all_available_models()) > 0

    for model in model_selector.get_all_available_models():
        print(model)

def test_model_selector_get_context_window_by_model_name():
    model_selector = ModelSelector(
        api_key=get_environment_variable("GROQ_API_KEY"))

    assert model_selector.get_context_window_by_model_name(
        "llama-3.3-70b-versatile") is not None
    assert model_selector.get_context_window_by_model_name(
        "llama-3.3-70b-versatile") == 131072