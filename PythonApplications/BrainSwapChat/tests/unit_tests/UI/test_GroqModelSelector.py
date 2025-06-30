from corecode.Utilities import (get_environment_variable, load_environment_file)

load_environment_file()

from brainswapchat.UI import GroqModelSelector

def test_GroqModelSelector_inits():
    groq_model_selector = GroqModelSelector()
    assert groq_model_selector._available_models is None
    assert groq_model_selector._current_model is None

def test_GroqModelSelector__get_available_models_works():
    groq_model_selector = GroqModelSelector()

    result = groq_model_selector._get_available_models()
    assert result == groq_model_selector._available_models

    for model in result.items():
        print(model)