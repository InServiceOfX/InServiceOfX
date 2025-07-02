from corecode.Utilities import (get_environment_variable, load_environment_file)

load_environment_file()

from brainswapchat.UI import GroqModelSelector

def test_GroqModelSelector_inits():
    groq_model_selector = GroqModelSelector()
