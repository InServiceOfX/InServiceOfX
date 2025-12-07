from morexai.Voice.TextToSpeech import TextToSpeechClient

from corecode.Utilities import (get_environment_variable, load_environment_file)
load_environment_file()

from pathlib import Path
from morexai.Configuration import TextToSpeechConfiguration

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"
text_file_path = test_data_directory / "AmericanPsychoHipToBeSquare.txt"
text_to_speech_configuration_path = test_data_directory / \
    "general_text_to_speech_configuration.yml"
from warnings import warn

def test_TextToSpeechClient_generates_speech_from_text_file():
    """
    Test that the TextToSpeechClient generates audio.
    """
    text_to_speech_configuration = TextToSpeechConfiguration.from_yaml(
        text_to_speech_configuration_path)
    text_to_speech_configuration.text_file_path = text_file_path

    text_to_speech_client = TextToSpeechClient(text_to_speech_configuration)
    xai_api_key = get_environment_variable("XAI_API_KEY")
    if xai_api_key is None or xai_api_key == "":
        warn("XAI_API_KEY is not set")
        return
    text_to_speech_client.text_to_speech(xai_api_key)
    assert True