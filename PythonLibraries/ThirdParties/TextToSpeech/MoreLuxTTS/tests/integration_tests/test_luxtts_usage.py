from corecode.Utilities import DataSubdirectories, is_model_there

data_subdirectories = DataSubdirectories()

relative_model_path = \
    "Models/Generative/TextToSpeech/YatharthS/LuxTTS"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

import soundfile as sf

# Here, you'll write the path to an example sound file outside of this
# repository; I demonstrate here that you can use DataSubdirectories to point to
# a data subdirectory.
example_sound_file_path = \
    data_subdirectories.Data / "Public" / "Audio" / "Voices" / \
        "FakeRogerVerbalKint-KevinSpacey.wav"

if not example_sound_file_path.exists():
    print(f"Example sound file {example_sound_file_path} not found")

from zipvoice.luxvoice import LuxTTS

def test_luxtts_load_model():
    lux_tts = LuxTTS

def test_luxtts_simple_inference():

    text = "Hey, what's up? I'm feeling really great if you ask me honestly!"
