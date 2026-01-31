from corecode.Utilities import DataSubdirectories, is_model_there

data_subdirectories = DataSubdirectories()

relative_model_path = \
    "Models/Generative/TextToSpeech/ResembleAI/chatterbox"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

import torchaudio
from chatterbox.tts import ChatterboxTTS

# Here, you'll write the path to an example sound file outside of this
# repository; I demonstrate here that you can use DataSubdirectories to point to
# a data subdirectory.
example_sound_file_path = \
    data_subdirectories.Data / "Public" / "Audio" / "Voices" / \
        "FakeRogerVerbalKint-KevinSpacey.wav"

if not example_sound_file_path.exists():
    print(f"Example sound file {example_sound_file_path} not found")

def test_chatterbox_load_model_locally():
    model = ChatterboxTTS.from_local(ckpt_dir=model_path, device="cuda:0")
    assert True

def test_chatterbox_generate_with_audio_prompt():
    model = ChatterboxTTS.from_local(ckpt_dir=model_path, device="cuda:0")
    text = (
        "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down "
        "the enemy's Nexus in an epic late-game pentakill.")

    wav = model.generate(text, audio_prompt_path=example_sound_file_path)
    torchaudio.save(
        "test_chatterbox_generate_with_audio_prompt.wav",
        wav,
        model.sr)
    assert True
