from corecode.Utilities import DataSubdirectories, is_model_there

data_subdirectories = DataSubdirectories()

relative_model_path = \
    "Models/Generative/TextToSpeech/ResembleAI/chatterbox"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

# Here, you'll write the path to an example sound file outside of this
# repository; I demonstrate here that you can use DataSubdirectories to point to
# a data subdirectory.
example_sound_file_path = \
    data_subdirectories.Data / "Public" / "Audio" / "Voices" / \
        "FakeRogerVerbalKint-KevinSpacey.wav"

from morechatterbox.Wrappers import ChatterboxTTSModel
from morechatterbox.Configurations import ChatterboxTTSConfiguration
from pathlib import Path

import tempfile

def test_ChatterboxTTSModel_generate_and_save():
    example_text = (
        "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down "
        "the enemy's Nexus in an epic late-game pentakill."
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        example_text_file_path = tmp / "example.txt"
        example_text_file_path.write_text(example_text)

        configuration = ChatterboxTTSConfiguration(
            model_dir=model_path,
            audio_prompt_path=example_sound_file_path,
            text_file_path=example_text_file_path,
            directory_path_to_save=Path.cwd(),
            base_saved_filename="ChatterboxTTSModelTest",
            sample_rate=24000
        )
        model = ChatterboxTTSModel(configuration)
        model.generate_and_save()
        assert True