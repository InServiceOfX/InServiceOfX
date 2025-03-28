from audiocraft.data.audio import audio_write
from corecode.Utilities import DataSubdirectories
from music.Configurations.audiocraft import (
    JASCOConfiguration,
    JASCOGenerationConfiguration)
from music.Wrappers.audiocraft import JASCOWrapper, JASCOGenerateMusicInputs
from pathlib import Path
import torch
import torchaudio

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "jasco-chords-drums-melody-1B"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "jasco-chords-drums-melody-1B"

path_to_chords_mapping = Path(
    "/ThirdParty/audiocraft/assets/chord_to_index_mapping.pkl")

test_data_dir = Path(__file__).resolve().parents[3] / "TestData"

def test_JASCOGenerateMusicInputs_drums_and_chords_conditioning_generation():

    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float
    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.chords_mapping_path = path_to_chords_mapping

    generation_configuration = JASCOGenerationConfiguration.from_yaml(
        test_data_dir / "JASCO_generation_configuration_minimal.yml")
    generation_configuration.cfg_coef_all = 1.5
    generation_configuration.cfg_coef_txt = 3.0

    wrapper = JASCOWrapper(configuration, generation_configuration)

    # set textual prompt
    text = "string quartet, orchestral, dramatic"
    descriptions = [text]

    generation_inputs = JASCOGenerateMusicInputs(descriptions=descriptions)

    generation_inputs.drums_wav, generation_inputs.drums_sample_rate = \
        torchaudio.load(Path("/ThirdParty/audiocraft/assets/sep_drums_1.mp3"))

    # define chord progression
    generation_inputs.chords = [
        ('C', 0.0), ('D', 2.0), ('F', 4.0), ('Ab', 6.0), ('Bb', 7.0), ('C', 8.0)]

    output = wrapper.model.generate_music(**generation_inputs.to_dict())

    audio_write(
        'jasco_generate_musicdrums_and_chords_conditioning_generation',
        output.cpu().squeeze(0),
        wrapper.model.sample_rate,
        strategy="loudness",
        loudness_compressor=True)