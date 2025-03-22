from corecode.Utilities import DataSubdirectories
from pathlib import Path
import torch

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "musicgen-style"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-style"

from music.Configurations import (
    MusicgenConfiguration,)

from music.Configurations.audiocraft import (
    MusicGenGenerationConfiguration,)

from music.Wrappers.audiocraft import (
    MusicGenWrapperForMelodyAndStyle,)

test_data_dir = Path(__file__).resolve().parents[3] / "TestData"

def test_with_musicgen_stereo_melody():
    configuration = MusicgenConfiguration.from_yaml(
        test_data_dir / "musicgen_configuration_minimal.yml")
    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float

    generation_configuration = MusicGenGenerationConfiguration.from_yaml(
        test_data_dir / "MusicGen_generation_configuration_minimal.yml")
    generation_configuration.duration = 16

    wrapper = MusicGenWrapperForMelodyAndStyle(
        configuration, generation_configuration)

    descriptions = [
        (
            "A jazzy piece with a smooth saxophone solo. The sound is both "
            "sophisticated and playful with a slow tempo."
        )
    ]
    wrapper.save_wav(
        wrapper.generate_with_chroma(
            descriptions,
            data_sub_dirs.Data / "Public" / "Music" / \
                "Carlos_Gardels_-_01_-_Bach_-_Book_I_Prelude_and_Fugue_No_8_in_E_Flat_Minor_BWV_853_Prelude(chosic.com).mp3"),
        "test_with_musicgen_stereo_melody")