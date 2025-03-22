from corecode.Utilities import DataSubdirectories
from pathlib import Path
import torch

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "musicgen-stereo-melody"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-stereo-melody"

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

from music.Configurations import MusicgenConfiguration

import torchaudio

test_data_dir = Path(__file__).resolve().parents[1] / "TestData"

def test_with_musicgen_stereo_melody():
    configuration = MusicgenConfiguration.from_yaml(
        test_data_dir / "musicgen_configuration_minimal.yml")

    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float16

    model = MusicGen.get_pretrained(
        pretrained_model_name_or_path,
        device=configuration.device_map)

    model.set_generation_params(duration=8)

    descriptions = ["happy rock", "energetic EDM", "sad jazz"]

    mp3_path = data_sub_dirs.Data / "Public" / "Music" / \
        "Carlos_Gardels_-_01_-_Bach_-_Book_I_Prelude_and_Fugue_No_8_in_E_Flat_Minor_BWV_853_Prelude(chosic.com).mp3"

    assert mp3_path.exists()

    melody, sample_rate = torchaudio.load(mp3_path)

    assert melody.dim() == 2
    assert melody.shape[0] == 2
    assert melody.shape[1] == 9555840

    wav = model.generate_with_chroma(
        descriptions,
        melody.to(configuration.device_map)[None].expand(3, -1, -1),
        sample_rate)

    for idx, one_wave in enumerate(wav):
        audio_write(
            f'{idx}.wav',
            one_wave.cpu(),
            model.sample_rate,
            strategy="loudness")

pretrained_large_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / \
        "musicgen-stereo-melody-large"
if not pretrained_large_model_name_or_path.exists():
    pretrained_large_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-stereo-melody-large"

def test_with_musicgen_stereo_melody_large():
    configuration = MusicgenConfiguration.from_yaml(
        test_data_dir / "musicgen_configuration_minimal.yml")

    configuration.pretrained_model_name_or_path = \
        pretrained_large_model_name_or_path
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float16

    model = MusicGen.get_pretrained(
        pretrained_large_model_name_or_path,
        device=configuration.device_map)

    model.set_generation_params(duration=12)

    # https://github.com/yzfly/awesome-music-prompts
    descriptions = [
        (
            "Optimistic melody about the arrival of spring, full of joy and hope, "
            "tranquil flute in the background, upbeat with a gentle guitar riff"
        ),
    ]

    mp3_path = data_sub_dirs.Data / "Public" / "Music" / \
        "Carlos_Gardels_-_01_-_Bach_-_Book_I_Prelude_and_Fugue_No_8_in_E_Flat_Minor_BWV_853_Prelude(chosic.com).mp3"

    melody, sample_rate = torchaudio.load(mp3_path)

    wav = model.generate_with_chroma(
        descriptions,
        melody.to(configuration.device_map)[None].expand(
            len(descriptions),
            -1,
            -1),
        sample_rate)

    for idx, one_wave in enumerate(wav):
        audio_write(
            f'{idx}-large.wav',
            one_wave.cpu(),
            model.sample_rate,
            strategy="loudness")