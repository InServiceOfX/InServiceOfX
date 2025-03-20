from corecode.Utilities import DataSubdirectories

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "musicgen-medium"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-medium"

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def test_MusicGen_get_pretrained_from_local_path():
    model = MusicGen.get_pretrained(pretrained_model_name_or_path)
    # generate 8 seconds
    model.set_generation_params(duration=8)

    descriptions = ["happy rock", "energetic EDM"]
    # Generates 2 samples.
    wav = model.generate(descriptions)

    for idx, one_wav in enumerate(wav):
        # Will save under [idx].wav, with loudness normalization at -14 db LUFS.
        audio_write(
            f'{idx}.wav',
            one_wav.cpu(),
            model.sample_rate,
            strategy="loudness")

# https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md

def test_MusicGen_get_pretrained_with_device():
    model = MusicGen.get_pretrained(
        pretrained_model_name_or_path,
        device="cuda:0")
    model.set_generation_params(duration=8)

    descriptions = ["happy rock", "energetic EDM", "sad jazz"]
    wav = model.generate(descriptions)

    for idx, one_wav in enumerate(wav):
        audio_write(
            f'{idx}.wav',
            one_wav.cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True)

