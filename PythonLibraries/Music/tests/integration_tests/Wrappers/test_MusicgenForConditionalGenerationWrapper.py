from corecode.Utilities import DataSubdirectories
from pathlib import Path
import torch

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "musicgen-stereo-large"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-stereo-large"

from music.Configurations import (
    MusicgenConfiguration,
    MusicgenGenerationConfiguration)

from music.Wrappers import (
    from_prompt_and_processor,
    MusicgenForConditionalGenerationWrapper)

test_data_dir = Path(__file__).resolve().parents[2] / "TestData"

def test_MusicgenForConditionalGenerationWrapper_with_stereo_large():
    configuration = MusicgenConfiguration.from_yaml(
        test_data_dir / "musicgen_configuration_minimal.yml")

    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float16

    # https://github.com/yzfly/awesome-music-prompts
    prompt = (
        "Sad and longing. The melody is slow and wistful creating a sense of "
        "melancholy and nostalgia. Simple arrangement on the piano. ")

    inputs = from_prompt_and_processor(prompt, configuration=configuration)

    wrapper = MusicgenForConditionalGenerationWrapper(configuration)

    generation_configuration = MusicgenGenerationConfiguration.from_yaml(
        test_data_dir / "musicgen_generation_configuration_minimal.yml")

    generation_configuration.max_new_tokens = 768

    audio_values = wrapper.generate(inputs, generation_configuration)

    wrapper.save_using_soundfile(
        audio_values=audio_values,
        output_path="musicgen_stereo_large_out2.wav",
        dtype=torch.float32)