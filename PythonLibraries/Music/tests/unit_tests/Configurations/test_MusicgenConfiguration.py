from music.Configurations import MusicgenConfiguration
from pathlib import Path
import torch

test_data_dir = Path(__file__).resolve().parents[2] / "TestData"

def test_MusicgenConfiguration_inits():
    configuration = MusicgenConfiguration.from_yaml(
        test_data_dir / "musicgen_configuration_minimal.yml")

    assert configuration.pretrained_model_name_or_path == \
        str(Path.cwd().resolve() / "musicgen-stereo-large")
    assert configuration.device_map == None
    assert configuration.torch_dtype == None
    assert configuration.attn_implementation == None

def test_fill_defaults():
    configuration = MusicgenConfiguration.from_yaml(
        test_data_dir / "musicgen_configuration_minimal.yml")
    configuration.fill_defaults()
    assert configuration.attn_implementation == "eager"
    assert configuration.torch_dtype == torch.float32
    assert configuration.device_map == None