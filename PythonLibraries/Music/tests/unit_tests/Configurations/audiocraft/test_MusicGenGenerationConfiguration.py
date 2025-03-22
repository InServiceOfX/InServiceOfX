from music.Configurations.audiocraft import MusicGenGenerationConfiguration
from pathlib import Path

test_data_dir = Path(__file__).resolve().parents[3] / "TestData"

def test_MusicGenGenerationConfiguration_inits():
    configuration = MusicGenGenerationConfiguration.from_yaml(
        test_data_dir / "MusicGen_generation_configuration_minimal.yml")

    assert configuration.duration == None
    assert configuration.temperature == None
    assert configuration.top_k == None
    assert configuration.top_p == None

def test_fill_defaults():
    configuration = MusicGenGenerationConfiguration.from_yaml(
        test_data_dir / "MusicGen_generation_configuration_minimal.yml")
    configuration.fill_defaults()
    assert configuration.duration == 30.0
    assert configuration.temperature == 1.0
    assert configuration.top_k == 250
    assert configuration.top_p == 0.0