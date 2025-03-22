from music.Configurations import MusicgenGenerationConfiguration
from pathlib import Path

test_data_dir = Path(__file__).resolve().parents[2] / "TestData"

def test_MusicgenGenerationConfiguration_inits():
    configuration = MusicgenGenerationConfiguration.from_yaml(
        test_data_dir / "musicgen_generation_configuration_minimal.yml")

    assert configuration.max_new_tokens == None
    assert configuration.temperature == None
    assert configuration.top_k == None
    assert configuration.top_p == None

def test_fill_defaults():
    configuration = MusicgenGenerationConfiguration.from_yaml(
        test_data_dir / "musicgen_generation_configuration_minimal.yml")
    configuration.fill_defaults()
    assert configuration.max_new_tokens == 512
    assert configuration.temperature == 1.0
    assert configuration.top_k == 50
    assert configuration.top_p == 1.0