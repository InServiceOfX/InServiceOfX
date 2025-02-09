from morelumaai.Configuration import GenerationConfiguration
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_GenerationConfiguration_loads_from_yaml():
    test_file_path = test_data_directory / "generation_configuration.yml"
    assert test_file_path.exists()

    configuration = GenerationConfiguration.from_yaml(test_file_path)
    assert configuration.temporary_save_path == "/Data/Private"
