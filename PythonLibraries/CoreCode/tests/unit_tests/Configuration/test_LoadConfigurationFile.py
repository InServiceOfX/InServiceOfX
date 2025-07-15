from corecode.Configuration import LoadConfigurationFile
from pathlib import Path
from warnings import warn

test_data_path = Path(__file__).parents[2] / "TestData"
if not test_data_path.exists():
    warn(f"Test data path {test_data_path} does not exist")

def test_load_configuration_file():
    config = LoadConfigurationFile.load_configuration_file(
        test_data_path / "example_config.config")
    print(config)
    assert config is not None
    assert config['BASE_DATA_PATH'] == '/Data/'
    assert config['PROMPTS_COLLECTION_PATH'] == '/Data/Prompts/PromptsCollection'
    assert config['BASE_DATA_PATH_1'] == '/Data1/'