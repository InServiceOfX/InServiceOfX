from clichat.Configuration import Configuration
from clichat.Utilities.FileIO import get_path_from_configuration
from pathlib import Path

import pytest

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

@pytest.fixture
def configuration():
    test_file_path = test_data_directory / "clichat_configuration.yml"
    assert test_file_path.exists()
    return Configuration(test_file_path)

def test_get_path_from_configuration_work(configuration):
    assert get_path_from_configuration(configuration, "chat_history_path") == \
        Path(test_data_directory / "chat_history.txt")

    assert Path(test_data_directory / "system_messages.json").exists()

    assert get_path_from_configuration(
        configuration,
        "system_messages_path") == Path(
            test_data_directory / "system_messages.json")

def test_get_path_from_configuration_on_nonexisting_paths(configuration):
    configuration.system_messages_path = Path(
        test_data_directory / "nonexisting.json")

    with pytest.raises(FileNotFoundError):
        get_path_from_configuration(configuration, "system_messages_path")
