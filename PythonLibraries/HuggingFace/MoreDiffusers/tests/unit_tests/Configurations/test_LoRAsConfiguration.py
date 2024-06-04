from morediffusers.Configurations import LoRAsConfiguration
from pathlib import Path
import pytest

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_LoRAsConfiguration_inits_for_empty_file():
    test_file_path = test_data_directory / "loras_configuration_empty.yml"
    assert test_file_path.exists()

    configuration = LoRAsConfiguration(test_file_path)

    assert configuration.lora_scale == None

    assert len(configuration.loras) == 0
    assert configuration.loras == {}

    keys = []
    lora_parameters = []

    for key, lora_parameter in configuration.loras.items():
        keys.append(key)
        lora_parameters.append(lora_parameter)
    assert keys == []
    assert lora_parameters == []