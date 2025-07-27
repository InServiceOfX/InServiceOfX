from corecode.FileIO import get_project_directory_path
from morediffusers.Configurations import NunchakuLoRAsConfiguration
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_NunchakuLoRAsConfiguration_from_yaml_for_empty_file():
    test_file_path = test_data_directory / "empty.yml"
    assert test_file_path.exists()

    configuration = NunchakuLoRAsConfiguration.from_yaml(test_file_path)

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

def test_NunchakuLoRAsConfiguration_inits():
    configuration = NunchakuLoRAsConfiguration()

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

def test_NunchakuLoRAsConfiguration_from_yaml_on_example_file():
    example_file_path = get_project_directory_path() / \
        "Configurations" / "HuggingFace" / "MoreDiffusers" / \
        "nunchaku_loras_configuration.yml.example"
    assert example_file_path.exists()

    configuration = NunchakuLoRAsConfiguration.from_yaml(example_file_path)

    assert configuration.lora_scale == 0.5

    assert len(configuration.loras) == 1
    assert configuration.to_dict() == {
        "lora_scale": 0.5,
        "loras": {
            "lora_1": {
                "directory_path": Path("/Data1/Models/Diffusion/LoRAs/pgc"),
                "filename": \
                    "qint4-SECRET-SAUCE-HERO-V2.1-diffusers.safetensors",
                "lora_strength": 1.5
            }
        }
    }