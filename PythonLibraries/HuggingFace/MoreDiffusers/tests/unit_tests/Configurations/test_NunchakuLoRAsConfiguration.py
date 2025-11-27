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

    assert len(configuration.loras) == 2
    configuration_as_dict = configuration.to_dict()
    assert configuration_as_dict["lora_scale"] == 0.5

    assert len(configuration_as_dict["loras"]) == 2
    assert configuration_as_dict["loras"][0][
        "nickname"] == "Secret Sauce [Hero]-V2.1"
    assert configuration_as_dict["loras"][0][
        "directory_path"] == Path("/Data1/Models/Diffusion/LoRAs/pgc")
    assert configuration_as_dict["loras"][0][
        "filename"] == "qint4-SECRET-SAUCE-HERO-V2.1-diffusers.safetensors"
    assert configuration_as_dict["loras"][0]["lora_strength"] == 1.5
    assert configuration_as_dict["loras"][0]["is_active"] == False
    assert configuration_as_dict["loras"][0][
        "description"] == "secret sauce http://civitai.com/213"
    assert configuration_as_dict["loras"][1][
        "nickname"] == "hero-v2.1"
    assert configuration_as_dict["loras"][1][
        "directory_path"] == Path("/Data1/Models/Diffusion/LoRAs/pgc")
    assert configuration_as_dict["loras"][1][
        "filename"] == "qint4-SECRET-SAUCE-HERO-V2.1-diffusers.safetensors"
    assert configuration_as_dict["loras"][1]["lora_strength"] == 0.9
    assert configuration_as_dict["loras"][1]["is_active"] == True
    assert configuration_as_dict["loras"][1]["description"] == None

def test_NunchakuLoRAsConfiguration_get_valid_loras_works():
    example_file_path = get_project_directory_path() / \
        "Configurations" / "HuggingFace" / "MoreDiffusers" / \
        "nunchaku_loras_configuration.yml.example"
    configuration = NunchakuLoRAsConfiguration.from_yaml(example_file_path)

    valid_loras = configuration.get_valid_loras()
    assert len(valid_loras) == 1
    print("valid_loras: ", valid_loras)
    assert valid_loras[0][0] == Path("/Data1/Models/Diffusion/LoRAs/pgc") / "qint4-SECRET-SAUCE-HERO-V2.1-diffusers.safetensors"
    assert valid_loras[0][1] == 0.9