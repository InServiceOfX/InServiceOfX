from morediffusers.Configurations import LoRAsConfiguration
from pathlib import Path

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


def test_LoRAsConfiguration_inits():
    test_file_path = test_data_directory / \
        "integration_loras_configuration_0.yml"
    assert test_file_path.exists()

    configuration = LoRAsConfiguration(test_file_path)

    assert configuration.lora_scale == 0.9

    assert len(configuration.loras) == 2

    assert "toy" in configuration.loras.keys()
    assert "pixel_art" in configuration.loras.keys()
    assert configuration.loras["toy"]["directory_path"] == \
        "/Data/Models/Diffusion/CiroN2022/toy-face"
    assert configuration.loras["toy"]["weight_name"] == "toy_face_sdxl.safetensors"
    assert configuration.loras["toy"]["adapter_name"] == "toy"
    assert configuration.loras["toy"]["adapter_weight"] == 0.85
    assert configuration.loras["pixel_art"]["directory_path"] == \
        "/Data/Models/Diffusion/nerijs/pixel-art-xl"
    assert configuration.loras["pixel_art"]["weight_name"] == "pixel-art-xl.safetensors"
    assert configuration.loras["pixel_art"]["adapter_name"] == "pixel_art"
    assert configuration.loras["pixel_art"]["adapter_weight"] == None

def test_LoRAsConfiguration_inits_with_sdxl_file():
    test_file_path = test_data_directory / "sdxl_loras_configuration.yml"
    assert test_file_path.exists()

    configuration = LoRAsConfiguration(test_file_path)
    assert configuration.lora_scale == None
    assert len(configuration.loras) == 2
    assert configuration.loras["add_detail"]["directory_path"] == \
        "/Data/Models/Diffusion/LoRAs/w4r10ck"
    assert configuration.loras["add_detail"]["weight_name"] == \
        "add-detail-xl.safetensors"
    assert configuration.loras["add_detail"]["adapter_name"] == "add_detail"
    assert configuration.loras["add_detail"]["adapter_weight"] == 1.0

def test_LoRAsConfiguration_inits_with_sdxl_file_complex():
    test_file_path = test_data_directory / "sdxl_loras_configuration_complex.yml"
    assert test_file_path.exists()

    configuration = LoRAsConfiguration(test_file_path)
    assert configuration.lora_scale == None
    assert len(configuration.loras) == 5
    assert configuration.loras["add_detail"]["directory_path"] == \
        "/Data/Models/Diffusion/LoRAs/w4r10ck"
    assert configuration.loras["add_detail"]["weight_name"] == \
        "add-detail-xl.safetensors"
    assert configuration.loras["add_detail"]["adapter_name"] == "add_detail"
    assert configuration.loras["add_detail"]["adapter_weight"] == 1.0
    assert configuration.loras["detailed_eyes"]["directory_path"] == \
        "/Data/Models/Diffusion/LoRAs/bdsqlsz"
