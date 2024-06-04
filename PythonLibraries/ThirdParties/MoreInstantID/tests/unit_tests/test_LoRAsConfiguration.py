from moreinstantid.LoRAsConfiguration import LoRAsConfiguration
from pathlib import Path
import pytest

test_data_directory = Path(__file__).resolve().parent.parent / "TestData"

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