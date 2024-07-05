from morediffusers.Configurations import IPAdapterConfiguration
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_IPAdapterConfiguration_inits_for_single_values():
    test_file_path = test_data_directory / "ip_adapter_configuration_single.yml"
    assert test_file_path.exists()

    configuration = IPAdapterConfiguration(test_file_path)

    assert configuration.path == "/Data/Models/Diffusion/h94/IP-Adapter"
    assert configuration.subfolder == "sdxl_models"
    assert configuration.weight_names == "ip-adapter_sdxl.safetensors"
    assert configuration.image_filepath == None
    assert configuration.scales == 0.5

def test_IPAdapterConfiguration_inits_for_faceid():

    test_file_path = test_data_directory / "ip_adapter_configuration_faceid.yml"
    assert test_file_path.exists()

    configuration = IPAdapterConfiguration(test_file_path)

    assert configuration.path == "/Data/Models/Diffusion/h94/IP-Adapter-FaceID"
    assert configuration.subfolder == None
    assert configuration.weight_names == "ip-adapter-faceid-portrait_sdxl_unnorm.bin"
    assert configuration.image_filepath == None
    assert configuration.scales == 0.6
