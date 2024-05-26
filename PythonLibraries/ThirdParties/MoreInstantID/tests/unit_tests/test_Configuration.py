from moreinstantid.Configuration import Configuration
from pathlib import Path
import pytest

test_data_directory = Path(__file__).resolve().parent.parent / "TestData"

def test_Configuration_inits():
    test_file_path = test_data_directory / "configuration.yml"
    assert test_file_path.exists()

    configuration = Configuration(test_file_path)

    assert configuration.face_analysis_model_name == "buffalo_l"
    assert configuration.face_analysis_model_directory_path == \
        "/Data/Models/Diffusion/InstantX"
    assert configuration.det_size == 640
    assert configuration.scheduler == None

def test_Configuration_inits_on_abnormal_yaml_file():
    test_file_path = test_data_directory / "not_normal_configuration.yml"
    assert test_file_path.exists()

    configuration = Configuration(test_file_path)

    assert configuration.face_analysis_model_name == "antelopev2"
    assert configuration.face_analysis_model_directory_path == ""
    assert configuration.diffusion_model_path == "diff"
    assert configuration.control_net_model_path == \
        "/Data/Models/Diffusion/InstantX/InstantID/ControlNetModel"
    assert configuration.ip_adapter_path == \
        "/Data/Models/Diffusion/h94/IP-Adapter-FaceID/ip-adapter-faceid-portrait_sdxl_unnorm.bin"
    assert configuration.face_image_path == \
        "/Data/Public/Images/LennaSjööblom/Lenna_(test_image).png"
    assert configuration.pose_image_path == None
    assert configuration.scheduler == "UniPCMultistepScheduler"