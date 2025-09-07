from corecode.Utilities import DataSubdirectories, is_model_there
from morediffusers.Configurations import NunchakuFluxControlConfiguration
from pathlib import Path

import torch

data_subdirectories = DataSubdirectories()

relative_flux_model_path = "Models/Diffusion/black-forest-labs/FLUX.1-dev"

is_flux_model_downloaded, flux_model_path = is_model_there(
    relative_flux_model_path,
    data_subdirectories)

relative_nunchaku_model_path = "Models/Diffusion/nunchaku-tech/nunchaku-flux.1-depth-dev"

is_nunchaku_model_downloaded, nunchaku_model_path = is_model_there(
    relative_nunchaku_model_path,
    data_subdirectories)

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_NunchakuFluxControlConfiguration_loads_from_yaml():
    test_file_path = test_data_directory / \
        "nunchaku_flux_control_configuration.yml"
    assert test_file_path.exists()

    configuration = NunchakuFluxControlConfiguration.from_yaml(test_file_path)

    assert configuration.flux_model_path == Path(
        "/Data1/Models/Diffusion/black-forest-labs/FLUX.1-dev")

    assert configuration.depth_model_path == Path(
        "/Data1/Models/Generation/LiheYoung/depth-anything-large-hf")

    assert configuration.nunchaku_t5_model_path == Path(
        "/Data1/Models/Diffusion/nunchaku-tech/svdq-flux.1-t5")

    assert configuration.nunchaku_flux_model_path == Path(  
        "/Data1/Models/Diffusion/nunchaku-tech/nunchaku-flux.1-depth-dev")

    assert configuration.torch_dtype == torch.bfloat16

    assert configuration.cuda_device == "cuda:0"

def test_NunchakuFluxControlConfiguration_inits():
    configuration = NunchakuFluxControlConfiguration(
        flux_model_path=flux_model_path,
        nunchaku_flux_model_path=nunchaku_model_path,
    )

    assert configuration.flux_model_path == flux_model_path
    assert configuration.depth_model_path is None
    assert configuration.nunchaku_t5_model_path is None
    assert configuration.nunchaku_flux_model_path == nunchaku_model_path
    assert configuration.torch_dtype is None
    assert configuration.cuda_device == 'cuda'
