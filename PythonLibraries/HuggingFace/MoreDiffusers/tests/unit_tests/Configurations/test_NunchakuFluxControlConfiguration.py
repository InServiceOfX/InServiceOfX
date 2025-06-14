from morediffusers.Configurations import NunchakuFluxControlConfiguration
from pathlib import Path

import torch

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
        "/Data1/Models/Diffusion/mit-han-lab/svdq-flux.1-t5")

    assert configuration.nunchaku_flux_model_path == Path(  
        "/Data1/Models/Diffusion/mit-han-lab/svdq-int4-flux.1-depth-dev")

    assert configuration.torch_dtype == torch.bfloat16

    assert configuration.cuda_device == "cuda:0"