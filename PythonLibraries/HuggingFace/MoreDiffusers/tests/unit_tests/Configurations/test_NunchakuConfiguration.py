from corecode.FileIO import get_project_directory_path
from morediffusers.Configurations import NunchakuConfiguration
from pathlib import Path

import torch

configurations_path = get_project_directory_path() / "Configurations" / \
    "HuggingFace" / "MoreDiffusers"


def test_NunchakuConfiguration_loads_from_yaml():
    test_file_path = configurations_path / "nunchaku_configuration.yml"
    assert test_file_path.exists()

    configuration = NunchakuConfiguration.from_yaml(test_file_path)

    assert configuration.flux_model_path == Path(
        "/Data1/Models/Diffusion/black-forest-labs/FLUX.1-dev")

    assert configuration.nunchaku_t5_model_path == Path(
        "/Data1/Models/Diffusion/mit-han-lab/svdq-flux.1-t5")

    assert configuration.nunchaku_model_path == Path(
        "/Data/Models/Diffusion/jib-mix-svdq")

    assert configuration.torch_dtype == torch.bfloat16

    assert configuration.cuda_device == "cuda:0"

    assert configuration.get_cuda_device_index() == 0