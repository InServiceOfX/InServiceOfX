from corecode.Utilities import (
    DataSubdirectories,
    )

from morediffusers.Configurations import Configuration

from morediffusers.Wrappers.pipelines import create_flux_pipeline
from morediffusers.Wrappers import change_pipe_to_cuda_or_not

from pathlib import Path

import pytest
import torch

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[3] / "TestData"


def test_create_flux_pipeline_default_init():

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-schnell"

    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-schnell"

    pipe = create_flux_pipeline(
        pretrained_diffusion_model_name_or_path,
        )

    assert pipe.scheduler.config._class_name == "EulerDiscreteScheduler"
    assert pipe.unet.config.time_cond_proj_dim == None

def test_create_flux_pipeline_init():

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-schnell"

    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-schnell"

    pipe = create_flux_pipeline(
        pretrained_diffusion_model_name_or_path,
        torch_dtype=torch.bfloat16,
        variant=None,
        use_safetensors=True,
        is_enable_cpu_offload=False,
        is_enable_sequential_cpu_offload=False
        )

    assert pipe.scheduler.config._class_name == "FlowMatchEulerDiscreteScheduler"

    with pytest.raises(
        AttributeError,
        match="'FluxPipeline' object has no attribute 'unet'"
        ):
        pipe.unet