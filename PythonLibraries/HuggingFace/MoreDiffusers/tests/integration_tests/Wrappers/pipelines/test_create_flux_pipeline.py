from corecode.Utilities import (
    DataSubdirectories,
    )

from morediffusers.Configurations import DiffusionPipelineConfiguration
from morediffusers.Wrappers.pipelines import create_flux_pipeline

from pathlib import Path

import pytest
import torch

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

def test_create_flux_pipeline_default_init():

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-schnell"

    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-schnell"

    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.is_to_cuda = True
    configuration.cuda_device = "cuda:0"

    pipe = create_flux_pipeline(configuration)

    assert pipe.scheduler.config._class_name == \
        "FlowMatchEulerDiscreteScheduler"
    
    with pytest.raises(
        AttributeError,
        match="'FluxPipeline' object has no attribute 'unet'"):
        pipe.unet.config.time_cond_proj_dim

