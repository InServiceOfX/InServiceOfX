from corecode.Utilities import (
    DataSubdirectories,
    )

from diffusers.hooks import apply_group_offloading

from morediffusers.Configurations import DiffusionPipelineConfiguration
from morediffusers.Wrappers.pipelines import create_flux_pipeline

from pathlib import Path

import torch

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

def test_offload_groups_of_internal_layers():
    """
    https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#group-offloading
    """

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"
    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"

    pipe = create_flux_pipeline(configuration)

    apply_group_offloading(
        pipe.transformer,
        offload_type="leaf_level",
        offload_device=torch.device("cpu"),
        onload_device=torch.device(configuration.cuda_device),
        use_stream=True,)
    apply_group_offloading(
        pipe.text_encoder,
        offload_type="leaf_level",
        offload_device=torch.device("cpu"),
        onload_device=torch.device(configuration.cuda_device),
        use_stream=True,)
    apply_group_offloading(
        pipe.text_encoder_2,
        offload_type="leaf_level",
        offload_device=torch.device("cpu"),
        onload_device=torch.device(configuration.cuda_device),
        use_stream=True,)
    apply_group_offloading(
        pipe.vae,
        offload_type="leaf_level",
        offload_device=torch.device("cpu"),
        onload_device=torch.device(configuration.cuda_device),
        use_stream=True,)


    