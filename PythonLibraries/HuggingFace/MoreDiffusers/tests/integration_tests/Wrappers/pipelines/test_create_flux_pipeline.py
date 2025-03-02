from corecode.Utilities import (
    DataSubdirectories,
    )

from morediffusers.Configurations import (
    DiffusionPipelineConfiguration,
    FluxGenerationConfiguration)
from morediffusers.Wrappers.pipelines import (
    create_flux_pipeline,
    change_pipe_to_cuda_or_not,
    prepare_flux_generation)
from morediffusers.Wrappers import create_seed_generator

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

def test_running_fp16_inference_as_is():
    """
    https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#running-fp16-inference

    It works, in that much of the work has been offloaded to the CPU, but is
    underutilizing the GPU.
    """
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    configuration.torch_dtype = torch.bfloat16
    configuration.is_enable_sequential_cpu_offload = True
    #configuration.cuda_device = "cuda:0"
    configuration.cuda_device = None

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"

    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path

    pipe = create_flux_pipeline(configuration)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe.to(torch.float16)

    test_file_path = test_data_directory / "flux_generation_configuration_empty.yml"
    generation_configuration = FluxGenerationConfiguration(test_file_path)

    generation_configuration.height = 1216
    generation_configuration.width = 832
    generation_configuration.num_inference_steps = 32
    generation_configuration.seed = 1073625683
    generation_configuration.guidance_scale = 8

    prompt = (
        "sfw, beautiful supermodel eatng a large colorful lollipop, smiling, "
        "wearing a t-shirt with text ‘BUZZ ME IM HAPPY’, blonde hair, BLUE EYES, "
        "hairstyle pigtails, cute face, buzz, seaside walk, sunny, outside bar, "
        "sunset, 8k uhd, dslr, soft lighting, high quality, film grain, shallow "
        "depth of field, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy")

    generation_kwargs = prepare_flux_generation(
        configuration,
        generation_configuration,
        prompt)

    image = pipe(**generation_kwargs).images[0]

    save_path = Path.cwd() / "test_flux_fp16_inference.png"
    image.save(str(save_path))

def test_running_fp16_inference():
    """
    https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#running-fp16-inference

    Faster (6 min vs 8 min on 3060), but still slow and GPU underutilized.
    """
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    configuration.torch_dtype = torch.bfloat16
    configuration.is_enable_sequential_cpu_offload = True
    configuration.cuda_device = "cuda:0"

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"

    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path

    pipe = create_flux_pipeline(configuration)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    change_pipe_to_cuda_or_not(configuration, pipe)

    test_file_path = test_data_directory / "flux_generation_configuration_empty.yml"
    generation_configuration = FluxGenerationConfiguration(test_file_path)

    generation_configuration.height = 1216
    generation_configuration.width = 832
    generation_configuration.num_inference_steps = 32
    generation_configuration.seed = 1073625683
    generation_configuration.guidance_scale = 8

    prompt = (
        "sfw, beautiful supermodel eatng a large colorful lollipop, smiling, "
        "wearing a t-shirt with text ‘BUZZ ME IM HAPPY’, blonde hair, BLUE EYES, "
        "hairstyle pigtails, cute face, buzz, seaside walk, sunny, outside bar, "
        "sunset, 8k uhd, dslr, soft lighting, high quality, film grain, shallow "
        "depth of field, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy")

    generation_kwargs = prepare_flux_generation(
        configuration,
        generation_configuration,
        prompt)

    image = pipe(**generation_kwargs).images[0]

    save_path = Path.cwd() / "test_flux_fp16_inference1.png"
    image.save(str(save_path))
