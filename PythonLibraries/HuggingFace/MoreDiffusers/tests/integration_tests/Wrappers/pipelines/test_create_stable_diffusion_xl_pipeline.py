from morediffusers.Configurations import DiffusionPipelineConfiguration
from morediffusers.Wrappers.pipelines import create_stable_diffusion_xl_pipeline

from pathlib import Path

from diffusers import StableDiffusionXLPipeline

import pytest

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

def test_StableDiffusionXLPipeline_from_pretrained_works():
    test_file_path = test_data_directory / "sdxl_pipeline_configuration.yml"
    assert test_file_path.exists()

    configuration = DiffusionPipelineConfiguration(test_file_path)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        configuration.diffusion_model_path,
        local_files_only=True,
        use_safetensors=True)

def test_StableDiffusionXLPipeline_from_pretrained_fails_with_device_map():
    test_file_path = test_data_directory / "sdxl_pipeline_configuration.yml"
    assert test_file_path.exists()

    configuration = DiffusionPipelineConfiguration(test_file_path)

    # device_map only can be "balanced" because SUPPORTED_DEVICE_MAP only has
    # "balanced".
    with pytest.raises(NotImplementedError):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            configuration.diffusion_model_path,
            local_files_only=True,
            use_safetensors=True,
            device_map="cuda:0")

# For NVIDIA GeForce RTX 3060 even with 12GB VRAM, this fails for
# torch.OutofMemoryError. :(
def test_StableDiffusionXLPipeline_to_specific_device():
    test_file_path = test_data_directory / "sdxl_pipeline_configuration.yml"
    assert test_file_path.exists()

    configuration = DiffusionPipelineConfiguration(test_file_path)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        configuration.diffusion_model_path,
        local_files_only=True,
        use_safetensors=True)

    pipe.to(device="cuda:0")

def test_StableDiffusionXLPipeline_to_specific_device_and_dtype():
    test_file_path = test_data_directory / "sdxl_pipeline_configuration.yml"
    assert test_file_path.exists()

    configuration = DiffusionPipelineConfiguration(test_file_path)

    import torch

    pipe = StableDiffusionXLPipeline.from_pretrained(
        configuration.diffusion_model_path,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16)

    pipe.to(device="cuda:0")

def test_create_stable_diffusion_xl_pipeline_inits():
    test_file_path = test_data_directory / "sdxl_pipeline_configuration.yml"

    configuration = DiffusionPipelineConfiguration(test_file_path)

    pipe = create_stable_diffusion_xl_pipeline(configuration)

    assert isinstance(pipe, StableDiffusionXLPipeline)