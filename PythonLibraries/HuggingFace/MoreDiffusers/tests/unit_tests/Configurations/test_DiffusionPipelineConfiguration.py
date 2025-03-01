from morediffusers.Configurations import DiffusionPipelineConfiguration
from pathlib import Path

import pytest
import torch

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_DiffusionPipelineConfiguration_fails_for_empty_values():
    test_file_path = test_data_directory / "empty.yml"
    assert test_file_path.exists()

    with pytest.raises(ValueError):
        configuration = DiffusionPipelineConfiguration(test_file_path)

def test_DiffusionPipelineConfiguration_inits_with_diffusion_model_path():
    test_file_path = test_data_directory / \
        "sdxl_pipeline_configuration_minimal.yml"
    assert test_file_path.exists()

    configuration = DiffusionPipelineConfiguration(test_file_path)

    assert configuration.diffusion_model_path == Path(
        "/Data1/Models/Diffusion/fluently/Fluently-XL-Final")

    assert configuration.torch_dtype == None

    assert configuration.is_enable_model_cpu_offload == None
    assert configuration.is_enable_sequential_cpu_offload == None
    assert configuration.is_to_cuda == None

    assert configuration.scheduler == None
    assert configuration.a1111_kdiffusion == None

def test_DiffusionPipelineConfiguration_get_pretrained_kwargs():
    test_file_path = test_data_directory / \
        "sdxl_pipeline_configuration_minimal.yml"
    assert test_file_path.exists()

    configuration = DiffusionPipelineConfiguration(test_file_path)
    kwargs = configuration.get_pretrained_kwargs()
    assert kwargs == {}

    test_file_path = test_data_directory / "sdxl_pipeline_configuration.yml"
    assert test_file_path.exists()

    configuration = DiffusionPipelineConfiguration(test_file_path)
    kwargs = configuration.get_pretrained_kwargs()
    assert kwargs == {
        "torch_dtype": torch.bfloat16,
    }