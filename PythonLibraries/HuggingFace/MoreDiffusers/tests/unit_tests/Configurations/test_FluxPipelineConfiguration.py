from morediffusers.Configurations import FluxPipelineConfiguration

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_FluxPipelineConfiguration_inits_for_empty_values():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    assert test_file_path.exists()

    configuration = FluxPipelineConfiguration(test_file_path)

    assert configuration.diffusion_model_path == None 
    assert configuration.temporary_save_path == None
    assert configuration.scheduler == None
    assert configuration.a1111_kdiffusion == 'None'
    assert configuration.height == None
    assert configuration.width == None
    assert configuration.seed == None
    assert configuration.max_sequence_length == 512
    assert configuration.torch_dtype == None
    assert configuration.is_enable_cpu_offload == None
    assert configuration.is_enable_sequential_cpu_offload == None
    assert configuration.is_to_cuda == None
    assert configuration.variant == None
    assert configuration.use_safetensors == None
