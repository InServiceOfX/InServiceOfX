from morediffusers.Configurations import VideoConfiguration
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_GenerateVideoConfiguration_inits_for_empty_values():
    test_file_path = test_data_directory / "video_configuration_empty.yml"
    assert test_file_path.exists()

    configuration = VideoConfiguration(test_file_path)

    assert configuration.diffusion_model_path == None 
    assert configuration.scheduler == None
    assert configuration.torch_dtype == None
    assert configuration.is_enable_cpu_offload == None
    assert configuration.is_enable_sequential_cpu_offload == None
    assert configuration.is_to_cuda == None
    assert configuration.variant == None
    assert configuration.use_safetensors == None
