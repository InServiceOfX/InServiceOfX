from morediffusers.Configurations import StableDiffusionXLGenerationConfiguration
from pathlib import Path

import pytest

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_StableDiffusionXLGenerationConfiguration_fails_for_empty_values():
    test_file_path = test_data_directory / "empty.yml"
    assert test_file_path.exists()

    with pytest.raises(ValueError):
        StableDiffusionXLGenerationConfiguration(test_file_path)

def test_StableDiffusionXLGenerationConfiguration_inits():
    test_file_path = test_data_directory / "sdxl_generation_configuration.yml"
    assert test_file_path.exists()

    configuration = StableDiffusionXLGenerationConfiguration(test_file_path)

    assert configuration.height == 1216
    assert configuration.width == 832
    assert configuration.num_inference_steps == 40
    assert configuration.num_images_per_prompt == None
    assert configuration.seed == 873546738
    assert configuration.clip_skip == 2
    assert str(configuration.temporary_save_path) == str(Path.cwd())

def test_StableDiffusionXLGenerationConfiguration_get_generation_kwargs():
    test_file_path = test_data_directory / "sdxl_generation_configuration.yml"
 
    configuration = StableDiffusionXLGenerationConfiguration(test_file_path)
    kwargs = configuration.get_generation_kwargs()
    assert kwargs == {
        "height": 1216,
        "width": 832,
        "num_inference_steps": 40,
        "guidance_scale": 7.0,
        "clip_skip": 2,
    }