from morediffusers.Configurations import StableDiffusionXLGenerationConfiguration
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_StableDiffusionXLGenerationConfiguration_inits_with_default_path():
    configuration = StableDiffusionXLGenerationConfiguration()
    assert configuration.height == 1216
    assert configuration.width == 832
    assert configuration.num_inference_steps == 30
    assert configuration.num_images_per_prompt == None
    assert configuration.seed == 458766893
    assert configuration.guidance_scale == 5.5
    assert configuration.clip_skip == 2
    assert str(configuration.temporary_save_path) == "/Data/Private"