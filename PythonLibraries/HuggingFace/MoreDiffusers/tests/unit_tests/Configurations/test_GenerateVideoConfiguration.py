from morediffusers.Configurations import GenerateVideoConfiguration
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_GenerateVideoConfiguration_inits_for_empty_values():
    test_file_path = test_data_directory / "generate_video_configuration_empty.yml"
    assert test_file_path.exists()

    configuration = GenerateVideoConfiguration(test_file_path)

    assert configuration.image_path == "/Data/Temporary/temporary_image.jpg" 
    assert isinstance(configuration.image_path, str)

    assert configuration.height == None
    assert configuration.width == None
    assert configuration.num_frames == None
    assert configuration.min_guidance_scale == 1.0
    assert configuration.max_guidance_scale == 3.0
    assert configuration.fps == 7
    assert configuration.motion_bucket_id == 127
    assert configuration.noise_aug_strength == 0.02
    assert configuration.num_videos_per_prompt == 1
    assert configuration.seed == None
    assert configuration.temporary_save_path == "/Data/Temporary"
    assert isinstance(configuration.temporary_save_path, str)
