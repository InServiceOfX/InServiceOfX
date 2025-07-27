from morediffusers.Configurations import CannyDetectorConfiguration
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_CannyDetectorConfiguration_from_yaml():
    test_file_path = test_data_directory / "canny_detector_configuration.yml"
    assert test_file_path.exists()

    configuration = CannyDetectorConfiguration.from_yaml(test_file_path)

    assert configuration.low_threshold == 50
    assert configuration.high_threshold == 100
    assert configuration.detect_resolution == 1024
    assert configuration.image_resolution == 1024
