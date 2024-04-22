from moreinstantid.Configuration import Configuration
from pathlib import Path
import pytest

test_data_directory = Path(__file__).resolve().parent.parent / "TestData"

def test_Configuration_inits():
	test_file_path = test_data_directory / "configuration.yml"
	assert test_file_path.exists()

	configuration = Configuration(test_file_path)

	assert configuration.face_analysis_model_name == "buffalo_l"
	assert configuration.face_analysis_directory_path == \
		"/Data/Models/Diffusion/InstantX"
