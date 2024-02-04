from CoreCode.Utilities.ConfigurePaths import (_setup_paths)
from CoreCode.Utilities.LoadEnvironmentFile import (load_environment_file)

import os
import pytest

def test_load_environment_file_loads_example():

	basic_project_paths = _setup_paths()	

	load_environment_file(str(basic_project_paths.project_path / ".envExample"))

	api_key = os.getenv('OPEN_AI_API_KEY')

	assert api_key == "your_api_key_here"