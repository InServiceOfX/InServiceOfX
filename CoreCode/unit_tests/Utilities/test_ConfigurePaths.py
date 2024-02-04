from CoreCode.Utilities.ConfigurePaths import (
	_setup_paths,
	default_path_to_env_file)

import pytest

def test_setup_paths_contains_expected_subdirectores():

	basic_project_paths = _setup_paths()

	assert "ConfigurePaths" in str(basic_project_paths.configure_paths_path)
	assert "Utilities" in str(basic_project_paths.configure_paths_path)
	assert "CoreCode" in str(basic_project_paths.configure_paths_path)
	assert "InServiceOfX" in str(basic_project_paths.configure_paths_path)

	assert "ConfigurePaths" not in str(
		basic_project_paths.core_code_python_path)
	assert "Utilities" not in str(basic_project_paths.core_code_python_path)
	assert "CoreCode" in str(basic_project_paths.core_code_python_path)
	assert "InServiceOfX" in str(basic_project_paths.core_code_python_path)

	assert "ConfigurePaths" not in str(
		basic_project_paths.core_code_path)
	assert "Utilities" not in str(basic_project_paths.core_code_path)
	assert "CoreCode" in str(basic_project_paths.core_code_path)
	assert "InServiceOfX" in str(basic_project_paths.core_code_path)

	assert "ConfigurePaths" not in str(basic_project_paths.project_path)
	assert "Utilities" not in str(basic_project_paths.project_path)
	assert "CoreCode" not in str(basic_project_paths.project_path)
	assert "InServiceOfX" in str(basic_project_paths.project_path)

def test_default_path_to_env_file_contains_expected_substrings():

	env_file_path = default_path_to_env_file()

	assert "InServiceOfX" in str(env_file_path)
	assert ".env" in str(env_file_path)