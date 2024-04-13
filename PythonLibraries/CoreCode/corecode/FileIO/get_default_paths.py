from corecode.FileIO.get_project_directory_path import get_project_directory_path

def get_default_path_to_config_file():
	return get_project_directory_path() / ".config"

def get_default_path_to_env_file():
	return get_project_directory_path() / ".env"