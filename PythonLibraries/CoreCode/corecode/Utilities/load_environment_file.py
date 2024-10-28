from dotenv import load_dotenv

from corecode.FileIO import get_default_path_to_env_file

import os

def load_environment_file(env_file_path=str(get_default_path_to_env_file())):
	"""
	Run this to load into the environment the variables in the .env file.
	"""
	load_dotenv(env_file_path)

def get_environment_variable(environment_variable_name):
	return os.environ[environment_variable_name]