from dotenv import load_dotenv
from pathlib import Path

import os

def load_environment_file(env_file_path=Path.cwd() / ".env"):
	"""
	Run this to load into the environment the variables in the .env file.
	"""
	load_dotenv(env_file_path)

def get_environment_variable(environment_variable_name):
	return os.environ[environment_variable_name]