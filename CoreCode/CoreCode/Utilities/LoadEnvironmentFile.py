from dotenv import load_dotenv
from .ConfigurePaths import default_path_to_env_file

def load_environment_file(env_file_path=str(default_path_to_env_file())):
	load_dotenv(env_file_path)