from dotenv import load_dotenv
from pathlib import Path
import os
from warnings import warn

try:
    from corecode.FileIO import get_project_directory_path
    CORECODE_AVAILABLE = True
except ImportError:
    CORECODE_AVAILABLE = False
    warn(
        "corecode is not available. This may cause issues with the path to the "
        "environment file.",
        UserWarning
    )

class LoadMoreXEnvironment:
    RELATIVE_PATH_TO_ENV_FILE = \
        "Configurations/ThirdParties/APIs/MoreX/morex.env"

    ENVIRONMENT_VARIABLE_NAMES = [
        "X_CONSUMER_API_KEY",
        "X_CONSUMER_SECRET",
        "X_BEARER_TOKEN",
        "X_ACCESS_TOKEN",
        "X_SECRET_TOKEN",
    ]

    def __init__(self, path_to_env_file: str | Path | None = None):
        if path_to_env_file is None:
            if CORECODE_AVAILABLE:
                path_to_env_file = get_project_directory_path() / \
                    LoadMoreXEnvironment.RELATIVE_PATH_TO_ENV_FILE
            else:
                path_to_env_file = Path(__file__).resolve().parent / \
                    "Configurations" / "morex.env"
        elif isinstance(path_to_env_file, str):
            path_to_env_file = Path(path_to_env_file)

        self._path_to_env_file = path_to_env_file

    def __call__(self):
        load_dotenv(self._path_to_env_file)

    def get_environment_variable(self, variable_name):
        if variable_name not in self.ENVIRONMENT_VARIABLE_NAMES:
            raise ValueError(
                f"Variable name {variable_name} not found in {self.ENVIRONMENT_VARIABLE_NAMES}")
        return os.environ[variable_name]