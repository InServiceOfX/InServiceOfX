from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict

import yaml

@dataclass
class BuildDockerConfigurationData:
    docker_image_name: str
    base_image: str
    build_args: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize the data after initialization."""
        # Ensure build_args is always a dict, even if None was passed
        if self.build_args is None:
            self.build_args = {}

class BuildDockerConfiguration:
    DEFAULT_FILE_NAME = "build_configuration.yml"
    DEFAULT_FILE_DIR = Path(__file__).parent

    @staticmethod
    def load_data(file_path: Optional[Union[Path, str]] = None) \
        -> BuildDockerConfigurationData:
        """
        Load build configuration data from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file. If None, uses the default
                      file in the DEFAULT_FILE_DIR.
        
        Returns:
            BuildDockerConfigurationData: The loaded configuration data.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If required fields are missing or invalid.
        """
        if file_path is None:
            file_path = (
                BuildDockerConfiguration.DEFAULT_FILE_DIR / 
                BuildDockerConfiguration.DEFAULT_FILE_NAME
            )
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Configuration file '{file_path}' does not exist."
            )

        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        if data is None:
            data = {}

        # Validate required fields
        if "docker_image_name" not in data:
            raise ValueError("Missing required field: 'docker_image_name'")
        if "base_image" not in data:
            raise ValueError("Missing required field: 'base_image'")

        # Extract build_args if present, default to empty dict
        build_args = data.pop("build_args", {})
        if build_args is None:
            build_args = {}

        return BuildDockerConfigurationData(
            docker_image_name=data["docker_image_name"],
            base_image=data["base_image"],
            build_args=build_args
        )