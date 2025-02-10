# You'll want to keep the imports of this class to be at a minimum because the
# application needs this to start and find modules and configuration files upon

from pathlib import Path
from dataclasses import dataclass

@dataclass
class ApplicationPaths:
    """Path configuration for the application"""
    application_path: Path
    project_path: Path
    third_party_paths: dict[str, Path]
    environment_file_path: Path
    configuration_file_path: Path
    lumaai_configuration_file_path: Path

    @classmethod
    def create(cls, is_development: bool = False) -> 'ApplicationPaths':
        app_path = Path(__file__).resolve().parents[2]
        project_path = app_path.parents[1]
        
        third_party_paths = {
            "morelumaai": \
                project_path / "PythonLibraries" / "ThirdParties" / \
                    "MoreLumaAI",
            "morefal": \
                project_path / "PythonLibraries" / "ThirdParties" / \
                    "Diffusion" / "MoreFal"
        }

        if is_development:
            environment_file_path = app_path / "Configurations" / ".env"
            configuration_file_path = app_path / "Configurations" / \
                "clivideo_configuration.yml"
            lumaai_configuration_file_path = app_path / "Configurations" / \
                "lumaai_generation_configuration.yml"
        else:
            config_dir = Path.home() / ".config" / "clivideo"
            environment_file_path = config_dir / "Configurations" / ".env"
            configuration_file_path = config_dir / "Configurations" / \
                "clivideo_configuration.yml"
            lumaai_configuration_file_path = config_dir / "Configurations" / \
                "lumaai_generation_configuration.yml"

        return cls(
            application_path=app_path,
            project_path=project_path,
            third_party_paths=third_party_paths,
            environment_file_path=environment_file_path,
            configuration_file_path=configuration_file_path,
            lumaai_configuration_file_path=lumaai_configuration_file_path
        )
