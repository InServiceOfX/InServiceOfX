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

    @classmethod
    def create(cls, is_development: bool = False) -> 'ApplicationPaths':
        app_path = Path(__file__).resolve().parents[2]
        project_path = app_path.parents[1]
        
        third_party_paths = {
            "moresglang": \
                project_path / "PythonLibraries" / "ThirdParties" / "MoreSGLang"
        }

        if is_development:
            environment_file_path = app_path / "Configurations" / ".env"
            configuration_file_path = app_path / "Configurations" / \
                "clichatlocal_configuration.yml"
        else:
            config_dir = Path.home() / ".config" / "clichatlocal"
            environment_file_path = config_dir / "Configurations" / ".env"
            configuration_file_path = config_dir / "Configurations" / \
                "clichatlocal_configuration.yml"

        return cls(
            application_path=app_path,
            project_path=project_path,
            third_party_paths=third_party_paths,
            environment_file_path=environment_file_path,
            configuration_file_path=configuration_file_path
        )
