# You'll want to keep the imports of this class to be at a minimum because the
# application needs this to start and find modules and configuration files upon
# running the main function.

from pathlib import Path
from dataclasses import dataclass

@dataclass
class ApplicationPaths:
    """Path configuration for the application"""
    application_path: Path
    project_path: Path
    inhouse_library_paths: dict[str, Path]
    configuration_file_paths: dict[str, Path]

    @classmethod
    def create(
            cls,
            is_development: bool = False,
            is_current_path: bool = False,
            configpath: str = None) -> 'ApplicationPaths':
        app_path = Path(__file__).resolve().parents[1]
        project_path = app_path.parents[1]
        
        inhouse_library_paths = {
            "CoreCode": \
                project_path / "PythonLibraries" / "CoreCode",
            "MoreTransformers": \
                project_path / "PythonLibraries" / "HuggingFace" / \
                    "MoreTransformers",
        }

        if configpath is not None:
            config_dir = Path(configpath).resolve()
            configuration_file_paths = {
                "cli_configuration": \
                    config_dir / "cli_configuration.yml"
            }
        elif is_current_path:
            current_path = Path.cwd()
            configuration_file_paths = {
                "cli_configuration": \
                    current_path / ".clitexttospeech" / "Configurations" / \
                        "cli_configuration.yml"
            }
        elif is_development:
            configuration_file_paths = {
                "cli_configuration": \
                    app_path / "Configurations" / "cli_configuration.yml"
            }
        else:
            config_dir = Path.home() / ".config" / "clitexttospeech"
            configuration_file_paths = {
                "cli_configuration": \
                    config_dir / "Configurations" / "cli_configuration.yml"
            }

        return cls(
            application_path=app_path,
            project_path=project_path,
            inhouse_library_paths=inhouse_library_paths,
            configuration_file_paths=configuration_file_paths
        )
