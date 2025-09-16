# You'll want to keep the imports of this class to be at a minimum because the
# application needs this to start and find modules and configuration files upon
# running the main function.

from pathlib import Path
from dataclasses import dataclass
import sys
from warnings import warn

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

        def _create_paths_from_base_path(base_path: Path | str):
            if isinstance(base_path, str):
                base_path = Path(base_path)
            configuration_file_paths = {
                "vibe_voice_model_configuration": \
                    base_path / "Configurations" / \
                        "vibe_voice_model_configuration.yml",
                "vibe_voice_configuration": \
                    base_path / "Configurations" / \
                        "vibe_voice_configuration.yml",
                "cli_configuration": \
                    base_path / "Configurations" / "cli_configuration.yml"
            }
            return configuration_file_paths

        if configpath is not None:
            config_dir = Path(configpath).resolve()
            configuration_file_paths = \
                _create_paths_from_base_path(config_dir)
        elif is_current_path:
            current_path = Path.cwd()
            configuration_file_paths = \
                _create_paths_from_base_path(current_path)
        elif is_development:
            configuration_file_paths = \
                _create_paths_from_base_path(app_path)
        else:
            config_dir = Path.home() / ".config" / "clitexttospeech"
            configuration_file_paths = \
                _create_paths_from_base_path(config_dir)

        return cls(
            application_path=app_path,
            project_path=project_path,
            inhouse_library_paths=inhouse_library_paths,
            configuration_file_paths=configuration_file_paths
        )

    def add_libraries_to_path(self):
        def add_path_to_sys_path(path: Path | str):
            if path.exists():
                if not str(path) in sys.path:
                    sys.path.append(str(path))
                    print(f"Added {path} to sys.path")
                else:
                    print(f"{path} already in sys.path")
            else:
                warn(f"{path} does not exist")

        path = self.inhouse_library_paths["CoreCode"]

        add_path_to_sys_path(path)

        path = self.inhouse_library_paths["MoreTransformers"]

        add_path_to_sys_path(path)