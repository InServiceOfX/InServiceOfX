# You'll want to keep the imports of this class to be at a minimum because the
# application needs this to start and find modules and configuration files upon

from pathlib import Path
from dataclasses import dataclass
from warnings import warn
import sys

@dataclass
class ApplicationPaths:
    """Path configuration for the application"""
    application_path: Path
    project_path: Path
    inhouse_library_paths: dict[str, Path]
    configuration_file_paths: dict[str, Path]
    logs_file_paths: dict[str, Path]

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
            "MoreDiffusers": \
                project_path / "PythonLibraries" / "HuggingFace" / \
                    "MoreDiffusers",
            "MoreTransformers": \
                project_path / "PythonLibraries" / "HuggingFace" / \
                    "MoreTransformers",
        }

        def create_paths_from_base_path(base_path: Path | str):
            if isinstance(base_path, str):
                base_path = Path(base_path)

            configuration_file_paths = {
                "model_list": \
                    base_path / "Configurations" / "model_list.yml",
                "nunchaku_configuration": \
                    base_path / "Configurations" / "nunchaku_configuration.yml",
                "nunchaku_flux_control_configuration": \
                    base_path / "Configurations" / \
                        "nunchaku_flux_control_configuration.yml",
                "flux_generation_configuration": \
                    base_path / "Configurations" / \
                        "flux_generation_configuration.yml",
                "nunchaku_loras_configuration": \
                    base_path / "Configurations" / \
                        "nunchaku_loras_configuration.yml",
                "pipeline_inputs": \
                    base_path / "Configurations" / "pipeline_inputs.yml",
                "batch_processing_configuration": \
                    base_path / "Configurations" / \
                        "batch_processing_configuration.yml",
                "cli_configuration": \
                    base_path / "Configurations" / "cli_configuration.yml"
            }

            logs_file_paths = {
                "nunchaku_generation_logs": \
                    base_path / "Logs" / "nunchaku_generations.yml"
            }

            return configuration_file_paths, logs_file_paths

        if configpath is not None:
            configuration_file_paths, logs_file_paths = \
                create_paths_from_base_path(configpath)
        elif is_current_path:
            configuration_file_paths, logs_file_paths = \
                create_paths_from_base_path(Path.cwd())
        elif is_development:
            configuration_file_paths, logs_file_paths = \
                create_paths_from_base_path(app_path)
        else:
            config_dir = Path.home() / ".config" / "cliimage"
            configuration_file_paths, logs_file_paths = \
                create_paths_from_base_path(config_dir)

        return cls(
            application_path=app_path,
            project_path=project_path,
            inhouse_library_paths=inhouse_library_paths,
            configuration_file_paths=configuration_file_paths,
            logs_file_paths=logs_file_paths
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

        path = self.inhouse_library_paths["MoreDiffusers"]

        add_path_to_sys_path(path)

        path = self.inhouse_library_paths["MoreTransformers"]

        add_path_to_sys_path(path)

