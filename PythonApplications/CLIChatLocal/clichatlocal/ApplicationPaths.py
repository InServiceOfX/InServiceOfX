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
    system_messages_file_path: Path
    conversations_file_path: Path

    @classmethod
    def create(
            cls,
            is_development: bool = False,
            is_current_path: bool = False,
            configpath: str = None) -> 'ApplicationPaths':
        """
        # Update to add a Python library, module to sys.path

        For each Python library, module you want to be able to import in, make
        sure to update inhouse_library_paths directly below and then
        add_library_to_paths function that will actually do the appending of the
        path to this Python library into the sys.path at runtime.

        Finally, you'll want to update ProcessConfigurations, under the function
        process_configurations 

        # Update to add a configuration file (typically YAML)

        Update configuration_file_paths local variable inside
        create_paths_from_base_path

        Finally, you'll want to update ProcessConfigurations, in
        process_configurations
        """
        app_path = Path(__file__).resolve().parents[1]
        project_path = app_path.parents[1]
        
        inhouse_library_paths = {
            "CommonAPI": \
                project_path / "PythonLibraries" / "ThirdParties" / "APIs" / \
                    "CommonAPI",
            "CoreCode": \
                project_path / "PythonLibraries" / "CoreCode",
            "Embeddings": \
                project_path / "PythonLibraries" / "Embeddings",
            "MoreSGLang": \
                project_path / "PythonLibraries" / "ThirdParties" / "APIs" / \
                    "MoreSGLang",
            "MoreTransformers": \
                project_path / "PythonLibraries" / "HuggingFace" / \
                    "MoreTransformers",
            "Tools": \
                project_path / "PythonLibraries" / "Tools"
        }

        def create_paths_from_base_path(base_path: Path | str):
            if isinstance(base_path, str):
                base_path = Path(base_path)

            configuration_file_paths = {
                "model_list": \
                    base_path / "Configurations" / "model_list.yml",
                "from_pretrained_model": \
                    base_path / "Configurations" / "from_pretrained_model.yml",
                "from_pretrained_tokenizer": \
                    base_path / "Configurations" / "from_pretrained_tokenizer.yml",
                "generation": \
                    base_path / "Configurations" / \
                        "generation_configuration.yml",
                "cli_configuration": \
                    base_path / "Configurations" / "cli_configuration.yml",
                "postgresql_configuration": base_path / "Configurations" / \
                    "postgresql_configuration.yml",
                "embedding_models_configuration": base_path / \
                    "Configurations" / "embedding_models_configuration.yml"
            }

            system_messages_file_path = \
                base_path / "Configurations" / "system_messages.json"
            conversations_file_path = \
                base_path / "Configurations" / "conversations.json"

            return (
                configuration_file_paths,
                system_messages_file_path,
                conversations_file_path)

        if configpath is not None:
            configuration_file_paths, system_messages_file_path, conversations_file_path = \
                create_paths_from_base_path(configpath)
        elif is_current_path:
            configuration_file_paths, system_messages_file_path, conversations_file_path = \
                create_paths_from_base_path(Path.cwd())
        elif is_development:
            configuration_file_paths, system_messages_file_path, conversations_file_path = \
                create_paths_from_base_path(app_path)
        else:
            config_dir = Path.home() / ".config" / "clichatlocal"
            configuration_file_paths, system_messages_file_path, conversations_file_path = \
                create_paths_from_base_path(config_dir)

        return cls(
            application_path=app_path,
            project_path=project_path,
            inhouse_library_paths=inhouse_library_paths,
            configuration_file_paths=configuration_file_paths,
            system_messages_file_path=system_messages_file_path,
            conversations_file_path=conversations_file_path
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

        path = self.inhouse_library_paths["CommonAPI"]

        add_path_to_sys_path(path)

        path = self.inhouse_library_paths["CoreCode"]

        add_path_to_sys_path(path)

        path = self.inhouse_library_paths["Embeddings"]

        add_path_to_sys_path(path)

        path = self.inhouse_library_paths["MoreSGLang"]

        add_path_to_sys_path(path)

        path = self.inhouse_library_paths["MoreTransformers"]

        add_path_to_sys_path(path)

        path = self.inhouse_library_paths["Tools"]

        add_path_to_sys_path(path)

    @staticmethod
    def _create_missing_file(path: Path) -> bool:
        """
        Create file for specified path if it doesn't exist.

        Args:
            path: Path to create file for

        Returns:
            True if file was created, False if it already existed.
        """
        if path.exists():
            return False
        else:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create the file if it doesn't exist.
            path.touch()
            return True

    def create_missing_system_messages_file(self) -> bool:
        return self._create_missing_file(self.system_messages_file_path)

