# You'll want to keep the imports of this class to be at a minimum because the
# application needs this to start and find modules and configuration files upon

from pathlib import Path
from dataclasses import dataclass

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
    def create(cls, is_development: bool = False) -> 'ApplicationPaths':
        app_path = Path(__file__).resolve().parents[1]
        project_path = app_path.parents[1]
        
        inhouse_library_paths = {
            "moresglang": \
                project_path / "PythonLibraries" / "ThirdParties" / "MoreSGLang",
            "CommonAPI": \
                project_path / "PythonLibraries" / "ThirdParties" / "APIs" / \
                    "CommonAPI",
            "MoreTransformers": \
                project_path / "PythonLibraries" / "HuggingFace" / \
                    "MoreTransformers",
            "CoreCode": \
                project_path / "PythonLibraries" / "CoreCode"
        }

        if is_development:
            configuration_file_paths = {
                "llama3_configuration": \
                    app_path / "Configurations" / "llama3_configuration.yml",
                "llama3_generation_configuration": \
                    app_path / "Configurations" / \
                        "llama3_generation_configuration.yml"
            }

            system_messages_file_path = \
                app_path / "Configurations" / "system_messages.json"
            conversations_file_path = \
                app_path / "Configurations" / "conversations.json"

        else:
            config_dir = Path.home() / ".config" / "clichatlocal"
            configuration_file_paths = {
                "llama3_configuration": \
                    config_dir / "Configurations" / "llama3_configuration.yml",
                "llama3_generation_configuration": \
                    config_dir / "Configurations" / \
                        "llama3_generation_configuration.yml"
            }

            system_messages_file_path = \
                config_dir / "system_messages.json"
            conversations_file_path = \
                config_dir / "conversations.json"

        return cls(
            application_path=app_path,
            project_path=project_path,
            inhouse_library_paths=inhouse_library_paths,
            configuration_file_paths=configuration_file_paths,
            system_messages_file_path=system_messages_file_path,
            conversations_file_path=conversations_file_path
        )
