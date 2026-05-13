from dataclasses import dataclass
from pathlib import Path
from warnings import warn
import sys


@dataclass
class ApplicationPaths:
    application_path: Path
    project_path: Path
    inhouse_library_paths: dict[str, Path]
    configuration_file_paths: dict[str, Path]

    @classmethod
    def create(
        cls,
        is_development: bool = False,
        is_current_path: bool = False,
        configpath: str | None = None,
    ) -> "ApplicationPaths":
        app_path = Path(__file__).resolve().parents[1]
        project_path = app_path.parents[1]

        inhouse_library_paths = {
            "CoreCode": project_path / "PythonLibraries" / "CoreCode",
            "MoreMinerU": (
                project_path / "PythonLibraries" / "HuggingFace" / "MoreMinerU"
            ),
        }

        def configuration_file_paths_from_base(
            base_path: Path | str,
        ) -> dict[str, Path]:
            base = Path(base_path) if isinstance(base_path, str) else base_path
            return {
                "colqwen2_5_configuration": (
                    base / "Configurations" / "colqwen2_5_configuration.yml"
                ),
                "pdf_index_configuration": (
                    base / "Configurations" / "pdf_index_configuration.yml"
                ),
            }

        if configpath is not None:
            configuration_file_paths = configuration_file_paths_from_base(
                configpath
            )
        elif is_current_path:
            configuration_file_paths = configuration_file_paths_from_base(
                Path.cwd()
            )
        elif is_development:
            configuration_file_paths = configuration_file_paths_from_base(
                app_path
            )
        else:
            config_dir = Path.home() / ".config" / "clipdfcolqwenindexer"
            configuration_file_paths = configuration_file_paths_from_base(
                config_dir
            )

        return cls(
            application_path=app_path,
            project_path=project_path,
            inhouse_library_paths=inhouse_library_paths,
            configuration_file_paths=configuration_file_paths,
        )

    def add_libraries_to_path(self) -> None:
        for name, path in self.inhouse_library_paths.items():
            if not path.exists():
                warn(f"{name} library path does not exist: {path}")
                continue
            if str(path) not in sys.path:
                sys.path.append(str(path))
                print(f"Added {path} to sys.path")
