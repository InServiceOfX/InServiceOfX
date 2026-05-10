from pathlib import Path
from typing import Any, Dict


class ProcessConfigurations:
    def __init__(self, application_paths):
        self._application_paths = application_paths
        self.configurations: Dict[str, Any] = {}

    def process_configurations(self) -> None:
        # Imports deferred until after ApplicationPaths.add_libraries_to_path()
        # has injected MoreMinerU into sys.path.
        from moremineru.Configurations import MinerUConfiguration
        from clipdfextraction.Core.PDFExtractionConfiguration import (
            PDFExtractionConfiguration,
        )

        mineru_path = self._application_paths.configuration_file_paths[
            "mineru_configuration"
        ]
        if not mineru_path.exists():
            raise FileNotFoundError(
                f"mineru_configuration.yml not found at {mineru_path}. "
                "Copy mineru_configuration.yml.example and edit."
            )
        self.configurations["mineru_configuration"] = (
            MinerUConfiguration.from_yaml(mineru_path, validate_paths=True)
        )
        print(f"Loaded mineru_configuration from {mineru_path}")

        pdf_path = self._application_paths.configuration_file_paths[
            "pdf_extraction_configuration"
        ]
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"pdf_extraction_configuration.yml not found at {pdf_path}. "
                "Copy pdf_extraction_configuration.yml.example and edit."
            )
        self.configurations["pdf_extraction_configuration"] = (
            PDFExtractionConfiguration.from_yaml(pdf_path)
        )
        print(f"Loaded pdf_extraction_configuration from {pdf_path}")
