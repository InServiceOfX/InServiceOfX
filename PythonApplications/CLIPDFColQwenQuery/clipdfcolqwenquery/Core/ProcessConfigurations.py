from typing import Any, Dict


class ProcessConfigurations:
    def __init__(self, application_paths):
        self._application_paths = application_paths
        self.configurations: Dict[str, Any] = {}

    def process_configurations(self) -> None:
        from moremineru.Configurations import ColQwen2_5Configuration
        from clipdfcolqwenquery.Core.PDFQueryConfiguration import (
            PDFQueryConfiguration,
        )

        colqwen_path = self._application_paths.configuration_file_paths[
            "colqwen2_5_configuration"
        ]
        if not colqwen_path.exists():
            raise FileNotFoundError(
                f"colqwen2_5_configuration.yml not found at {colqwen_path}. "
                "Copy colqwen2_5_configuration.yml.example and edit."
            )
        self.configurations["colqwen2_5_configuration"] = (
            ColQwen2_5Configuration.from_yaml(
                colqwen_path,
                validate_paths=True,
            )
        )
        print(f"Loaded colqwen2_5_configuration from {colqwen_path}")

        pdf_path = self._application_paths.configuration_file_paths[
            "pdf_query_configuration"
        ]
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"pdf_query_configuration.yml not found at {pdf_path}. "
                "Copy pdf_query_configuration.yml.example and edit."
            )
        self.configurations["pdf_query_configuration"] = (
            PDFQueryConfiguration.from_yaml(pdf_path)
        )
        print(f"Loaded pdf_query_configuration from {pdf_path}")
