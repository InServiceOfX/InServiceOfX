from typing import Any, Dict


class ProcessConfigurations:
    def __init__(self, application_paths):
        self._application_paths = application_paths
        self.configurations: Dict[str, Any] = {}

    def process_configurations(self) -> None:
        from paddleocrvl_api import PaddleOCRVLAPIConfiguration
        from clipdfpaddleocrvlchat.Core.PDFChatConfiguration import (
            PDFChatConfiguration,
        )

        api_path = self._application_paths.configuration_file_paths[
            "paddleocrvl_api_configuration"
        ]
        if not api_path.exists():
            raise FileNotFoundError(
                f"paddleocrvl_api_configuration.yml not found at {api_path}."
            )
        self.configurations["paddleocrvl_api_configuration"] = (
            PaddleOCRVLAPIConfiguration.from_yaml(api_path)
        )
        print(f"Loaded paddleocrvl_api_configuration from {api_path}")

        pdf_path = self._application_paths.configuration_file_paths[
            "pdf_chat_configuration"
        ]
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"pdf_chat_configuration.yml not found at {pdf_path}."
            )
        self.configurations["pdf_chat_configuration"] = (
            PDFChatConfiguration.from_yaml(pdf_path)
        )
        print(f"Loaded pdf_chat_configuration from {pdf_path}")
