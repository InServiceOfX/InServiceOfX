from typing import Any, Dict


class ProcessConfigurations:
    def __init__(self, application_paths):
        self._application_paths = application_paths
        self.configurations: Dict[str, Any] = {}

    def process_configurations(self) -> None:
        # Imports deferred until after ApplicationPaths.add_libraries_to_path()
        # has injected MoreMinerU into sys.path.
        from moremineru.Configurations import Qwen3VLConfiguration
        from clipdfqwen3vlchat.Core.PDFChatConfiguration import (
            PDFChatConfiguration,
        )

        qwen_path = self._application_paths.configuration_file_paths[
            "qwen3vl_configuration"
        ]
        if not qwen_path.exists():
            raise FileNotFoundError(
                f"qwen3vl_configuration.yml not found at {qwen_path}. "
                "Copy qwen3vl_configuration.yml.example and edit."
            )
        self.configurations["qwen3vl_configuration"] = (
            Qwen3VLConfiguration.from_yaml(qwen_path, validate_paths=True)
        )
        print(f"Loaded qwen3vl_configuration from {qwen_path}")

        pdf_path = self._application_paths.configuration_file_paths[
            "pdf_chat_configuration"
        ]
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"pdf_chat_configuration.yml not found at {pdf_path}. "
                "Copy pdf_chat_configuration.yml.example and edit."
            )
        self.configurations["pdf_chat_configuration"] = (
            PDFChatConfiguration.from_yaml(pdf_path)
        )
        print(f"Loaded pdf_chat_configuration from {pdf_path}")
