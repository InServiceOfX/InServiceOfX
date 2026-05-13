from clipdfpaddleocrvlchat.Core import ChatRunner, ProcessConfigurations


class CLIPDFPaddleOCRVLChat:
    def __init__(self, application_paths):
        self._process_configurations = ProcessConfigurations(application_paths)
        self._process_configurations.process_configurations()

        from paddleocrvl_api import PaddleOCRVLVLLMClient

        self._client = PaddleOCRVLVLLMClient(
            self._process_configurations.configurations[
                "paddleocrvl_api_configuration"
            ]
        )
        self._runner = ChatRunner(
            client=self._client,
            pdf_configuration=self._process_configurations.configurations[
                "pdf_chat_configuration"
            ],
        )

    def run(self) -> None:
        self._runner.run()
