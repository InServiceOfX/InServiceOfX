from clipdfqwen3vlchat.Core import ChatRunner, ProcessConfigurations


class CLIPDFQwen3VLChat:
    def __init__(self, application_paths):
        self._application_paths = application_paths

        self._process_configurations = ProcessConfigurations(application_paths)
        self._process_configurations.process_configurations()

        # Imported here so ApplicationPaths.add_libraries_to_path() has already
        # made MoreMinerU importable.
        from moremineru.Applications import Qwen3VLVLLM

        self._qwen3vl_runner = Qwen3VLVLLM(
            self._process_configurations.configurations[
                "qwen3vl_configuration"
            ]
        )

        self._chat_runner = ChatRunner(
            qwen3vl_runner=self._qwen3vl_runner,
            pdf_configuration=self._process_configurations.configurations[
                "pdf_chat_configuration"
            ],
        )

    def run(self) -> None:
        try:
            self._chat_runner.run()
        finally:
            self._qwen3vl_runner.release()
