from clipdfextraction.Core import ExtractionRunner, ProcessConfigurations


class CLIPDFExtraction:
    def __init__(self, application_paths):
        self._application_paths = application_paths

        self._process_configurations = ProcessConfigurations(application_paths)
        self._process_configurations.process_configurations()

        # Imported here so ApplicationPaths.add_libraries_to_path() has already
        # made MoreMinerU importable.
        from moremineru.Applications import MinerU2_5ProVLLM

        self._mineru_runner = MinerU2_5ProVLLM(
            self._process_configurations.configurations["mineru_configuration"]
        )

        self._extraction_runner = ExtractionRunner(
            mineru_runner=self._mineru_runner,
            pdf_configuration=self._process_configurations.configurations[
                "pdf_extraction_configuration"
            ],
        )

    def run(self) -> None:
        try:
            self._extraction_runner.run()
        finally:
            self._mineru_runner.release()
