from clipdfcolqwenindexer.Core import IndexRunner, ProcessConfigurations


class CLIPDFColQwenIndexer:
    def __init__(self, application_paths):
        self._application_paths = application_paths

        self._process_configurations = ProcessConfigurations(application_paths)
        self._process_configurations.process_configurations()

        from moremineru.Applications import ColQwen2_5Embedder

        self._embedder = ColQwen2_5Embedder(
            self._process_configurations.configurations[
                "colqwen2_5_configuration"
            ]
        )
        self._index_runner = IndexRunner(
            embedder=self._embedder,
            pdf_configuration=self._process_configurations.configurations[
                "pdf_index_configuration"
            ],
        )

    def run(self) -> None:
        try:
            self._index_runner.run()
        finally:
            self._embedder.release()
