from clipdfcolqwenquery.Core import ProcessConfigurations, QueryRunner


class CLIPDFColQwenQuery:
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
        self._query_runner = QueryRunner(
            embedder=self._embedder,
            query_configuration=self._process_configurations.configurations[
                "pdf_query_configuration"
            ],
        )

    def run(self) -> None:
        try:
            self._query_runner.run()
        finally:
            self._embedder.release()
