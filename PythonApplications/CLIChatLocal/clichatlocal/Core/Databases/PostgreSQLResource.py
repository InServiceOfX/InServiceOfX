from clichatlocal.Core import ProcessConfigurations
from tools.Databases.PostgreSQLSetup import PostgreSQLSetup

class PostgreSQLResource:
    def __init__(self, process_configurations: ProcessConfigurations):
        """
        Args:
            process_configurations: This should be the variable name of the
            single ProcessConfigurations object, instance, that we'll
            exclusively use for this application. We'll then create a reference
            to it as a "private" class data member.
        """
        self._process_configurations = process_configurations
        self._pgs_setup = None
        self._pgs_setup_data = None

    def _load_setup_configuration(self):
        if self._process_configurations.configurations is None:
            self._process_configurations.process_configurations()

        self._pgs_setup_data = self._process_configurations.configurations[
            "postgresql_configuration"]

        return self._pgs_setup_data

    async def load_configuration_and_create_pool(self):
        self._pgs_setup = PostgreSQLSetup(self._load_setup_configuration())
        self._pgs_setup.create_connections_for_all_databases()
        await self._pgs_setup.create_pool_for_all_databases()

    def get_connection(self, database_type: str):
        return self._pgs_setup._connections[database_type]
