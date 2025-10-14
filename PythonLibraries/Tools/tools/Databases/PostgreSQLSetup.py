from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tools.Databases import PostgreSQLConnection
from typing import AsyncGenerator, Dict, Optional
import yaml

@dataclass
class PostgreSQLSetupData:
    database_port: int
    ip_address: str
    postgres_user: str
    postgres_password: str
    database_names: Dict[str, str]

    @classmethod
    def from_yaml(cls, yaml_path: Path | str):
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        
        return cls(
            database_port=config["database_port"],
            ip_address=config["ip_address"],
            postgres_user=config["postgres_user"],
            postgres_password=config["postgres_password"],
            database_names=config["database_names"]
        )

    @classmethod
    def from_default_values(cls):
        database_names = {"PermanentConversation": "permanent_conversation"}

        return cls(
            database_port=5432,
            ip_address="local-llm-full-postgres",
            postgres_user="inserviceofx",
            postgres_password="inserviceofx",
            database_names=database_names
        )        

    def get_dsn(self) -> str:
        return \
            f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.ip_address}:{self.database_port}"

    def get_database_name(self, database_type: str) -> str:
        """Get database name by type."""
        if database_type not in self.database_names:
            raise KeyError(
                f"Database type '{database_type}' not found in configuration")
        return self.database_names[database_type]

    def list_database_types(self) -> list[str]:
        """List all available database types."""
        return list(self.database_names.keys())

class PostgreSQLSetup:
    def __init__(self, setup_data: PostgreSQLSetupData):
        self._setup_data = setup_data
        self._connections: Dict[str, PostgreSQLConnection] = {}

    def _get_dsn(self) -> str:
        return self._setup_data.get_dsn()

    def create_postgresql_connection(
            self, 
            database_type: Optional[str] = None):
        """Create a PostgreSQL connection for a specific database type. If no
        database type is specified, then we do not make a connection for a
        specific database."""
        if database_type is not None:
            database_name = self._setup_data.get_database_name(database_type)
        else:
            database_name = None

        if database_type is not None:
            self._connections[database_type] = \
                PostgreSQLConnection(
                    self._get_dsn(),
                    database_name)
        else:
            connection = PostgreSQLConnection(
                self._get_dsn(),
                database_name)
            return connection

    def create_connections_for_all_databases(self):
        for database_type in self._setup_data.database_names:
            self.create_postgresql_connection(database_type)

    async def create_pool_for_database(self, database_type: str):
        """
        Note that this function creates the extension for "vector"; if there is
        another asynchronous function that is also creating this extension, the
        2 processes could "collide" and result in this error:
        Error checking extension vector: cannot perform operation: another operation is in progress
        Error creating extension vector: cannot perform operation: another operation is in progress
        """
        if database_type not in self._connections:
            raise ValueError(
                f"Database type '{database_type}' not found in connections")

        try:
            database_name = self._setup_data.database_names[database_type]
        except KeyError as err:
            raise KeyError(
                f"Database type '{database_type}' not found in configuration;" + \
                    str(err))

        connection = self._connections[database_type]

        if not await connection.database_exists(
            database_name):
            await connection.create_database(database_name)

        await connection.create_new_pool(database_name)
        # We assume here that all databases should have the "vector" extension.
        # But beware if another asynchronous function is running that is trying
        # to create this extension, you could get this error:
        # Error checking extension vector: cannot perform operation: another operation is in progress
        # Error creating extension vector: cannot perform operation: another operation is in progress
        await connection.create_extension("vector")

    async def create_pool_for_all_databases(self):
        for database_type in self._setup_data.database_names:
            await self.create_pool_for_database(database_type)
