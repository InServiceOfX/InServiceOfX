from dataclasses import dataclass
from commonapi.Databases import PostgreSQLConnection
from pathlib import Path

import asyncpg
import pytest_asyncio
import yaml

@dataclass
class PostgreSQLDatabaseSetupData:
    def __init__(self):
        configuration_file_path = Path(__file__).parent / \
            "postgresql_test_configuration.yml"

        with open(configuration_file_path, "r") as f:
            configuration = yaml.safe_load(f)
            self.database_port = configuration["database_port"]
            self.ip_address = configuration["ip_address"]
            self.postgres_user = configuration["postgres_user"]
            self.postgres_password = configuration["postgres_password"]


            self.test_dsn = \
                f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.ip_address}:{self.database_port}"

async def cleanup_test_database(dsn: str, db_name: str):
    """Clean up test database by dropping it and terminating all connections."""
    # Connect to default postgres database to drop the test database
    sys_conn = await asyncpg.connect(
        dsn + "/" + PostgreSQLConnection.DEFAULT_SYSTEM_DB)
    try:
        # First terminate all connections to the test database
        await sys_conn.execute(f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = $1
        """, db_name)
        # Then drop the database
        await sys_conn.execute(f'DROP DATABASE IF EXISTS {db_name}')
    finally:
        await sys_conn.close()

@pytest_asyncio.fixture(scope="function")
async def postgres_connection(test_dsn: str, test_db_name: str):
    """Create a PostgreSQLConnection instance for testing."""
    conn = PostgreSQLConnection(test_dsn, test_db_name)
    yield conn
    # Cleanup after test
    await cleanup_test_database(test_dsn, test_db_name)
