from dataclasses import dataclass
from commonapi.Databases import PostgreSQLConnection

import asyncpg
import pytest_asyncio

@dataclass
class PostgreSQLDatabaseSetupData:
    # port number is found in docker-compose.yml file for postgres service.
    DATABASE_PORT = 5432
    # Because we're using docker-compose and if we're connecting from another
    # running Docker container, IP address may not be localhost; try the IP
    # address of host machine.
    IP_ADDRESS = "192.168.86.91"
    # TODO: Have docker-compose.yml have user and password as environment variables.
    POSTGRES_USER = "inserviceofx"
    POSTGRES_PASSWORD = "inserviceofx"

    TEST_DSN = \
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{IP_ADDRESS}:{DATABASE_PORT}"

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
