"""
USAGE:
You have to have a PostgreSQL database running; I suggest using
docker-compose.yml in
InServiceOfX/Scripts/DockerBuilds/Builds/LLM/LocalLLMFull/Databases/docker-compose.yml.example

Do (if you want to use GPU #1 out of a 2 GPU setup; if you only have 1 GPU use
--gpu 0)
InServiceOfX/Scripts/DockerBuilds/Builds/LLM/LocalLLMFull$ python ../../../CommonFiles/RunDocker.py --gpu 1 .

You can use pytest to run individual tests:
root@5a53362ce66a:/InServiceOfX/PythonLibraries/ThirdParties/APIs/MorePydanticAI/tests# 
    pytest -s -v ./integration_tests/Database/test_PostgreSQLConnection.py::test_list_all_databases
"""
import pytest
import pytest_asyncio
import asyncpg
from morepydanticai.Database import PostgreSQLConnection

# port number is found in docker-compose.yml file for postgres service.
DATABASE_PORT = 5432
# Because we're using docker-compose and if we're connecting from another
# running Docker container, IP address may not be localhost; try the IP address
# of host machine.
IP_ADDRESS = "192.168.86.201"
# TODO: Have docker-compose.yml have user and password as environment variables.
POSTGRES_USER = "inserviceofx"
POSTGRES_PASSWORD = "mypassword"
# Connect to default postgres database
TEST_DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{IP_ADDRESS}:{DATABASE_PORT}"
TEST_DB_NAME = "test_pydantic_ai_database"


@pytest_asyncio.fixture(scope="function")
async def postgres_connection(db_name: str = TEST_DB_NAME):
    """Create a PostgreSQLConnection instance for testing."""
    conn = PostgreSQLConnection(TEST_DSN, db_name)
    yield conn
    # Cleanup after test
    await cleanup_test_database(db_name)

async def cleanup_test_database(db_name: str = TEST_DB_NAME):
    """Clean up test database by dropping it and terminating all connections."""
    # Connect to default postgres database to drop the test database
    sys_conn = await asyncpg.connect(
        TEST_DSN + "/" + PostgreSQLConnection.DEFAULT_SYSTEM_DB)
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

@pytest.mark.asyncio
async def test_cleanup_test_database():
    database_name = TEST_DB_NAME
    await cleanup_test_database(database_name)

    postgres_connection = PostgreSQLConnection(TEST_DSN, database_name)
    assert await postgres_connection.database_exists(database_name) is False, \
        f"Database {database_name} should not exist!"

@pytest.mark.asyncio
async def test_list_all_databases():
    postgres_connection = PostgreSQLConnection(TEST_DSN, TEST_DB_NAME)

    databases = await postgres_connection.list_all_databases()
    for db in databases:
        print(f"Database: {db}")

    assert "postgres" in databases, \
        "postgres database should be in the list of databases"

    assert postgres_connection._connection is None, \
        "Connection should be None since the class member wasn't set"

@pytest.mark.asyncio
async def test_database_exists():
    postgres_connection = PostgreSQLConnection(TEST_DSN, TEST_DB_NAME)

    database_list = await postgres_connection.list_all_databases()
    for db in database_list:
        assert await postgres_connection.database_exists(db) is True, \
            f"Database {db} should exist!"

    assert await postgres_connection.database_exists(
        "nonexistent_database") is False, \
        "Nonexistent database should not exist!"

    assert postgres_connection._connection is None, \
        "Connection should be None since the class member wasn't set"

@pytest.mark.asyncio
async def test_connect():
    postgres_connection = PostgreSQLConnection(TEST_DSN, TEST_DB_NAME)

    databases = await postgres_connection.list_all_databases()
    database_name = databases[-1]
    
    # Use async with instead of await since connect has asynccontextmanager.
    async with postgres_connection.connect(database_name) as conn:
        assert conn is not None, \
            "Connection should be established"
        assert postgres_connection._connection is not None, \
            "Connection should be set"

        # Test a simple query
        result = await conn.fetchval("SELECT 1")
        assert result == 1

@pytest.mark.asyncio
async def test_connection_lifecycle(postgres_connection: PostgreSQLConnection):
    """Test the complete lifecycle of database connection."""

    # Create the database
    await postgres_connection.create_database(TEST_DB_NAME)

    # Verify database was created
    assert await postgres_connection.database_exists(TEST_DB_NAME) is True, \
        f"Database {TEST_DB_NAME} should exist!"

    # Test connection to the new database
    async with postgres_connection.connect() as conn:
        assert conn is not None, "Connection should be established"
        # Test a simple query
        result = await conn.fetchval("SELECT 1")
        assert result == 1

        # Test database operations
        # Create a test table
        await conn.execute("""
            CREATE TABLE test_table (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)
        # Insert some data
        await conn.execute(
            "INSERT INTO test_table (name) VALUES ($1)",
            "test_value"
        )
        # Query the data
        result = await conn.fetchval(
            "SELECT name FROM test_table WHERE id = 1"
        )
        assert result == "test_value"

    # Cleanup is handled by the fixture

