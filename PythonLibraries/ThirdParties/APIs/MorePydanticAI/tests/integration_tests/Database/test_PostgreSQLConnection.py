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
    sys_conn = await asyncpg.connect(TEST_DSN)
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
async def test_connection_lifecycle(postgres_connection: PostgreSQLConnection):
    """Test the complete lifecycle of database connection."""
    # # # First verify the database doesn't exist
    # # sys_conn = await asyncpg.connect(TEST_DSN)
    # # try:
    # #     exists = await sys_conn.fetchval(
    # #         'SELECT 1 FROM pg_database WHERE datname = $1',
    # #         TEST_DB_NAME
    # #     )
    # #     assert exists is None, f"Test database {TEST_DB_NAME} already exists!"
    # # finally:
    # #     await sys_conn.close()

    # # Try to connect (should fail as database doesn't exist)
    # async with postgres_connection.connect() as conn:
    #     assert conn is None, \
    #         "Connection should be None for non-existent database"

    # Create the database
    await postgres_connection.create_database(TEST_DB_NAME)

    # Verify database was created
    sys_conn = await asyncpg.connect(TEST_DSN)
    try:
        exists = await sys_conn.fetchval(
            'SELECT 1 FROM pg_database WHERE datname = $1',
            TEST_DB_NAME
        )
        assert exists is not None, \
            f"Database {TEST_DB_NAME} was not created!"
    finally:
        await sys_conn.close()

    # Test connection to the new database
    async with postgres_connection.connect() as conn:
        assert conn is not None, "Connection should be established"
        # Test a simple query
        result = await conn.fetchval("SELECT 1")
        assert result == 1

    # Test database operations
    async with postgres_connection.connect() as conn:
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

print(f"Using TEST_DSN: {TEST_DSN}")
