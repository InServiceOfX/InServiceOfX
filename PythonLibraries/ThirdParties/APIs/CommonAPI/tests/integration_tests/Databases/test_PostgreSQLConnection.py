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

20250701: I've also tried running all the tests at once; it seems that pytest
runs tests synchronously (i.e. sequentially one by one).
"""
from commonapi.Databases import PostgreSQLConnection
import pytest
import pytest_asyncio

from TestSetup.PostgreSQLDatabaseSetup import (
    cleanup_test_database,
    PostgreSQLDatabaseSetupData,
    postgres_connection
)

postgresql_database_setup_data = PostgreSQLDatabaseSetupData()

@pytest_asyncio.fixture(scope="session")
def test_dsn():
    """Provide the test database connection string."""
    # You can customize this based on environment or configuration
    return postgresql_database_setup_data.test_dsn

@pytest_asyncio.fixture(scope="function")
def test_db_name():
    # The name is arbitrary, can be anything you want.
    return "test_pydantic_ai_database"

@pytest.mark.asyncio
async def test_cleanup_test_database_drops_only_if_database_exists(
        test_dsn: str,
        test_db_name: str):
    database_name = test_db_name
    await cleanup_test_database(test_dsn, database_name)

    postgres_connection = PostgreSQLConnection(test_dsn, database_name)
    assert await postgres_connection.database_exists(database_name) is False, \
        f"Database {database_name} should not exist!"

@pytest.mark.asyncio
async def test_list_all_databases(test_dsn: str):
    postgres_connection = PostgreSQLConnection(test_dsn)

    databases = await postgres_connection.list_all_databases()
    for db in databases:
        print(f"Database: {db}")

    assert "postgres" in databases, \
        "postgres database should be in the list of databases"

    assert postgres_connection._connection is None, \
        "Connection should be None since the class member wasn't set"

@pytest.mark.asyncio
async def test_database_exists(test_dsn: str):
    postgres_connection = PostgreSQLConnection(test_dsn)

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
async def test_connect(test_dsn: str):
    postgres_connection = PostgreSQLConnection(test_dsn, test_db_name)

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

        assert postgres_connection._database_name is not None
        assert postgres_connection._database_name == database_name

@pytest.mark.asyncio
async def test_connection_lifecycle(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):
    """Test the complete lifecycle of database connection."""

    # Create the database
    await postgres_connection.create_database(test_db_name)

    # Verify database was created
    assert await postgres_connection.database_exists(test_db_name) is True, \
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

@pytest.mark.asyncio
async def test_create_new_pool(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):
    await postgres_connection.create_database(test_db_name)
    
    pool = await postgres_connection.create_new_pool(test_db_name)
    assert pool is not None
    assert postgres_connection._pool is not None
    assert postgres_connection._pool == pool
    
    async with postgres_connection.connect() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1

@pytest.mark.asyncio
async def test_connect_uses_pool_when_available(
        test_dsn: str,
        test_db_name: str,
        postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)
    
    # Connect should use pool
    async with postgres_connection.connect() as conn:
        assert conn is not None
        result = await conn.fetchval("SELECT 1")
        assert result == 1
        
        # Verify we're using pool (connection should be from pool)
        assert postgres_connection._pool is not None

@pytest.mark.asyncio
async def test_connect_falls_back_to_individual_connection_when_no_pool(
        test_dsn: str,
        test_db_name: str,
        postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    
    # Connect should use individual connection
    async with postgres_connection.connect() as conn:
        assert conn is not None
        result = await conn.fetchval("SELECT 1")
        assert result == 1
        
        # Verify we're using individual connection
        assert postgres_connection._pool is None

@pytest.mark.asyncio
async def test_close_pool(
        test_dsn: str,
        test_db_name: str,
        postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)
    
    assert postgres_connection._pool is not None
    
    # Close pool
    await postgres_connection.close_pool()
    assert postgres_connection._pool is None

@pytest.mark.asyncio
async def test_create_new_pool_closes_existing_pool(
        test_dsn: str,
        test_db_name: str,
        postgres_connection: PostgreSQLConnection):

    # Create database and first pool
    await postgres_connection.create_database(test_db_name)
    pool1 = await postgres_connection.create_new_pool(test_db_name)
    
    # Create second pool (should close first one)
    pool2 = await postgres_connection.create_new_pool(test_db_name)
    
    assert pool1 != pool2
    assert postgres_connection._pool == pool2

# TODO: Check the following tests individually.

@pytest.mark.asyncio
async def test_pool_concurrent_connections(test_dsn: str, test_db_name: str):
    """Test that pool can handle multiple concurrent connections."""
    postgres_connection = PostgreSQLConnection(test_dsn, test_db_name)
    
    # Create database and pool
    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name, min_size=2, max_size=5)
    
    # Create test table
    async with postgres_connection.connect() as conn:
        await conn.execute("""
            CREATE TABLE concurrent_test (
                id SERIAL PRIMARY KEY,
                value TEXT
            )
        """)
    
    # Test multiple concurrent connections
    import asyncio
    
    async def insert_value(value: str):
        async with postgres_connection.connect() as conn:
            await conn.execute(
                "INSERT INTO concurrent_test (value) VALUES ($1)",
                value
            )
    
    # Run multiple concurrent inserts
    tasks = [insert_value(f"value_{i}") for i in range(10)]
    await asyncio.gather(*tasks)
    
    # Verify all inserts worked
    async with postgres_connection.connect() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM concurrent_test")
        assert count == 10

@pytest.mark.asyncio
async def test_pool_connection_reuse(test_dsn: str, test_db_name: str):
    """Test that pool reuses connections efficiently."""
    postgres_connection = PostgreSQLConnection(test_dsn, test_db_name)
    
    # Create database and pool
    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name, min_size=1, max_size=3)
    
    # Create test table
    async with postgres_connection.connect() as conn:
        await conn.execute("""
            CREATE TABLE reuse_test (
                id SERIAL PRIMARY KEY,
                value TEXT
            )
        """)
    
    # Perform multiple operations (should reuse connections from pool)
    for i in range(5):
        async with postgres_connection.connect() as conn:
            await conn.execute(
                "INSERT INTO reuse_test (value) VALUES ($1)",
                f"reuse_test_{i}"
            )
    
    # Verify all operations worked
    async with postgres_connection.connect() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM reuse_test")
        assert count == 5

@pytest.mark.asyncio
async def test_close_cleans_up_both_pool_and_connection(test_dsn: str, test_db_name: str):
    """Test that close() method cleans up both pool and individual connection."""
    postgres_connection = PostgreSQLConnection(test_dsn, test_db_name)
    
    # Create database and pool
    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)
    
    # Also create an individual connection
    async with postgres_connection.connect() as conn:
        await conn.fetchval("SELECT 1")
    
    # Close everything
    await postgres_connection.close()
    
    assert postgres_connection._pool is None
    assert postgres_connection._connection is None

@pytest.mark.asyncio
async def test_pool_with_different_database_names(test_dsn: str, test_db_name: str):
    """Test pool behavior when connecting to different databases."""
    postgres_connection = PostgreSQLConnection(test_dsn, test_db_name)
    
    # Create two databases
    db1 = f"{test_db_name}_1"
    db2 = f"{test_db_name}_2"
    
    await postgres_connection.create_database(db1)
    await postgres_connection.create_database(db2)
    
    # Create pool for db1
    await postgres_connection.create_new_pool(db1)
    
    # Connect to db1 (should use pool)
    async with postgres_connection.connect(db1) as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1
    
    # Connect to db2 (should use individual connection since no pool for db2)
    async with postgres_connection.connect(db2) as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1

@pytest.mark.asyncio
async def test_pool_parameters(test_dsn: str, test_db_name: str):
    """Test creating pool with custom parameters."""
    postgres_connection = PostgreSQLConnection(test_dsn, test_db_name)
    
    # Create database
    await postgres_connection.create_database(test_db_name)
    
    # Create pool with custom parameters
    pool = await postgres_connection.create_new_pool(
        test_db_name, 
        min_size=5, 
        max_size=15
    )
    
    assert pool is not None
    assert postgres_connection._pool == pool
    
    # Test that pool works with custom parameters
    async with postgres_connection.connect() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1

@pytest.mark.asyncio
async def test_list_extension(
        test_dsn: str,
        test_db_name: str,
        postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)

    list_of_extensions = await postgres_connection.list_extensions()

    for extension in list_of_extensions:
        print(extension)

@pytest.mark.asyncio
async def test_create_extension(
        test_dsn: str,
        test_db_name: str,
        postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)

    create_result = await postgres_connection.create_extension("vector")

    assert create_result == True

    assert await postgres_connection.extension_exists("vector")