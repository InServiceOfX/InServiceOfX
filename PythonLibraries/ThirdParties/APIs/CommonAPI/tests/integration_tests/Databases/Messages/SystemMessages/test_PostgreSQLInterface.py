from commonapi.Databases import PostgreSQLConnection
from commonapi.Databases.Messages.SystemMessages \
    import PostgreSQLInterface as SystemMessagesPostgreSQLInterface
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
    return postgresql_database_setup_data.test_dsn

@pytest_asyncio.fixture(scope="function")
def test_db_name():
    return "test_system_messages_database"

@pytest.mark.asyncio
async def test_system_messages_database_lifecycle(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)

    assert await postgres_connection.database_exists(test_db_name) is True, \
        f"Database {test_db_name} should exist!"

    system_messages_postgres_interface = \
        SystemMessagesPostgreSQLInterface(postgres_connection)

    assert system_messages_postgres_interface.create_table() is True, \
        "Table should be created"

    assert system_messages_postgres_interface.table_exists() is True, \
        "Table should exist"

    assert system_messages_postgres_interface.drop_table() is True, \
        "Table should be dropped"

    assert system_messages_postgres_interface.table_exists() is False, \
        "Table should not exist"

    # Cleanup is handled by the fixture
