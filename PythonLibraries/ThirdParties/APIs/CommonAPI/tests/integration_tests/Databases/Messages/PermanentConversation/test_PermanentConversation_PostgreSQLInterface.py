from commonapi.Databases import PostgreSQLConnection
from commonapi.Databases.Messages.PermanentConversation \
    import PostgreSQLInterface as PermanentConversationPostgreSQLInterface

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
    return "test_permanent_conversation_database"

@pytest.mark.asyncio
async def test_permanent_conversation_database_lifecycle(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)
    await postgres_connection.create_extension("vector")

    assert await postgres_connection.database_exists(test_db_name) is True, \
        f"Database {test_db_name} should exist!"

    pcpsqli = PermanentConversationPostgreSQLInterface(
        postgres_connection)

    assert await pcpsqli.create_tables() is True, "Tables should be created"

    assert await pcpsqli.table_exists(
        PermanentConversationPostgreSQLInterface.MESSAGES_TABLE_NAME) is True, \
        "Table messages should exist"

    assert await pcpsqli.table_exists(
        PermanentConversationPostgreSQLInterface.MESSAGE_PAIRS_TABLE_NAME) \
            is True, \
        "Table message pairs should exist"

    # Cleanup is handled by the fixture

