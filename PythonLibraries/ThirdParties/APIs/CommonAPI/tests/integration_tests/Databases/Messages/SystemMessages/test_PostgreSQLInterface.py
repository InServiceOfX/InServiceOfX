from commonapi.Databases import PostgreSQLConnection
from commonapi.Databases.Messages.SystemMessages \
    import PostgreSQLInterface as SystemMessagesPostgreSQLInterface

from commonapi.Messages import (RecordedSystemMessage, SystemMessagesManager)
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

    assert await system_messages_postgres_interface.create_table() is True, \
        "Table should be created"

    assert await system_messages_postgres_interface.table_exists() is True, \
        "Table should exist"

    assert await system_messages_postgres_interface.drop_table() is True, \
        "Table should be dropped"

    assert await system_messages_postgres_interface.table_exists() is False, \
        "Table should not exist"

    # Cleanup is handled by the fixture

def create_example_content():
    content_1 = "You are a helpful, knowledgeable assistant."
    content_2 = (
        "You are a math tutor. Always show your reasoning step by step before "
        "giving the final answer.")
    content_3 = \
        "You are a concise summarizer. Respond in no more than two sentences."
    
    hash_1 = RecordedSystemMessage.create_hash(content_1)
    hash_2 = RecordedSystemMessage.create_hash(content_2)
    hash_3 = RecordedSystemMessage.create_hash(content_3)

    content = [content_1, content_2, content_3]
    hashes = [hash_1, hash_2, hash_3]

    return content, hashes

@pytest.mark.asyncio
async def test_system_messages_manager_to_database(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    system_messages_manager = SystemMessagesManager()
    system_messages_manager.clear()

    content, _ = create_example_content()
    system_messages_manager.add_message(content[0])
    system_messages_manager.add_message(content[1])
    system_messages_manager.add_message(content[2])

    await postgres_connection.create_database(test_db_name)

    system_messages_postgres_interface = \
        SystemMessagesPostgreSQLInterface(postgres_connection)

    assert await system_messages_postgres_interface.create_table() is True, \
        "Table should be created"

    assert await system_messages_postgres_interface.insert_message(
        system_messages_manager.messages[0]) is not None, \
        "Message should be inserted"

    assert await system_messages_postgres_interface.insert_message(
        system_messages_manager.messages[1]) is not None, \
        "Message should be inserted"

    assert await system_messages_postgres_interface.insert_message(
        system_messages_manager.messages[2]) is not None, \
        "Message should be inserted"

    list_of_messages = \
        await system_messages_postgres_interface.load_all_messages()
    assert len(list_of_messages) == 3, \
        "There should be 3 messages in the database"

    assert list_of_messages[0].content == content[0], \
        "First message should be the first message in the list"

    assert list_of_messages[1].content == content[1], \
        "Second message should be the second message in the list"
    
    assert list_of_messages[2].content == content[2], \
        "Third message should be the third message in the list"

    # Clean up.
    assert await system_messages_postgres_interface.drop_table() is True, \
        "Table should be dropped"

@pytest.mark.asyncio
async def test_database_to_system_messages_manager(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    system_messages_manager = SystemMessagesManager()
    system_messages_manager.clear()

    content, _ = create_example_content()

    await postgres_connection.create_database(test_db_name)

    system_messages_postgres_interface = \
        SystemMessagesPostgreSQLInterface(postgres_connection)

    assert await system_messages_postgres_interface.create_table() is True, \
        "Table should be created"

    assert await system_messages_postgres_interface.insert_message(
        content[0]) is not None, \
        "Message should be inserted"

    assert await system_messages_postgres_interface.insert_message(
        content[1]) is not None, \
        "Message should be inserted"

    assert await system_messages_postgres_interface.insert_message(
        content[2]) is not None, \
        "Message should be inserted"

    list_of_messages = \
        await system_messages_postgres_interface.load_all_messages()
    assert len(list_of_messages) == 3, \
        "There should be 3 messages in the database"

    for message in list_of_messages:
        assert system_messages_manager.add_previously_recorded_message(message) is True, \
            "Message should be added"

    assert len(system_messages_manager.messages) == 3, \
        "There should be 3 messages in the manager"
    
    assert system_messages_manager.messages[0].content == content[0], \
        "First message should be the first message in the manager"
    
    assert system_messages_manager.messages[1].content == content[1], \
        "Second message should be the second message in the manager"
    
    assert system_messages_manager.messages[2].content == content[2], \
        "Third message should be the third message in the manager"

    # Clean up.
    assert await system_messages_postgres_interface.drop_table() is True, \
        "Table should be dropped"
