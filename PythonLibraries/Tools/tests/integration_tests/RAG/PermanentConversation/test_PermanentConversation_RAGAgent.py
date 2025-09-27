from TestSetup.setup_PermanentConversation_RAG \
    import (
        setup_PermanentConversation_RAG,
        setup_PermanentConversation_RAG_dependences,
        )

model_path, is_model_downloaded, model_is_not_downloaded_message, \
    test_db_name, load_test_conversation, cleanup_test_database, \
    postgres_connection = \
        setup_PermanentConversation_RAG_dependences()

from tools.Databases import PostgreSQLConnection
from tools.RAG.PermanentConversation import (
    PostgreSQLInterface,
    RAGAgent,
    RAGProcessor,
    )

import pytest
import pytest_asyncio

@pytest_asyncio.fixture(scope="function")
def test_db_name():
    return "test_permanent_conversation_database"

from TestSetup.PostgreSQLDatabaseSetup import (
    # cleanup_test_database is run in postgres_connection, but you still need to
    # import it so that postgres_connection can use it.
    cleanup_test_database,
    PostgreSQLDatabaseSetupData,
    postgres_connection
)
postgresql_database_setup_data = PostgreSQLDatabaseSetupData()
@pytest_asyncio.fixture(scope="session")
def test_dsn():
    return postgresql_database_setup_data.test_dsn

@pytest.mark.asyncio
async def test_PermanentConversation_RAGAgent_with_conversation(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    pgsql_interface, embed_pc, cleanup_test_database, \
        postgres_connection = await setup_PermanentConversation_RAG()

    rag_processor = RAGProcessor(
        pgsql_interface,
        embed_pc)
