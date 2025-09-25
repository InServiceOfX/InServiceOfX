from TestSetup.setup_PermanentConversation_RAG \
    import setup_PermanentConversation_RAG

model_path, is_model_downloaded, model_is_not_downloaded_message, \
    test_dsn, test_db_name, load_test_conversation, cleanup_test_database, \
    postgres_connection = \
        setup_PermanentConversation_RAG()

from tools.Databases import PostgreSQLConnection
from tools.RAG.PermanentConversation.RAGAgent import RAGAgent

import pytest

@pytest.mark.asyncio
async def test_PermanentConversation_RAGAgent_with_conversation(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)
    await postgres_connection.create_extension("vector")
