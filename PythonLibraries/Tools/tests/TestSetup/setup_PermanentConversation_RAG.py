from commonapi.Messages import (
    ConversationSystemAndPermanent,
    AssistantMessage,
    UserMessage)
from corecode.Utilities import DataSubdirectories, is_model_there
from embeddings.TextSplitters import TextSplitterByTokens
from sentence_transformers import SentenceTransformer

from pathlib import Path

from TestSetup.PostgreSQLDatabaseSetup import (
    # cleanup_test_database is run in postgres_connection, but you still need to
    # import it so that postgres_connection can use it.
    cleanup_test_database,
    PostgreSQLDatabaseSetupData,
    postgres_connection
)

from tools.RAG.PermanentConversation.EmbedPermanentConversation \
    import EmbedPermanentConversation
from tools.RAG.PermanentConversation import PostgreSQLInterface

import json

def setup_PermanentConversation_RAG_dependences(
        test_database_name: str = "test_permanent_conversation_database"):
    data_subdirectories = DataSubdirectories()
    relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"
    is_model_downloaded, model_path = is_model_there(
        relative_model_path,
        data_subdirectories)
    model_is_not_downloaded_message = \
        f"Model {relative_model_path} not downloaded"

    postgresql_database_setup_data = PostgreSQLDatabaseSetupData()
    
    python_libraries_path = Path(__file__).parents[3]

    test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
        "CommonAPI" / "tests" / "TestData"
    test_conversation_path = test_data_path / "test_enable_thinking_true.json"
    
    def load_test_conversation():
        with open(test_conversation_path, "r") as f:
            return json.load(f)

    return (
        model_path,
        is_model_downloaded,
        model_is_not_downloaded_message,
        test_database_name,
        load_test_conversation,
        cleanup_test_database,
        postgres_connection,
        postgresql_database_setup_data.test_dsn)

async def setup_PermanentConversation_RAG(
        input_postgres_connection,
        test_db_name,
        model_path,
        load_test_conversation):

    await input_postgres_connection.create_database(test_db_name)
    await input_postgres_connection.create_new_pool(test_db_name)
    await input_postgres_connection.create_extension("vector")

    pgsql_interface = PostgreSQLInterface(input_postgres_connection)
    assert await pgsql_interface.create_tables() is True, \
        "Tables should be created"

    text_splitter = TextSplitterByTokens(model_path=model_path)
    embedding_model = SentenceTransformer(str(model_path), device = "cuda:0",)
    csp = ConversationSystemAndPermanent()
    conversation = load_test_conversation()
    for message in conversation:
        if message["role"] == "user":
            csp.append_message(UserMessage(message["content"]))
        elif message["role"] == "assistant":
            csp.append_message(AssistantMessage(message["content"]))
        elif message["role"] == "system":
            csp.add_system_message(message["content"])
    embed_pc = EmbedPermanentConversation(
        text_splitter,
        embedding_model,
        csp.pc)
    message_chunks, message_pair_chunks = embed_pc.embed_conversation()

    # Insert chunks into database.
    for message_chunk in message_chunks:
        await pgsql_interface.insert_message_chunk(message_chunk)
    for message_pair_chunk in message_pair_chunks:
        await pgsql_interface.insert_message_pair_chunk(message_pair_chunk)

    return (
        pgsql_interface,
        embed_pc,
    )