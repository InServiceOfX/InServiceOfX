from corecode.Utilities import DataSubdirectories, is_model_there
data_subdirectories = DataSubdirectories()
relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)
model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

from pathlib import Path
from tools.Databases import PostgreSQLConnection
from tools.RAG.PermanentConversation import PostgreSQLInterface, RAGProcessor
import pytest
import pytest_asyncio

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

from commonapi.Messages import (
    ConversationSystemAndPermanent,
    AssistantMessage,
    UserMessage)
from embeddings.TextSplitters import TextSplitterByTokens
from sentence_transformers import SentenceTransformer
from tools.RAG.PermanentConversation.EmbedPermanentConversation \
    import EmbedPermanentConversation
import json

python_libraries_path = Path(__file__).parents[5]
test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
    "CommonAPI" / "tests" / "TestData"
test_conversation_path = test_data_path / "test_enable_thinking_true.json"
def load_test_conversation():
    with open(test_conversation_path, "r") as f:
        return json.load(f)

@pytest_asyncio.fixture(scope="function")
def test_db_name():
    return "test_permanent_conversation_database"

test_queries = [
    "HTML CSS JavaScript ball hexagon animation physics gravity",
    "Go language recursive function maze solving algorithm", 
    "transformer neural network attention mechanism machine learning"
]

@pytest.mark.asyncio
async def test_PermanentConversation_RAGProcessor_process_query_to_context(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)
    await postgres_connection.create_extension("vector")

    pgsql_interface = PostgreSQLInterface(postgres_connection)
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

    rag_processor = RAGProcessor(
        pgsql_interface,
        embed_pc)

    contexts = []
    for query in test_queries:
        context = await rag_processor.process_query_to_context(query, limit=3)
        contexts.append(context)

    # for index, context in enumerate(contexts):
    #     print(f"Query {index}: {test_queries[index]}")
    #     print(f"Context {index}: {context}")
    #     print("--------------------------------")

    contexts = []
    for query in test_queries:
        context = await rag_processor.process_query_to_context(
            query,
            role_filter="assistant",
            limit=2)
        contexts.append(context)

    for index, context in enumerate(contexts):
        print(f"Query {index}: {test_queries[index]}")
        print(f"Context {index}: {context}")
        print("--------------------------------")



