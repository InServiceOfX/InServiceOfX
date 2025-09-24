"""
USAGE:
You'll also want to monitor real-time changes to the PostgreSQL database server
you are running with Docker. Follow the instructions in the README.md of this
(sub-)project but otherwise, this is a summary:

Get the name and container ID of the Docker image that has postgresql running
docker ps
`exec` into the running Docker container, e.g.
docker exec -it local-llm-full-postgres psql -U inserviceofx -d local_llm_full_database
where you'll get the username and database name from the docker-compose.yml file.

You should see something like this:
psql (16.8 (Debian 16.8-1.pgdg120+1))
Type "help" for help.

Then you can do something like this:

# list all databases
\l
"""
from corecode.Utilities import DataSubdirectories, is_model_there
data_subdirectories = DataSubdirectories()
relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

from pathlib import Path
from tools.Databases import PostgreSQLConnection
from tools.RAG.PermanentConversation import PostgreSQLInterface as \
    PermanentConversationPostgreSQLInterface
import pytest
import pytest_asyncio

from TestSetup.PostgreSQLDatabaseSetup import (
    # cleanup_test_database is run in postgres_connection, but you still need to
    # import it so that postgres_connection can use it.
    cleanup_test_database,
    PostgreSQLDatabaseSetupData,
    postgres_connection
)
from commonapi.Messages import (
    ConversationSystemAndPermanent,
    AssistantMessage,
    UserMessage)
from embeddings.TextSplitters import TextSplitterByTokens
from sentence_transformers import SentenceTransformer
from tools.RAG.PermanentConversation.EmbedPermanentConversation \
    import EmbedPermanentConversation
import json

postgresql_database_setup_data = PostgreSQLDatabaseSetupData()

python_libraries_path = Path(__file__).parents[5]
test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
    "CommonAPI" / "tests" / "TestData"
test_conversation_path = test_data_path / "test_enable_thinking_true.json"
def load_test_conversation():
    with open(test_conversation_path, "r") as f:
        return json.load(f)

@pytest_asyncio.fixture(scope="session")
def test_dsn():
    return postgresql_database_setup_data.test_dsn

@pytest_asyncio.fixture(scope="function")
def test_db_name():
    return "test_permanent_conversation_database"

@pytest.mark.asyncio
async def test_permanent_conversation_database_inserts_and_persists(
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
        PermanentConversationPostgreSQLInterface.MESSAGE_CHUNKS_TABLE_NAME) is True, \
        "Table chunks should exist"

    assert await pcpsqli.table_exists(
        PermanentConversationPostgreSQLInterface.MESSAGE_PAIR_CHUNKS_TABLE_NAME) is True, \
        "Table message pair chunks should exist"

    # Set up test conversation.

    conversation = load_test_conversation()
    text_splitter = TextSplitterByTokens(model_path=model_path)
    embedding_model = SentenceTransformer(str(model_path), device = "cuda:0",)
    csp = ConversationSystemAndPermanent()
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
        await pcpsqli.insert_message_chunk(message_chunk)
    for message_pair_chunk in message_pair_chunks:
        await pcpsqli.insert_message_pair_chunk(message_pair_chunk)

    # Check if we can retrieve all message chunks.

    all_message_chunks = await pcpsqli.get_all_message_chunks()

    for index, message_chunk in enumerate(all_message_chunks):
        assert message_chunk.content == message_chunks[index].content
        assert message_chunk.role == message_chunks[index].role
        assert message_chunk.datetime == message_chunks[index].datetime
        assert message_chunk.hash == message_chunks[index].hash
        assert message_chunk.conversation_id == \
            message_chunks[index].conversation_id
        assert message_chunk.chunk_type == "message"
        assert message_chunk.embedding == pytest.approx(
            message_chunks[index].embedding)

    assert len(all_message_chunks) == len(message_chunks), \
        "All message chunks should be retrieved"

    reconstructed_messages = \
        EmbedPermanentConversation.recreate_conversation_messages_from_chunks(
            all_message_chunks)

    assert len(reconstructed_messages) == len(csp.pc.messages), \
        "All messages should be reconstructed"

    for i in range(len(reconstructed_messages)):
        assert reconstructed_messages[i].content == csp.pc.messages[i].content
        assert reconstructed_messages[i].role == csp.pc.messages[i].role
        assert reconstructed_messages[i].datetime == csp.pc.messages[i].datetime
        assert reconstructed_messages[i].hash == csp.pc.messages[i].hash
        assert reconstructed_messages[i].conversation_id == csp.pc.messages[i].conversation_id

    # Check if we can retrieve all message pair chunks.

    all_message_pair_chunks = await pcpsqli.get_all_message_pair_chunks()

    for index, message_pair_chunk in enumerate(all_message_pair_chunks):
        assert message_pair_chunk.content == message_pair_chunks[index].content
        assert message_pair_chunk.role == message_pair_chunks[index].role
        assert message_pair_chunk.datetime == message_pair_chunks[index].datetime
        assert message_pair_chunk.hash == message_pair_chunks[index].hash
        assert message_pair_chunk.conversation_id == \
            message_pair_chunks[index].conversation_id
        assert message_pair_chunk.chunk_type == "message_pair"
        assert message_pair_chunk.embedding == pytest.approx(
            message_pair_chunks[index].embedding)

    # Cleanup is handled by the fixture

test_queries = [
    "HTML CSS JavaScript ball hexagon animation physics gravity",
    "Go language recursive function maze solving algorithm", 
    "transformer neural network attention mechanism machine learning"
]

@pytest.mark.asyncio
async def test_permanent_conversation_database_does_vector_similarity_search(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)
    await postgres_connection.create_extension("vector")

    pcpsqli = PermanentConversationPostgreSQLInterface(
        postgres_connection)

    assert await pcpsqli.create_tables() is True, "Tables should be created"

    # Set up test conversation.

    conversation = load_test_conversation()
    text_splitter = TextSplitterByTokens(model_path=model_path)
    embedding_model = SentenceTransformer(str(model_path), device = "cuda:0",)
    csp = ConversationSystemAndPermanent()
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
        await pcpsqli.insert_message_chunk(message_chunk)
    for message_pair_chunk in message_pair_chunks:
        await pcpsqli.insert_message_pair_chunk(message_pair_chunk)

    results_for_queries = []

    for query in test_queries:
        query_results = []
        query_embeddings = text_splitter.text_to_embedding(
            embedding_model,
            query)
        for embedding in query_embeddings:
            results = await pcpsqli.vector_similarity_search_message_chunks(
                query_embedding=embedding)
            query_results.append(results)
        results_for_queries.append(query_results)

    assert len(results_for_queries) == len(test_queries)
    assert type(results_for_queries[0]) == list

    # 1 chunk for the 0th query (no text splitting needed)
    assert len(results_for_queries[0]) == 1
    # Number of closest matches
    assert len(results_for_queries[0][0]) == 10
    assert results_for_queries[0][0][0]["similarity_score"] == pytest.approx(
        0.8292039138553209,)

    # Uncomment out print statements to see actual content.

    # print(
    #     "results_for_queries[0][0][0]['content']",
    #     results_for_queries[0][0][0]["content"])

    assert results_for_queries[0][0][1]["similarity_score"] == pytest.approx(
        0.826344689566536,)
    # print(
    #     "results_for_queries[0][0][1]['content']",
    #     results_for_queries[0][0][1]["content"])

    assert results_for_queries[0][0][2]["similarity_score"] == pytest.approx(
        0.8079813833247805,)
    # print(
    #     "results_for_queries[0][0][2]['content']",
    #     results_for_queries[0][0][2]["content"])

    assert len(results_for_queries[1]) == 1
    # Number of closest matches
    assert len(results_for_queries[1][0]) == 10
    assert results_for_queries[1][0][0]["similarity_score"] == pytest.approx(
        0.7993406773006526,)
    # print(
    #     "results_for_queries[1][0][0]['content']",
    #     results_for_queries[1][0][0]["content"])

    assert results_for_queries[1][0][1]["similarity_score"] == pytest.approx(
        0.7295023202896118,)
    # print(
    #     "results_for_queries[1][0][1]['content']",
    #     results_for_queries[1][0][1]["content"])

    assert results_for_queries[1][0][2]["similarity_score"] == pytest.approx(
        0.6303956884381596,)
    # print(
    #     "results_for_queries[1][0][2]['content']",
    #     results_for_queries[1][0][2]["content"])


async def setup_test_database(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection,
    model_path,
    conversation):

    await postgres_connection.create_database(test_db_name)
    await postgres_connection.create_new_pool(test_db_name)
    await postgres_connection.create_extension("vector")

    pcpsqli = PermanentConversationPostgreSQLInterface(
        postgres_connection)

    assert await pcpsqli.create_tables() is True, "Tables should be created"

    text_splitter = TextSplitterByTokens(model_path=model_path)
    embedding_model = SentenceTransformer(str(model_path), device = "cuda:0",)
    csp = ConversationSystemAndPermanent()
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
        await pcpsqli.insert_message_chunk(message_chunk)
    for message_pair_chunk in message_pair_chunks:
        await pcpsqli.insert_message_pair_chunk(message_pair_chunk)

    return pcpsqli, text_splitter, embedding_model

def make_query_embeddings(text_splitter, embedding_model, query):
    return text_splitter.text_to_embedding(
        embedding_model,
        query)

@pytest.mark.asyncio
async def test_vector_similarity_search_with_role(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    conversation = load_test_conversation()

    pcpsqli, text_splitter, embedding_model = await setup_test_database(
        test_dsn,
        test_db_name,
        postgres_connection,
        model_path,
        conversation)

    results_for_queries = []

    for query in test_queries:
        query_results = []
        query_embeddings = make_query_embeddings(
            text_splitter,
            embedding_model,
            query)
        for embedding in query_embeddings:
            results = await pcpsqli.vector_similarity_search_message_chunks(
                query_embedding=embedding,
                role_filter="user",
                limit=5)
            query_results.append(results)
        results_for_queries.append(query_results)

    for query_results in results_for_queries:
        # 1 chunk for each query (no text splitting needed)
        assert len(query_results) == 1
        assert len(query_results[0]) == 5
        for result in query_results[0]:
            assert result["role"] == "user"
