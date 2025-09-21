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

    # Cleanup is handled by the fixture

