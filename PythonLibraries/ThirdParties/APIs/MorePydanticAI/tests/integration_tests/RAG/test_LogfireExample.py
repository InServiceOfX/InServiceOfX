from corecode.Utilities import DataSubdirectories
from morepydanticai.Database import PostgreSQLConnection
from morepydanticai.RAG.LogfireExample import (
    create_type_adapter,
    insert_doc_section)
from morepydanticai.RAG import parse_online_json

from pathlib import Path
from sentence_transformers import SentenceTransformer
from TestSetup.PostgreSQLDatabaseSetup import (
    PostgreSQLDatabaseSetupData,
    postgres_connection
)
from TestSetup.TestData import pydantic_ai_rag_test_data

import pytest
import pytest_asyncio
import asyncio

@pytest_asyncio.fixture(scope="session")
def test_dsn():
    return PostgreSQLDatabaseSetupData.TEST_DSN

@pytest_asyncio.fixture(scope="function")
def test_db_name():
    return pydantic_ai_rag_test_data()[0]

data_sub_dirs = DataSubdirectories()
MODEL_DIR = data_sub_dirs.Models / "Embeddings" / "BAAI" / \
    "bge-large-en-v1.5"
if not Path(MODEL_DIR).exists():
    print("for MODEL_DIR:", MODEL_DIR)
    print("MODEL_DIR.exists(): ", MODEL_DIR.exists())
    MODEL_DIR = Path("/Data1/Models/Embeddings/BAAI/bge-large-en-v1.5")

@pytest.mark.asyncio
async def test_create_test_table(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):
    await postgres_connection.create_database(test_db_name)
    pool = await postgres_connection.create_new_pool(test_db_name)

    list_of_tables = await postgres_connection.list_table_names(test_db_name)
    assert len(list_of_tables) == 0
    list_of_table_names = await postgres_connection.list_table_names(
        test_db_name)
    assert len(list_of_table_names) == 0
    is_table_exists = await postgres_connection.is_table_exists(
        "doc_sections", test_db_name)
    assert not is_table_exists

    await postgres_connection.create_table_from_schema(
        pydantic_ai_rag_test_data()[2])

    list_of_tables = await postgres_connection.list_table_names(test_db_name)
    assert len(list_of_tables) == 1
    list_of_table_names = await postgres_connection.list_table_names(
        test_db_name)
    assert len(list_of_table_names) == 1
    assert list_of_table_names[0] == "doc_sections"
    is_table_exists = await postgres_connection.is_table_exists(
        "doc_sections", test_db_name)
    assert is_table_exists

@pytest.mark.asyncio
async def test_insert_doc_section(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):
    """
    Ref:
    https://ai.pydantic.dev/examples/rag/#example-code
    """
    embedding_model = SentenceTransformer(str(MODEL_DIR), device="cuda:0",)
    await postgres_connection.create_database(test_db_name)

    assert postgres_connection._database_name == test_db_name
    pool = await postgres_connection.create_new_pool(test_db_name)

    await postgres_connection.create_table_from_schema(
        pydantic_ai_rag_test_data()[2])

    # Get example online documentation as DocsSections, which have str data
    # members.
    url = pydantic_ai_rag_test_data()[3]
    type_adapter = create_type_adapter()
    sections = await parse_online_json(url, type_adapter)

    # This then allows only 10 concurrent connections.
    sem = asyncio.Semaphore(10)
    async with asyncio.TaskGroup() as tg:
        for section in sections:
            tg.create_task(insert_doc_section(
                sem,
                embedding_model,
                pool,
                section))

    # Test 1: Count total rows
    row_count = await postgres_connection.fetch_value(
        "SELECT COUNT(*) FROM doc_sections"
    )
    assert row_count == len(sections), \
        f"Expected {len(sections)} rows, got {row_count}"

    # Test 2: Check for specific content
    sample_section = sections[0]
    sample_row = await postgres_connection.fetch_row(
        "SELECT title, content FROM doc_sections WHERE title = $1",
        None,  # database_name parameter
        sample_section.title  # query parameter
    )
    assert sample_row is not None, "Sample section not found in database"
    assert sample_row['title'] == sample_section.title
    assert sample_row['content'] == sample_section.content

    # Test 3: Check embedding was created
    embedding_row = await postgres_connection.fetch_row(
        "SELECT embedding FROM doc_sections WHERE title = $1",
        None,  # database_name parameter
        sample_section.title  # query parameter
    )
    assert embedding_row is not None, "Embedding not found"
    assert embedding_row['embedding'] is not None, "Embedding is null"

