from corecode.Utilities import DataSubdirectories, is_model_there
from pathlib import Path

from TestSetup.PostgreSQLDatabaseSetup import (
    # cleanup_test_database is run in postgres_connection, but you still need to
    # import it so that postgres_connection can use it.
    cleanup_test_database,
    PostgreSQLDatabaseSetupData,
    postgres_connection
)

import json
import pytest_asyncio

def setup_PermanentConversation_RAG(
        test_database_name: str = "test_permanent_conversation_database"):
    data_subdirectories = DataSubdirectories()
    relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"
    is_model_downloaded, model_path = is_model_there(
        relative_model_path,
        data_subdirectories)
    model_is_not_downloaded_message = \
        f"Model {relative_model_path} not downloaded"

    postgresql_database_setup_data = PostgreSQLDatabaseSetupData()
    @pytest_asyncio.fixture(scope="session")
    def test_dsn():
        return postgresql_database_setup_data.test_dsn

    python_libraries_path = Path(__file__).parents[3]

    test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
        "CommonAPI" / "tests" / "TestData"
    test_conversation_path = test_data_path / "test_enable_thinking_true.json"
    def load_test_conversation():
        with open(test_conversation_path, "r") as f:
            return json.load(f)

    @pytest_asyncio.fixture(scope="function")
    def test_db_name():
        return test_database_name

    return (
        model_path,
        is_model_downloaded,
        model_is_not_downloaded_message,
        test_dsn,
        test_db_name,
        load_test_conversation,
        cleanup_test_database,
        postgres_connection)