from commonapi.Messages import (
    AssistantMessage,
    UserMessage)
from clichatlocal import ApplicationPaths
from clichatlocal.Configuration.CLIConfiguration import CLIConfiguration
from clichatlocal.Core import (
    ModelConversationAndToolsManager,
    ProcessConfigurations)
from clichatlocal.Core.Databases import PostgreSQLResource
from clichatlocal.Core.RAG.PermanentConversation import PostgreSQLAndEmbedding
from clichatlocal.Terminal import TerminalUI
from tools.Databases.PostgreSQLSetup import PostgreSQLSetup

from pathlib import Path

import json, pytest

def load_test_conversation():
    python_libraries_path = Path(__file__).parents[7] / "PythonLibraries"
    assert python_libraries_path.exists()

    test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
        "CommonAPI" / "tests" / "TestData"
    test_conversation_path = test_data_path / "test_enable_thinking_true.json"
    with open(test_conversation_path, "r") as f:
        return json.load(f)

class TestApplication:
    def __init__(self, application_paths, process_configurations):
        self._application_paths = application_paths
        self._process_configurations = process_configurations

@pytest.mark.asyncio
async def test_PostgreSQLAndEmbedding_sets_up():
    application_paths = ApplicationPaths.create(
        is_development=True,
        is_current_path=False,
        configpath=None)

    cli_configuration = CLIConfiguration()
    terminal_ui = TerminalUI(cli_configuration)

    process_configurations = ProcessConfigurations(
        application_paths,
        terminal_ui)
    process_configurations.process_configurations()

    test_application = TestApplication(
        application_paths,
        process_configurations)

    model_conversation_and_tools_manager = ModelConversationAndToolsManager(
        test_application)

    postgresql_resource = PostgreSQLResource(process_configurations)

    await postgresql_resource.load_configuration_and_create_pool()

    connection = postgresql_resource._pgs_setup._connections[
        "PermanentConversation"]

    postgresql_and_embedding = PostgreSQLAndEmbedding(
        connection,
        process_configurations)

    await postgresql_and_embedding.create_tables()
    postgresql_and_embedding.setup_embedding_model()
    postgresql_and_embedding.create_EmbedPermanentConversation(
        model_conversation_and_tools_manager._csp.pc)

    assert len(
        model_conversation_and_tools_manager._csp.get_permanent_conversation_messages()) \
            == 0

    message_chunks, message_pair_chunks = \
        postgresql_and_embedding.embed_conversation()

    assert len(message_chunks) == 0
    assert len(message_pair_chunks) == 0

@pytest.mark.asyncio
async def test_PostgreSQLAndEmbedding_embeds_permanent_conversation():
    application_paths = ApplicationPaths.create(
        is_development=True,
        is_current_path=False,
        configpath=None)

    cli_configuration = CLIConfiguration()
    terminal_ui = TerminalUI(cli_configuration)

    process_configurations = ProcessConfigurations(
        application_paths,
        terminal_ui)
    process_configurations.process_configurations()

    test_application = TestApplication(
        application_paths,
        process_configurations)

    model_conversation_and_tools_manager = ModelConversationAndToolsManager(
        test_application)

    postgresql_resource = PostgreSQLResource(process_configurations)

    await postgresql_resource.load_configuration_and_create_pool()

    connection = postgresql_resource._pgs_setup._connections[
        "PermanentConversation"]

    postgresql_and_embedding = PostgreSQLAndEmbedding(
        connection,
        process_configurations)

    await postgresql_and_embedding.create_tables()
    postgresql_and_embedding.setup_embedding_model()
    postgresql_and_embedding.create_EmbedPermanentConversation(
        model_conversation_and_tools_manager._csp.pc)

    conversation = load_test_conversation()
    for message in conversation:
        if message["role"] == "user":
            model_conversation_and_tools_manager._csp.append_message(
                UserMessage(message["content"]))
        elif message["role"] == "assistant":
            model_conversation_and_tools_manager._csp.append_message(
                AssistantMessage(message["content"]))
        elif message["role"] == "system":
            model_conversation_and_tools_manager._csp.add_system_message(
                message["content"])

    message_chunks, message_pair_chunks = \
        postgresql_and_embedding.embed_conversation()

    assert len(message_chunks) == 19
    assert len(message_pair_chunks) == 11