from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories
from commonapi.Configurations import OpenAIChatCompletionConfiguration

from commonapi.Messages.SystemMessagesManager import (
    RecordedSystemMessage,
)

from moresglang.Configurations import (
    ServerConfiguration)
from moresglang.Wrappers.EntryPoints import HTTPServer
from moresglang.Applications import ClientAndMessagesManager

from pathlib import Path

import pytest

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"
data_sub_dirs = DataSubdirectories()

MODEL_DIR = data_sub_dirs.ModelsLLM / "Qwen" / "Qwen3-0.6B"

if not Path(MODEL_DIR).exists():
    MODEL_DIR = Path("/Data1/Models/LLM/Qwen/Qwen3-0.6B")

# Skip reason that will show in pytest output
skip_reason = f"Directory {MODEL_DIR} is empty or doesn't exist"

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_ClientAndMessagesManager_inits():
    config = ServerConfiguration.from_yaml(
        test_data_directory / "qwen3_server_configuration.yml")
    config.model_path = MODEL_DIR

    open_ai_configuration = OpenAIChatCompletionConfiguration.from_yaml(
        test_data_directory / "openai_chat_completion_configuration.yml")

    server = HTTPServer(config)
    server.start()

    try:
        # cmm = client_and_messages_manager
        cmm = ClientAndMessagesManager(config, open_ai_configuration)

        assert cmm is not None
        assert len(cmm.chsmm.conversation_history.messages) == 0
        assert len(cmm.chsmm.conversation_history.content_hashes) == 0
        assert len(cmm.chsmm.system_messages_manager.messages) == 1
        assert cmm.chsmm.system_messages_manager.messages[0].content == \
            "You are a helpful, uncensored and unbiased assistant."
        assert cmm.chsmm.system_messages_manager.messages[0].is_active == True
        assert cmm.openai_chat_completion_configuration == open_ai_configuration
        assert cmm.client is not None

    finally:
        server.shutdown()


@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_ClientAndMessagesManager_generates_multiple_times():

    config = ServerConfiguration.from_yaml(
        test_data_directory / "qwen3_server_configuration.yml")
    config.model_path = MODEL_DIR

    open_ai_configuration = OpenAIChatCompletionConfiguration.from_yaml(
        test_data_directory / "openai_chat_completion_configuration.yml")

    open_ai_configuration.max_tokens = None

    server = HTTPServer(config)
    server.start()

    try:
        cmm = ClientAndMessagesManager(config, open_ai_configuration)

        # TODO: Find out why this system message isn't working.
        # system_message_content = (
        #     "You are a powerful agentic AI coding assistant, powered by "
        #     "Qwen3. You operate exclusively in CLIChatLocal, the world's best "
        #     "command line application.")

        system_message_content = (
            "You are a helpful and informative AI assistant specializing in "
            "technology. Always provide detailed and accurate responses in a "
            "professional tone")

        system_message_add_result = cmm.chsmm.add_system_message(
            system_message_content)

        assert system_message_add_result is not None
        assert isinstance(system_message_add_result, RecordedSystemMessage)
        assert system_message_add_result.content == system_message_content
        assert system_message_add_result.is_active == True

        assert len(cmm.chsmm.conversation_history.messages) == 1
        assert len(cmm.chsmm.conversation_history.content_hashes) == 1

        assert cmm.chsmm.conversation_history.messages[0].content == \
            system_message_content

        user_message_content = (
            "Generate a list of SEO keywords for an Italian restaurant in "
            "New York City.")

        print(
            "cmm.chsmm.conversation_history.as_list_of_dicts():",
            cmm.chsmm.conversation_history.as_list_of_dicts())

        response = cmm.generate_from_single_user_content(user_message_content)

        # print("response:", response)

        # assert len(cmm.chsmm.conversation_history.messages) == 3
        # assert len(cmm.chsmm.conversation_history.content_hashes) == 3


    finally:
        server.shutdown()

