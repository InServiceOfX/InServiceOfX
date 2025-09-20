from commonapi.Messages import (
    ConversationSystemAndPermanent,
    ParsePromptsCollection,
    AssistantMessage,
    UserMessage)

from corecode.Utilities import DataSubdirectories, is_model_there

from pathlib import Path
import json

python_libraries_path = Path(__file__).parents[5]

test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
    "CommonAPI" / "tests" / "TestData"

test_conversation_path = test_data_path / "test_enable_thinking_true.json"

def load_test_conversation():
    with open(test_conversation_path, "r") as f:
        return json.load(f)

def test_EmbedPermanentConversation_works():
    conversation = load_test_conversation()
    