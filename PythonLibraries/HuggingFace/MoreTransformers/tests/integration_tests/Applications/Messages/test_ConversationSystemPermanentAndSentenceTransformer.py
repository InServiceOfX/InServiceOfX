from moretransformers.Applications.Messages import \
    ConversationSystemPermanentAndSentenceTransformer
from commonapi.Messages.Messages import AssistantMessage, UserMessage
from corecode.Utilities import DataSubdirectories
from sentence_transformers import SentenceTransformer
from pathlib import Path
from warnings import warn
import pytest
import sys
import json

# To import CreateExampleConversation
common_api_test_data_path = Path(__file__).parents[6] / "ThirdParties" / \
    "APIs" / "CommonAPI" / "tests" / "TestData"
if common_api_test_data_path.exists() and \
    str(common_api_test_data_path) not in sys.path:
    sys.path.append(str(common_api_test_data_path))
elif not common_api_test_data_path.exists():
    warn(
        f"CommonAPI test data path does not exist: {common_api_test_data_path}")
from CreateExampleConversation import CreateExampleConversation

data_sub_dirs = DataSubdirectories()

EMBEDDING_MODEL_DIR = data_sub_dirs.Models / "Embeddings" / "BAAI" / \
    "bge-large-en-v1.5"

def test_ConversationSystemPermanentAndSentenceTransformer_works():
    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    cspst = \
        ConversationSystemPermanentAndSentenceTransformer(embedding_model)

    cspst.add_system_message(example_conversation[0]["content"])

    for message in example_conversation[1:]:
        if message["role"] == "user":
            cspst.append_message(UserMessage(message["content"]))
        elif message["role"] == "assistant":
            cspst.append_message(AssistantMessage(message["content"]))

    list_of_messages_dict = cspst.get_conversation_as_list_of_dicts()

    assert len(list_of_messages_dict) == len(example_conversation)

    for index, message in enumerate(list_of_messages_dict):
        assert message["role"] == example_conversation[index]["role"]
        assert message["content"] == example_conversation[index]["content"]

    assert len(cspst.csp.permanent_conversation.messages) == \
        len(example_conversation)

    for index, message_pair in enumerate(
        cspst.csp.permanent_conversation.message_pairs):
        i = 2 * index + 1
        assert message_pair.conversation_pair_id == index
        assert message_pair.content_0 == example_conversation[i]["content"]
        assert message_pair.content_1 == example_conversation[i+1]["content"]
        assert message_pair.role_0 == example_conversation[i]["role"]
        assert message_pair.role_1 == example_conversation[i+1]["role"]
        assert message_pair.embedding is not None

    message_embedding = cspst.csp.permanent_conversation.messages[0].embedding
    loaded_message_embedding = json.loads(message_embedding)
    assert loaded_message_embedding is not None
    assert len(loaded_message_embedding) == 1024
    assert loaded_message_embedding[0] == pytest.approx(0.026325367391109467)
    assert loaded_message_embedding[1] == pytest.approx(-0.0014000479131937027)

    message_embedding = cspst.csp.permanent_conversation.messages[1].embedding
    loaded_message_embedding = json.loads(message_embedding)
    assert loaded_message_embedding is not None
    assert len(loaded_message_embedding) == 1024
    assert loaded_message_embedding[0] == pytest.approx(0.04801274091005325)
    assert loaded_message_embedding[1] == pytest.approx(0.011487155221402645)

    message_embedding = cspst.csp.permanent_conversation.message_pairs[0].embedding
    loaded_message_embedding = json.loads(message_embedding)
    assert loaded_message_embedding is not None
    assert len(loaded_message_embedding) == 1024
    assert loaded_message_embedding[0] == pytest.approx(0.053619712591171265)
    assert loaded_message_embedding[1] == pytest.approx(0.01472424902021885)

    for message in cspst.csp.permanent_conversation.messages:
        assert message.embedding is not None
        assert len(json.loads(message.embedding)) == 1024