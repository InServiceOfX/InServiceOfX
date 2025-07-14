from commonapi.Messages import ConversationSystemAndPermanent
from commonapi.Messages.Messages import AssistantMessage, UserMessage
from corecode.Utilities import DataSubdirectories
from moretransformers.Applications.Messages import (
    create_missing_embeddings,
    MakeMessageEmbeddingsWithSentenceTransformer)
from sentence_transformers import SentenceTransformer
from pathlib import Path
from warnings import warn
import sys
import time

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

def setup_conversation_system_and_permanent():
    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0
    csp = ConversationSystemAndPermanent()
    csp.add_system_message(example_conversation[0]["content"])
    for message in example_conversation[1:]:
        if message["role"] == "user":
            csp.append_message(UserMessage(message["content"]))
        elif message["role"] == "assistant":
            csp.append_message(AssistantMessage(message["content"]))

    return csp, example_conversation

def test_create_missing_embeddings_works():
    csp, _ = setup_conversation_system_and_permanent()

    for message in csp.permanent_conversation.messages:
        assert message.embedding is None
    for message_pair in csp.permanent_conversation.message_pairs:
        assert message_pair.embedding is None

    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    mmewst = MakeMessageEmbeddingsWithSentenceTransformer(embedding_model)

    start_time = time.time()
    create_missing_embeddings(csp, mmewst)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    for message in csp.permanent_conversation.messages:
        assert message.embedding is not None
    for message_pair in csp.permanent_conversation.message_pairs:
        assert message_pair.embedding is not None

def test_with_database_interface():
    csp, example_conversation = setup_conversation_system_and_permanent()

    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    mmewst = MakeMessageEmbeddingsWithSentenceTransformer(embedding_model)
    create_missing_embeddings(csp, mmewst)
