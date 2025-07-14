from commonapi.Messages.Messages import (
    AssistantMessage,
    UserMessage,
    SystemMessage)
from corecode.Utilities import DataSubdirectories
from moretransformers.Applications.Messages \
    import MakeMessageEmbeddingsWithSentenceTransformer
from pathlib import Path
from sentence_transformers import SentenceTransformer
from warnings import warn

import json
import pytest
import sys

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

def test_make_embedding_from_message_works():
    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    maker = \
        MakeMessageEmbeddingsWithSentenceTransformer(embedder=embedding_model)

    example_message_0 = SystemMessage(
        content=example_conversation[0]["content"])
    example_message_1 = UserMessage(
        content=example_conversation[1]["content"])
    example_message_2 = AssistantMessage(
        content=example_conversation[2]["content"])

    embedding_0 = maker.make_embedding_from_message(example_message_0)
    embedding_1 = maker.make_embedding_from_message(example_message_1)
    embedding_2 = maker.make_embedding_from_message(example_message_2)

    assert isinstance(embedding_0, str)
    assert isinstance(embedding_1, str)
    assert isinstance(embedding_2, str)

    assert embedding_0 is not None
    assert embedding_1 is not None
    assert embedding_2 is not None

    embedding_0_list = json.loads(embedding_0)
    embedding_1_list = json.loads(embedding_1)
    embedding_2_list = json.loads(embedding_2)

    assert len(embedding_0_list) == 1024
    assert len(embedding_1_list) == 1024
    assert len(embedding_2_list) == 1024

    assert embedding_0_list[0] == pytest.approx(0.026325367391109467)
    assert embedding_0_list[1] == pytest.approx(-0.0014000479131937027)
    assert embedding_1_list[0] == pytest.approx(0.04801274091005325)
    assert embedding_1_list[1] == pytest.approx(0.011487155221402645)
    assert embedding_2_list[0] == pytest.approx(0.04269658774137497)
    assert embedding_2_list[1] == pytest.approx(0.008013206534087658)

def test_make_embedding_from_message_pair_works():
    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    maker = \
        MakeMessageEmbeddingsWithSentenceTransformer(embedder=embedding_model)

    example_message_1 = UserMessage(
        content=example_conversation[1]["content"])
    example_message_2 = AssistantMessage(
        content=example_conversation[2]["content"])

    embedding_0 = maker.make_embedding_from_message_pair(
        example_message_1,
        example_message_2)

    assert isinstance(embedding_0, str)

    embedding_0_list = json.loads(embedding_0)

    assert len(embedding_0_list) == 1024

    assert embedding_0_list[0] == pytest.approx(0.053619712591171265)
    assert embedding_0_list[1] == pytest.approx(0.01472424902021885)

def test_make_embedding_from_content_works():
    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    maker = \
        MakeMessageEmbeddingsWithSentenceTransformer(embedder=embedding_model)

    embedding_0 = maker.make_embedding_from_content(
        content=example_conversation[0]["content"],
        role=example_conversation[0]["role"])
    embedding_1 = maker.make_embedding_from_content(
        content=example_conversation[1]["content"],
        role=example_conversation[1]["role"])
    embedding_2 = maker.make_embedding_from_content(
        content=example_conversation[2]["content"],
        role=example_conversation[2]["role"])

    assert isinstance(embedding_0, str)
    assert isinstance(embedding_1, str)
    assert isinstance(embedding_2, str)

    embedding_0_list = json.loads(embedding_0)
    embedding_1_list = json.loads(embedding_1)
    embedding_2_list = json.loads(embedding_2)

    assert len(embedding_0_list) == 1024
    assert len(embedding_1_list) == 1024
    assert len(embedding_2_list) == 1024

    assert embedding_0_list[0] == pytest.approx(0.026325367391109467)
    assert embedding_0_list[1] == pytest.approx(-0.0014000479131937027)
    assert embedding_1_list[0] == pytest.approx(0.04801274091005325)
    assert embedding_1_list[1] == pytest.approx(0.011487155221402645)
    assert embedding_2_list[0] == pytest.approx(0.04269658774137497)
    assert embedding_2_list[1] == pytest.approx(0.008013206534087658)

def test_make_embedding_from_content_pair_works():
    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    maker = \
        MakeMessageEmbeddingsWithSentenceTransformer(embedder=embedding_model)

    embedding_0 = maker.make_embedding_from_content_pair(
        content_0=example_conversation[1]["content"],
        content_1=example_conversation[2]["content"],
        role_0=example_conversation[1]["role"],
        role_1=example_conversation[2]["role"])

    assert isinstance(embedding_0, str)

    embedding_0_list = json.loads(embedding_0)

    assert len(embedding_0_list) == 1024

    assert embedding_0_list[0] == pytest.approx(0.053619712591171265)
    assert embedding_0_list[1] == pytest.approx(0.01472424902021885)