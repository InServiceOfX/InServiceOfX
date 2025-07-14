from commonapi.Messages.PermanentConversation import PermanentConversation
from corecode.Utilities import DataSubdirectories
from moretransformers.Applications.Messages \
    import MakeMessageEmbeddingsWithSentenceTransformer
from pathlib import Path
from sentence_transformers import SentenceTransformer
from warnings import warn
import time
import statistics
from dataclasses import dataclass
from typing import List

@dataclass
class TimingResult:
    role_length: int
    content_length: int
    embedding_time: float
    chars_per_second: float

import sys

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

def test_iterate_through_messages_and_make_embeddings():
    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    maker = \
        MakeMessageEmbeddingsWithSentenceTransformer(embedder=embedding_model)

    permanent_conversation = PermanentConversation()

    for message in example_conversation:
        permanent_conversation.add_message_as_content(
            content=message["content"],
            role=message["role"])

    assert len(permanent_conversation.messages) == len(example_conversation)

    timing_results: List[TimingResult] = []

    for index, message in enumerate(permanent_conversation.messages):
        role_length = len(message.role)
        content_length = len(message.content)
        total_length = role_length + content_length

        start_time = time.time()
        embedding = maker.make_embedding_from_content(
            content=message.content,
            role=message.role)
        message.embedding = embedding
        end_time = time.time()
        embedding_time = end_time - start_time
        chars_per_second = \
            total_length / embedding_time if embedding_time > 0 else 0
        timing_results.append(TimingResult(
            role_length=role_length,
            content_length=content_length,
            embedding_time=embedding_time,
            chars_per_second=chars_per_second))

        print(
            f"Message {index+1}: {embedding_time:.4f}s ({total_length} chars, {chars_per_second:.1f} chars/s)")

    embedding_times = [r.embedding_time for r in timing_results]
    total_lengths = [r.role_length + r.content_length for r in timing_results]
    chars_per_second = [r.chars_per_second for r in timing_results]

    for message in permanent_conversation.messages:
        assert message.embedding is not None

    print(f"\n=== Performance Statistics ===")
    print(f"Total messages processed: {len(timing_results)}")
    print(f"Total time: {sum(embedding_times):.4f}s")
    print(f"Average time per message: {statistics.mean(embedding_times):.4f}s")
    print(f"Min time: {min(embedding_times):.4f}s")
    print(f"Max time: {max(embedding_times):.4f}s")
    print(f"Standard deviation: {statistics.stdev(embedding_times):.4f}s")
    print(f"Average chars per second: {statistics.mean(chars_per_second):.1f}")
    print(f"Average length per second: {sum(total_lengths) / sum(embedding_times):.1f}")

def test_iterate_through_messages_and_make_embeddings_all_together():
    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cuda:0",)

    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    maker = \
        MakeMessageEmbeddingsWithSentenceTransformer(embedder=embedding_model)

    permanent_conversation = PermanentConversation()

    for message in example_conversation:
        permanent_conversation.add_message_as_content(
            content=message["content"],
            role=message["role"])

    start_time = time.time()
    for message in permanent_conversation.messages:
        embedding = maker.make_embedding_from_content(
            content=message.content,
            role=message.role)
        message.embedding = embedding
    end_time = time.time()
    embedding_time = end_time - start_time
    print(f"Total time: {embedding_time:.4f}s")
