"""
USAGE:
Despite being "integration" tests, running an actual model, it should be ok
running *all* the tests asynchronously:

/InServiceOfX/PythonLibraries/Tools/tests# pytest -s ./integration_tests/RAG/PermanentConversation/test_EmbedPermanentConversation.py
"""
from commonapi.Messages import (
    ConversationSystemAndPermanent,
    AssistantMessage,
    UserMessage)

from corecode.Utilities import DataSubdirectories, is_model_there

from embeddings.TextSplitters import TextSplitterByTokens
from sentence_transformers import SentenceTransformer

from tools.RAG.PermanentConversation.EmbedPermanentConversation \
    import EmbedPermanentConversation

from CreateExampleConversation import CreateExampleConversation

from dataclasses import dataclass
from pathlib import Path
import json, pytest, time, statistics

@dataclass
class ChunkingTimingResult:
    """Statistics for chunking and embedding performance."""
    total_messages: int
    total_message_pairs: int
    total_message_chunks: int
    total_message_pair_chunks: int
    total_chunks: int
    total_embedding_time: float
    total_content_length: int
    chars_per_second: float
    avg_time_per_chunk: float

@dataclass
class IndividualChunkResult:
    """Individual chunk statistics."""
    chunk_index: int
    content_length: int
    embedding_time: float
    chars_per_second: float
    chunk_type: str
    role: str

data_subdirectories = DataSubdirectories()
relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

python_libraries_path = Path(__file__).parents[5]

test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
    "CommonAPI" / "tests" / "TestData"

test_conversation_path = test_data_path / "test_enable_thinking_true.json"

import sys
if str(test_data_path) not in sys.path:
    sys.path.append(str(test_data_path))

def load_test_conversation():
    with open(test_conversation_path, "r") as f:
        return json.load(f)

def test_EmbedPermanentConversation_works():
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

    assert len(message_chunks) == 19
    assert len(message_pair_chunks) == 11

def setup_test(conversation):
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

    return embed_pc

def test_embed_conversation_makes_embeddings_for_messages():
    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    embed_pc = setup_test(example_conversation)

    message_chunks, message_pair_chunks = embed_pc.embed_conversation()

    assert len(message_chunks) == 8
    assert len(message_pair_chunks) == 3

    assert type(message_chunks[0].embedding) == list
    assert type(message_chunks[0].embedding[0]) == float
    assert len(message_chunks[0].embedding) == 1024
    assert len(message_chunks[1].embedding) == 1024
    assert len(message_chunks[2].embedding) == 1024

    assert message_chunks[0].embedding[0] == pytest.approx(0.03347434476017952)
    assert message_chunks[0].embedding[1] == pytest.approx(
        -0.0017530706245452166)
    assert message_chunks[1].embedding[0] == pytest.approx(0.03339601680636406)
    assert message_chunks[1].embedding[1] == pytest.approx(
        -0.018528396263718605)
    assert message_chunks[2].embedding[0] == pytest.approx(0.04801274091005325)
    assert message_chunks[2].embedding[1] == pytest.approx(0.011487155221402645)

    assert message_chunks[3].embedding[0] == pytest.approx(0.04269658774137497)
    assert message_chunks[3].embedding[1] == pytest.approx(0.008013206534087658)

    assert message_pair_chunks[0].embedding[0] == pytest.approx(
        0.053619712591171265)
    assert message_pair_chunks[0].embedding[1] == pytest.approx(
        0.01472424902021885)
    assert message_pair_chunks[1].embedding[0] == pytest.approx(
        0.05208650603890419)
    assert message_pair_chunks[1].embedding[1] == pytest.approx(
        0.009506979957222939)
    assert message_pair_chunks[2].embedding[0] == pytest.approx(
        0.05018622428178787)
    assert message_pair_chunks[2].embedding[1] == pytest.approx(
        -0.0003014350950252265)

@pytest.mark.skipif(
    not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_embed_conversation_performance_statistics():
    """Test performance statistics for embedding conversation with chunking."""
    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    embed_pc = setup_test(example_conversation)

    message_chunks, message_pair_chunks = embed_pc.embed_conversation()

    # Measure total time for embedding conversation
    start_time = time.time()
    message_chunks, message_pair_chunks = embed_pc.embed_conversation()
    end_time = time.time()
    
    total_embedding_time = end_time - start_time
    
    # Calculate statistics
    total_messages = len(embed_pc._pc.messages)
    total_message_pairs = len(embed_pc._pc.message_pairs)
    total_message_chunks = len(message_chunks)
    total_message_pair_chunks = len(message_pair_chunks)
    total_chunks = total_message_chunks + total_message_pair_chunks
    
    # Calculate total content length
    total_content_length = 0
    for chunk in message_chunks + message_pair_chunks:
        total_content_length += len(chunk.content)
    
    chars_per_second = total_content_length / total_embedding_time \
        if total_embedding_time > 0 else 0
    avg_time_per_chunk = total_embedding_time / total_chunks \
        if total_chunks > 0 else 0
    
    # Create timing result
    timing_result = ChunkingTimingResult(
        total_messages=total_messages,
        total_message_pairs=total_message_pairs,
        total_message_chunks=total_message_chunks,
        total_message_pair_chunks=total_message_pair_chunks,
        total_chunks=total_chunks,
        total_embedding_time=total_embedding_time,
        total_content_length=total_content_length,
        chars_per_second=chars_per_second,
        avg_time_per_chunk=avg_time_per_chunk,
    )

    # Print performance statistics
    print(f"\n=== EmbedPermanentConversation Performance Statistics ===")
    print(f"Total messages processed: {timing_result.total_messages}")
    print(f"Total message pairs processed: {timing_result.total_message_pairs}")
    print(f"Total message chunks created: {timing_result.total_message_chunks}")
    print(
        f"Total message pair chunks created: {timing_result.total_message_pair_chunks}")
    print(f"Total chunks created: {timing_result.total_chunks}")
    print(f"Total embedding time: {timing_result.total_embedding_time:.4f}s")
    print(
        f"Total content length: {timing_result.total_content_length:,} characters")
    print(f"Average chars per second: {timing_result.chars_per_second:.1f}")
    print(f"Average time per chunk: {timing_result.avg_time_per_chunk:.4f}s")
    print(
        f"Chunking ratio (chunks/messages): {timing_result.total_chunks / timing_result.total_messages:.2f}")

@pytest.mark.skipif(
    not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_embed_conversation_detailed_chunking_analysis():
    """Test detailed analysis of chunking behavior and individual chunk
    statistics."""
    example_conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    embed_pc = setup_test(example_conversation)

    # Measure embedding time
    start_time = time.time()
    message_chunks, message_pair_chunks = embed_pc.embed_conversation()
    end_time = time.time()
    
    total_embedding_time = end_time - start_time
    
    # Analyze individual chunks
    all_chunks = message_chunks + message_pair_chunks
    chunk_results = []
    
    for i, chunk in enumerate(all_chunks):
        chunk_result = IndividualChunkResult(
            chunk_index=i,
            content_length=len(chunk.content),
            # Approximate
            embedding_time=total_embedding_time / len(all_chunks),
            chars_per_second=\
                len(chunk.content) / (total_embedding_time / len(all_chunks)) \
                    if total_embedding_time > 0 else 0,
            chunk_type="message" if chunk in message_chunks else "message_pair",
            role=chunk.role
        )
        chunk_results.append(chunk_result)
    
    # Calculate chunking statistics
    message_chunk_lengths = [len(chunk.content) for chunk in message_chunks]
    message_pair_chunk_lengths = [
        len(chunk.content) for chunk in message_pair_chunks]
    
    # Token counts (approximate)
    message_chunk_tokens = [
        embed_pc._text_splitter.get_token_count(chunk.content) \
            for chunk in message_chunks]
    message_pair_chunk_tokens = [
        embed_pc._text_splitter.get_token_count(chunk.content) \
            for chunk in message_pair_chunks]
    
    # Print detailed analysis
    print(f"\n=== Detailed Chunking Analysis ===")
    print(f"Total embedding time: {total_embedding_time:.4f}s")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Message chunks: {len(message_chunks)}")
    print(f"Message pair chunks: {len(message_pair_chunks)}")
    
    if message_chunk_lengths:
        print(f"\n--- Message Chunks ---")
        print(
            f"Average length: {statistics.mean(message_chunk_lengths):.1f} chars")
        print(f"Min length: {min(message_chunk_lengths)} chars")
        print(f"Max length: {max(message_chunk_lengths)} chars")
        print(f"Std dev: {statistics.stdev(message_chunk_lengths):.1f} chars")
        print(f"Average tokens: {statistics.mean(message_chunk_tokens):.1f}")
        print(f"Max tokens: {max(message_chunk_tokens)}")
    
    if message_pair_chunk_lengths:
        print(f"\n--- Message Pair Chunks ---")
        print(
            f"Average length: {statistics.mean(message_pair_chunk_lengths):.1f} chars")
        print(f"Min length: {min(message_pair_chunk_lengths)} chars")
        print(f"Max length: {max(message_pair_chunk_lengths)} chars")
        print(
            f"Std dev: {statistics.stdev(message_pair_chunk_lengths):.1f} chars")
        print(
            f"Average tokens: {statistics.mean(message_pair_chunk_tokens):.1f}")
        print(f"Max tokens: {max(message_pair_chunk_tokens)}")
    
    # Analyze chunking behavior
    original_messages = embed_pc._pc.messages
    original_pairs = embed_pc._pc.message_pairs
    
    messages_split = \
        sum(1 for msg in original_messages \
            if embed_pc._text_splitter.get_token_count(msg.content) > \
                embed_pc._text_splitter.max_tokens)
    pairs_split = \
        sum(1 for pair in original_pairs \
            if embed_pc._text_splitter.get_token_count(
                f"{pair.content_0}\n{pair.content_1}") > \
                    embed_pc._text_splitter.max_tokens)
    
    print(f"\n--- Chunking Behavior ---")
    print(
        f"Messages that needed splitting: {messages_split}/{len(original_messages)}")
    print(
        f"Message pairs that needed splitting: {pairs_split}/{len(original_pairs)}")
    print(
        f"Split rate: {(messages_split + pairs_split) / (len(original_messages) + len(original_pairs)) * 100:.1f}%")

def test_recreate_conversation_messages_from_chunks_works():
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
    message_chunks, _ = embed_pc.embed_conversation()

    reconstructed_messages = EmbedPermanentConversation.recreate_conversation_messages_from_chunks(
        message_chunks)

    assert len(reconstructed_messages) == len(csp.pc.messages)
    for i in range(len(reconstructed_messages)):
        assert reconstructed_messages[i].content == csp.pc.messages[i].content
        assert reconstructed_messages[i].role == csp.pc.messages[i].role
        assert reconstructed_messages[i].datetime == csp.pc.messages[i].datetime
        assert reconstructed_messages[i].hash == csp.pc.messages[i].hash
        assert reconstructed_messages[i].conversation_id == csp.pc.messages[i].conversation_id

def test_recreate_conversation_message_pairs_from_chunks_works():
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
    _, message_pair_chunks = embed_pc.embed_conversation()

    reconstructed_message_pairs = \
        EmbedPermanentConversation.recreate_conversation_messages_pairs_from_chunks(
            message_pair_chunks)

    assert len(reconstructed_message_pairs) == len(csp.pc.message_pairs)
    for i in range(len(reconstructed_message_pairs)):
        content = reconstructed_message_pairs[i]["content"]
        content_0 = content.split("assistant:")[0][6:-1]
        content_1 = content.split("assistant:")[1]
        # TODO: Originally we added a '\n' and we hadn't been able to split
        # correctly to get rid of this extra '\n'.
        assert content_0 in csp.pc.message_pairs[i].content_0
        assert content_1 == csp.pc.message_pairs[i].content_1
        assert reconstructed_message_pairs[i]["datetime"] == csp.pc.message_pairs[i].datetime
        assert reconstructed_message_pairs[i]["hash"] == csp.pc.message_pairs[i].hash
        assert reconstructed_message_pairs[i]["conversation_pair_id"] == \
            csp.pc.message_pairs[i].conversation_pair_id