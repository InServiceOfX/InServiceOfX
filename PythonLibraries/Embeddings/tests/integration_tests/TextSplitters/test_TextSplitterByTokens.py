from corecode.Utilities import DataSubdirectories, is_model_there

from embeddings.TextSplitters import (get_token_count, TextSplitterByTokens)
from sentence_transformers import SentenceTransformer

from tools.YouTubeTranscripts import ProcessYouTube
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api import YouTubeTranscriptApi

from pathlib import Path
import json

import pytest

data_subdirectories = DataSubdirectories()

youtube_url_1 = "https://www.youtube.com/watch?v=F7MxPxNbFUw&t=303s"

relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

python_libraries_path = Path(__file__).parents[4]
test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
    "CommonAPI" / "tests" / "TestData"
test_conversation_path = test_data_path / "test_enable_thinking_true.json"

def load_test_conversation():
    with open(test_conversation_path, "r") as f:
        return json.load(f)

@pytest.mark.skipif(
    not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_TextSplitterByTokens_inits():
    text_splitter = TextSplitterByTokens(model_path=model_path)

    assert text_splitter.max_tokens == 512
    assert text_splitter.model_tokenizer is not None
    assert text_splitter.add_special_tokens is None

def test_TextSplitterByTokens_split_text_works():
    video_id = ProcessYouTube.extract_video_id(youtube_url_1)
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)

    text_formatted = TextFormatter().format_transcript(fetched_transcript)

    raw_text_length = len(text_formatted)

    text_splitter = TextSplitterByTokens(model_path=model_path)
    assert text_splitter.max_tokens == 512
    assert text_splitter.model_tokenizer is not None
    assert text_splitter.add_special_tokens is None

    print("raw_text_length: ", raw_text_length)
    raw_token_count = get_token_count(  
        text_splitter.model_tokenizer,
        text_formatted,
        text_splitter.add_special_tokens)

    assert raw_token_count > text_splitter.max_tokens
    print("raw_token_count: ", raw_token_count)

    split_text = text_splitter.split_text(text_formatted)
    assert isinstance(split_text, list)
    assert isinstance(split_text[0], str)
    assert len(split_text) > 0
    print("split_text length: ", len(split_text))

    reconstructed_text = "".join(split_text)
    assert reconstructed_text == text_formatted

    for chunk in split_text:
        chunk_token_count = get_token_count(
            text_splitter.model_tokenizer,
            chunk,
            text_splitter.add_special_tokens)
        assert chunk_token_count <= text_splitter.max_tokens
        print("chunk_token_count: ", chunk_token_count)

    embedding_model = SentenceTransformer(str(model_path), device = "cuda:0",)

    # This still won't warn.
    failed_embed = embedding_model.encode(
        text_formatted,
        normalize_embeddings=True,
        )
    assert failed_embed.shape == (1024,)

    print((
        "Double check if we get any warnings when we attempt to create an "
        "embedding over token size"))

    for chunk in split_text:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        assert embedding.shape == (1024,)

def test_TextSplitterByTokens_splits_long_text():
    conversation = load_test_conversation()

    text_splitter = TextSplitterByTokens(model_path=model_path)
    assert text_splitter.max_tokens == 512
    assert text_splitter.model_tokenizer is not None
    assert text_splitter.add_special_tokens is None

    split_chunks = []
    for message in conversation:
        split_chunks.append(text_splitter.split_text(message["content"]))

    assert split_chunks[0] == [conversation[0]["content"],]
    assert split_chunks[2] == [conversation[2]["content"],]
    assert split_chunks[3] == [conversation[3]["content"],]
    assert split_chunks[4] == [conversation[4]["content"],]
    assert split_chunks[5] == [conversation[5]["content"],]
    assert split_chunks[6] == [conversation[6]["content"],]
    assert split_chunks[7] == [conversation[7]["content"],]
    assert split_chunks[8] == [conversation[8]["content"],]
    assert split_chunks[9] == [conversation[9]["content"],]
    assert split_chunks[10] == [conversation[10]["content"],]
    assert split_chunks[12] == [conversation[12]["content"],]
    assert split_chunks[13] == [conversation[13]["content"],]
    assert split_chunks[14] == [conversation[14]["content"],]
    assert split_chunks[15] == [conversation[15]["content"],]

    assert len(split_chunks[1]) == 3
    assert text_splitter.get_token_count(split_chunks[1][0]) == 470
    assert text_splitter.get_token_count(split_chunks[1][1]) == 499
    assert text_splitter.get_token_count(split_chunks[1][2]) == 120

    assert len(split_chunks[11]) == 2
    assert text_splitter.get_token_count(split_chunks[11][0]) == 472
    assert text_splitter.get_token_count(split_chunks[11][1]) == 190

    reconstructed_text = "".join(split_chunks[1])
    assert reconstructed_text == conversation[1]["content"]

    reconstructed_text = "".join(split_chunks[11])
    assert reconstructed_text == conversation[11]["content"]

def test_text_to_embedding_works():
    conversation = load_test_conversation()

    text_splitter = TextSplitterByTokens(model_path=model_path)

    embedding_model = SentenceTransformer(str(model_path), device = "cuda:0",)

    embeddings = []
    for message in conversation:
        embeddings.append(
            text_splitter.text_to_embedding(
                embedding_model,
                message["content"]))

    assert len(embeddings) == len(conversation)
    for i in range(len(embeddings)):
        for j in range(len(embeddings[i])):
            assert len(embeddings[i][j])== 1024
