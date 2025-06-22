from corecode.Utilities import DataSubdirectories

from embeddings.TextSplitters import (get_token_count, TextSplitterByTokens)
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tools.YoutubeTranscripts import extract_video_id
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api import YouTubeTranscriptApi

data_sub_dirs = DataSubdirectories()
MODEL_DIR = data_sub_dirs.Models / "Embeddings" / "BAAI" / \
    "bge-large-en-v1.5"
if not Path(MODEL_DIR).exists():
    print("for MODEL_DIR:", MODEL_DIR)
    print("MODEL_DIR.exists(): ", MODEL_DIR.exists())
    MODEL_DIR = Path("/Data1/Models/Embeddings/BAAI/bge-large-en-v1.5")

youtube_url_1 = "https://www.youtube.com/watch?v=F7MxPxNbFUw&t=303s"

def test_TextSplitterByTokens():
    video_id = extract_video_id(youtube_url_1)
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)

    text_formatted = TextFormatter().format_transcript(fetched_transcript)

    raw_text_length = len(text_formatted)

    text_splitter = TextSplitterByTokens(model_path=MODEL_DIR)
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

    embedding_model = SentenceTransformer(str(MODEL_DIR), device = "cuda:0",)

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
