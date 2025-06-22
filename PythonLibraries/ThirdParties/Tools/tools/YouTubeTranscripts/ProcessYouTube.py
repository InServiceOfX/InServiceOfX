from pathlib import Path

from embeddings.TextSplitters import TextSplitterByTokens
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api import YouTubeTranscriptApi

import pydantic_core
import re

def extract_video_id(url):
    if not url:
        return None
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def simple_youtube_video_id_extraction(url):
    """
    See
    https://build5nines.com/python-get-youtube-video-transcript-from-url-for-use-in-generative-ai-and-rag-summarization/    
    """
    if not url:
        return None
    video_id = url.replace('https://www.youtube.com/watch?v=', '')
    # Split by '&' to remove additional parameters and take only the video ID.
    video_id = video_id.split('&')[0]
    return video_id

class ProcessYouTube:
    def __init__(self, url: str, model_path: str | Path):
        self.url = url
        self.model_path = model_path
        self._video_id = extract_video_id(url)
        self._text_splitter = TextSplitterByTokens(model_path=model_path)
        self._chunks = None
        self._embeddings = None
        self._embedding_jsons = None

    def get_and_process_transcript(self):
        ytt_api = YouTubeTranscriptApi()
        # TODO: the result of fetch(..) contains a lot of useful metadata about
        # each section; consider using in the future.
        fetched_transcript = ytt_api.fetch(self._video_id)
        text_formatted = TextFormatter().format_transcript(fetched_transcript)
        self._chunks = self._text_splitter.split_text(text_formatted)

        self._embeddings = []
        self._embedding_jsons = []
        for chunk in self._chunks:
            embedding = self._embedding_model.encode(
                chunk,
                normalize_embeddings=True)
            self._embeddings.append(embedding)
            self._embedding_jsons.append(
                pydantic_core.to_json(embedding.tolist()).decode())

