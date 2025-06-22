from tools.YouTubeTranscripts.ProcessYouTube import extract_video_id
from youtube_transcript_api.formatters import (JSONFormatter, TextFormatter)
from youtube_transcript_api import (
    FetchedTranscript,
    FetchedTranscriptSnippet,
    YouTubeTranscriptApi,
)

import json
import pytest

youtube_url_1 = "https://www.youtube.com/watch?v=F7MxPxNbFUw&t=303s"

def test_YouTubeTranscriptApi_get_transcript_fails():

    video_id = extract_video_id(youtube_url_1)
    # https://neo4j.com/blog/developer/youtube-transcripts-knowledge-graphs-rag/
    # https://github.com/ganesh3/rag-youtube-assistant/blob/08dfe3fffee0d8f247c0bc9def492dd8e8373f41/app/transcript_extractor.py
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    assert isinstance(transcript, list)
    assert len(transcript) > 0
    print("length of transcript: ", len(transcript))
    assert isinstance(transcript[0], dict)
    # Only these 3 "keys".
    assert 'text' in transcript[0]
    assert 'start' in transcript[0]
    assert 'duration' in transcript[0]

    raw_character_count = 0
    for item in transcript:
        raw_character_count += len(item['text'])
    print("raw_character_count: ", raw_character_count)

    # FAILED unit_tests/test_youtube_transcript_api.py::test_YouTubeTranscriptApi - AttributeError: 'dict' object has no attribute 'text'

    with pytest.raises(AttributeError):
        test_transcript = TextFormatter().format_transcript(transcript)

    # # Format the transcript as JSON.
    # json_transcript = JSONFormatter().format_transcript(transcript)
    # assert isinstance(json_transcript, str)

def test_YouTubeTranscriptApi_fetch():
    video_id = extract_video_id(youtube_url_1)
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)

    assert isinstance(fetched_transcript, FetchedTranscript)

    # https://github.com/jdepoix/youtube-transcript-api/blob/master/youtube_transcript_api/_transcripts.py

    fetched_snippets = getattr(fetched_transcript, "snippets", None)
    assert isinstance(fetched_snippets, list)
    assert len(fetched_snippets) > 0
    assert isinstance(fetched_snippets[0], FetchedTranscriptSnippet)
    assert hasattr(fetched_snippets[0], "text")
    assert hasattr(fetched_snippets[0], "start")
    assert hasattr(fetched_snippets[0], "duration")

    raw_character_count = 0
    for snippet in fetched_snippets:
        raw_character_count += len(snippet.text)
    print("raw_character_count: ", raw_character_count)

    assert hasattr(fetched_transcript, "video_id")
    assert hasattr(fetched_transcript, "language")
    assert hasattr(fetched_transcript, "language_code")
    assert hasattr(fetched_transcript, "is_generated")

    # https://github.com/jdepoix/youtube-transcript-api#formatter-example

    text_formatted = TextFormatter().format_transcript(fetched_transcript)

    assert isinstance(text_formatted, str)
    assert len(text_formatted) > 0

    print("text_formatted length: ", len(text_formatted))

    assert len(text_formatted) >= raw_character_count

    json_formatted = JSONFormatter().format_transcript(fetched_transcript)
    assert isinstance(json_formatted, str)
    assert len(json_formatted) > 0

    json_formatted_dict = json.loads(json_formatted)
    assert isinstance(json_formatted_dict, list)
    assert len(json_formatted_dict) == len(fetched_snippets)

    assert isinstance(json_formatted_dict[0], dict)
    assert 'text' in json_formatted_dict[0]
    assert 'start' in json_formatted_dict[0]
    assert 'duration' in json_formatted_dict[0]