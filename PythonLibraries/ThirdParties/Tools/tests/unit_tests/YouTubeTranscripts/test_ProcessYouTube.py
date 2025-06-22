from tools.YouTubeTranscripts.ProcessYouTube import (
    extract_video_id,
    simple_youtube_video_id_extraction)

youtube_url_1 = "https://www.youtube.com/watch?v=F7MxPxNbFUw&t=303s"

def test_extract_video_id():
    video_id_1 = extract_video_id(youtube_url_1)
    assert video_id_1 == "F7MxPxNbFUw"

def test_simple_youtube_video_id_extraction():
    video_id_1 = simple_youtube_video_id_extraction(youtube_url_1)
    assert video_id_1 == "F7MxPxNbFUw"