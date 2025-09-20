from googleapiclient.discovery import build
import googleapiclient.http

from tools.YoutubeTranscripts import extract_video_id

youtube_url_1 = "https://www.youtube.com/watch?v=F7MxPxNbFUw&t=303s"

def test_googleapiclient_for_youtube_data():
    """
    https://github.com/ganesh3/rag-youtube-assistant/blob/08dfe3fffee0d8f247c0bc9def492dd8e8373f41/app/transcript_extractor.py#L3    
    """
    http = googleapiclient.http.build_http()

    # Without an API key, you get this error:
    # googleapiclient.errors.HttpError: <HttpError 403 when requesting https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id=F7MxPxNbFUw&alt=json returned "Method doesn't allow unregistered callers (callers without established identity). Please use API Key or other form of API consumer identity to call this API.". Details: "[{'message': "Method doesn't allow unregistered callers (callers without established identity). Please use API Key or other form of API consumer identity to call this API.", 'domain': 'global', 'reason': 'forbidden'}]">

    youtube_client = build('youtube', 'v3', http=http)

    video_id = extract_video_id(youtube_url_1)

    request = youtube_client.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()
    print(response)
