
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)
from youtube_transcript_api._errors import IpBlocked

video_id = "tqJk6vaWgiE"

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id)

    transcript = " ".join(
        snippet.text for snippet in transcript_list
    )
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

except IpBlocked:
    print("YouTube is blocking requests from your IP.")
