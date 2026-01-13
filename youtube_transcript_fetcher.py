from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json

json_formatter = JSONFormatter()

def get_youtube_transcript(video_link):
    video_id = video_link.split("v=")[-1].split("&")[0]
    transcript = YouTubeTranscriptApi().fetch(video_id)
    if not transcript:
        return "No transcript available for this video."
    transcript_str = json_formatter.format_transcript(transcript)
    transcript = json.loads(transcript_str)
    text_only = " ".join([entry['text'] for entry in transcript])
    return text_only