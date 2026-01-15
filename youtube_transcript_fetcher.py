from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json
import re

json_formatter = JSONFormatter()

def extract_video_id(video_link):
    """Extract video ID from various YouTube URL formats"""
    # Handle youtu.be links
    if 'youtu.be/' in video_link:
        video_id = video_link.split('youtu.be/')[-1].split('?')[0].split('&')[0]
    # Handle youtube.com/watch?v= links (desktop and mobile)
    elif 'v=' in video_link:
        video_id = video_link.split('v=')[-1].split('&')[0]
    # Handle youtube.com/shorts/ links
    elif '/shorts/' in video_link:
        video_id = video_link.split('/shorts/')[-1].split('?')[0].split('&')[0]
    # Handle embed links
    elif 'youtube.com/embed/' in video_link:
        video_id = video_link.split('/embed/')[-1].split('?')[0].split('&')[0]
    else:
        # Try regex as fallback
        pattern = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, video_link)
        if match:
            video_id = match.group(1)
        else:
            return None
    return video_id

def get_youtube_transcript(video_link):
    video_id = extract_video_id(video_link)
    if not video_id:
        return "Invalid YouTube URL format."
    
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        if not transcript:
            return "No transcript available for this video."
        transcript_str = json_formatter.format_transcript(transcript)
        transcript = json.loads(transcript_str)
        text_only = " ".join([entry['text'] for entry in transcript])
        return text_only
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"