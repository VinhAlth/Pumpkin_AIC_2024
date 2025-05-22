import subprocess
import json
from moviepy.editor import VideoFileClip
import datetime
from pytube import YouTube, Search  

def get_creation_time_ffmpeg(file_path):
    # Run ffmpeg command to get metadata in JSON format
    result = subprocess.run(
        ['ffmpeg', '-i', file_path, '-f', 'ffmetadata', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Parse the output
    metadata = result.stderr
    creation_time = None
    duration = None
    for line in metadata.split('\n'):
        if 'creation_time' in line:
            creation_time = line.split(':')[1].strip()[:-3]
    
    creation_time = datetime.datetime.strptime(creation_time, "%Y-%m-%d")
    
    return creation_time

def get_duration(file_path):
    # Get video duration using moviepy
    video = VideoFileClip(file_path)
    duration = video.duration  # Duration in seconds
    video.close()
    return duration

def get_youtube_date_length(video_url):

    # Create a YouTube object
    yt = YouTube(video_url)

    # Access various metadata
    video_info = {
        "title": yt.title,
        "author": yt.author,
        "publish_date": yt.publish_date,
        "views": yt.views,
        "length": yt.length,
        "description": yt.description,
        "keywords": yt.keywords,
        "rating": yt.rating,
        "thumbnail_url": yt.thumbnail_url
    }
    
    return video_info['publish_date'], video_info['length']

# Example usage
file_path = '/dataset/AIC2024/original_dataset/0/videos/Videos_L01/video/L01_V001.mp4'
# creation_time = get_creation_time_ffmpeg(file_path)
# duration = get_duration(file_path)
# print(f"Creation Time: {creation_time - datetime.timedelta(days=1)}")
# print(f"Duration: {round(duration)}")

# print(type(get_youtube_info("https://www.youtube.com/watch?v=1yHly8dYhIQ")))

def search_and_match(video_path):
    local_creation_time = get_creation_time_ffmpeg(video_path)
    local_duration = get_duration(video_path)
    
    search_prompt = "Chương trình 60s " + str(local_creation_time)[:10]
    #s = Search(search_prompt + str(local_creation_time))
    print(search_prompt)

search_and_match(file_path)