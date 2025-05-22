from pytube import YouTube

# Replace the URL with the YouTube video link you're interested in
video_url = "https://www.youtube.com/watch?v=1yHly8dYhIQ"

# Create a YouTube object
yt = YouTube(video_url)

# Access various metadata

print(yt)

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

# Print the metadata
for key, value in video_info.items():
    print(f"{key}: {value}")