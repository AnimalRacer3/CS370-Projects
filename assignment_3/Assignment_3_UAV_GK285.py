import requests
import os
import re
from pytube import YouTube

# Timeout time in second till it times out
timeout_sec = 900

# Regular expression pattern for matching YouTube video URLs
youtube_url_pattern = re.compile(r'^https?://(?:www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+')

# URL of the videos to download
vid_urls = [
    'https://drive.google.com/uc?export=download&id=1VkvhMaZfV4If_qiup4XNlVSNO2rUkynr',
    'https://drive.google.com/uc?export=download&id=1JYATx5H1L99ke-lKSsuJuyTBLLh8ohtV'
]

# Create a dir to store the videos
vid_dir = 'assignment_3/videos'
os.makedirs(vid_dir, exist_ok=True)

try:
    for url in vid_urls:
        # Check if the URL is a valid YouTube video link
        if youtube_url_pattern.match(url):
            # This is a YouTube video URL
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            if os.path.isfile(os.path.join(vid_dir, yt.title + ".mp4")):
                print(f'{yt.title} already exists in {vid_dir}. Skipping download.')
                continue
            stream.download(output_path=vid_dir, timeout=timeout_sec)
            print(f'{yt.title} has been successfully downloaded to {vid_dir}')
        else:
            # This is a generic web URL
            response = requests.get(url, timeout=(timeout_sec, timeout_sec))
            if response.status_code == 200:
                content_disposition = response.headers.get('Content-Disposition')
                if content_disposition:
                    vid_name = content_disposition.split(';')[1].split('=')[1].strip('"')
                else:
                    # If the 'Content-Disposition' header is not present, use the last part of the URL as the file name
                    vid_name = url.split('/')[-1]

                if os.path.isfile(os.path.join(vid_dir, vid_name)):
                    print(f'{vid_name} already exists in {vid_dir}. Skipping download.')
                    continue
                
                vid_path = os.path.join(vid_dir, vid_name)

                with open(vid_path, 'wb') as file:
                    file.write(response.content)

                print(f'{vid_name} downloaded successfully to {vid_dir}')
            else:
                print(f'Failed to download {url}')
except requests.Timeout:
    print(f"Request timed out. Try increasing the timeout to {timeout_sec} seconds.")