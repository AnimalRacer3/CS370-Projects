import requests
import os
import re
import tensorflow as tf
import cv2
import warnings
import numpy as np
from pytube import YouTube

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

#region Download Videos
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

vidNum = 0

try:
    for url in vid_urls:
        vid_name = f"video_{vidNum}.mp4"
        # Check if the URL is a valid YouTube video link
        if youtube_url_pattern.match(url):
            # This is a YouTube video URL
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            stream.download(output_path=vid_dir, timeout=timeout_sec)
            print(f'{vid_name} downloaded successfully')
        else:
            # This is a generic web URL
            response = requests.get(url, timeout=(timeout_sec, timeout_sec))
            if response.status_code == 200:              
                vid_path = os.path.join(vid_dir, vid_name)

                with open(vid_path, 'wb') as file:
                    file.write(response.content)

                print(f'{vid_name} downloaded successfully')
            else:
                print(f'Failed to download {url}')
        vidNum += 1
except requests.Timeout:
    print(f"Request timed out. Try increasing the timeout to {timeout_sec} seconds.")
#endregion

#region Load Model
model_path = 'assignment_3/Model'
model = tf.saved_model.load(model_path)
#endregion

#region Split Video Into Frames
target_frame_size = (1024, 1024)

frame_dir = 'assignment_3/frames'
os.makedirs(frame_dir, exist_ok=True)

detections_dir = 'assignment_3/detections'
os.makedirs(detections_dir, exist_ok=True)

for i in range(len(vid_urls)):
    vid_frame_dir = os.path.join(frame_dir, f"video_{i}")
    os.makedirs(vid_frame_dir, exist_ok=True)

    vid_path = os.path.join(vid_dir, f"video_{i}.mp4")
    cap = cv2.VideoCapture(vid_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, target_frame_size)

        frame_filename = f"frame_{frame_count:06d}.jpg"
        frame_filepath = os.path.join(vid_frame_dir, frame_filename)
        
        cv2.imwrite(frame_filepath, frame)

        frame_count += 1
    cap.release()

    print(f"Saved frames in video {i}")

    #region Object Detection
    for frame_filename in os.listdir(vid_frame_dir):
        image = tf.image.decode_image(tf.io.read_file(os.path.join(vid_frame_dir, frame_filename)))

        image = tf.expand_dims(image, axis=0)
        image_np = image.numpy()

        detections = model(image_np)

        # Process and filter the detection results


        # Save the frames with detection if any objects are detected
        if len(detections['detection_boxes']) > 0:
            detection_frame_path = os.path.join(detections_dir, frame_filename)
            cv2.imwrite(detection_frame_path, image_np[0])
    #endregion
#endregion