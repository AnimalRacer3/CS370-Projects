#region Imports
import requests
import os
import re
import tensorflow as tf
import cv2
import warnings
import numpy as np
import zipfile
import random
import shutil
import yaml
import appdirs
import xml.etree.ElementTree as ET
from pytube import YouTube
from PIL import Image
from ultralytics import YOLO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
#endregion

#region Setting yaml
project_dir = os.path.join(os.getcwd(),"assignment_3")
setting_dir = os.path.join(os.environ['APPDATA'], 'Ultralytics')
settings_yaml_path = os.path.join(setting_dir, 'settings.yaml')

with open(settings_yaml_path, 'r') as f:
    settings = yaml.safe_load(f)

settings['datasets_dir'] = os.path.join(project_dir, "dataset")
settings['runs_dir'] = os.path.join(project_dir, "model", "runs")
settings['weights_dir'] = os.path.join(project_dir, "model", "weights")

with open(settings_yaml_path, 'w') as f:
    yaml.dump(settings, f)
#endregion

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

model_dir = "assignment_3/model"

if not os.path.exists(model_dir):

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

#region Download Dataset
dataset_url = 'https://drive.google.com/u/4/uc?id=16CMtbV2XoZvIrVLOOjlzICNesZmGbQM_&export=download&confirm=t&uuid=e54f6130-6999-414c-aea2-555b49e873ed&at=AB6BwCBRCmV2MdO8SgQv15mSXYuB:1699477691972'
dataset_dir = 'assignment_3\dataset'
zip_file_path = os.path.join(dataset_dir,'drone_dataset.zip')

dataset_yolo_dir = os.path.join(dataset_dir, "drone_dataset_yolo\dataset_txt")
classes_path = os.path.join(dataset_yolo_dir, "classes.txt")

if not os.path.exists(model_dir):
    os.makedirs(dataset_dir, exist_ok=True)

    response = requests.get(dataset_url)

    if response.status_code == 200:
        with open(zip_file_path, 'wb') as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

    os.remove(classes_path)
#endregion

#region Splitting Data
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")
train_image_dir = os.path.join(train_dir, "images")
train_txt_dir = os.path.join(train_dir, "labels")
val_image_dir = os.path.join(val_dir, "images")
val_txt_dir = os.path.join(val_dir, "labels")
test_image_dir = os.path.join(test_dir, "images")
test_txt_dir = os.path.join(test_dir, "labels")
dataset_yaml_path = os.path.join(dataset_dir, 'data.yaml')

if not os.path.exists(model_dir):
    yaml_lines = ["names:", "- Drone", "nc: 1", f"test: {os.join(settings['datasets_dir'],'/test/images')}", f"train: {os.join(settings['datasets_dir'],'/train/images')}", f"val: {os.join(settings['datasets_dir'],'/val/images')}"]

    os.makedirs(test_txt_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(val_txt_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_txt_dir, exist_ok=True)
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    with open(dataset_yaml_path, 'w') as f:
        f.write('\n'.join(yaml_lines))

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 1 - (train_ratio + val_ratio)

    image_files = [f for f in os.listdir(dataset_yolo_dir) if f.lower().endswith('.jpg')]

    random.shuffle(image_files)

    total_images = len(image_files)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * (train_ratio + val_ratio))

    for i, image_file in enumerate(image_files):
        source_path = os.path.join(dataset_yolo_dir, image_file)
        txt_file = os.path.splitext(image_file)[0] + ".txt"
        txt_path = os.path.join(dataset_yolo_dir, txt_file)

        if i < train_split:
            destination_dir = train_dir
        elif i < val_split:
            destination_dir = val_dir
        else:
            destination_dir = test_dir

        destination_path = os.path.join(destination_dir, "images", image_file)
        destination_txt_path = os.path.join(destination_dir, "labels", txt_file)

        shutil.move(source_path, destination_path)
        shutil.move(txt_path, destination_txt_path)
#endregion

#region Clean Directories
if not os.path.exists(model_dir):
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

    xml_dir = os.path.join(dataset_dir, "dataset_xml_format")

    if os.path.exists(xml_dir):
        shutil.rmtree(xml_dir)
    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)
#endregion

#region Augmenting Data
# Turns annotations into an array of dicts
def parse_annotations(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        val = line.strip().split()
        annotation = {
            'class': int(val[0]),
            'x': float(val[1]),
            'y': float(val[2]),
            'width': float(val[3]),
            'height': float(val[4])
        }
        annotations.append(annotation)
    return annotations

# Augments the image then updates the annotations
def data_augmentation(input_image_path, input_txt_path, output_image_path, output_txt_path):
    image = cv2.imread(input_image_path)
    annotations = parse_annotations(input_txt_path)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image).numpy()
        for i in range(len(annotations)):
            annotations[i] = {
                'class': annotations[i]['class'],
                'x': 1 - annotations[i]['x'],
                'y': annotations[i]['y'],
                'width': 1 - annotations[i]['width'],
                'height': annotations[i]['height']
            }
        
    elif tf.random.uniform(()) > 0.5:
        # Apply random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.25).numpy()
    
    else:
        # Apply random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5).numpy()

    save_annotations(annotations, output_txt_path)
    cv2.imwrite(output_image_path, image)

# Resizes Image
def resize_image(image_path, target_size):
    try:
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, target_size)
        cv2.imwrite(image_path, resized_image)
        
    except Exception as e:
        print(f"Error: {e}")

def save_annotations(annotations, txt_path):
    with open(txt_path, 'w') as f:
        for ann in annotations:
            f.write(f"{ann['class']} {ann['x']} {ann['y']} {ann['width']} {ann['height']}")

def augment_and_resize_data(dir, target_frame_size, aug_amount = 1):
    image_dir = os.path.join(dir, "images")
    labels_dir = os.path.join(dir, "labels")
    for i in range(aug_amount):
        for image_file in os.listdir(image_dir):
            if image_file.lower().endswith('.jpg'):
                image_path = os.path.join(image_dir, image_file)
                txt_file = os.path.splitext(image_file)[0] + '.txt'
                txt_path = os.path.join(labels_dir, txt_file)

                aug_image_path = os.path.join(image_dir, f"aug_{image_file}")
                aug_txt_path = os.path.join(labels_dir, f"aug_{txt_file}")

                data_augmentation(image_path, txt_path, aug_image_path, aug_txt_path)

                resize_image(image_path, target_frame_size)
                resize_image(aug_image_path, target_frame_size)

target_frame_size = (640, 640)
if not os.path.exists(model_dir):
    augment_and_resize_data(train_dir, target_frame_size, 2)
    augment_and_resize_data(val_dir, target_frame_size, 2)
    augment_and_resize_data(test_dir, target_frame_size, 2)
#endregion

#region YOLO Model
os.makedirs(model_dir, exist_ok=True)

model = YOLO(model_dir + '/yolov8n.pt')
model.train(
    data=dataset_yaml_path,
    epochs=3,
    patience = 50,
    batch = -1,
    imgsz = 640,
    save = True
)

metrics = model.val()

print(metrics.box.map)
print(metrics.box.map50)
print(metrics.box.map75)
print(metrics.box.maps)
#endregion