'''
In this python file it will run and download the videos to test the AI then using the pretrained model drone_obj_det.pt 
it will go and run the videos and any detection with confidence above the conf_threshold will be made into a jpg and saved 
in detections folder. 

In case you run this script without the pretrained model it will go and download the dataset I used for making drone_obj_det.pt 
then augment the data twice and train the model using that data. it is currently set to 45 epochs with a patience of 5 it takes 
about 7 minutes per epoch. After training it will take the best pt file and rename it drone_obj_det.pt and put it in the 
assignments_3 folder.

The model used in this is the YOLOv8 I chose this model over the Faster_RCNN due to a few factors. First is more documentation 
on the YOLO models, second is the ease of use and of making the annotations, lastly it is capable of being given a video for a 
source and automatically putting it into frames and ran live for you to see as it detects and tracks the drone. 

Since we were given an extension for this project. I plan on improving the augmentation methods and retraining it to attempt at
getting better results and capable to track the drone more accuratly in video_1.
'''


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
import supervision as sv
import xml.etree.ElementTree as ET
from pytube import YouTube
from PIL import Image
from ultralytics import YOLO
from matplotlib import pyplot as plt

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

vid_dir = 'assignment_3/videos'
drone_det_dir = os.path.join("assignment_3", "drone_obj_det.pt")

if not os.path.exists(vid_dir):

    # Create a dir to store the videos
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

if not os.path.exists(drone_det_dir):
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

if not os.path.exists(drone_det_dir):
    yaml_lines = [
        "names:", "- Drone", "nc: 1", 
        f"test: {os.path.join(project_dir,'dataset', 'test', 'images')}", 
        f"train: {os.path.join(project_dir,'dataset', 'train', 'images')}", 
        f"val: {os.path.join(project_dir,'dataset', 'val', 'images')}"
        ]

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
if not os.path.exists(drone_det_dir):
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
if not os.path.exists(drone_det_dir):
    augment_and_resize_data(train_dir, target_frame_size, 2)
    augment_and_resize_data(val_dir, target_frame_size, 2)
    augment_and_resize_data(test_dir, target_frame_size, 2)
#endregion

#region YOLO Model
model_dir = "assignment_3/model"
os.makedirs(model_dir, exist_ok=True)

model = YOLO(os.path.join(model_dir, 'yolov8n.pt'))
model = YOLO(os.path.join(model_dir, 'yolov8n-seg.pt'))

if not os.path.exists(drone_det_dir):
    model.train(
        data=dataset_yaml_path,
        epochs=45,
        patience = 5,
        batch = -1,
        imgsz = 640,
        save = True
    )
    
    train_folder_dir = os.path.join(model_dir, 'runs', 'detect')

    if os.path.exists(train_folder_dir):
        for folder in sorted(os.listdir(train_folder_dir), reverse=True):
            weights_folder_path = os.path.join(train_folder_dir, folder, 'weights')
            if not os.path.exists(weights_folder_path):
                continue
            if 'best.pt' in os.listdir(weights_folder_path):
                source_path = os.path.join(weights_folder_path, 'best.pt')
                shutil.move(source_path, drone_det_dir)
                break

    metrics = model.val()

    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)
else:
    model = YOLO(drone_det_dir)

#endregion

#region Tracking and Detection
conf_threshold = 0.3
det_dir = os.path.join('assignment_3', 'detections')
os.makedirs(det_dir, exist_ok = True)

frame_counter = 0

box_ann = sv.BoxAnnotator(
    thickness = 2,
    text_thickness = 1,
    text_scale = 0.5
)

for video in os.listdir(vid_dir):
    for result in model.track(source = os.path.join(vid_dir, video), show = True, stream = True):
        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)

        detections = detections[detections.confidence >= conf_threshold]

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(detections.confidence, detections.class_id)
        ]

        frame = box_ann.annotate(scene = frame, detections = detections, labels = labels)

        cv2.imshow('yolov8', frame)

        if len(detections) > 0:
            frame_filename =f"{os.path.splitext(video)[0]}_frame_{frame_counter}.jpg"
            frame_path = os.path.join(det_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

        frame_counter += 1

        #breaks loop if you press esc incase you set source to webcam
        if(cv2.waitKey(30) == 27):
            break
#endregion