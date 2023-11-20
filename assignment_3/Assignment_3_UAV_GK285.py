#region Imports
import requests
import os
import re
import cv2
import warnings
import zipfile
import random
import shutil
import yaml
import time
import keyboard
import numpy as np
import supervision as sv
import albumentations as A
from pytube import YouTube
from ultralytics import YOLO
from random import randrange
from filterpy.kalman import KalmanFilter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
#endregion

def main():
    #region Setting yaml
    print("Setting yaml...")

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
    print("Setting up videos...")
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
    print("setting up dataset...")

    dataset_url = 'https://drive.google.com/u/4/uc?id=16CMtbV2XoZvIrVLOOjlzICNesZmGbQM_&export=download&confirm=t&uuid=e54f6130-6999-414c-aea2-555b49e873ed&at=AB6BwCBRCmV2MdO8SgQv15mSXYuB:1699477691972'
    dataset_dir = 'assignment_3\dataset'
    zip_file_path = os.path.join(dataset_dir,'drone_dataset.zip')

    dataset_yolo_dir = os.path.join(dataset_dir, "drone_dataset_yolo\dataset_txt")
    classes_path = os.path.join(dataset_yolo_dir, "classes.txt")

    if not os.path.exists(dataset_dir):
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
    print("Splitting Dataset...")

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

    if os.path.exists(dataset_yolo_dir):
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
            if os.path.splitext(image_file)[0] in ["0013", "0086", "0295", "foto02843", "foto04235", "pic_071", "pic_074", "yoto01161"]:
                os.remove(source_path)
                os.remove(txt_path)
                continue

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

    drone_det_dir = os.path.join("assignment_3", "drone_obj_det.pt")

    #region Clean Directories
    print("Cleaning directories...")
    def zip_and_delete(path, zip_path):
        shutil.make_archive(zip_path, 'zip', path)
        shutil.rmtree(path)

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
    print("Data augmenting...")
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

    def annotations_to_bboxes(txt_path = None, annotations = None):
        if txt_path is None and annotations is None:
            return None, None
        elif annotations is None:
            annotations = parse_annotations(txt_path=txt_path)
        bboxes = []
        for annotation in annotations:
            bbox = [annotation['x'], annotation['y'], annotation['width'], annotation['height'], annotation['class']]
            bboxes.append(bbox)
        
        return bboxes

    def bboxes_to_annotations(bboxes):
        annotations = []

        for bbox in bboxes:
            annotation = {
                'class': int(bbox[4]),
                'x': float(bbox[0]),
                'y': float(bbox[1]),
                'width': float(bbox[2]),
                'height': float(bbox[3])
            }
            annotations.append(annotation)

        return annotations

    def get_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # Resizes Image
    def resize_image(image_path, target_size):
        image = get_image(image_path)
        resized_image = cv2.resize(image, target_size)
        return resized_image

    # Augments the image then updates the annotations
    def data_augmentation(input_image_path, input_txt_path, output_image_path, output_txt_path, target_size):
        image = resize_image(input_image_path, target_size)
        bboxes = annotations_to_bboxes(txt_path=input_txt_path)
        temp_bboxes = []
        for bbox in bboxes:
            bbox = [max(0.0, min(float(bbox[0]), 1)), max(0.0, min(float(bbox[1]), 1)), 
                    max(0.0, min(float(bbox[2]), 1)), max(0.0, min(float(bbox[3]), 1)),
                    int(bbox[4])]
            temp_bboxes.append(bbox)
        bboxes = temp_bboxes

        # Augmentation with Albumentations
        transform = A.Compose([
            A.Blur(p = 0.2),
            A.RandomCrop(width = int(randrange(60, 80)*target_size[0]/100), height = int(randrange(60, 80)*target_size[1]/100), p = 0.2),
            A.Rotate(limit = 20, p = 0.2, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p = 0.5),
            A.RGBShift(r_shift_limit = 25, g_shift_limit = 25, b_shift_limit = 25, p = 0.2),
            A.ColorJitter(p = 0.2),
            A.MotionBlur(p = 0.2),
            A.RandomFog(p = 0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomRain(p = 0.2),
            A.Downscale(p=0.2),
            A.ZoomBlur(p = 0.2),
            A.RandomScale(scale_limit=(0.8, 1.2), p = 2),
            A.RandomSizedBBoxSafeCrop(width=target_size[0], height=target_size[1], p = 0.2),
            A.ElasticTransform(alpha=120, sigma = 120*0.05, alpha_affine=120*0.03, p = 0.2),
            A.RandomContrast(limit=0.2, p=0.2),
            A.GridDistortion(p=0.2)
        ], bbox_params=A.BboxParams(format="yolo", min_area= 1024, min_visibility= 0.6))

        transformed = transform(image=image, bboxes = bboxes)
        image = transformed["image"]
        bboxes = transformed['bboxes']

        bboxes = np.clip(bboxes, 0, 1)

        save_annotations(bboxes_to_annotations(bboxes), output_txt_path)
        cv2.imwrite(output_image_path, image)

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

                    data_augmentation(image_path, txt_path, aug_image_path, aug_txt_path, target_frame_size)
                    
                    cv2.imwrite(image_path, resize_image(image_path, target_frame_size))

    target_frame_size = (384, 640)

    dirs_to_loop = [train_dir, val_dir, test_dir]
    dirs_to_check = [train_image_dir, val_image_dir, test_image_dir]

    for dir, image_dir in zip(dirs_to_loop, dirs_to_check):
        matching_files = [file for file in os.listdir(image_dir) if file.startswith("aug_")]
        if not matching_files:
            print("\t- Augmenting dataset from dir...")
            augment_and_resize_data(dir, target_frame_size, 8)
    #endregion

    #region YOLO Model
    print("Creating YOLOv8 model...")
    model_dir = "assignment_3/model"
    os.makedirs(model_dir, exist_ok=True)

    model = YOLO(os.path.join(model_dir, 'yolov8n.pt'))
    #model = YOLO(os.path.join(model_dir, 'yolov8n-seg.pt'))

    def save_model(train_folder_dir, drone_det_dir):
        for folder in sorted(os.listdir(train_folder_dir), reverse=True):
            weights_folder_path = os.path.join(train_folder_dir, folder, 'weights')
            if not os.path.exists(weights_folder_path):
                continue
            if 'best.pt' in os.listdir(weights_folder_path):
                source_path = os.path.join(weights_folder_path, 'best.pt')
                shutil.move(source_path, drone_det_dir)
                break
    
    def train_yolov8(model, dataset_yaml_path, model_dir, drone_det_dir, seed_given = 0):
        paused = False
        def on_key_event(e):
            nonlocal paused
            if e.name == 'end':
                paused = not paused
                print(f"Training {'paused' if paused else 'resumed'}.")
        
        keyboard.hook(on_key_event)

        model.train(
            data=dataset_yaml_path,
            epochs=10000,
            patience = 100,
            batch = -1,
            imgsz = 640,
            save = True,
            device = '0',
            seed = seed_given,
            dropout = 0.2
        )
            
        train_folder_dir = os.path.join(model_dir, 'runs', 'detect')

        if os.path.exists(train_folder_dir):
            save_model(train_folder_dir, drone_det_dir)

        metrics = model.val()

        print(metrics.box.map)
        print(metrics.box.map50)
        print(metrics.box.map75)
        print(metrics.box.maps)

        return model

    if not os.path.exists(drone_det_dir):
        model = train_yolov8(model, dataset_yaml_path, model_dir, drone_det_dir)  
    else:
        model = YOLO(drone_det_dir)
    #endregion

    #region Tracking and Detection
    print("Object tracking and detection...")
    global prev_time, prev_pos, known_size, prev_dist
    prev_time = time.time()
    prev_pos = np.array([0,0])

    known_size = 31
    prev_dist = 0

    def est_relative_dist(apparent_size):
        global prev_dist, known_size
        focal_length = 500

        cur_dist = (known_size * focal_length) / apparent_size
        relative_distance_change = cur_dist - prev_dist

        prev_dist = cur_dist

        return relative_distance_change

    def calc_speed(cur_pos):
        global prev_pos, prev_time
        cur_time = time.time()
        time_elapsed = cur_time - prev_time

        displacement = np.linalg.norm(cur_pos - prev_pos)

        speed = displacement / time_elapsed if time_elapsed > 0 and time_elapsed < 3 else 0

        prev_pos = cur_pos
        prev_time = cur_time

        return speed

    # Kalman filter initialization
    kf = KalmanFilter(dim_x=4, dim_z=2)

    measurement_noise = 25
    process_noise = 50

    line_thickness = 2
    line_color = (0, 0, 255)

    # Set initial parameters for the Kalman filter
    initial_state = np.array([0, 0, 0, 0])  # Initial state (x, y, vx, vy)
    kf.x = initial_state
    kf.F = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])  # Observation matrix
    kf.P *= 1e3  # Covariance matrix
    kf.R = np.diag([measurement_noise, measurement_noise])  # Measurement noise
    kf.Q = np.diag([process_noise, process_noise, process_noise, process_noise])  # Process noise

    conf_threshold = 0.2
    det_dir = os.path.join('assignment_3', 'detections')
    os.makedirs(det_dir, exist_ok = True)

    frame_counter = 0

    box_ann = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 1,
        text_scale = 0.5
    )

    for video in os.listdir(vid_dir):
        history = []
        video_path = os.path.join(vid_dir, video)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release the video capture object
        cap.release()
        for result in model.track(source = video_path, show = True, stream = True):
            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)

            detections = detections[detections.confidence >= conf_threshold]
            if len(detections) < 1:
                continue
            best_detection_index = 0
            for i in range(len(detections.confidence)):
                if detections.confidence[best_detection_index] < detections.confidence[i]:
                    best_detection_index = i
            detection = detections[best_detection_index]

            # Kalman filter update with detection
            bbox_coordinates = detection.xyxy[0]
            conf = detection.confidence[0]
            x, y, w, h = bbox_coordinates
            width = w - x
            height = h - y

            measurement = np.array([x, y])
                
            speed = calc_speed(measurement)
            distance = est_relative_dist(width)
            size = (width + height) / 2
            kf.Q = np.diag([speed, distance, size, process_noise])

            kf.R = np.diag([conf, measurement_noise])

            kf.predict()
            kf.update(measurement)

            # Retrieve the esitmated status from Kalman filter
            estimated_x, estimated_y, _, _ = kf.x
            detection.xyxy = np.array([[estimated_x, estimated_y, w, h]])

            bbox_middle_width = int(estimated_x + (width)/2)
            bbox_middle_height = int(estimated_y + (height)/2)

            line_thickness = int(max(2, min(int(width * height / 400), 8)))

            completion_percentage = frame_counter/total_frames

            red = int(255 * np.cos(completion_percentage * np.pi / 2))
            green = int(255 * np.sin(completion_percentage * np.pi / 2))
            blue = int(255 * np.cos((completion_percentage + 1/3) * np.pi / 2))

            line_color = (
                red,
                green,
                blue
            )

            history.append([(bbox_middle_width, bbox_middle_height), line_color, line_thickness])

            for i in range(1, len(history)):
                point1, _, _ = history[i-1]
                point2, color, thickness = history[i-1]
                cv2.line(frame, point1, point2, color, thickness)

            label = [f"{model.model.names[detection.class_id[0]]} {detection.confidence[0]:0.2f}"]

            frame = box_ann.annotate(scene = frame, detections = detection, labels = label)

            cv2.imshow('yolov8', frame)

            if len(detections) > 0:
                frame_filename =f"{os.path.splitext(video)[0]}_frame_{frame_counter}.jpg"
                frame_path = os.path.join(det_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

            frame_counter += 1

            #breaks loop if you press esc incase you set source to webcam
            if(cv2.waitKey(30) == 27):
                break

    zip_and_delete(det_dir, det_dir)
    #endregion

if __name__ == '__main__':
    main()