import requests
import os
import re
import tensorflow as tf
import cv2
import warnings
import numpy as np
import tarfile
import zipfile
import random
import shutil
import xml.etree.ElementTree as ET
from pytube import YouTube
from PIL import Image
from tensorflow import keras
from keras import metrics

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

#region Download Dataset
dataset_url = 'https://drive.google.com/u/4/uc?id=16CMtbV2XoZvIrVLOOjlzICNesZmGbQM_&export=download&confirm=t&uuid=e54f6130-6999-414c-aea2-555b49e873ed&at=AB6BwCBRCmV2MdO8SgQv15mSXYuB:1699477691972'
dataset_dir = 'assignment_3\dataset'
zip_file_path = os.path.join(dataset_dir,'drone_dataset.zip')

dataset_xml_format_dir = os.path.join(dataset_dir, "dataset_xml_format\dataset_xml_format")

os.makedirs(dataset_dir, exist_ok=True)

response = requests.get(dataset_url)

if response.status_code == 200:
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

for filename in os.listdir(dataset_xml_format_dir):
    if filename.lower().endswith(".png"):
        img = Image.open(os.path.join(dataset_xml_format_dir, filename))
        img = img.convert("RGB")

        img.info.pop('icc', None)

        img.save(os.path.join(dataset_xml_format_dir, os.path.splitext(filename)[0] + ".jpg"))

        os.remove(os.path.join(dataset_xml_format_dir, filename))
#endregion

#region Splitting Data
train_dir = os.path.join(dataset_dir, "train")
os.makedirs(train_dir, exist_ok=True)

val_dir = os.path.join(dataset_dir, "validation")
os.makedirs(val_dir, exist_ok=True)

test_dir = os.path.join(dataset_dir, "test")
os.makedirs(test_dir, exist_ok=True)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 1 - (train_ratio + val_ratio)

image_files = [f for f in os.listdir(dataset_xml_format_dir) if f.lower().endswith('.jpg')]

random.shuffle(image_files)

total_images = len(image_files)
train_split = int(total_images * train_ratio)
val_split = int(total_images * (train_ratio + val_ratio))

for i, image_file in enumerate(image_files):
    source_path = os.path.join(dataset_xml_format_dir, image_file)
    xml_file = os.path.splitext(image_file)[0] + ".xml"
    xml_path = os.path.join(dataset_xml_format_dir, xml_file)

    if i < train_split:
        destination_dir = train_dir
    elif i < val_split:
        destination_dir = val_dir
    else:
        destination_dir = test_dir

    destination_path = os.path.join(destination_dir, image_file)
    destination_xml_path = os.path.join(destination_dir, xml_file)

    shutil.move(source_path, destination_path)
    shutil.move(xml_path, destination_xml_path)
#endregion

#region Clean Directories
for root, dirs, files in os.walk(dataset_dir, topdown=False):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if not os.listdir(dir_path):
            os.rmdir(dir_path)

yolo_dir = os.path.join(dataset_dir, "drone_dataset_yolo")

if os.path.exists(yolo_dir):
    shutil.rmtree(yolo_dir)
#endregion

#region Augmenting Data
def parse_xml_annotations(xml_file_path):
    annotations = []

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        class_label = obj.find('name').text

        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        
        annotation = {
            'class': class_label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }
        annotations.append(annotation)

    return annotations

def save_annotations_to_xml(annotations, output_path):
    root = ET.Element('annotations')

    for annotation in annotations:
        object_elem = ET.SubElement(root, 'object')

        ET.SubElement(object_elem, 'name').text = annotation['class']

        bndbox_elem = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox_elem, 'xmin').text = str(annotation['xmin'])
        ET.SubElement(bndbox_elem, 'ymin').text = str(annotation['ymin'])
        ET.SubElement(bndbox_elem, 'xmax').text = str(annotation['xmax'])
        ET.SubElement(bndbox_elem, 'ymax').text = str(annotation['ymax'])
    tree = ET.ElementTree(root)
    tree.write(output_path)

def data_augmentation(input_image_path, input_xml_path):
    image = cv2.imread(input_image_path)
    annotations = parse_xml_annotations(input_xml_path)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        for i in range(len(annotations)):
            annotations[i]['xmin'], annotations[i]['xmax'] = 1 - annotations[i]['xmax'], 1 - annotations[i]['xmin']

    if tf.random.uniform(()) > 0.5:
        # Apply random rotation to the image
        degrees = tf.cast(tf.random.uniform([], -45, 45), tf.int32)
        image = tf.image.rot90(image, k=degrees // 90)
        
        for i in range(len(annotations)):
            annotations[i]['xmin'], annotations[i]['ymin'], annotations[i]['xmax'], annotations[i]['ymax'] = (
                annotations[i]['ymin'],
                1 - annotations[i]['xmax'],
                annotations[i]['ymax'],
                1 - annotations[i]['xmin'],
            )
        
    if tf.random.uniform(()) > 0.5:
        # Apply random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.2)
    
    if tf.random.uniform(()) > 0.5:
        # Apply random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    return image, annotations

def resize_image(input_image_path, input_xml_path, output_image_path, output_xml_path, target_size):
    try:
        image = cv2.imread(input_image_path)
        xml_tree = ET.parse(input_xml_path)
        xml_root = xml_tree.getroot()

        resized_image = cv2.resize(image, target_size)

        og_width, og_height = image.shape[1], image.shape[0]
        new_width, new_height = target_size[0], target_size[1]

        for obj in xml_root.findall('object'):
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            obj.find('bndbox/xmin').text = str(int(xmin * new_width / og_width))
            obj.find('bndbox/ymin').text = str(int(ymin * new_height / og_height))
            obj.find('bndbox/xmax').text = str(int(xmax * new_width / og_width))
            obj.find('bndbox/ymax').text = str(int(ymax * new_height / og_height))

        cv2.imwrite(output_image_path, resized_image)
        xml_tree.write(output_xml_path)
        
    except Exception as e:
        print(f"Error: {e}")

def augment_and_resize_data(dir, target_frame_size):
    for image_file in os.listdir(dir):
        if image_file.lower().endswith('.jpg'):
            image_path = os.path.join(dir, image_file)
            xml_file = os.path.splitext(image_file)[0] + '.xml'
            xml_path = os.path.join(dir, xml_file)

            aug_image_path = os.path.join(dir, f"aug_{image_file}")
            aug_xml_path = os.path.join(dir, f"aug_{xml_file}")

            aug_image, aug_ann = data_augmentation(image_path, xml_path)

            aug_image_np = np.array(aug_image)

            cv2.imwrite(aug_image_path, aug_image_np)
            save_annotations_to_xml(aug_ann, aug_xml_path)

            resize_image(image_path, xml_path, image_path, xml_path, target_frame_size)
            resize_image(aug_image_path, aug_xml_path, aug_image_path, aug_xml_path, target_frame_size)

target_frame_size = (1024, 1024)
augment_and_resize_data(train_dir, target_frame_size)
augment_and_resize_data(val_dir, target_frame_size)
augment_and_resize_data(test_dir, target_frame_size)
#endregion

#region Load Model
model_dir = 'assignment_3\Models'
os.makedirs(model_dir, exist_ok=True)

model_original_dir = os.path.join(model_dir, "original")
os.makedirs(model_original_dir, exist_ok=True)

model_url = 'https://drive.google.com/uc?export=download&id=1b6FGwj7vkHl9WgGEw3tURjmnWA8hLjrP&confirm=t&uuid=30206700-aa8e-42ea-ab21-1da45585054e&at=AB6BwCCd5oeaaZOkuo_tZUKGazei:1699494607929'

model_path = os.path.join(model_original_dir, "Faster_RCNN.tar.gz")

response = requests.get(model_url, stream=True)
if response.status_code == 200:
    with open(model_path, 'wb') as f:
        f.write(response.raw.read())

with tarfile.open(model_path, 'r:gz') as model_file:
    model_file.extractall(model_original_dir)

model = tf.saved_model.load(model_original_dir)
#endregion

#region Model Training
learning_rate = 0.001
batch_size = 32
num_epochs = 10

def create_head(num_classes):
    head = keras.Sequential()

    head.add(keras.layers.GlobalAveragePooling2D())

    head.add(keras.layers.Dense(num_classes, activation = 'softmax'))

    return head

def load_image_and_annotations(image_path, xml_path):
    image = cv2.imread(image_path)
    objects = parse_xml_annotations(xml_path)

    return image, objects

def get_dataset(data_dir):
    images = []
    annotations = []

    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(data_dir, filename)
            xml_path = os.path.join(data_dir, os.path.splitext(filename)[0] + '.xml')
            image, objects = load_image_and_annotations(image_path, xml_path)
            images.append(image)
            annotations.append(objects)

    data = list(zip(images, annotations))
    random.shuffle(data)

    return data

def object_detection_loss(y_true, y_pred, lambda_coord=5.0, lambda_noobj=0.5):
    # Split the predicted and ground truth values
    pred_box_confidence = y_pred[..., 4:5]  # Objectness score
    pred_class_probs = y_pred[..., 5:]     # Class probabilities
    pred_box_coords = y_pred[..., 0:4]     # Bounding box coordinates
    
    true_box_confidence = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]
    true_box_coords = y_true[..., 0:4]
    
    # Calculate localization loss (MSE) for both x, y, w, and h
    loc_loss = lambda_coord * tf.reduce_sum(
        true_box_confidence * tf.square(pred_box_coords - true_box_coords), axis=[1, 2, 3, 4]
    )
    
    # Calculate objectness loss (MSE) for background (no object)
    no_obj_loss = lambda_noobj * tf.reduce_sum(
        (1 - true_box_confidence) * tf.square(pred_box_confidence), axis=[1, 2, 3, 4]
    )
    
    # Calculate objectness loss (MSE) for the correct object
    obj_loss = tf.reduce_sum(
        true_box_confidence * tf.square(pred_box_confidence - true_box_confidence), axis=[1, 2, 3, 4]
    )
    
    # Calculate classification loss using cross-entropy
    class_loss = tf.reduce_sum(
        true_box_confidence * tf.keras.losses.categorical_crossentropy(true_class_probs, pred_class_probs, from_logits=True),
        axis=[1, 2, 3, 4]
    )
    
    # Sum all the losses to get the total object detection loss
    detection_loss = loc_loss + no_obj_loss + obj_loss + class_loss
    
    return detection_loss

def calculate_association_loss(predicted_associations, true_associations):
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    loss = bce_loss(true_associations, predicted_associations)
    return loss

def calculate_smoothness_loss(predicted_positions):
    l2_loss = tf.keras.losses.MeanSquaredError()
    loss = l2_loss(predicted_positions[:, :-1], predicted_positions[:, 1:])
    return loss

def calculate_appearance_loss(predicted_features, true_features):
    cosine_similarity = tf.keras.losses.CosineSimilarity()
    loss = cosine_similarity(true_features, predicted_features)
    return loss

def tracking_loss(predicted_tracks, true_tracks, features):
    loss = 0.0

    for pred_track, true_track in zip(predicted_tracks, true_tracks):
        # Calculate loss based on data association, smoothness, and appearance
        association_loss = calculate_association_loss(pred_track, true_track)
        smoothness_loss = calculate_smoothness_loss(pred_track)
        appearance_loss = calculate_appearance_loss(pred_track, features)
        
        track_loss = association_loss + smoothness_loss + appearance_loss
        loss += track_loss

    return loss

def mot_loss(detection_loss, tracking_loss, lambda_detection, lambda_tracking):
    # Combine detection and tracking loss with appropriate weights
    loss = lambda_detection * detection_loss + lambda_tracking * tracking_loss
    return loss

def train_model():
    num_classes = 2 # Drones and not Drones
    new_head = create_head(num_classes)

    model.head = new_head

    train_dataset = get_dataset(train_dir)
    val_dataset = get_dataset(val_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=mot_loss, metrics=['accuracy'])

    for epoch in range(num_epochs):
        for batch in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i + batch_size]
            with tf.GradientTape() as tape:
                images, annotations = zip(*batch)
                predictions = model(images, training = True)
                loss_value = mot_loss(annotations, predictions)

            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        val_loss = 0.0
        for val_batch in val_dataset:
            val_images, val_annotations = zip(*val_batch)
            val_predictions = model(val_images, training = False)
            val_loss += mot_loss(val_annotations, val_predictions)

        val_loss /= len(val_dataset)
        print(f"Epoch {epoch+1}: Loss = {loss_value.numpy()}, Val Loss = {val_loss.numpy()}")

    # Save the fine-tuned model
    model.save(model_fine_tuned_dir)
    return model

model_fine_tuned_dir = os.path.join(model_dir, "fine_tuned", 'fine_tuned_model')
os.makedirs(f"{model_dir}/fine_tuned", exist_ok=True)

if os.path.exists(model_fine_tuned_dir):
    model = tf.saved_model.load(model_fine_tuned_dir)
else:
    model = train_model()

#endregion

#region Test Model
test_dataset = get_dataset(test_dir)

test_loss = 0.0
for i in range(0, len(test_dataset), batch_size):
    batch = test_dataset[i:i + batch_size]
    images, annotations = zip(*batch)

    test_pred = model(images, training = False)

    test_batch_loss = mot_loss(annotations, test_pred)
    test_loss += test_batch_loss

test_loss /= len(test_dataset)

print(f"Test loss: {test_loss.numpy()}")
#endregion

#region Split Video Into Frames
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