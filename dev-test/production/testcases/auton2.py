import os
import csv
import cv2
import torch
from glob import glob
from pathlib import Path

model = torch.hub.load("ultralytics/yolov5", "custom", f"../models/yolov5m6_640x640_batch_1.engine") # This line is important since it contains RT execution
input_params = model.model.bindings["images"].shape  # Retrieve input size of the model
img_size = input_params[-1] # One dimension since n x n shape

# images = glob("../datasets/*/")
# images = images[:2]
# print(images)
# # for image in images:
# # cams = [cv2.imread(file) for file in glob(f"{images[0]}*.jpg")]
# # print(cams[0])
# print(Path(glob(f"{images[0]}*.jpg")[0]).with_suffix(".txt"))

thickness = 4
fontscale = 0.75
fontthick = 2

def make_label_folders():
    with open("../datasets/all_classes.txt", "r") as all:
        classes = all.read().split("\n")
        print(classes)
        [Path(f"../datasets/{folder}/labels/").mkdir(parents=True, exist_ok=True) for folder in classes]

def write_detections_x86_64(detection, stream, class_id, label):
    # height, width = stream.shape[:2]
    dets = []
    scores = []
    for i in range(detection.shape[0]):    
        if detection.iloc[i]["class"] != 8: # 8 - boat, ship, vessel
            continue

        xmin = float(detection.iloc[i]["xmin"] / (img_size))
        xmax = float(detection.iloc[i]["xmax"] / (img_size))
        ymin = float(detection.iloc[i]["ymin"] / (img_size))
        ymax = float(detection.iloc[i]["ymax"] / (img_size))
        confidence = detection.iloc[i]['confidence']
        score_txt = f"{(confidence * 100.0):.0f}%"
        scores.append(score_txt)
        dets.append([class_id, xmin, ymin, xmax, ymax])

    with open(label, "w") as f:
        writer = csv.writer(f)
        writer.writerows(dets)
        
    print(scores)
def main():
    image_folders = glob("../datasets/*/")
    all_classes = os.listdir("../datasets")[1:]
    # image_folders = image_folders[:1]
    # all_classes = all_classes[0]
    # print(image_folders, all_classes)
    for class_id, (folder, class_name) in enumerate(zip(image_folders, all_classes)):
        image_labels = [Path(image).with_suffix(".txt") for image in glob(f"{folder}*.jpg")]
        images = [cv2.imread(file) for file in glob(f"{folder}*.jpg")]
        # print(class_id)
        # print(image_labels[0])
        inputs = [cv2.resize(image, (img_size, img_size)) for image in images]
        for input, label in zip(inputs, image_labels):
            results = model(input, size=img_size)
            detections = results.pandas().xyxy
            # print(detections[0])
            write_detections_x86_64(detections[0], input, class_id, label)
    # print(all_classes)

main()