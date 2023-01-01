import os
import cv2
import torch
from glob import glob
from pathlib import Path

# images = glob("../datasets/*/")
# images = images[:2]
# print(images)
# # for image in images:
# # cams = [cv2.imread(file) for file in glob(f"{images[0]}*.jpg")]
# # print(cams[0])
# print(Path(glob(f"{images[0]}*.jpg")[0]).with_suffix(".txt"))

# model = torch.hub.load("ultralytics/yolov5", "custom", f"../models/yolov5x6_1280x1280_batch_3.engine") # This line is important since it contains RT execution
# input_params = model.model.bindings["images"].shape  # Retrieve input size of the model
# img_size = input_params[-1] # One dimension since n x n shape

thickness = 4
fontscale = 0.75
fontthick = 2

def make_label_folders():
    with open("../datasets/all_classes.txt", "r") as all:
        classes = all.read().split("\n")
        print(classes)
        [Path(f"../datasets/{folder}/labels/").mkdir(parents=True, exist_ok=True) for folder in classes]

def plotdetections_x86_64(detection, stream):
    height, width = stream.shape[:2]
    for i in range(detection.shape[0]):    
        if detection.iloc[i]["class"] != 8: # 8 - boat, ship, vessel
            continue

        xmin = float(detection.iloc[i]["xmin"] / (img_size))
        xmax = float(detection.iloc[i]["xmax"] / (img_size))
        ymin = float(detection.iloc[i]["ymin"] / (img_size))
        ymax = float(detection.iloc[i]["ymax"] / (img_size))
        # print(xmin, xmax)
        confidence = detection.iloc[i]['confidence']
        score_txt = f"{(confidence * 100.0):.0f}%"
        
        # label = detection.iloc[i]["name"]

        cv2.rectangle(stream, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), (0, int(confidence * 255), int(255 - confidence * 255)), thickness)
        # cv2.rectangle(stream, (xmin, ymin), (xmax, ymax), colour, thickness)
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # This block for seeing the values on detection boxes
        # (w, h), _ = cv2.getTextSize(f"{label}: " + score_txt, font, fontscale, fontthick)
        # cv2.rectangle(streams, (xmin, ymax - h - 10), (xmin + w, ymax), colour, -1) # -1 to fill the rectangle
        # cv2.putText(stream, f"{label}: {score_txt}", (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)
        # cv2.putText(stream, score_txt, (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)
    
    return stream

def main():
    image_folders = glob("../datasets/*/")
    all_classes = os.listdir("../datasets")[1:]
    for class_id, (folder, class_name) in enumerate(zip(image_folders, all_classes)):
        image_labels = [Path(image).with_suffix(".txt") for image in glob(f"{folder}*.jpg")]
        print(class_id)
        print(image_labels[0])
    print(all_classes)
        # images = [cv2.imread(file) for file in glob(f"{folder}*.jpg")]

# cv2.imwrite(Path(glob(f"{images[0]}*.jpg")[0]), plotdetections_x86_64(detection, stream))

main()