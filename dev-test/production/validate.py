#!/usr/bin/env python3
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Description: Perform model accuracy validation on predefined, annotated datasets

import cv2
import os
import torch

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

imgs = load_images_from_folder("/workspace/torching/datasets/typesofships.v6i.yolov5pytorch/valid/images/")
num_of_imgs = len(imgs)

yolo = "l l6 m m6 n n6 s s6 x x6".split()
imsize = [640, 1280]

outdata = ["Model name, Input size, Accuracy"]

# yolo = ["s"]
# imsize = [640]
for size in imsize:
    for suffix in yolo:
        counter = 0
        model = torch.hub.load("ultralytics/yolov5", f"yolov5{suffix}", pretrained=True) # Load unoptimised model from Ultralytics's servers
        for image in imgs:
            model_input = cv2.resize(image, (size, size))
            outs = model(model_input, size=size)
            outs = outs.pandas().xyxy
            if "boat" in [i for i in outs[0]["name"]]:
                counter += 1
        print(f"Done with yolov5{suffix}, {size}x{size}, Acc {counter / num_of_imgs}")
        outdata.append(f"yolov5{suffix}, {size}x{size}, {counter / num_of_imgs}")

print("Writing to file")
with open(f"resultings.csv", "w") as out:
    out.write("\n".join(outdata))
print("Done!")
            