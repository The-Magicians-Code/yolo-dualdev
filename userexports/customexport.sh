#!/usr/bin/env bash
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Description: Export user trained model as ONNX model using pretrained weights and classes file


# Model weights file name
model_name=latest.pt
# Model input size (a x a)
size=640
# Input batch size
batch=1
# Classes file from dataset if model is custom trained else please set it to default as ../data/coco128.yaml
data=data.yaml

python ../export.py --data "$data" --weights "$model_name" --batch-size "$batch" --include onnx --imgsz "$size"
