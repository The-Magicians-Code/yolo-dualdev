#!/usr/bin/env bash
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Script: shellexport.sh
# @Description: Automatically export user defined yolov5 models as ONNX formatted models for later TensorRT conversion
# @Last modified: 2023/03/18

# Define YOLOv5 model type
declare -a models=(
   # "n" 
   # "s" 
   # "m" 
   # "l" 
   # "x" 
   # "n6" 
   # "s6"
   # "m6" 
   # "l6" 
   "x6"
)
# Model input size a, since (a x a)
size=1280
# Input batch size
batch=3

echo "Making the folder for models"
mkdir models_"$size"x"$size"_batch_"$batch"
echo "Starting model exports"
for i in "${models[@]}"
do
   python ../export.py --weights yolov5"$i".pt --batch-size "$batch" --include onnx --imgsz "$size"
   mv yolov5"$i".onnx yolov5"$i"_"$size"x"$size"_batch_"$batch".onnx
   mv yolov5"$i".pt yolov5"$i"_"$size"x"$size"_batch_"$batch".pt
done
echo "Moving exported models to user specified folder"
mv yolov5*.* models_"$size"x"$size"_batch_"$batch"/
echo "Done!"