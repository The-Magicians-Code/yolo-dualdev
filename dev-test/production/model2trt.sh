#!/usr/bin/env bash
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Description: Convert user specified ONNX model to TensorRT model

if [ ! -f /usr/local/bin/trtexec ]; then
    echo "trtexec not set up. initialising"
    sudo cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/
fi
# User defined ONNX model path without .onnx suffix
model="models/yolov5m6_640x640_batch_1"
# workspace_size=$((1<<30))
trtexec --onnx="$model".onnx --saveEngine="$model".engine --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 # --workspace=$workspace_size