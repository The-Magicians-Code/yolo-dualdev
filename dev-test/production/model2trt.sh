#!/usr/bin/env bash
if [ ! -f /usr/local/bin/trtexec ]; then
    echo "trtexec not set up. initialising"
    sudo cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/;
fi
# User defined ONNX model path without .onnx suffix
model=""
trtexec --onnx="$model".onnx --saveEngine="$model".engine --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16