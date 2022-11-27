#!/usr/bin/env bash
# Automatically converts all ONNX models in /models/ folder to TensorRT .engine format

if [ ! -f /usr/local/bin/trtexec ]; then
    echo "trtexec not set up. initialising"
    sudo cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/
fi
models=$(ls models/*.onnx)
echo $models

for model in $models; do
    # Remove suffix from filename
    model=${model%.*}
    # Check if there exists a corresponding .engine file, if not execute trtexec
    if [ ! -f "$model".engine ]; then
        echo "$model.engine does not exist"
        trtexec --onnx="$model".onnx --saveEngine="$model".engine --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
    fi
done