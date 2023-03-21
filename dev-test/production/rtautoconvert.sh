#!/usr/bin/env bash
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Script: rtautoconvert.sh
# @Description: Automatically converts all ONNX models in /models/ folder to TensorRT format
# @Last modified: 2023/03/22

if [ ! -f /usr/local/bin/trtexec ]; then
    echo "trtexec not set up. initialising"
    sudo cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/
fi

models=$(ls models/*.onnx)
not_converted=()

for model in $models; do
    # Remove suffix from filename
    model=${model%.*}
    # Check if there exists a corresponding .engine file, if not execute trtexec
    if [ ! -f "$model".engine ]; then
        echo "$model.engine does not exist"
        not_converted+=($model)
    fi
done

echo Number of ONNX models to convert ${#not_converted[@]}
for model in $not_converted; do
    trtexec --onnx="$model".onnx --saveEngine="$model".engine --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
    echo Conversion done!
done
echo Done!