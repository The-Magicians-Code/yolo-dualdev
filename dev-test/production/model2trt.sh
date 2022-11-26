#!/usr/bin/env bash

# User defined ONNX model path without .onnx suffix
model=""
trtexec --onnx="$model".onnx --saveEngine="$model".engine --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16