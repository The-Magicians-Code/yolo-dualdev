#!/usr/bin/env python3
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Script: benchmark.py
# @Description: Perform performance benchmark on user defined neural network model
# @Last modified: 2022/11/30

import time
import torch
import argparse
import numpy as np
from pathlib import Path
from subprocess import Popen, DEVNULL, STDOUT

parser = argparse.ArgumentParser(description="Neural Network benchmark script")
parser.add_argument('--model', help="YOLOv5 unoptimised Neural Network for object detection --model yolov5m6")
parser.add_argument('--imsize', help="YOLOv5 unoptimised Neural Network input size --imsize 640")
parser.add_argument('--batch', default = 1, help="YOLOv5 unoptimised Neural Network batch size --batch 1")
parser.add_argument('--precision', choices=["fp32", "fp16", "int8"], help="YOLOv5 TensorRT optimised Neural Network precision --precision fp32")
parser.add_argument('--rt-model', help="YOLOv5 TensorRT Neural Network for object detection --rt-model yolov5m6_640x640_batch_1")
parser.add_argument('-v', '--verbose', action="store_true", help="Benchmark verbose, default False")
parser.add_argument('-e', '--export', action="store_true", help="Benchmark export results into .txt file, default False")
args = parser.parse_args()

try:
    # When ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
    Popen("export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0".split(), shell=True, stdout=DEVNULL, stderr=STDOUT)
except ImportError as e:
    print(e)

if args.rt_model:
    if not args.precision:
        parser.error("--rt-model argument requires --precision argument")
    if not Path(f"models/{args.rt_model}.engine").exists():
        raise FileNotFoundError(f"Did not find RT model {args.rt_model}.engine in /models/ folder. Specify model name ONLY without .engine suffix and make sure it is located in models/ folder")
    model_name = f"{args.rt_model}.engine"
    model = torch.hub.load("ultralytics/yolov5", "custom", f"models/{args.rt_model}.engine") # This line is important since it contains RT execution
    input_params = model.model.bindings["images"].shape  # Retrieve input size of the model
    precision = args.precision
    outfile = f"{args.rt_model}_RT.txt"

    print(f"[o] {args.rt_model} model optimised with TensorRT @{precision} with input: {input_params}")
elif args.model:
    if not args.imsize:
        parser.error("--model argument requires --imsize argument")
    model_name = args.model
    model = torch.hub.load("ultralytics/yolov5", args.model, pretrained=True) # Load unoptimised model from Ultralytics's servers
    precision = "fp32"
    input_params = (int(args.batch), 3, int(args.imsize), int(args.imsize))
    outfile = f"{args.model}_{args.imsize}x{args.imsize}_batch_{args.batch}.txt"
    print(f"[o] {args.model} model with input: {input_params}")
else:
    parser.error("No model received!")

def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000, verbose=False, export=False):
    model_input = torch.randn(input_shape)
    model_input = model_input.to("cuda")
    
    if dtype == 'fp16':
        model_input = model_input.half()

    print(f"Warming up for {nwarmup} iterations...")
    with torch.no_grad():
        for _ in range(nwarmup):
            outs = model(model_input, size=input_shape[-1])
    torch.cuda.synchronize()
    print("Starting benchmark...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            outs = model(model_input, size=input_shape[-1])
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if not i % 10 and verbose:
                print('Iteration %d/%d, avg inference time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Output shape:", outs.shape)
    print("Input shape:", model_input.size())
    print("Average inference time: %.2f ms" % (np.mean(timings) * 1000))
    print("Average FPS: %0.0f" % (1.0 / np.mean(timings)))

    if export:
        outdata = [
            f"Model name: {model_name}",
            f"Input shape: {model_input.size()}",
            f"Model precision: {dtype}",
            f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}",
            f"Stats for {nruns} runs:",
            "Average inference time: %.2f ms" % (np.mean(timings) * 1000),
            "Average FPS: %0.0f" % (1.0 / np.mean(timings))
        ]
        with open(f"benchmarks/{outfile}", "w") as out:
            out.write("\n".join(outdata))

benchmark(model, input_shape=input_params, dtype=precision, verbose=args.verbose, export=args.export)