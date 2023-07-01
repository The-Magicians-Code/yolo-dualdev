#!/usr/bin/env python3
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Description: Benchmark multiple all neural network models sequentially

from glob import glob
from subprocess import call
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Benchmark multiple neural networks at once")
parser.add_argument('-m', '--merge', action="store_true", help="Merge benchmark results into a .csv file, if not requested, then benchmarks will be restarted and then merged into a file, default False")
parser.add_argument('--runs', default = 1000, type=int, help="How many benchmarking runs on a model, default 1000")
args = parser.parse_args()

def autobenchmark():
    rt_models = glob("models/*.engine")
    models = glob("models/*.onnx")

    models2 = [i.split(".")[0].split("/")[1] for i in models]
    models_as_args = [i.split("_")[0] for i in models2]

    params = [i.replace("batch_", "").split("_")[1:] for i in models2]
    imsizes, batchsizes = map(list, zip(*params))
    imsizes = [i.split("x")[0] for i in imsizes]
    print(f"Benchmarking models: {rt_models, models}")

    for model in rt_models:
        call(f"python3 benchmark.py --rt-model {model} --precision fp16 --export --runs {args.runs}".split())

    for model, imsize, batchsize in zip(models_as_args, imsizes, batchsizes):
        call(f"python3 benchmark.py --model {model} --imsize {imsize} --batch {batchsize} --export --runs {args.runs}".split())

def mergedata():
    files = glob("benchmarks/*.txt")
    dataframes = []
    for file in files:
        dataframes.append(pd.read_csv(file, skiprows=10, nrows=12))

    df = pd.concat(dataframes)

    df.to_csv("model_benchmarks.csv", index=False)
    print(df.head())


if args.merge:
    print("Benchmark skipped, merging existing results...")
    mergedata()
else:
    print("Starting benchmarks")
    autobenchmark()
    mergedata()