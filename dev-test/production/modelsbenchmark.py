from glob import glob
from subprocess import call

rt_models = glob("models/*.engine")
models = glob("models/*.onnx")

rt_models2 = [i.split(".")[0].split("/")[1] for i in rt_models]
models2 = [i.split(".")[0].split("/")[1] for i in models]
models_as_args = [i.split("_")[0] for i in models2]

params = [i.replace("batch_", "").split("_")[1:] for i in models2]
imsizes, batchsizes = map(list, zip(*params))
# imsizes = [i.split("x")[0] for i in imsizes]
print(f"Benchmarking models: {rt_models, models}")

# for model in rt_models2:
#     call(f"python3 benchmark.py --rt-model {model} --precision fp16 --export".split())

# for model, imsize, batchsize in zip(models_as_args, imsizes, batchsizes):
#     call(f"python3 benchmark.py --model {model} --imsize {imsize} --batch {batchsize} --export".split())