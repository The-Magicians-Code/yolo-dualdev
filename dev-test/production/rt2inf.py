import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
import numpy as np
import torch
import os
from collections import namedtuple, OrderedDict
# check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
# if device.type == 'cpu':
    # device = torch.device('cuda:0')
device = "cuda"

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)

w = os.path.join("models/yolov5m6_640x640_batch_3.engine")

with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
context = model.create_execution_context()
bindings = OrderedDict()
output_names = []
fp16 = False  # default updated below
dynamic = False
for i in range(model.num_bindings):
    name = model.get_binding_name(i)
    dtype = trt.nptype(model.get_binding_dtype(i))
    if model.binding_is_input(i):
        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
            dynamic = True
            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
        if dtype == np.float16:
            fp16 = True
    else:  # output
        output_names.append(name)
    shape = tuple(context.get_binding_shape(i))
    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
print(binding_addrs)
print(batch_size)