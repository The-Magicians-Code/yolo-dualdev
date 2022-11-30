import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
import numpy as np
import torch
import os
from collections import namedtuple, OrderedDict
# check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
device = torch.device("cpu")
if device.type == 'cpu':
    device = torch.device('cuda:0')

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
print(bindings)

# def forward(im):
#     YOLOv5 MultiBackend inference
#     b, ch, h, w = im.shape  # batch, channel, height, width
#     if fp16 and im.dtype != torch.float16:
#         im = im.half()  # to FP16
#     if nhwc:
#         im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
#     if dynamic and im.shape != bindings['images'].shape:
#         i = model.get_binding_index('images')
#         context.set_binding_shape(i, im.shape)  # reshape if dynamic
#         bindings['images'] = bindings['images']._replace(shape=im.shape)
#         for name in output_names:
#             i = model.get_binding_index(name)
#             bindings[name].data.resize_(tuple(context.get_binding_shape(i)))
#     s = bindings['images'].shape
#     assert im.shape == s, f"input size {im.shape} {'>' if dynamic else 'not equal to'} max model size {s}"
#     binding_addrs['images'] = int(im.data_ptr())
#     context.execute_v2(list(binding_addrs.values()))
#     y = [bindings[x].data for x in sorted(output_names)]

def from_numpy(x):
    return torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x

if isinstance(y, (list, tuple)):
    outs = from_numpy(y[0]) if len(y) == 1 else [from_numpy(x) for x in y]
else:
    outs = from_numpy(y)
    
def warmup(imgsz=(1, 3, 640, 640)):
    # Warmup model by running inference once
    if device.type != 'cpu':
        im = torch.empty(*imgsz, dtype=torch.half if fp16 else torch.float, device=device)  # input
        for _ in range(2 if jit else 1):  #
            forward(im)  # warmup
