import tensorrt as trt
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
            
    def __call__(self, x:np.ndarray, batch_size=2):
        x = x.astype(self.dtype)        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)     
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]

if __name__ == "__main__":
    model_path = "models/yolov5m6_640x640_batch_1.engine"
    batch_size = int(model_path.split(".")[0][-1])
    trt_engine_path = os.path.join(model_path)
    model = TrtModel(trt_engine_path, dtype=np.float16)
    shape = model.engine.get_binding_shape(0)
    
    import cv2
    img = cv2.imread("ship2.jpeg")
    print(img.shape)
    pic = img[...,::-1]

    size = shape[2:]
    data = cv2.resize(pic, size)

    # data = np.random.randint(0, 255, (batch_size, *shape[1:])) / 255
    result = model(data, batch_size)
    dimensions = 85
    rows = int(result[0].shape[1] / dimensions)
    batch_size_out = result[0].shape[0]
    confidence_index = 4
    label_index = 5

    result = np.reshape(result[0], (batch_size_out, rows, dimensions))

    locations = []
    labels = []
    confidences = []

    # print(result[0].shape)
    # print(result[0][0][np.argmax(result[0][0])])
    # print(result.shape)

    print(result.shape)

    i = 0
    classes = result[0][i][5:]
    boxes = result[0][i][:4]
    confidence = result[0][i][4]
    # print(boxes, confidence)
    # print(classes.shape)
    best_class_index = np.argmax(classes)
    # print(best_class_index)
    # print(classes[best_class_index])
    dh, dw = img.shape[:2]
    for i in range(1):
        continue
        classes = result[0][i][5:]
        boxes = result[0][i][:4]
        confidence = result[0][i][4]

        best_class_index = np.argmax(classes)
        best_class = classes[best_class_index]

        print(f"Best class: {best_class_index}, score: {best_class}\nBoxes: {boxes}\nConfidence: {confidence}")
        # print(f"Classes: {classes}\nBoxes: {boxes}\nConfidence: {confidence}")

        x, y, w, h = boxes

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
        cv2.imwrite("ohcock.jpg", img)
    import torch
    from nms import *
    batch_detections = torch.from_numpy(np.array(result))
    # print(batch_detections)
    batch_detections = non_max_suppression(batch_detections)
    print(batch_detections[0].shape)