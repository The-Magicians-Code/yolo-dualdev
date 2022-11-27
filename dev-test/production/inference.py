#!/usr/bin/env python3
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Script: inference.py
# @Description: Perform inference on user defined input streams, using preconfigured YOLOv5 TensorRT model
# @Last modified: 2022/11/27

import cv2
import time
import torch
from flask import Flask, render_template, Response

app = Flask(__name__)

# User defined variables
model_path = "models/yolov5m6_640x640_batch_3.engine"
videos = ["video.mp4", "videoplay.mp4", "videoplayback.mp4"]

model = torch.hub.load("ultralytics/yolov5", "custom", model_path) # This line is important since it contains RT execution
input_params = model.model.bindings["images"].shape  # Retrieve input size of the model
img_size = input_params[-1] # One dimension since n x n shape
batch_size = input_params[0]
model.eval().to("cuda")

# For x86 systems
cams = [cv2.VideoCapture(f'filesrc location={video} ! qtdemux ! queue ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER) for video in videos]
# For Jetson
# cams = [cv2.VideoCapture('filesrc location=video.mp4 ! qtdemux ! queue ! h264parse ! nvv4l2dec ! nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER) for video in videos]

# colour = (B, G, R)
colour = (0, 140, 255)
thickness = 4
fontscale = 0.75
fontthick = 2

def openstreams(cameras):
    """Open every available stream and return it's state and current frame

    Args:
        cameras (list): list of cv2 VideoCapture objects

    Returns:
        states, frames (list, list): stream status True / False and current frame
    """
    states, frames = zip(*[camera.read() for camera in cameras])
    return states, frames

def plotdetections(detection, stream):
    height, width = stream.shape[:2]
    for i in range(detection.shape[0]):    
        if detection.iloc[i]["class"] != 8: # 8 - boat, ship, vessel
            continue

        xmin = int(detection.iloc[i]["xmin"] / (img_size) * (width))
        xmax = int(detection.iloc[i]["xmax"] / (img_size) * (width))
        ymin = int(detection.iloc[i]["ymin"] / (img_size) * (height))
        ymax = int(detection.iloc[i]["ymax"] / (img_size) * (height))
        # print(xmin, xmax)
        confidence = detection.iloc[i]['confidence']
        score_txt = f"{int(confidence * 100.0)}%"
        
        # label = "" # detection.iloc[i]["name"]
        
        cv2.rectangle(stream, (xmin, ymin), (xmax, ymax), (0, int(confidence * 255), int(255 - confidence * 255)), thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # (w, h), _ = cv2.getTextSize(
        #         f"{label}: " + score_txt, font, fontscale, fontthick)

        # This block for seeing the values on detection boxes
        # cv2.rectangle(streams, (xmin, ymax - h - 10), (xmin + w, ymax), colour, -1) # -1 to fill the rectangle
        # cv2.putText(stream, f"{label}: {score_txt}", (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)
        cv2.putText(stream, score_txt, (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)
    
    return stream

def gen_frames():
    if len(cams) != batch_size:
        raise ValueError(f"Number of input streams has to be equal to the model's batch size {len(cams)} != {batch_size}")
    online, frame = openstreams(cams)
    height, width = frame[0].shape[:2]
    # streams_ok = all(online)
    fps = 0
    tau = time.time()
    smoothing = 0.9
    fps_pos = (32, height - 32)
    top_left = (32, 38)

    with torch.no_grad():
        while True:
            #Capture frame-by-frame
            online, streams = openstreams(cams)
            streams_ok = all(online)
            if not streams_ok:
                break
            # FPS calculation
            now = time.time()
            if now > tau:  # avoid div0
                fps = fps*smoothing + 0.1/(now - tau)
            tau = now
            
            inputs = [cv2.resize(stream, (img_size, img_size)) for stream in streams]
            results = model(inputs, size=img_size)
            detections = results.pandas().xyxy

            streams = [plotdetections(detection, stream) for detection, stream in zip(detections, streams)]
            # Display fps
            [cv2.putText(stream, pos, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2) for pos, stream in zip(["Port", "Bow", "Starboard"], streams)]
            outs = cv2.hconcat([cv2.putText(stream, f"{int(fps)}", fps_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2) for stream in streams])
            online, buffer = cv2.imencode('.jpg', outs)
            outs = buffer.tobytes()
            # if cv2.waitKey(1) & 0xFF == ord('q'): # Disabled for performance
            #     break
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + outs + b'\r\n')
            # print(f"{fps:.2f}")   # Display fps
            
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
    # app.run(host='0.0.0.0', port=3030) # For Jetson
# gen_frames()