#!/usr/bin/env python3
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Script: inference.py
# @Description: Perform inference on user defined input streams, using preconfigured YOLOv5 TensorRT model
# @Last modified: 2022/11/29

import cv2
import time
import torch
import argparse
import platform
from pathlib import Path
from flask import Flask, render_template, Response

parser = argparse.ArgumentParser(description="Neural Network inference on video stream(s)")
parser.add_argument('--rec', help="Record the stream into a video file --rec out.mp4")
parser.add_argument('--input-video', required=True, nargs="+", help="Read video file(s) --input-video video0.mp4 video1.mp4 ...")
parser.add_argument('--model', help="YOLOv5 unoptimised Neural Network for object detection --model yolov5m6")
parser.add_argument('--imsize', help="YOLOv5 unoptimised Neural Network input size --imsize 640")
parser.add_argument('--perf', action='store_true')
parser.add_argument('--rt-model', help="YOLOv5 RT Neural Network for object detection --rt-model yolov5m6_640x640_batch_1")
args = parser.parse_args()

app = Flask(__name__)

if args.rec:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_file = Path().cwd() / args.rec
    print("Saving output video to:", video_file)

if args.input_video:
    videos = args.input_video
else:
    parser.error("--input-video requires an argument")

if args.rt_model:
    if not Path(f"models/{args.rt_model}.engine").exists():
        raise FileNotFoundError(f"Did not find RT model {args.rt_model}.engine in /models/ folder. Specify model name ONLY without .engine suffix and make sure it is located in models/ folder")
    
    model = torch.hub.load("ultralytics/yolov5", "custom", f"models/{args.rt_model}.engine") # This line is important since it contains RT execution
    input_params = model.model.bindings["images"].shape  # Retrieve input size of the model
    img_size = input_params[-1] # One dimension since n x n shape
    batch_size = input_params[0]
elif args.model and args.imsize:
    model = torch.hub.load("ultralytics/yolov5", args.model, pretrained=True) # Load unoptimised model from Ultralytics servers
    img_size = int(args.imsize)
    batch_size = len(videos)
elif args.model and not args.imsize:
    parser.error("--model argument requires --imsize argument")
    
model.eval().to("cuda")

if platform.machine() == "x86_64":  # For PC
    decoder = "avdec_h264"
    video_converter = "videoconvert"
    app_port = 3000
elif platform.machine() == "aarch64":   # For Jetson
    decoder = "nvv4l2decoder"
    video_converter = "nvvidconv"
    app_port = 3030

# Setup the cameras
cams = [cv2.VideoCapture(f'filesrc location={video} ! qtdemux ! queue ! h264parse ! {decoder} ! {video_converter} ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER) for video in videos]

    
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

        # This block for seeing the values on detection boxes
        # (w, h), _ = cv2.getTextSize(f"{label}: " + score_txt, font, fontscale, fontthick)
        # cv2.rectangle(streams, (xmin, ymax - h - 10), (xmin + w, ymax), colour, -1) # -1 to fill the rectangle
        # cv2.putText(stream, f"{label}: {score_txt}", (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)
        cv2.putText(stream, score_txt, (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)
    
    return stream

def main():
    if len(cams) != batch_size:
        raise ValueError(f"Number of input streams has to be equal to the model's batch size {len(cams)} != {batch_size}")
    
    online, frame = openstreams(cams)
    if not all(online):
        print("At least one stream can not be captured")
        return
    
    height, width = frame[0].shape[:2]

    if args.rec:
        output_video = cv2.VideoWriter(str(video_file), fourcc, 10, (len(frame) * width, height))
    
    fps = 0
    tau = time.time()
    smoothing = 0.9
    fps_pos = (32, height - 32)
    top_left = (32, 38)

    with torch.no_grad():
        while True:
            # Capture frame-by-frame
            online, streams = openstreams(cams)
            streams_ok = all(online)
            if not streams_ok:
                print("At least one stream went offline")
                if output_video:
                    output_video.release()
                    [camera.release() for camera in cams]
                    cv2.destroyAllWindows()
                break
            # FPS calculation
            now = time.time()
            if now > tau:  # avoid div0
                fps = fps * smoothing + 0.1/(now - tau)
            tau = now
            
            if args.model or args.rt_model:
                inputs = [cv2.resize(stream, (img_size, img_size)) for stream in streams]
                results = model(inputs, size=img_size)
                detections = results.pandas().xyxy

                streams = [plotdetections(detection, stream) for detection, stream in zip(detections, streams)]
                # Display fps
            [cv2.putText(stream, pos, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2) for pos, stream in zip(["Port", "Bow", "Starboard"], streams)]
            outs = cv2.hconcat([cv2.putText(stream, f"{int(fps)}", fps_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2) for stream in streams])
            
            if args.rec:
                output_video.write(outs)
            
            if args.perf:
                print(f"{fps:.2f}")   # Display fps  
            else:
                online, buffer = cv2.imencode('.jpg', outs)
                outs = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + outs + b'\r\n')
            # print(f"{fps:.2f}")   # Display fps        
            
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=app_port)
