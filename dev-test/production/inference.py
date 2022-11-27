import cv2
import time
import torch
# import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

model_path = "models/yolov5m6_640x640_batch_1.engine"

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)
model = torch.hub.load("ultralytics/yolov5", "custom", model_path) # This line is important since it contains RT execution
in_size = model.model.bindings["images"].shape[-1]  # Retrieve input size of the model
model.eval().to("cuda")

# cap = cv2.VideoCapture('filesrc location=../video.mp4 ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture("video.mp4")

def gen_frames():
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    fps = 0
    tau = time.time()
    smoothing = 0.9
    text_pos = (32, height - 32)

    # Parameters for drawing
    #colour = (B, G, R)
    colour = (0, 140, 255)
    thickness = 4
    fontscale = 0.5
    fontthick = 1

    with torch.no_grad():
        while(ret):
            #Capture frame-by-frame
            ret, pic = cap.read()
            
            # FPS calculation
            now = time.time()
            if now > tau:  # avoid div0
                fps = fps*smoothing + 0.1/(now - tau)
            tau = now
            
            pic = cv2.resize(pic, (in_size, in_size))
            results = model(pic, size=in_size)
            detections = results.pandas().xyxy[0]

            for i in range(detections.shape[0]):
                if detections.iloc[i]["class"] != 8:
                    continue

                xmin = int(detections.iloc[i]["xmin"])
                xmax = int(detections.iloc[i]["xmax"])
                ymin = int(detections.iloc[i]["ymin"])
                ymax = int(detections.iloc[i]["ymax"])

                score_txt = f"{round(detections.iloc[i]['confidence']*100.0, 2)}%"
                
                label = detections.iloc[i]["name"]

                cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), colour, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (w, h), _ = cv2.getTextSize(
                        f"{label}: " + score_txt, font, fontscale, fontthick)

                # This block for seeing the values on detection boxes
                # cv2.rectangle(pic, (xmin, ymax - h - 10), (xmin + w, ymax), colour, -1) # -1 to fill the rectangle
                cv2.putText(pic, f"{label}: " + score_txt, (xmin, ymax - 5), font, fontscale, (255, 255, 255), fontthick, cv2.FILLED)

            # Display fps
            cv2.putText(pic, str(int(fps)), \
                        text_pos, \
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', pic)
            pic = buffer.tobytes()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + pic + b'\r\n')  # concat frame one by one and show result
            # print(f"{fps:.2f}")
            
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