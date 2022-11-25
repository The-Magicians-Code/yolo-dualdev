FROM ultralytics/yolov5

WORKDIR /usr/src/app
RUN mkdir hostitems
EXPOSE 3001