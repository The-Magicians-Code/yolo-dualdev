#!/usr/bin/env bash
docker build -t YOLO_models .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD/userexports:/usr/src/app/hostitems -p 3001:3001 YOLO_models
