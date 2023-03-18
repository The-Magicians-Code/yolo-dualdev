#!/usr/bin/env bash
docker build -t yolo_models . -f Dockerfile
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -i -d -v $PWD/userexports:/usr/src/app/hostitems -v /mnt/d/yolodatasets:/usr/src/app/hostitems/datasets -p 3001:3001 --name model_prep yolo_models