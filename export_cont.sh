#!/usr/bin/env bash
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Description: Automate the process of building and running a Docker container on PC

docker build -t yolo_models . -f Dockerfile
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -i -d -v $PWD/userexports:/usr/src/app/hostitems -v /mnt/d/yolodatasets:/usr/src/app/hostitems/datasets -p 3001:3001 --name model_prep yolo_models