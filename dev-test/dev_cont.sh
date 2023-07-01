#!/usr/bin/env bash
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Description: Automate the process of building and running a Docker container on PC

docker build -t pc_dev .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -i -d -v $PWD/production:/workspace/torching -p 3000:3000 --name dev_cont pc_dev
# docker run -it --rm -p 3000:3000 PC_dev
