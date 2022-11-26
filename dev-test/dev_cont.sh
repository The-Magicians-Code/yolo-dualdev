#!/usr/bin/env bash
docker build -t pc_dev .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -d --rm -v $PWD/production:/workspace/torching -p 3000:3000 pc_dev
# docker run -it --rm -p 3000:3000 PC_dev
