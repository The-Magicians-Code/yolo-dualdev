#!/usr/bin/env bash
docker build -t pc_dev .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -i -d -v $PWD/production:/workspace/torching -p 3000:3000 --name dev_cont pc_dev
# docker run -it --rm -p 3000:3000 PC_dev
