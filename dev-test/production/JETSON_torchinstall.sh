#!/usr/bin/env bash
# This script is meant solely for running on Nvidia JETSON inference device
wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-1.11.0-cp38-cp38-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev -y
pip3 install Cython
pip3 install numpy torch-1.11.0-cp38-cp38-linux_aarch64.whl

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev -y
git clone --branch v0.12.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.12.0
python3 setup.py install
cd ../