#!/usr/bin/env bash
# cat /etc/nv_tegra_release
# apt-cache show nvidia-jetpack

# Set versions according to your Jetson Jetpack info
l4tversion=32.7.1
torchversion=1.10

# sudo docker pull nvcr.io/nvidia/l4t-pytorch:r${l4tversion}-pth${torchversion}-py3
# sudo docker run -it --rm --runtime nvidia --network host nvcr.io/nvidia/l4t-pytorch:r${l4tversion}-pth${torchversion}-py3

sudo docker build -t cont_test . -f prod.Dockerfile --build-arg l4tversion=$l4tversion --build-arg torchversion=$torchversion
sudo docker run --rm -i -d --runtime nvidia --name torchcont cont_test