#!/usr/bin/env bash
# cat /etc/nv_tegra_release
# apt-cache show nvidia-jetpack

# Automatically determine L4T version
l4t=($(dpkg-query --show nvidia-l4t-core))
l4t=${l4t[1]%-*}

sudo docker build -t cont_test . -f prod.Dockerfile --build-arg l4tversion=$l4t
sudo docker run --rm -i -d --runtime nvidia -v $PWD/production:/code -p 3000:3000 --name torchcont cont_test