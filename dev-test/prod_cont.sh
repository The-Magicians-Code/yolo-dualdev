#!/usr/bin/env bash
# cat /etc/nv_tegra_release
# apt-cache show nvidia-jetpack

# Automatically determine L4T version
# If container could not be built, please check the L4T and Jetpack version from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml
# and overwrite the l4t variable manually
l4t=($(dpkg-query --show nvidia-l4t-core))
l4t=${l4t[1]%-*}
jetpack=($(apt-cache show nvidia-jetpack | grep "Version"))
jetpack=${jetpack[-1]%-*}

echo -e "L4T: ${l4t}\nJetpack: ${jetpack}"
if sudo docker build -t cont_test . -f prod.Dockerfile --build-arg l4tversion=$l4t; then
    echo Successfully built container!
    sudo docker run --rm -i -d --runtime nvidia -v $PWD/production:/code -p 3000:3000 --name torchcont cont_test
else
    echo Please check for the corresponding L4T version using Jetpack: ${jetpack} version from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml and overwrite the l4t variable manually
fi