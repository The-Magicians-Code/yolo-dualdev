#!/usr/bin/env bash
# @Author: Tanel Treuberg
# @Github: https://github.com/The-Magicians-Code
# @Description: Automate the process of building and running a Docker container on Nvidia Jetson

# Determine the L4T version and Jetpack version
# If container could not be built, please check the L4T and Jetpack version from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml
# and overwrite the l4t variable manually
l4t_raw=($(dpkg-query --show nvidia-l4t-core))
l4t=${l4t_raw[1]%-*} # Extract the version number from the package name
jetpack_raw=($(apt-cache show nvidia-jetpack | grep "Version"))
jetpack=${jetpack_raw[-1]%-*} # Extract the version number from the package name

# Print the L4T version and Jetpack version to the console
echo -e "L4T: ${l4t}\nJetpack: ${jetpack}"

# Build the Docker container
if sudo docker build -t cont_test . -f prod.Dockerfile --build-arg l4tversion=$l4t; then
    # If the container is successfully built
    echo Successfully built container!
    # Run the container with NVIDIA runtime, mount the current directory as a volume,
    # map port 3000 on the host to port 3000 in the container, and name the container "torchcont"
    sudo docker run --rm -i -d --runtime nvidia -v $PWD/production:/code -p 3000:3000 --name torchcont cont_test
else
    # If the container cannot be built
    echo Please check for the corresponding L4T version using Jetpack: ${jetpack} version from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml and overwrite the l4t variable manually
fi