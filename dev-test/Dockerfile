FROM nvcr.io/nvidia/pytorch:22.01-py3

ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get update -y

# Works, but without Gstreamer
# RUN apt-get install ffmpeg libsm6 libxext6 -y
# RUN apt-get install -y libsm6 libxext6 libxrender-dev
# RUN pip install opencv-python==4.5.5.64

RUN pip install scikit-image pycuda seaborn

# OpenCV custom build instructions from:
# https://medium.com/@galaktyk01/how-to-build-opencv-with-gstreamer-b11668fa09c
# https://github.com/junjuew/Docker-OpenCV-GStreamer/blob/master/opencv3-gstreamer1.0-Dockerfile

# Install base dependencies + gstreamer
RUN pip uninstall -y opencv-python
RUN apt-get -y update
RUN apt-get -y install build-essential
RUN apt-get -y install pkg-config
RUN apt-get -y install libgstreamer1.0-0 \
gstreamer1.0-plugins-base \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
gstreamer1.0-doc \
gstreamer1.0-tools \
libgstreamer1.0-dev \
libgstreamer-plugins-base1.0-dev \
cmake \
protobuf-compiler \
libgtk2.0-dev \
ocl-icd-opencl-dev

# Clone OpenCV repo
WORKDIR /
RUN git clone https://github.com/opencv/opencv.git
WORKDIR /opencv
RUN git checkout 4.2.0

# Build OpenCV
RUN mkdir /opencv/build
WORKDIR /opencv/build
RUN ln -s /opt/conda/lib/python3.6/site-packages/numpy/core/include/numpy /usr/include/numpy
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python) \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D WITH_GSTREAMER=ON \
-D BUILD_EXAMPLES=ON ..
RUN make -j$(nproc)

# Install OpenCV
RUN make install
RUN ldconfig

RUN pip install seaborn tqdm

WORKDIR /workspace/torching
EXPOSE 3000