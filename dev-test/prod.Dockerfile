ARG l4tversion
# ARG torchversion
# FROM l4t-pytorch:r$l4tversion-pth$torchversion-py3
# FROM nvcr.io/nvidia/l4t-pytorch:r$l4tversion-pth$torchversion-py3
FROM nvcr.io/nvidia/l4t-ml:r$l4tversion-py3

# ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
# ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

# COPY opencv_install.sh /tmp/opencv_install.sh
# RUN cd /tmp && bash opencv_install.sh ${OPENCV_URL} ${OPENCV_DEB}
# This is to avoid ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
# This is to avoid ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
# RUN export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0

ENV LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0

RUN apt-get update && apt-get install python3-tk -y && \
pip3 install psutil tqdm flask cryptography && \
pip3 install seaborn flask-opencv-streamer --no-dependencies

WORKDIR /code
EXPOSE 3000
# RUN ["python3"]