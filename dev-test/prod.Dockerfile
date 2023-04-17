ARG l4tversion
FROM nvcr.io/nvidia/l4t-ml:r$l4tversion-py3

ENV LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0

RUN apt-get update && apt-get install python3-tk -y && \
pip3 install psutil tqdm flask cryptography && \
pip3 install seaborn flask-opencv-streamer --no-dependencies && \
cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/

WORKDIR /code
EXPOSE 3030