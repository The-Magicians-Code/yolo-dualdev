ARG l4tversion
FROM nvcr.io/nvidia/l4t-ml:r$l4tversion-py3

ENV LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0

RUN apt-get update && apt-get install python3-tk -y && \
pip3 install psutil tqdm flask cryptography || true && \
pip3 install seaborn flask-opencv-streamer --no-dependencies
RUN cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/ || true && echo "Could not create trtexec symlink!"
RUN pip install ultralytics --no-deps && \
pip install thop

WORKDIR /code
EXPOSE 3030