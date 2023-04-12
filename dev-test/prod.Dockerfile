ARG l4tversion
# ARG torchversion
# FROM l4t-pytorch:r$l4tversion-pth$torchversion-py3
# FROM nvcr.io/nvidia/l4t-pytorch:r$l4tversion-pth$torchversion-py3
FROM nvcr.io/nvidia/l4t-ml:r$l4tversion-py3

# ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
# ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

# COPY opencv_install.sh /tmp/opencv_install.sh
# RUN cd /tmp && bash opencv_install.sh ${OPENCV_URL} ${OPENCV_DEB}
RUN apt-get update && apt-get install python3-tk -y
RUN pip3 install psutil tqdm flask
RUN pip3 install seaborn flask-opencv-streamer --no-dependencies

RUN ["python3"]