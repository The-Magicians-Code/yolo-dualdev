ARG l4tversion
ARG torchversion
FROM nvcr.io/nvidia/l4t-pytorch:r$l4tversion-pth$torchversion-py3
RUN ["python3"]