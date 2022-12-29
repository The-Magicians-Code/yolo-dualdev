docker build -t dataset_downloader .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v /mnt/d/yolodatasets:/home/datasets -p 3003:3003 dataset_downloader