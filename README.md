This project contains tools for developing YOLOv5 models for inference on Nvidia Jetson devices. 
Dev-test folder has user tuned Docker container which has been configured to run with most 
compatible version of software on PC, which is also installed on Jetson device. 
Production folder contains scripts that are capable of running on Jetson natively and on PC in 
dev container, with minimal modifications to ensure the best possible compatibility.

# How to run?
Simple
````bash export_cont.sh```` for initialising YOLO models container
````bash dev-test/dev_cont.sh```` for initialising development container
> Code to be rearranged and additional documentation added in the near future
