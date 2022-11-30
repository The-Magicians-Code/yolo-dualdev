Scripts in this environment must be developed in such a way that they are capable of running both on x86 (PC) and arm (Jetson) architecture with minimal modifications, make sure that PyTorch versions match in torchinstall script and dev container to avoid incompatibiliy and version specific nuances on each architecture. model2trt.sh must be run on the specified device which performs inference, since TensorRT models are compiled for the specified GPU architecture.
## How to run?
>Make sure that you are at the root of the project!
* Dev container:  
``cd dev-test/``  
``bash dev_cont.sh``  
Done!  
Scripts are in the production environment, now named as torching/ for reality check

* Yolo models export container:  
``bash export_cont.sh``  
Done!
Scripts are in the export environment, now named as hostitems/ for reality check  
That's all good and stuff, but how the fuck am I going to export my models?  
Speak no more:  
``cd hostitems/``  
``bash shellexport.sh``  
Open the shellexport file with nano for example to edit the requested model types and their parameters  
``nano shellexport.sh``  
## Tips?
Sure! If you happen to use VSCode, then check out [Developing in Docker](https://code.visualstudio.com/docs/devcontainers/containers)  
Once you have remote development extensions installed and you've started a container, press ``F1`` search  
for ``Dev Containers: Attach to a running container`` and select the active container, you'll be in the container in VSCode