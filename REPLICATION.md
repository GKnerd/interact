# Changes in the Fork

The original repository does not provide a Dockerfile to replicate the code. From the `requirements.txt` and the `README.md` the original code was built with these system dependencies:

```
python == 3.8.16 

torch == 1.10.0+cu113

nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-cupti-cu11==11.7.101
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.2.10.91
nvidia-cusolver-cu11==11.4.0.1
nvidia-cusparse-cu11==11.7.4.91
nvidia-nccl-cu11==2.14.3
nvidia-nvtx-cu11==11.7.91
```
i.e. python version 3.8.16, CUDA version of 11.7.99 and torch version of 1.10.0

The repository link`https://github.com/cnstark/pytorch-docker` provides different images with prebuilt OS, python, pytorch and CUDA versions. 
The image `cnstark/pytorch:1.13.1-py3.8.16-cuda11.7.1-devel-ubuntu20.04` is used as a baseline to create an isolated docker container. It is the closest to the needed requirements. Of course you need the NVIDIA Container Toolit installed locally to be able to talk to the GPU. This will otherwise fail.

The image needs to be installed locally via: 

```bash
docker pull cnstark/pytorch:1.13.1-py3.8.16-cuda11.7.1-devel-ubuntu20.04
```

After that you can use **from the root** your working directory the command: 
```bash
./.docker/build_image.sh
```
to build the image and from the root again use, 

```bash
./.docker/run_container.sh
```
to run the container. 

Disclaimer: You might need to modify the data directories mounted to your container, other than that you should be good to go.
