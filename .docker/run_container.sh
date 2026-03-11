#!/bin/sh
echo "Run Container"
xhost + local:root

docker run --name hpi_interact \
    --privileged \
    --cap-add=sys_nice \
    -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/home/docker_panda/.Xauthority \
    -v /dev:/dev \
    -v $(pwd):/home/torch_1_13_1_docker/interact/ \
    -v /home/georgios-katranis/Projects/datasets/raw:/home/torch_1_13_1_docker/interact/data \
    --net host \
    --rm \
    --ipc host \
    hpi_interact/py:3_8_16
