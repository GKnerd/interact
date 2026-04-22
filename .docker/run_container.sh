#!/bin/sh
echo "Run Container"
# xhost +local:root

docker run --name hpi_interact \
    --runtime=nvidia \
    --privileged \
    --cap-add=sys_nice \
    -it \
    -v $(pwd):/home/torch_orin_docker/interact/ \
    -v /home/cognisafe/datasets/:/home/torch_orin_docker/interact/data \
    --net host \
    --rm \
    --ipc host \
    hpi_interact/jetson_orin

# docker run --name hpi_interact \
#     --runtime=nvidia \
#     --privileged \
#     --cap-add=sys_nice \
#     -it \
#     -e DISPLAY=$DISPLAY \
#     -e QT_X11_NO_MITSHM=1 \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v ~/.Xauthority:/home/torch_orin_docker/.Xauthority \
#     -v /dev:/dev \
#     -v $(pwd):/home/torch_orin_docker/interact/ \
#     -v /home/cognisafe/datasets/:/home/torch_orin_docker/interact/data \
#     -p 6007:6007 \
#     --net host \
#     --rm \
#     --ipc host \
#     hpi_interact/jetson_orin
