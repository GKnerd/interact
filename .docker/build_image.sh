#!/bin/sh

uid=$(eval "id -u")
gid=$(eval "id -g")

echo "$(pwd "$0")"

docker build \
    --build-arg UID="$uid" \
    --build-arg GID="$gid" \
    --network host \
    -f ./.docker/Dockerfile \
    -t hpi_interact/py:3_8_16 .
