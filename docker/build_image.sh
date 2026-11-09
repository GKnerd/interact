#!/bin/bash
# Build the training image. Runs from root of the 'interact' package.

PACKAGE_NAME="interact"
CONTAINER_USER="interact"
IMAGE="$PACKAGE_NAME/train:$(uname -m)"

uid=$(eval "id -u")
gid=$(eval "id -g")

# ANSI escape codes
RED_BOLD="\033[1;31m"
YELLOW_BOLD="\033[1;33m"
GREEN_BOLD="\033[1;32m"
RESET="\033[0m"

# Set Package root
if [[ "$(pwd)" == *"/$PACKAGE_NAME/"* ]]; then
    # Case A: Inside a subdirectory
    echo -e "${YELLOW_BOLD}Inside subdirectory. Navigating to root...${RESET}"
    # Strip everything after the package name to find the root
    _cwd="$(pwd)"
    PACKAGE_ROOT="${_cwd%%/$PACKAGE_NAME/*}/$PACKAGE_NAME"
    cd "$PACKAGE_ROOT" || exit 1
elif [[ "$(pwd)" == *"/$PACKAGE_NAME" ]]; then
    # Case B: Already at the root
    echo -e "${GREEN_BOLD}Already at package root.${RESET}"
    PACKAGE_ROOT="$(pwd)"
else
    # Case C: Not in the package at all
    echo -e "${RED_BOLD}Error: You are not inside the directory '$PACKAGE_NAME'.${RESET}"
    echo "Current path: $(pwd)"
    exit 1
fi

# One Dockerfile per architecture: Dockerfile.x86_64, Dockerfile.aarch64 (Jetson Orin, L4T)
DOCKERFILE="./docker/Dockerfile.$(uname -m)"
echo -e "${GREEN_BOLD}Building $IMAGE from $DOCKERFILE${RESET}"

docker build \
    --build-arg USER_UID="$uid" \
    --build-arg USER_GID="$gid" \
    --build-arg USER="$CONTAINER_USER" \
    --network host \
    -f "$DOCKERFILE" \
    -t $IMAGE .
