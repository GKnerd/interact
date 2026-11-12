#!/bin/bash

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

# Check and create necessary folders
for FOLDER in log data outputs; do
    HOST_PATH="$PACKAGE_ROOT/$FOLDER"
    if [ ! -d "$HOST_PATH" ]; then
        echo -e "${YELLOW_BOLD}Warning: $HOST_PATH does not exist. Creating it...${RESET}"
        mkdir -p "$HOST_PATH"
    fi
done

# Create the .claude_container, so sessions with claude inside docker persist
for FOLDER in .claude_container; do 
    HOST_PATH="$PACKAGE_ROOT/$FOLDER"
    if [ ! -d "$HOST_PATH" ]; then
        echo -e "${YELLOW_BOLD}Warning: $HOST_PATH does not exist. Creating it...${RESET}"
        mkdir -p "$HOST_PATH"
    fi
done

# GPU access: the Jetson (aarch64, L4T) only supports the NVIDIA runtime, not --gpus
case "$(uname -m)" in
    aarch64) GPU_ARGS=(--runtime nvidia) ;;
    *)       GPU_ARGS=(--gpus all) ;;
esac

docker run \
    --name $PACKAGE_NAME \
    -it \
    "${GPU_ARGS[@]}" \
    --ipc host \
    --net host \
    -v "$PACKAGE_ROOT":/home/${CONTAINER_USER}/interact \
    -v "$PACKAGE_ROOT/data":/home/${CONTAINER_USER}/data \
    -v "$PACKAGE_ROOT/outputs":/home/${CONTAINER_USER}/outputs \
    -v $PACKAGE_ROOT/.claude_container:/home/${CONTAINER_USER}/.claude \
    --rm \
    $IMAGE
