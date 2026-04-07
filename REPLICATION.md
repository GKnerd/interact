# Replication Guide: InteRACT Environment

This guide outlines the steps to replicate the environment for the InteRACT project. Since the original repository does not provide a Dockerfile, this fork introduces a containerized workflow to ensure version consistency, dependency isolation, and GPU compatibility.

---

## Prerequisites

Before building the image, ensure your host machine has the following installed:
1.  **Docker Engine** (v20.10+)
2.  **NVIDIA Container Toolkit:** This is strictly required for the container to "talk" to the host GPU. Without it, the environment will default to CPU-only and the code will fail.

## Environment Requirements

The original project was developed using **Python 3.8.16** and **PyTorch 1.10.0**. Analysis of the `requirements.txt` and `README.md` indicates that the system was built against the following CUDA 11.7/11.3 stack:

* **Python:** `3.8.16`
* **PyTorch:** `1.10.0+cu113`
* **NVIDIA CUDA/cuDNN:** ~11.7.x (specified via `nvidia-*-cu11` wheels)

## Docker Base Image

To simplify the setup, we utilize a pre-configured image from the [cnstark/pytorch-docker](https://github.com/cnstark/pytorch-docker) repository. This provides a stable, high-performance baseline for the OS, Python, and CUDA layers.

The chosen image is:
`cnstark/pytorch:1.13.1-py3.8.16-cuda11.7.1-devel-ubuntu20.04`

> **Note 1:** This is the closest stable match to the original project's requirements. While it defaults to PyTorch 1.13.1, it maintains the exact Python and CUDA versions used in the original project.

> **Note 2:** Hardware-Specific Adaptation (RTX 40-Series & Newer GPUs):
Newer GPU architectures, specifically NVIDIA RTX 40-series (Ada Lovelace), strictly require CUDA 11.8 or higher. Attempting to run the default 11.7 environment on these modern GPUs will result in fatal `cuFFT` errors during training. To ensure cross-hardware compatibility, we inject a modern PyTorch ecosystem (2.4.1+cu118) using the following command:

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
Why this approach? 
This method allows us to provide modern GPU support while strictly preserving the legacy Python 3.8.16 foundation. By upgrading only the PyTorch/CUDA binaries, we prevent cascading dependency failures in the  codebase.

>**Warning**: Downloading and bundling these updated CUDA 11.8 wheels during the build process does result in a slightly larger final Docker image.

## Setup Instructions

### Step 1: Pull the Base Image
Download the base image to your local machine to speed up the build process:

```bash
docker pull cnstark/pytorch:1.13.1-py3.8.16-cuda11.7.1-devel-ubuntu20.04
```

### Step 2: Build the Image
From the **project root** directory, execute the build script. This script configures a local user (matching your host UID/GID), installs system-level dependencies (like cmake and git), and sets up the Python environment:

```bash
./.docker/build_image.sh
```

### Step 3: Run the Container
Launch the interactive environment from the project root:

```bash
./.docker/run_container.sh
```

## Important Considerations

- Data Mounts: You must modify the volume mappings in .docker/run_container.sh to point to your local directories for AMASS datasets and SMPL body models.

- User Permissions: The build script automatically maps your host UID and GID to the internal torch_1_13_1_docker user. This ensures that any files created by the container (like logs or checkpoints) are owned by you on the host machine.

- Dependencies: The Dockerfile automatically handles the removal of conflicting nvidia- and torch lines from requirements.txt to ensure the container's optimized drivers remain intact.
