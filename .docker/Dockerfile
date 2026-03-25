##############################################################################
##                                 Base Image                               ##
##############################################################################
FROM cnstark/pytorch:1.13.1-py3.8.16-cuda11.7.1-devel-ubuntu20.04

##############################################################################
##                                 Global Dependecies                       ##
##############################################################################

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    sudo \
    # libgl1-mesa-glx \
    # libosmesa6-dev \
    # freeglut3-dev \
    # libgles2-mesa-dev \
    # libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

##############################################################################
##                                 Create User                              ##
##############################################################################
ARG USERNAME=torch_1_13_1_docker
ARG USER_UID=1000
ARG USER_GID=1000

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi
# Create the User again
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER ${USERNAME}
RUN mkdir -p /home/${USERNAME}/interact/

##############################################################################
##                                 User Dependecies                         ##
##############################################################################

WORKDIR /home/${USERNAME}/
COPY --chown=${USERNAME} requirements.txt ./


RUN sed -i '/nvidia-/d' requirements.txt && \
    sed -i '/torch/d' requirements.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch-dct

RUN rm requirements.txt

WORKDIR /home/${USERNAME}/interact
CMD ["/bin/bash"]
