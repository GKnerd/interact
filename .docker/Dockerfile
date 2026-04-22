##############################################################################
##                                 Base Image                               ##
##############################################################################
# Taken from https://github.com/dusty-nv/jetson-containers?tab=readme-ov-file (last access 22.04.2026)
FROM dustynv/l4t-pytorch:r36.4.0 

##############################################################################
##                                 Global Dependecies                       ##
##############################################################################

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    sudo \
    && rm -rf /var/lib/apt/lists/*

##############################################################################
##                                 Create User                              ##
##############################################################################
ARG USERNAME=torch_orin_docker
ARG UID=1000
ARG GID=1000

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $UID ; then userdel `id -un $UID` ; fi
# Create the User again
RUN groupadd --gid $GID $USERNAME \
    && useradd --uid $UID --gid $GID -m $USERNAME \
    && usermod -aG video $USERNAME \
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
    sed -i '/mkl/d' requirements.txt && \
    sed -i '/triton/d' requirements.txt && \
    pip install --index-url https://pypi.org/simple --no-cache-dir -r requirements.txt && \
    pip install --index-url https://pypi.org/simple --no-cache-dir torch-dct && \
    rm requirements.txt

CMD ["/bin/bash"]
