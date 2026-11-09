##############################################################################
##                                 Base Image                               ##
##############################################################################
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

##############################################################################
##                                 PyTorch                                  ##
##############################################################################
# Same versions as the deployment image. The wheels bring their own CUDA runtime;
# the host needs a driver for CUDA 13.0
RUN pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 \
    && pip install numpy==1.26.4 torch-dct==0.1.6

##############################################################################
##                                 Create User                              ##
##############################################################################
ARG USERNAME=interact
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

##############################################################################
##                                 interact                                 ##
##############################################################################
# Installed editable; run_container.sh mounts the repo over /home/<user>/interact,
# so the container always runs the current code. Dependency changes need a rebuild.
COPY pyproject.toml README.md /home/${USERNAME}/interact/
COPY interact /home/${USERNAME}/interact/interact
RUN pip install -e "/home/${USERNAME}/interact[train,test]" numpy==1.26.4

USER ${USERNAME}
WORKDIR /home/${USERNAME}/interact

CMD ["/bin/bash"]
