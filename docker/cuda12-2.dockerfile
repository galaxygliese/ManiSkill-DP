FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ENV NVIDIA_DRIVER_CAPABILITIES=all
ARG PYTHON_VERSION=3.9

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    # vulkan-utils \
    wget \
    xvfb \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# # Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# install dependencies
RUN pip install --upgrade mani-skill==3.0.0b20 && pip cache purge
RUN pip install torch 

# download physx GPU binary via sapien
RUN python -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"

# zsh (from [1]) ----------------
RUN apt-get update && apt-get install -y wget git zsh
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN sed -i 's/# DISABLE_AUTO_UPDATE="true"/DISABLE_AUTO_UPDATE="true"/g' ~/.zshrc

# Install Mambaforge
COPY conda_environment.yaml /root/
ENV MAMBA_DIR=/opt/mamba
RUN wget https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p $MAMBA_DIR \
    && rm Miniforge3-Linux-x86_64.sh

# Initialize Mamba environment
ENV PATH="$MAMBA_DIR/bin:$PATH"
RUN mamba init zsh && \
    mamba env create -f /root/conda_environment.yaml --yes

RUN echo 'mamba init' >> /root/.zshrc
RUN echo 'mamba activate msdp' >> /root/.zshrc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
    
# finalize image  ----------------
RUN apt-get clean && \
        rm -rf /var/lib/apt/lists/*
    
RUN source /root/.zshrc
    
WORKDIR /root
CMD ["zsh"]