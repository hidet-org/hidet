FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y \
    ccache \
    git \
    graphviz \
    graphviz-dev \
    software-properties-common

# Install Python 3.9
RUN apt-get update && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install system dependencies and CMake 3.19.8
RUN apt-get update && \
    apt-get install -y wget gnupg software-properties-common && \
    wget https://github.com/Kitware/CMake/releases/download/v3.19.8/cmake-3.19.8-Linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-3.19.8-Linux-x86_64.sh --skip-license --prefix=/opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm cmake-3.19.8-Linux-x86_64.sh
# Test CMake Installation
RUN cmake --version

# Install pip
RUN python -m pip install --upgrade pip

# Add source files
ADD . /workspace/hidet

# Set working directory
WORKDIR /workspace

# Install Hidet Docs requirements
RUN pip install --no-cache-dir -r hidet/docs/requirements.txt

# Build and install Hidet
RUN bash hidet/scripts/wheel/build_wheel.sh

RUN WHEEL=$(find hidet/scripts/wheel/built_wheel -maxdepth 1 -name '*.whl') && \
    if [ -z "$WHEEL" ]; then echo "❌ No wheel file found!"; exit 1; fi && \
    echo "✅ Found wheel: $WHEEL" && \
    # pip install --force-reinstall "${WHEEL}[dev]"
    pip install "${WHEEL}[dev]"
