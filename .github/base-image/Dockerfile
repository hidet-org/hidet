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

# Install pip
RUN python -m pip install --upgrade pip

# Add source files
ADD . /workspace/hidet

# Set working directory
WORKDIR /workspace

# Install CI requirements
RUN pip install -r hidet/.github/requirements-ci.txt

# Install Hidet Docs requirements
RUN pip install --no-cache-dir -r hidet/docs/requirements.txt

# Build and install Hidet
RUN bash hidet/scripts/wheel/build_wheel.sh

RUN WHEEL=$(find hidet/scripts/wheel/built_wheel -maxdepth 1 -name '*.whl') && \
    if [ -z "$WHEEL" ]; then echo "❌ No wheel file found!"; exit 1; fi && \
    echo "✅ Found wheel: $WHEEL" && \
    # pip install --force-reinstall "${WHEEL}[dev]"
    pip install "${WHEEL}[dev]"
