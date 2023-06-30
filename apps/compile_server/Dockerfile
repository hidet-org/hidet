FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

COPY ./run.py /app/run.py
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && python -m pip install filelock requests

RUN apt-get update && apt-get install -y \
    git \
    && python -m pip install gunicorn flask \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 3281

CMD ["python", "run.py"]
