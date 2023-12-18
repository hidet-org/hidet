FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

COPY ./run.py /app/run.py
WORKDIR /app

ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    vim \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && python -m pip install filelock requests gunicorn flask cmake

EXPOSE 3281

CMD ["python", "run.py"]
