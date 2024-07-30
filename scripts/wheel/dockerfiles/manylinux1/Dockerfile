FROM quay.io/pypa/manylinux1_x86_64 as base

RUN yum install -y vim htop

ENV PATH=$PATH:/opt/python/cp38-cp38/bin

RUN PIP_ONLY_BINARY=cmake python3 -m pip install cmake
