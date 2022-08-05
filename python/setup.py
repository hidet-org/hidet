from typing import List
import tempfile
import os
import shutil
import pathlib
from setuptools import setup, Extension, find_packages
import subprocess


class Cwd:
    def __init__(self, name):
        self.name = name
        self.old_cwd = None

    def __enter__(self):
        self.old_cwd = os.getcwd()
        os.chdir(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.old_cwd)


cur_dir = os.path.dirname(os.path.abspath(__file__))


def build_cpp():
    cmake_dir = os.path.join(cur_dir, '..')

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        with Cwd(tmp_dir):
            subprocess.run('cmake {}'.format(cmake_dir).split(), check=True)
            subprocess.run('cmake --build . -- hidet -j4'.split(), check=True)
            subprocess.run('cp -r ./lib {}'.format(cur_dir).split(), check=True)


build_cpp()

setup(
    name="hidet",
    version="0.0.1",
    description="Hidet: a compilation-based DNN inference framework.",
    packages=find_packages(),
    include_dirs=[os.path.join(cur_dir, './lib')],
    include_package_data=True,
    data_files=[
        ('hidet', ['./lib/libhidet.so', './lib/libhidet_runtime.so']),
    ],
    install_requires=[
        "onnx",
        "numpy",
        "psutil",
        "tqdm",
        "nvtx",
        "tabulate"
    ]
)

if __name__ == '__main__':
    pass
