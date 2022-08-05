import tempfile
import os
from setuptools import setup, find_packages
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


def build_cpp():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cmake_dir = os.path.join(cur_dir, '..')

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        with Cwd(tmp_dir):
            subprocess.run('cmake {}'.format(cmake_dir).split(), check=True)
            subprocess.run('cmake --build . -- hidet -j4'.split(), check=True)
            subprocess.run('cp -a ./lib/. {}'.format(os.path.join(cur_dir, 'hidet')).split(), check=True)


build_cpp()

setup(
    name="hidet",
    version="0.0.1",
    description="Hidet: a compilation-based DNN inference framework.",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'hidet': ['*.so']
    },
    zip_safe=False,
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
