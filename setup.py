import tempfile
import os
import shutil
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
    root_dir = os.path.dirname(os.path.abspath(__file__))

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        with Cwd(tmp_dir):
            lib_dir = os.path.join(root_dir, 'python', 'hidet', 'lib')
            subprocess.run('cmake {}'.format(root_dir).split(), check=True)
            subprocess.run('cmake --build . -- hidet -j4'.split(), check=True)
            subprocess.run('mkdir -p {}'.format(lib_dir).split(), check=True)
            subprocess.run('cp -a ./lib/. {}'.format(lib_dir).split(), check=True)


def copy_include_dir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    include_dir = os.path.join(root_dir, 'include')
    dst_include_dir = os.path.join(root_dir, 'python', 'hidet', 'include')
    if os.path.exists(dst_include_dir) and os.path.isdir(dst_include_dir):
        shutil.rmtree(dst_include_dir)
    shutil.copytree(include_dir, dst_include_dir)


# build_cpp()
# copy_include_dir()

setup(
    name="hidet",
    version="0.0.1",
    description="Hidet: a compilation-based DNN inference framework.",
    python_requires='>=3.7',
    packages=find_packages(where='python'),
    package_dir={"": "python"},
    include_package_data=True,
    package_data={
        'hidet': [
            'lib/*.so',
            'include/**/*.h'
        ]
    },
    zip_safe=False,
    install_requires=[
        "onnx",
        "numpy",
        "psutil",
        "tqdm",
        "nvtx",
        "tabulate",
        "astunparse"
    ]
)
