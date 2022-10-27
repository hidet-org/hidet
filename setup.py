from glob import glob
from setuptools import setup, find_packages, Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup(
    name="hidet",
    version="0.0.1",
    description="Hidet: a compilation-based DNN inference framework.",
    python_requires='>=3.7',
    packages=find_packages(where='python'),
    package_dir={"": "python"},
    include_package_data=True,
    # package_data={
    #     'hidet': [
    #         *glob('lib/*.so'),
    #         *glob('include/**/*.h', recursive=True),
    #     ]
    # },
    install_requires=[
        "onnx",
        "numpy",
        "psutil",
        "tqdm",
        "nvtx",
        "tabulate",
        "astunparse"
    ],
    distclass=BinaryDistribution,
)
