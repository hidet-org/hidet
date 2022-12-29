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
    python_requires='>=3.8',
    packages=find_packages(where='python'),
    package_dir={"": "python"},
    include_package_data=True,
    install_requires=[
        "numpy>=1.23",  # for from_dlpack
        "psutil",
        "tqdm",
        "nvtx",
        "tabulate",
        "astunparse",
        "click",
        "cuda-python"
    ],
    distclass=BinaryDistribution,
    entry_points={
        'console_scripts': [
            'hidet = hidet.cli.main:main',
        ],
    },
)
