from setuptools import setup, find_packages, Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup(
    name="hidet",
    version="0.1.dev0",
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
    url="docs.hidet.org",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    license='Apache-2.0',
    keywords='deep learning, machine learning, neural network, inference, compiler',
)
