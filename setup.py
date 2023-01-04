# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_packages, Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup(
    name="hidet",
    version="0.1",
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
    ],
    license='Apache-2.0',
    keywords='deep learning, machine learning, neural network, inference, compiler',
)
