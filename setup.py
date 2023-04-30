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
from setuptools import setup, find_packages


setup(
    name="hidet",
    version="0.2.4.dev",
    description="Hidet: a compilation-based DNN inference framework.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
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
        "packaging",
        "cuda-python>=11.6.1; platform_system=='Linux'",
    ],
    platforms=["linux"],
    entry_points={
        'console_scripts': [
            'hidet = hidet.cli.main:main',
        ],
        'torch_dynamo_backends': [
            'hidet = hidet.graph.frontend.torch.dynamo_backends:hidet_backend',
        ]
    },
    url="https://docs.hidet.org",
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
