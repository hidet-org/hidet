from setuptools import setup, find_packages

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
