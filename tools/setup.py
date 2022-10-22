from setuptools import setup, find_packages

setup(
    name="hidet-cli",
    version="0.0.1",
    description="A command line interface for Hidet.",
    python_requires='>=3.7',
    packages=find_packages(where='.'),
    install_requires=[
        # "hidet",
        "numpy",
        "click"
    ],
    entry_points={
        'console_scripts': [
            'hidet = hidet_cli.main:cli',
        ],
    },
)
