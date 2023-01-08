#!/usr/bin/python3
"""
Get the current the version string in python/hidet/version.py

Usage
-----
$ python scripts/wheel/current_version.py

"""
import os
import argparse

parser = argparse.ArgumentParser("current_version.py")

parser.add_argument(
    "--root",
    type=str,
    default="./",
    help="root directory of the project, under which setup.py is located. Default: ./",
)


def main():
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.expanduser(args.root))
    version_py = os.path.realpath(
        os.path.join(root_dir, "python", "hidet", "version.py")
    )
    if not os.path.exists(version_py) or not os.path.isfile(version_py):
        raise FileNotFoundError(version_py)
    with open(version_py, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("__version__ = "):
                version = line.split("=")[1].strip()[1:-1]
                print(version)
                break


if __name__ == "__main__":
    main()
