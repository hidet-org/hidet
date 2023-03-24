#!/usr/bin/python3
"""
Update the version string in setup.py and python/hidet/version.py

Usage
-----
$ python scripts/wheel/update_version.py <version>

For example, to update the version to 0.0.2, run
$ python scripts/wheel/update_version.py 0.0.2
"""
import os
import argparse
import re

parser = argparse.ArgumentParser("update_version.py")

parser.add_argument(
    "--root",
    type=str,
    default="./",
    help="root directory of the project, under which setup.py is located. Default: ./",
)
parser.add_argument("--version", type=str, required=True, help="Version to update to (e.g., 0.2.3 or 0.2.3.dev).")


def update_setup_py(setup_py, version):
    print("Updating version in {} to {}".format(setup_py, version))

    with open(setup_py, "r") as f:
        lines = f.readlines()

    count = 0
    for i, line in enumerate(lines):
        if line.startswith('    version="'):
            lines[i] = '    version="{}",\n'.format(version)
            count += 1
    if count != 1:
        raise RuntimeError("The occurrence of version= in setup.py is not 1")

    with open(setup_py, "w") as f:
        f.writelines(lines)


def update_version_py(version_py, version):
    print("Updating version in {} to {}".format(version_py, version))

    with open(version_py, "r") as f:
        lines = f.readlines()

    count = 0
    for i, line in enumerate(lines):
        if line.startswith("__version__ = "):
            lines[i] = '__version__ = "{}"\n'.format(version)
            count += 1
    if count != 1:
        raise RuntimeError('The occurrence of "__version__ = " in version.py is not 1')

    with open(version_py, "w") as f:
        f.writelines(lines)


def check_version(version: str):
    patterns = [
        r"^\d+\.\d+(\.\d+)?$",  # 0.1.1 or 0.2
        r"^\d+\.\d+(\.\d+)?(\.dev\d*)?$",  # 0.1.1.dev1 or 0.2.dev2 or 0.2.dev20220101
    ]
    for pattern in patterns:
        if re.match(pattern, version):
            return
    raise ValueError("Invalid version: {}".format(version))


def main():
    args = parser.parse_args()

    version = args.version

    check_version(version)

    # Update version in setup.py
    root_dir = os.path.abspath(os.path.expanduser(args.root))
    setup_py = os.path.realpath(os.path.join(root_dir, "setup.py"))
    version_py = os.path.realpath(
        os.path.join(root_dir, "python", "hidet", "version.py")
    )
    if not os.path.exists(setup_py) or not os.path.isfile(setup_py):
        raise FileNotFoundError(setup_py)
    if not os.path.exists(version_py) or not os.path.isfile(version_py):
        raise FileNotFoundError(version_py)
    update_setup_py(setup_py, version)
    update_version_py(version_py, version)


if __name__ == "__main__":
    main()
