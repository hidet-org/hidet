#!/usr/bin/python3
"""
Get the current the version string in python/hidet/version.py

Usage
-----
$ python scripts/wheel/current_version.py

"""
import os
import argparse
import datetime

parser = argparse.ArgumentParser("current_version.py")

parser.add_argument(
    "--root",
    type=str,
    default="./",
    help="root directory of the project, under which setup.py is located. Default: ./",
)
parser.add_argument(
    "--nightly",
    action="store_true",
    help="If set, the version will be set to the nightly version. ",
)


def main():
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.expanduser(args.root))
    version_py = os.path.realpath(
        os.path.join(root_dir, "python", "hidet", "version.py")
    )
    if not os.path.exists(version_py) or not os.path.isfile(version_py):
        raise FileNotFoundError(version_py)
    output_version = None
    with open(version_py, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("__version__ = "):
                version = line.split("=")[1].strip()[1:-1]
                if args.nightly:
                    date_string = datetime.datetime.now().strftime("%Y%m%d")
                    if version.endswith(".dev"):
                        version = version.replace(".dev", ".dev{}".format(date_string))
                output_version = version
                break
    if output_version is None:
        raise RuntimeError('The occurrence of "__version__ = " in version.py is not')
    print(output_version, end="")


if __name__ == "__main__":
    main()
