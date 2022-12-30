"""
Update the version string in setup.py and python/hidet/version.py

Usage
-----
$ python scripts/update_version.py <version>

For example, to update the version to 0.0.2, run
$ python scripts/update_version.py 0.0.2
"""
import os
import argparse
from packaging.version import InvalidVersion, parse

parser = argparse.ArgumentParser('update_version.py')

parser.add_argument('version', type=str, help='Version to update to')


def update_setup_py(setup_py, version):
    print('Updating version in {} to {}'.format(setup_py, version))

    with open(setup_py, 'r') as f:
        lines = f.readlines()

    count = 0
    for i, line in enumerate(lines):
        if line.startswith('    version="'):
            lines[i] = '    version="{}",\n'.format(version)
            count += 1
    if count != 1:
        raise RuntimeError('The occurrence of version= in setup.py is not 1')

    with open(setup_py, 'w') as f:
        f.writelines(lines)


def update_version_py(version_py, version):
    print('Updating version in {} to {}'.format(version_py, version))

    with open(version_py, 'r') as f:
        lines = f.readlines()

    count = 0
    for i, line in enumerate(lines):
        if line.startswith('__version__ = '):
            lines[i] = '__version__ = "{}"\n'.format(version)
            count += 1
    if count != 1:
        raise RuntimeError('The occurrence of "__version__ = " in version.py is not 1')

    with open(version_py, 'w') as f:
        f.writelines(lines)


def main():
    args = parser.parse_args()

    version = args.version

    try:
        parse(version)
    except InvalidVersion as e:
        raise e

    # Update version in setup.py
    script_dir = os.path.dirname(os.path.realpath(__file__))
    setup_py = os.path.realpath(os.path.join(script_dir, '..', 'setup.py'))
    version_py = os.path.realpath(os.path.join(script_dir, '..', 'python', 'hidet', 'version.py'))
    update_setup_py(setup_py, version)
    update_version_py(version_py, version)


if __name__ == '__main__':
    main()
