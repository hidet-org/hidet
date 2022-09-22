from typing import List
import os


def get_include_dirs():
    cur_file = os.path.abspath(__file__)
    hidet_package_root = os.path.dirname(cur_file)
    include_dirs = []

    # include dir in a git repo
    dir_path = os.path.abspath(os.path.join(hidet_package_root, '..', '..', 'include'))
    include_dirs.append(dir_path)

    # include dir in a python package
    dir_path = os.path.abspath(os.path.join(hidet_package_root, 'include'))
    include_dirs.append(dir_path)

    exist_include_dirs = [dir_path for dir_path in include_dirs if os.path.exists(dir_path)]
    if len(exist_include_dirs) == 0:
        raise ValueError('Can not find the c/c++ include path, tried:\n{}'.format('\n'.join(include_dirs)))
    return exist_include_dirs


def get_library_search_dirs() -> List[str]:
    cur_file = os.path.abspath(__file__)
    root = os.path.dirname(cur_file)
    relative_dirs = [
        './',
        './lib',
        '../',
        '../lib',
        '../../',
        '../../lib',
        '../../build/lib',
        '../../build-release/lib',
        '../../build-debug/lib',
    ]
    return [os.path.abspath(os.path.join(root, relative)) for relative in relative_dirs]


if __name__ == '__main__':
    print(get_include_dirs())
