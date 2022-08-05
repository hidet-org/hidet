from typing import List
import os


def get_include_dir():
    cur_file = os.path.abspath(__file__)
    root = os.path.join(os.path.dirname(cur_file), '..', '..')
    include_dir = os.path.join(root, 'include')
    return os.path.abspath(include_dir)


def get_library_search_dirs() -> List[str]:
    cur_file = os.path.abspath(__file__)
    root = os.path.dirname(cur_file)
    relative_dirs = [
        './',
        '../',
        '../../',
        '../../lib',
        '../../build/lib',
        '../../build-release/lib',
        '../../build-debug/lib',
    ]
    return [os.path.abspath(os.path.join(root, relative)) for relative in relative_dirs]


if __name__ == '__main__':
    print(get_include_dir())
