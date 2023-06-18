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
"""
Query the search directories for dynamic shared library, and the runtime header include path.
"""
from typing import List
import os


def get_include_dirs():
    """
    Get the include directories for the runtime header files.

    Returns
    -------
    include_dirs : List[str]
        The include directories.
    """
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
    """
    Get the library search directories for the dynamic libraries of hidet.

    Returns
    -------
    lib_dirs : List[str]
        The library search directories.
    """
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
