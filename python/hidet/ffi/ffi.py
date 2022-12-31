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
from typing import List, Dict, Optional
import os
import os.path
import ctypes
from hidet.libinfo import get_library_search_dirs

_LIB: Optional[ctypes.CDLL] = None
_LIB_RUNTIME: Optional[ctypes.CDLL] = None


library_paths: Dict[str, Optional[str]] = {'hidet': None, 'hidet_runtime': None}


def load_library():
    global _LIB, _LIB_RUNTIME
    if _LIB:
        return
    library_dirs = get_library_search_dirs()
    for library_dir in library_dirs:
        libhidet_path = os.path.join(library_dir, 'libhidet.so')
        libhidet_runtime_path = os.path.join(library_dir, 'libhidet_runtime.so')
        if not os.path.exists(libhidet_path) or not os.path.exists(libhidet_runtime_path):
            continue
        _LIB_RUNTIME = ctypes.cdll.LoadLibrary(libhidet_runtime_path)
        _LIB = ctypes.cdll.LoadLibrary(libhidet_path)
        library_paths['hidet_runtime'] = libhidet_runtime_path
        library_paths['hidet'] = libhidet_path
        break
    if _LIB is None:
        raise OSError('Can not find library in the following directory: \n' + '\n'.join(library_dirs))


def get_last_error() -> Optional[str]:
    func = _LIB['hidet_get_last_error']
    func.restype = ctypes.c_char_p
    ret = func()
    if isinstance(ret, bytes):
        return ret.decode('utf-8')
    else:
        return None


class BackendException(Exception):
    pass


def func_exists(func_name: str, shared_lib: ctypes.CDLL) -> bool:
    try:
        getattr(shared_lib, func_name)
        return True
    except AttributeError:
        return False


def get_func(func_name, arg_types: List, restype):
    if func_exists(func_name, _LIB):
        func = getattr(_LIB, func_name)
    elif func_exists(func_name, _LIB_RUNTIME):
        func = getattr(_LIB_RUNTIME, func_name)
    else:
        raise ValueError(
            'Can not find function "{}" in hidet libraries:\n{}\n{}'.format(
                func_name, library_paths['hidet'], library_paths['hidet_runtime']
            )
        )

    func.argtypes = arg_types
    func.restype = restype

    def func_with_check(*args):
        try:
            ret = func(*args)
        except TypeError as e:
            raise TypeError('The argument or return type of function {} does not match.'.format(func_name)) from e
        status = get_last_error()
        if status is not None:
            msg = 'Calling {} with arguments {} failed. error:\n{}'.format(func_name, args, status)
            raise BackendException(msg)
        return ret

    return func_with_check


load_library()
