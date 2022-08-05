from typing import List, Dict
import os
import os.path
from typing import Optional
import ctypes
from hidet.libinfo import get_library_search_dirs

_LIB: Optional[ctypes.CDLL] = None


library_paths: Dict[str, Optional[str]] = {
    'hidet': None,
    'hidet_runtime': None
}


def load_library():
    global _LIB
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


def get_func(func_name, arg_types: List, restype):
    func = _LIB[func_name]
    func.argtypes = arg_types
    func.restype = restype

    def func_with_check(*args):
        ret = func(*args)
        status = get_last_error()
        if status is not None:
            msg = 'Calling {} with arguments {} failed. error:\n{}'.format(func_name, args, status)
            raise BackendException(msg)
        return ret

    return func_with_check


load_library()
