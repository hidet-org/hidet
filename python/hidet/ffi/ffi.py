import os
import os.path
from typing import Optional
import ctypes

_LIB: Optional[ctypes.CDLL] = None


def load_library():
    global _LIB
    if _LIB:
        return
    paths = ['./libhidet.so',
             '../build-release/lib/libhidet.so',
             '../../build-release/lib/libhidet.so',
             '../../../build-release/lib/libhidet.so',
             '../../../../build-release/lib/libhidet.so',
             '../build/lib/libhidet.so',
             '../../build/lib/libhidet.so',
             '../../../build/lib/libhidet.so',
             '../../../../build/lib/libhidet.so',
             ]
    cwd = os.getcwd()
    paths = [os.path.realpath(os.path.join(cwd, path)) for path in paths]
    for path in paths:
        try:
            _LIB = ctypes.cdll.LoadLibrary(path)
        except OSError as e:
            pass
    if _LIB is None:
        raise OSError('Can not load library: \n' + '\n'.join(paths))


load_library()
