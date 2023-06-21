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
from typing import Dict, Optional, Callable
import os
import pickle
import time
import warnings
import ctypes
from hidet.ir.type import FuncType, PointerType, DataType, BaseType, VoidType, TensorPointerType
from hidet.ffi.shared_lib import SharedLibrary
from hidet.ffi.utils import c_pointer_compatible


class CompiledModuleLoadError(Exception):
    pass


class CompiledFunction:
    """
    A compiled function that can be directly called.
    """

    def __init__(self, name, func_type: FuncType, ctypes_func):
        self.name: str = name
        self.func_type: FuncType = func_type
        self.ctypes_func: Callable = ctypes_func

        self._update_func_signature()

    def __call__(self, *args):
        from hidet.ffi.ffi import BackendException, get_last_error

        ret = self.ctypes_func(*args)

        status = get_last_error()
        if status is not None:
            msg = 'Calling {} with arguments {} failed. error:\n{}'.format(self.name, args, status)
            raise BackendException(msg)

        return ret

    def _parse_type(self, hidet_type: BaseType):
        if isinstance(hidet_type, DataType):
            from hidet.ir import dtypes

            mapping = {
                dtypes.int8: ctypes.c_int8,
                dtypes.int16: ctypes.c_int16,
                dtypes.int32: ctypes.c_int32,
                dtypes.int64: ctypes.c_int64,
                dtypes.uint8: ctypes.c_uint8,
                dtypes.uint16: ctypes.c_uint16,
                dtypes.uint32: ctypes.c_uint32,
                dtypes.uint64: ctypes.c_uint64,
                # dtypes.float16: sadly, there is no float16 in ctypes for now, we might need to create a custom type
                dtypes.float32: ctypes.c_float,
                dtypes.float64: ctypes.c_double,
                dtypes.boolean: ctypes.c_bool,
                # dtypes.complex64:
                # dtypes.complex128:
            }
            if hidet_type not in mapping:
                raise NotImplementedError('Unsupported type {}'.format(hidet_type))
            return mapping[hidet_type]
        elif isinstance(hidet_type, VoidType):
            return None
        elif isinstance(hidet_type, (PointerType, TensorPointerType)):
            return c_pointer_compatible
        else:
            raise NotImplementedError('Unsupported type {}'.format(hidet_type))

    def _update_func_signature(self):
        self.ctypes_func.argtypes = [self._parse_type(hidet_type) for hidet_type in self.func_type.param_types]
        self.ctypes_func.restype = self._parse_type(self.func_type.ret_type)

    def profile(self, *args, warmup=1, number=2, repeat=10):
        from hidet.cuda import current_stream

        for _ in range(warmup):
            self.ctypes_func(*args)

        results = []
        for _ in range(repeat):
            current_stream().synchronize()
            start = time.time()
            for _ in range(number):
                self.ctypes_func(*args)
            current_stream().synchronize()
            end = time.time()
            results.append((end - start) / number * 1000)

        return results


class CompiledModule:
    def __init__(self, module_dir: str):
        self.module_dir: str = module_dir
        self.shared_library: SharedLibrary = self._load_shared_library()
        self.functions: Dict[str, CompiledFunction] = self._load_functions()

    def __call__(self, *args):
        if 'launch' not in self.functions:
            raise RuntimeError('Launch function not found.')
        return self.functions['launch'](*args)

    def __getitem__(self, item: str) -> CompiledFunction:
        return self.functions[item]

    def _load_shared_library(self):
        lib_path = os.path.join(self.module_dir, 'lib.so')
        if not os.path.exists(lib_path):
            raise CompiledModuleLoadError('Shared library {} does not exist.'.format(lib_path))
        return SharedLibrary(lib_path)

    def _load_functions(self):
        func_types_path = os.path.join(self.module_dir, 'func_types.pickle')
        if not os.path.exists(func_types_path):
            raise CompiledModuleLoadError('Function types {} does not exist.'.format(func_types_path))
        with open(func_types_path, 'rb') as f:
            func_types: Dict[str, FuncType] = pickle.load(f)
        functions: Dict[str, CompiledFunction] = {}
        for name, func_type in func_types.items():
            functions[name] = CompiledFunction(name, func_type, self.shared_library['hidet_' + name])
        return functions

    def source(self, color=False) -> Optional[str]:
        if os.path.exists(os.path.join(self.module_dir, 'source.cc')):
            src_path = os.path.join(self.module_dir, 'source.cc')
        elif os.path.exists(os.path.join(self.module_dir, 'source.cu')):
            src_path = os.path.join(self.module_dir, 'source.cu')
        else:
            src_path = None

        if src_path is None:
            return None
        with open(src_path, 'r') as f:
            src_code = f.read()

        if color:
            import importlib.util

            if importlib.util.find_spec('pygments'):
                from pygments import highlight
                from pygments.lexers import CudaLexer
                from pygments.formatters import Terminal256Formatter

                return highlight(src_code, CudaLexer(), Terminal256Formatter(style='autumn'))
            else:
                warnings.warn('pygments is not installed, please install it to enable colorized source code.')
        return src_code

    def profile(self, *args, warmup=1, number=2, repeat=10):
        return self['launch'].profile(*args, warmup=warmup, number=number, repeat=repeat)


def load_compiled_module(module_dir: str) -> CompiledModule:
    return CompiledModule(module_dir)


def compiled_module_exists(module_dir: str) -> bool:
    required_files = ['lib.so', 'func_types.pickle']
    for file in required_files:
        if not os.path.exists(os.path.join(module_dir, file)):
            return False
    return True
