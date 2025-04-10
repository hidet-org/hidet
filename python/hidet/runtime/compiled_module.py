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
from typing import Dict, Optional, Callable, Sequence
import os
import pickle
import time
import warnings

from hidet.ffi.convert import ctypes_type, to_ctypes_arg, from_ctypes_return
from hidet.ir.type import FuncType, BaseType
from hidet.ffi.shared_lib import SharedLibrary


class CompiledModuleLoadError(Exception):
    pass


class CompiledFunction:
    """
    A compiled function that can be directly called.

    This class should not be instantiated directly. Instead, use the :attr:`CompiledModule.functions` attribute of
    :class:`CompiledModule` to get the compiled function.

    Parameters
    ----------
    name: str
        The name of the function.

    func_type: FuncType
        The type of the function.

    ctypes_func:
        The ctypes function, which holds the actual function pointer loaded in memory.
    """

    def __init__(self, name, func_type: FuncType, ctypes_func, lib_path: str):
        self.name: str = name
        self.param_types: Sequence[BaseType] = func_type.param_types
        self.ret_type: BaseType = func_type.ret_type
        self.ctypes_func: Callable = ctypes_func
        self.lib_path: str = lib_path

        ctypes_func.argtypes = [ctypes_type(t) for t in self.param_types]
        ctypes_func.restype = ctypes_type(self.ret_type)

    def __call__(self, *args):
        """
        Call the compiled function.

        Parameters
        ----------
        args: a sequence of int, float, bool, hidet.Tensor or hidet.ffi.utils.Array
            The arguments to the function.

        Returns
        -------
        ret: Optional[Union[int, float, bool]]
            The return value of the function.
        """
        from hidet.ffi.ffi import BackendException, get_last_error

        cargs = []
        for typ, arg in zip(self.param_types, args):
            cargs.append(to_ctypes_arg(hidet_type=typ, obj=arg))

        val = self.ctypes_func(*cargs)
        ret = from_ctypes_return(hidet_type=self.ret_type, val=val)

        status = get_last_error()
        if status is not None:
            from hidet.graph.tensor import Tensor

            args_items = []
            for arg in args:
                if isinstance(arg, Tensor):
                    args_items.append(arg.signature())
                elif hasattr(arg, '__module__') and arg.__module__ == 'torch':
                    import torch

                    assert isinstance(arg, torch.Tensor)
                    args_items.append(
                        'torch.Tensor(shape={}, dtype={}, device={})'.format(tuple(arg.shape), arg.dtype, arg.device)
                    )
                else:
                    args_items.append(str(arg))
            args_str = ', '.join(args_items)
            msg = (
                'Calling {} with arguments ({}) failed. error:\n'.format(self.name, args_str)
                + '{}\n'.format(status)
                + 'Function @ {}'.format(self.lib_path)
            )
            raise BackendException(msg)

        return ret

    def profile(self, *args, warmup=1, number=2, repeat=10):
        """
        Profile the compiled function.

        In total, the function will be called warmup + number * repeat times. We will group the calls into repeat
        groups, and measure the average time of each group.

        Parameters
        ----------
        args: a sequence of int, float, bool or hidet.Tensor
            The arguments to the function.

        warmup: int
            The number of warmup runs.

        number: int
            The number of runs to measure the average time.

        repeat: int
            The number of times to repeat the measurement.

        Returns
        -------
        results: List[float]
            The measured time in milliseconds (we have len(results) == repeat)).
        """
        from hidet.cuda import current_stream

        for _ in range(warmup):
            self(*args)

        results = []
        for _ in range(repeat):
            current_stream().synchronize()
            start = time.time()
            for _ in range(number):
                self(*args)
            current_stream().synchronize()
            end = time.time()
            results.append((end - start) / number * 1000)

        return results


class CompiledModule:
    """
    A compiled module that can be directly called.

    We can use the module like:

    .. code-block:: python

        module: CompiledModule = load_compiled_module(...)

        # check the function names
        print(module.functions.keys())

        # call the `launch` function
        module(...)
        # or equivalently
        module['launch'](...)

        # call the `foo` function (if exists)
        module['foo'](...)

        # get the source code
        source: str = module.source(color=True)  # colorized source code

    This class should not be instantiated directly. Instead, use the `load_compiled_module` function to load a compiled
    module from the given directory. Or build a CompiledModule from scratch like

    .. code-block:: python

      with hidet.script_module() as script_module:
          ...

      compiled_module = script_module.build()


    Parameters
    ----------
    module_dir: str
        The directory of the module.

    Attributes
    ----------
    module_dir: str
        The directory of the module.

    shared_library: SharedLibrary
        The shared library of the module.

    functions: Dict[str, CompiledFunction]
        The functions in the module.
    """

    def __init__(self, module_dir: str):
        """
        Construct a compiled module.

        """
        self.module_dir: str = module_dir
        self.shared_library: SharedLibrary = self._load_shared_library()
        self.functions: Dict[str, CompiledFunction] = self._load_functions()

    def __call__(self, *args):
        """
        Call the `launch` function in the module.

        Parameters
        ----------
        args: a sequence of int, float, bool or hidet.Tensor
            The arguments to the function.

        Returns
        -------
        ret: Optional[Union[int, float, bool]]
            The return value of the function.
        """
        if 'launch' not in self.functions:
            raise RuntimeError('Launch function not found.')
        return self.functions['launch'](*args)

    def __getitem__(self, item: str) -> CompiledFunction:
        """
        Get the function with the given name.

        Parameters
        ----------
        item: str
            The name of the function.

        Returns
        -------
        func: CompiledFunction
            The compiled function.
        """
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
            functions[name] = CompiledFunction(
                name, func_type, self.shared_library['hidet_' + name], self.shared_library.lib_path
            )
        return functions

    def source(self, color=False) -> Optional[str]:
        """
        Get the source code of the module.

        Parameters
        ----------
        color: bool
            Whether to colorize the source code.

        Returns
        -------
        source: Optional[str]
            The source code of the module if the source file exists, otherwise None.
        """
        if os.path.exists(os.path.join(self.module_dir, 'source.cc')):
            src_path = os.path.join(self.module_dir, 'source.cc')
        elif os.path.exists(os.path.join(self.module_dir, 'source.cu')):
            src_path = os.path.join(self.module_dir, 'source.cu')
        elif os.path.exists(os.path.join(self.module_dir, 'source.hip.cpp')):
            src_path = os.path.join(self.module_dir, 'source.hip.cpp')
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
        """
        Profile the `launch` function in the module.

        In total, the function will be called warmup + number * repeat times. We will group the calls into repeat
        groups, and measure the average time of each group.

        Parameters
        ----------
        args: a sequence of int, float, bool or hidet.Tensor
            The arguments to the function.

        warmup: int
            The number of warmup runs.

        number: int
            The number of runs to measure the average time.

        repeat: int
            The number of times to repeat the measurement.

        Returns
        -------
        results: List[float]
            The measured time in milliseconds (we have len(results) == repeat)).
        """
        return self['launch'].profile(*args, warmup=warmup, number=number, repeat=repeat)


def load_compiled_module(module_dir: str) -> CompiledModule:
    """
    Load a compiled module from the given directory.

    Parameters
    ----------
    module_dir: str
        The directory of the module.

    Returns
    -------
    module: CompiledModule
        The compiled module.
    """
    return CompiledModule(module_dir)


def compiled_module_exists(module_dir: str) -> bool:
    """
    Check whether the compiled module exists in the given module directory.

    Parameters
    ----------
    module_dir: str
        The directory of the module.

    Returns
    -------
    exists: bool
        Whether the compiled module exists.
    """
    required_files = ['lib.so', 'func_types.pickle']
    for file in required_files:
        if not os.path.exists(os.path.join(module_dir, file)):
            return False
        # if the lib.so file is empty, we consider it as not exists since it is a broken library
        # due to interrupted compilation
        if file == 'lib.so' and os.path.getsize(os.path.join(module_dir, file)) == 0:
            return False
    return True
