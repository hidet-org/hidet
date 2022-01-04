import ctypes
import numpy as np
from pycuda.gpuarray import GPUArray

from .ffi import _LIB
from ctypes import c_int32, c_void_p, pointer, c_float, cast
from ctypes import POINTER, Structure
from hidet.ir.type import ScalarType, TensorType, Type, tensor_type, PointerType
from hidet.runtime.value import Value, ScalarValue, TensorValue


c_int32_p = POINTER(c_int32)


class ArgType:
    INT32 = 1
    FLOAT32 = 2
    POINTER = 3


class CPackedFunc(Structure):
    _fields_ = [("num_args", c_int32),
                ("arg_types", c_int32_p),
                ("func_pointer", c_void_p)]


class PackedFunc:
    def __init__(self, param_types, c_func_pointer):
        self.param_types = param_types
        self.c_func_pointer = c_func_pointer

        n = len(self.param_types)
        num_args = c_int32(n)
        arg_types = cast(pointer((c_int32 * n)(*[self._type_code(param_type) for param_type in self.param_types])), POINTER(c_int32))
        func_pointer = cast(self.c_func_pointer, c_void_p)
        self.c_packed_func = CPackedFunc(num_args, arg_types, func_pointer)

    def _convert_arg(self, idx, arg: Value):
        """
        convert arg to a c_void_p
        """
        if isinstance(arg, ScalarValue):
            assert isinstance(self.param_types[idx], ScalarType)
            arg_type = arg.type
            if arg_type.name == 'int32':
                assert isinstance(self.param_types[idx].name, 'int32')
                return cast(pointer(c_int32(arg.value)), c_void_p)
            elif arg_type.name == 'float32':
                assert isinstance(self.param_types[idx].name, 'float32')
                return cast(pointer(c_float(arg.value)), c_void_p)
            else:
                raise NotImplementedError()
        elif isinstance(arg, TensorValue):
            assert isinstance(self.param_types[idx], PointerType)
            arg_type = arg.type
            if arg_type.scalar_type.name == 'float32':
                if isinstance(arg.array, GPUArray):
                    rt = cast(int(arg.array.gpudata), c_void_p)
                else:
                    rt = cast(int(arg.array.__array_interface__['data'][0]), c_void_p)
                return rt
            else:
                raise NotImplementedError()

    def _type_code(self, param_type):
        type_map = {
            'int32': c_int32(1),
            'float32': c_int32(2),
            'pointer': c_int32(3)
        }
        if isinstance(param_type, ScalarType):
            type_name = param_type.name
        elif isinstance(param_type, PointerType):
            type_name = 'pointer'
        else:
            raise NotImplementedError()
        return type_map[type_name]

    def _convert_args(self, args):
        n = len(args)
        p_args = cast(pointer((ctypes.c_void_p * n)(*[self._convert_arg(idx, arg) for idx, arg in enumerate(args)])), c_void_p)
        return p_args

    def __call__(self, *args):
        _LIB.CallPackedFunc(self.c_packed_func, self._convert_args(args))

