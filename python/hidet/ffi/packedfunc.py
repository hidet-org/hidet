from typing import Dict, Sequence, Union
import ctypes
from pycuda.gpuarray import GPUArray

from .ffi import _LIB
from ctypes import c_int32, c_void_p, pointer, c_float, cast, c_bool
from ctypes import POINTER, Structure
from hidet.ir.type import ScalarType, TensorType
from hidet.ir.dialects.lowlevel import PointerType, TensorPointerType
from hidet.runtime.value import Value, ScalarValue, TensorValue

c_int32_p = POINTER(c_int32)
c_float_p = POINTER(c_float)


class ArgType:
    INT32 = 1
    FLOAT32 = 2
    POINTER = 3


class CPackedFunc(Structure):
    _fields_ = [("num_args", c_int32),
                ("arg_types", c_int32_p),
                ("func_pointer", c_void_p)]


class PackedFunc:
    def __init__(self, param_types, c_func_pointer, ret_type=None, default_args: Dict[int, object] = None):
        self.param_types = param_types
        self.ret_type = ret_type
        self.c_func_pointer = c_func_pointer
        self.default_args = default_args if default_args is not None else {}

        type_codes = [self._type_code(param_type) for param_type in self.param_types]
        if self.ret_type:
            type_codes.append(self._type_code(self.ret_type))
        n = len(type_codes)
        num_args = c_int32(n)
        arg_types = cast(pointer((c_int32 * n)(*type_codes)), POINTER(c_int32))
        func_pointer = cast(self.c_func_pointer, c_void_p)
        self.c_packed_func = CPackedFunc(num_args, arg_types, func_pointer)

    def _convert_arg(self, idx, arg: Union[Value, int, float]):
        """
        convert arg to a c_void_p
        """
        if isinstance(arg, int):
            return cast(pointer(c_int32(arg)), c_void_p)
        elif isinstance(arg, float):
            return cast(pointer(c_float(arg)), c_void_p)
        elif isinstance(arg, ScalarValue):
            assert isinstance(self.param_types[idx], ScalarType)
            arg_type = arg.type
            if arg_type.name == 'int32':
                assert self.param_types[idx].name == 'int32'
                return cast(pointer(c_int32(arg.value)), c_void_p)
            elif arg_type.name == 'float32':
                assert self.param_types[idx].name == 'float32'
                return cast(pointer(c_float(arg.value)), c_void_p)
            else:
                raise NotImplementedError()
        elif isinstance(arg, TensorValue):
            assert isinstance(self.param_types[idx], (PointerType, TensorPointerType, TensorType))
            arg_type = arg.type
            if arg_type.scalar_type.name == 'float32':
                if isinstance(arg.array, GPUArray):
                    rt = cast(int(arg.array.gpudata), c_void_p)
                else:
                    rt = cast(int(arg.array.__array_interface__['data'][0]), c_void_p)
                return rt
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def _type_code(self, param_type):
        type_map = {
            'bool': c_int32(1),
            'int32': c_int32(1),
            'float32': c_int32(2),
            'pointer': c_int32(3)
        }
        if param_type is bool or param_type is int:
            type_name = 'int32'
        elif isinstance(param_type, ScalarType):
            type_name = param_type.name
        elif isinstance(param_type, PointerType):
            type_name = 'pointer'
        elif isinstance(param_type, TensorType):
            type_name = 'pointer'
        elif isinstance(param_type, TensorPointerType):
            type_name = 'pointer'
        else:
            raise NotImplementedError(param_type)
        return type_map[type_name]

    def _convert_args(self, orig_args: Sequence):
        n = len(orig_args) + len(self.default_args)
        args = []
        orig_args = list(reversed(orig_args))
        for i in range(n):
            if i in self.default_args:
                args.append(self.default_args[i])
            else:
                args.append(orig_args.pop())
        assert len(args) == len(self.param_types)

        converted_args = [self._convert_arg(idx, arg) for idx, arg in enumerate(args)]
        if self.ret_type is not None:
            if self.ret_type is bool:
                ret_arg = c_int32()
            else:
                raise NotImplementedError()
            n += 1
            converted_args.append(cast(pointer(ret_arg), c_void_p))
        else:
            ret_arg = None
        p_args = cast(pointer((ctypes.c_void_p * n)(*converted_args)), c_void_p)
        return p_args, ret_arg

    def __call__(self, *args):
        p_args, ret_arg = self._convert_args(args)
        _LIB.CallPackedFunc(self.c_packed_func, p_args)
        if ret_arg is not None:
            if issubclass(self.ret_type, bool):
                return bool(ret_arg.value)
            else:
                raise NotImplementedError()
        else:
            return None

    def profile(self, *args, warmup: int = 1, number: int = 1, repeat: int = 10):
        results = (c_float * repeat)()
        p_args, ret_arg = self._convert_args(args)
        _LIB.ProfilePackedFunc(self.c_packed_func, p_args, warmup, number, repeat, cast(pointer(results), c_float_p))
        return [float(v) for v in results]
