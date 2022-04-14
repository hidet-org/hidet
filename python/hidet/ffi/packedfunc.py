from typing import Dict, Sequence, Union, Type
import ctypes

from .ffi import _LIB
from ctypes import c_int32, c_void_p, pointer, c_float, cast, c_bool
from ctypes import POINTER, Structure
from hidet.ir.type import TypeNode, ScalarType, TensorType
from hidet.ir.dialects.lowlevel import PointerType, TensorPointerType

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

    def _convert_arg(self, param_type, arg: Union[int, float, 'Tensor']):
        """
        convert arg to a c_void_p
        """
        from hidet.tos.tensor import Tensor
        if isinstance(arg, int):
            assert isinstance(param_type, ScalarType)
            if param_type.name == 'int32':
                return cast(pointer(c_int32(arg)), c_void_p)
        elif isinstance(arg, float):
            if param_type.name == 'float32':
                return cast(pointer(c_float(arg)), c_void_p)
        elif isinstance(arg, Tensor):
            return cast(arg.storage.addr, c_void_p)
        raise NotImplementedError("Call PackedFunc with argument type: '{}' has not been implemented yet.".format(type(arg)))

    def _type_code(self, param_type: Union[Type[Union[bool, int, TypeNode]]]):
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
        elif isinstance(param_type, (PointerType, TensorType, TensorPointerType)):
            type_name = 'pointer'
        else:
            raise NotImplementedError(param_type)
        return type_map[type_name]

    def _apply_default_args(self, orig_args):
        n = len(orig_args) + len(self.default_args)
        args = []
        orig_args = list(reversed(orig_args))
        for i in range(n):
            if i in self.default_args:
                args.append(self.default_args[i])
            else:
                args.append(orig_args.pop())
        return args

    def _convert_args(self, args: Sequence):
        args = self._apply_default_args(args)
        assert len(args) == len(self.param_types)
        converted_args = [self._convert_arg(param_type, arg) for param_type, arg in zip(self.param_types, args)]
        if self.ret_type is not None:
            if self.ret_type is bool:
                ret_arg = c_int32()
            else:
                raise NotImplementedError("Currently do not support return type '{}' in packed function.".format(self.ret_type))
            converted_args.append(cast(pointer(ret_arg), c_void_p))
        else:
            ret_arg = None
        p_args = cast(pointer((ctypes.c_void_p * len(converted_args))(*converted_args)), c_void_p)
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
        return [float(v) / number for v in results]
