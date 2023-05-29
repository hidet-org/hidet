from typing import Sequence, Union, Type
import ctypes
import struct
from hidet.ir.type import BaseType, DataType, TensorType, TensorPointerType, PointerType, VoidType, ArrayType
from hidet.ir.type import void_p
from hidet.ir.dtypes import i8, i16, i32, i64, u8, u16, u32, u64, f32, f64

TypeAnnotation = Union[BaseType, Type[Union[int, float, bool, str]], None]


class c_pointer_compatible:
    @staticmethod
    def from_param(obj):
        from hidet.graph.tensor import Tensor

        if isinstance(obj, Tensor):
            return ctypes.c_void_p(obj.storage.addr)
        elif isinstance(obj, int):
            return ctypes.c_void_p(obj)
        elif obj.__class__.__name__ == 'Tensor' and obj.__module__ == 'torch':
            return ctypes.c_void_p(obj.data_ptr())
        elif isinstance(obj, Array):
            char_array = (ctypes.c_char * obj.nbytes).from_buffer(obj.buffer)
            return ctypes.cast(char_array, ctypes.c_void_p)
        elif isinstance(obj, str):
            return ctypes.c_char_p(obj.encode('utf-8'))
        elif isinstance(obj, ctypes.c_void_p):
            return obj
        else:
            raise ValueError(f"Argument type '{type(obj)}' can not converted to a pointer.")


class Array:
    format_character = {
        void_p: 'P',
        i8: 'b',
        i16: 'h',
        i32: 'i',
        i64: 'q',
        u8: 'B',
        u16: 'H',
        u32: 'I',
        u64: 'Q',
        f32: 'f',
        f64: 'd',
    }

    def __init__(self, base_type, length: int):
        self.item_format: str = self.format_character[base_type]
        self.item_nbytes: int = struct.calcsize(self.item_format)
        self.format: str = str(length) + self.item_format
        self.nbytes: int = struct.calcsize(self.format)
        self.buffer = bytearray(self.nbytes)
        self.length: int = length

    def __getitem__(self, item):
        return struct.unpack_from(self.item_format, self.buffer, item * self.item_nbytes)[0]

    def __setitem__(self, key, value):
        return struct.pack_into(self.item_format, self.buffer, key * self.item_nbytes, value)

    def __iter__(self):
        return iter(v[0] for v in struct.iter_unpack(self.item_format, self.buffer))

    def __len__(self):
        return self.length

    @staticmethod
    def from_int_list(data: Sequence):
        array = Array(i32, len(data))
        struct.pack_into(array.format, array.buffer, 0, *data)
        return array


def _convert_type(hidet_type: TypeAnnotation):
    if isinstance(hidet_type, (TensorType, TensorPointerType, PointerType, ArrayType)) or hidet_type in [str]:
        return c_pointer_compatible
    elif isinstance(hidet_type, DataType):
        return {
            'int8': ctypes.c_int8,
            'int16': ctypes.c_int16,
            'int32': ctypes.c_int32,
            'int64': ctypes.c_int64,
            'uint8': ctypes.c_uint8,
            'uint16': ctypes.c_uint16,
            'uint32': ctypes.c_uint32,
            'uint64': ctypes.c_uint64,
            'float32': ctypes.c_float,
            'float64': ctypes.c_double,
            'bool': ctypes.c_bool,
        }[hidet_type.name]
    elif hidet_type in [bool, int, float]:
        return {bool: ctypes.c_bool, int: ctypes.c_int32, float: ctypes.c_float}[hidet_type]
    elif isinstance(hidet_type, VoidType):
        return None
    else:
        raise NotImplementedError()


def annotate_type(ctypes_func, param_types: Sequence[TypeAnnotation], ret_type: TypeAnnotation = None):
    ctypes_func.argtypes = [_convert_type(hidet_type) for hidet_type in param_types]
    ctypes_func.restype = _convert_type(ret_type)
    return ctypes_func


def ctypes_func_pointer(ctypes_func) -> int:
    return ctypes.cast(ctypes_func, ctypes.c_void_p).value
