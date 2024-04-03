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
import ctypes

from hidet.ffi.array import Array
from hidet.ir import BaseType, dtypes, DataType, VoidType, PointerType, TensorPointerType


_dtypes_mapping = {
    dtypes.int8: ctypes.c_int8,
    dtypes.int16: ctypes.c_int16,
    dtypes.int32: ctypes.c_int32,
    dtypes.int64: ctypes.c_int64,
    dtypes.uint8: ctypes.c_uint8,
    dtypes.uint16: ctypes.c_uint16,
    dtypes.uint32: ctypes.c_uint32,
    dtypes.uint64: ctypes.c_uint64,
    # dtypes.float16: no float16 in ctypes for now, we might need a custom type
    dtypes.float32: ctypes.c_float,
    dtypes.float64: ctypes.c_double,
    dtypes.boolean: ctypes.c_bool,
    # dtypes.complex64:
    # dtypes.complex128:
}


def ctypes_type(hidet_type: BaseType):
    """
    Map a Hidet type to its ctypes representation; i.e. the type that can be used
    in a ctypes function signature.
    """
    if isinstance(hidet_type, DataType):
        ctype = _dtypes_mapping.get(hidet_type)
        if ctype is None:
            raise TypeError(f"The Hidet type {hidet_type} cannot be used as an FFI parameter/return")
        return ctype

    # Otherwise, not a data type
    if isinstance(hidet_type, VoidType):
        return None
    elif isinstance(hidet_type, (PointerType, TensorPointerType)):
        return ctypes.c_void_p
    else:
        raise TypeError(f"The Hidet type {hidet_type} cannot be used as an FFI parameter/return")


def to_ctypes_arg(hidet_type: BaseType, obj):
    """
    Given a Python value obj, convert it to a corresponding ctypes value which can
    be used as a ctypes FFI argument.
    """
    from hidet import Tensor

    if isinstance(hidet_type, DataType):
        if hidet_type not in _dtypes_mapping:
            raise TypeError(f"The Hidet type {hidet_type} cannot be used as an FFI parameter")
        return obj

    # Only datatypes + pointer/tensor pointer arguments are supported
    if not isinstance(hidet_type, (PointerType, TensorPointerType)):
        raise TypeError(f"The Hidet type {hidet_type} cannot be used as an FFI parameter")

    # PointerType or TensorType
    if isinstance(obj, Tensor):
        return obj.storage.addr
    elif isinstance(obj, int):
        return obj
    elif obj.__class__.__name__ == 'Tensor' and obj.__module__ == 'torch':
        return obj.data_ptr()
    elif isinstance(obj, Array):
        return obj.data_ptr()
    elif isinstance(obj, str):
        return obj.encode('utf-8')
    elif isinstance(obj, ctypes.c_void_p):
        return obj
    else:
        raise TypeError(f"The Hidet type {hidet_type} cannot be used as an FFI parameter")


def from_ctypes_return(hidet_type, val):
    """
    Given a ctypes return value, convert it to a Python value which corresponds to
    the user-facing function return value.
    """
    if isinstance(hidet_type, DataType):
        if hidet_type not in _dtypes_mapping:
            raise TypeError(f"The Hidet type {hidet_type} cannot be used as an FFI return")
        return val

    # Otherwise, not a data type
    elif isinstance(hidet_type, VoidType):
        return None
    else:
        raise TypeError(f"The Hidet type {hidet_type} cannot be used as an FFI return")
