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
from __future__ import annotations
from typing import Sequence, List
import time
import ctypes
from enum import Enum

from ctypes import c_int32, c_void_p, pointer, c_float, cast
from ctypes import POINTER, Structure

import hidet.graph.frontend.torch
from hidet.ir.type import TypeNode, DataType, TensorType, PointerType, TensorPointerType
from hidet.ir.expr import Constant
from .ffi import _LIB

c_int32_p = POINTER(c_int32)
c_float_p = POINTER(c_float)


class ArgTypeCode(Enum):
    INT32 = 1
    FLOAT32 = 2
    POINTER = 3
    FLOAT16 = 4

    @staticmethod
    def from_type(type_node: TypeNode) -> ArgTypeCode:
        if isinstance(type_node, DataType):
            if type_node.name == 'int32':
                return ArgTypeCode.INT32
            elif type_node.name == 'float32':
                return ArgTypeCode.FLOAT32
            elif type_node.name == 'float16':
                return ArgTypeCode.FLOAT16
            else:
                raise NotImplementedError('Unsupported scalar type: {}'.format(type_node.name))
        elif isinstance(type_node, (TensorType, PointerType, TensorPointerType)):
            return ArgTypeCode.POINTER
        else:
            raise NotImplementedError('Unsupported type: {}'.format(type_node))


class CPackedFunc(Structure):
    _fields_ = [("num_args", c_int32), ("arg_types", c_int32_p), ("func_pointer", c_void_p)]


def make_c_packed_func(param_types: Sequence[TypeNode], c_func_pointer) -> CPackedFunc:
    type_codes = [ArgTypeCode.from_type(param_type).value for param_type in param_types]
    n = len(type_codes)
    num_args = c_int32(n)
    arg_types = cast(pointer((c_int32 * n)(*type_codes)), POINTER(c_int32))
    func_pointer = cast(c_func_pointer, c_void_p)
    return CPackedFunc(num_args, arg_types, func_pointer)


class PackedFunc:
    def __init__(self, param_types: Sequence[TypeNode], c_func_pointer):
        self.param_types: List[TypeNode] = list(param_types)
        self.c_packed_func: CPackedFunc = make_c_packed_func(param_types, c_func_pointer)

    def convert_args(self, args: Sequence):
        """
        Convert arguments to a list of c_void_p.

        Parameters
        ----------
        args: Sequence[Union[int, float, hidet.Tensor]]
            Arguments to be converted.

        Returns
        -------
        ret: c_void_p
            A pointer that points to a c-array, while each element of the c-array is also a pointer that points to the
            converted arguments.
        """
        from hidet.graph import Tensor

        if len(args) != len(self.param_types):
            raise ValueError('The callee expects {} arguments, but got {}.'.format(len(self.param_types), len(args)))

        converted_args: List[ctypes.c_void_p] = []
        for i, (param_type, arg) in enumerate(zip(self.param_types, args)):
            if isinstance(arg, Constant):
                if arg.is_scalar():
                    arg = arg.value
                else:
                    assert arg.is_tensor()
                    raise NotImplementedError('Constant tensor is not supported.')
            if not isinstance(arg, Tensor) and arg.__class__.__name__ == 'Tensor':
                if hidet.graph.frontend.torch.available() and arg.__class__.__name__ == 'Tensor':
                    arg = hidet.from_torch(arg)

            if isinstance(arg, (float, int)):
                if not isinstance(param_type, DataType):
                    raise ValueError(
                        'The callee expects the {}-th element to be a {}, but got a {}.'.format(
                            i + 1, param_type, type(arg)
                        )
                    )
                if param_type.name == 'int32':
                    assert isinstance(arg, int), 'Expect an int, but got a {}.'.format(type(arg))
                    converted_args.append(cast(pointer(c_int32(arg)), c_void_p))
                elif param_type.name == 'float32':
                    assert isinstance(arg, float), 'Expect a float, but got a {}.'.format(type(arg))
                    converted_args.append(cast(pointer(c_float(arg)), c_void_p))
                else:
                    raise NotImplementedError(f"PackedFunc does not support argument type '{param_type.name}'.")
            elif isinstance(arg, Tensor):
                if isinstance(param_type, TensorType):
                    expect_dtype = param_type.dtype
                elif isinstance(param_type, TensorPointerType):
                    expect_dtype = param_type.tensor_type.dtype
                elif isinstance(param_type, PointerType):
                    isinstance(param_type.base_type, DataType)
                    expect_dtype = param_type.base_type
                else:
                    raise ValueError(
                        'The callee expects the {}-th element to be a {}, but got a {}.'.format(
                            i + 1, param_type, type(arg)
                        )
                    )
                if arg.dtype != expect_dtype:
                    raise ValueError(
                        'The callee expects the {}-th element to be a {} tensor, but got a {} tensor.'.format(
                            i + 1, expect_dtype, arg.dtype
                        )
                    )
                converted_args.append(cast(arg.storage.addr, c_void_p))
            else:
                raise ValueError(f"Argument type '{type(arg)}' is not supported.")
        return cast((c_void_p * len(converted_args))(*converted_args), c_void_p)

    def __call__(self, *args):
        p_args = self.convert_args(args)
        _LIB.CallPackedFunc(self.c_packed_func, p_args)

    def profile(self, *args, warmup: int = 1, number: int = 1, repeat: int = 10) -> List[float]:
        from hidet.cuda import current_stream

        p_args = self.convert_args(args)

        for _ in range(warmup):
            _LIB.CallPackedFunc(self.c_packed_func, p_args)

        results = []
        for _ in range(repeat):
            current_stream().synchronize()
            start = time.time()
            for _ in range(number):
                _LIB.CallPackedFunc(self.c_packed_func, p_args)
            current_stream().synchronize()
            end = time.time()
            results.append((end - start) / number)

        return results
