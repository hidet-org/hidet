import numpy as np
from hidet.ir.type import scalar_type, pointer_type
from hidet.ffi import PackedFunc, _LIB


def softmax_cudnn() -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # n
            scalar_type('int32'),  # c
            scalar_type('int32'),  # h
            scalar_type('int32'),  # w
            pointer_type(scalar_type('float32')),  # x
            pointer_type(scalar_type('float32')),  # y
        ],
        c_func_pointer=_LIB.SoftmaxCudnn
    )
