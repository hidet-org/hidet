import numpy as np
from hidet.ir.type import scalar_type
from hidet.ir.dialects.lowlevel import pointer_type
from hidet.ffi import PackedFunc, _LIB


def softmax_cudnn() -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # m
            scalar_type('int32'),  # n
            pointer_type(scalar_type('float32')),  # x
            pointer_type(scalar_type('float32')),  # y
        ],
        c_func_pointer=_LIB.SoftmaxCudnn
    )
