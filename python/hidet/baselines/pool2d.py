import numpy as np
from hidet.ir.type import scalar_type, pointer_type
from hidet.ffi import PackedFunc, _LIB

cudnn_pooling_mode_dict = {
    'max': 0,
    'avg_include_pad': 1,
    'avg': 2,
    'max_deterministic': 3
}


def pool2d_cudnn(pooling_mode='max') -> PackedFunc:
    """
    pooling_mode: 'max', 'avg', 'avg_include_pad', 'max_deterministic'
    """
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # 0: batch_size
            scalar_type('int32'),  # 1: in_channels
            scalar_type('int32'),  # 2: height
            scalar_type('int32'),  # 3: width
            scalar_type('int32'),  # 4: kernel_h
            scalar_type('int32'),  # 5: kernel_w
            scalar_type('int32'),  # 6: padding_h
            scalar_type('int32'),  # 7: padding_w
            scalar_type('int32'),  # 8: stride_h
            scalar_type('int32'),  # 9: stride_w
            scalar_type('int32'),  # 10: mode
            pointer_type(scalar_type('float32')),  # 11: x
            pointer_type(scalar_type('float32')),  # 12: y
        ],
        c_func_pointer=_LIB.Pool2dCudnn,
        default_args={
            10: cudnn_pooling_mode_dict[pooling_mode.lower()],
        }
    )


def max_pool2d_cudnn() -> PackedFunc:
    return pool2d_cudnn(pooling_mode='max')


def avg_pool2d_cudnn() -> PackedFunc:
    return pool2d_cudnn(pooling_mode='avg')

