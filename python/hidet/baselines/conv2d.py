import numpy as np
from hidet.ir.type import scalar_type
from hidet.ir.dialects.lowlevel import pointer_type
from hidet.ffi import PackedFunc, _LIB

cudnn_math_mode_dict = {
    'default': 0,
    'tensor_core': 1,
    'tensor_core_allow_conversion': 2,
    'fma': 3
}
cudnn_algo_dict = {
    'auto': -1,
    'implicit_gemm': 0,
    'implicit_precomp_gemm': 1,
    'gemm': 2,
    'direct': 3,
    'fft': 4,
    'fft_tiling': 5,
    'winograd': 6,
    'winograd_nofused': 7
}


def conv2d_cudnn_available(math_mode: str = 'default', algo: str = 'auto') -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # 0: batch_size
            scalar_type('int32'),  # 1: in_channels
            scalar_type('int32'),  # 2: height
            scalar_type('int32'),  # 3: width
            scalar_type('int32'),  # 4: out_channels
            scalar_type('int32'),  # 5: kernel_h
            scalar_type('int32'),  # 6: kernel_w
            scalar_type('int32'),  # 7: padding_h
            scalar_type('int32'),  # 8: padding_w
            scalar_type('int32'),  # 9: stride_h
            scalar_type('int32'),  # 10: stride_w
            scalar_type('int32'),  # 11: math_mode
            scalar_type('int32'),  # 12: algo
        ],
        ret_type=bool,
        c_func_pointer=_LIB.Conv2DCudnnAvailable,
        default_args={
            11: cudnn_math_mode_dict[math_mode.lower()],
            12: cudnn_algo_dict[algo.lower()],
        }
    )


def conv2d_cudnn(math_mode: str = 'default', algo: str = 'auto') -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # 0: batch_size
            scalar_type('int32'),  # 1: in_channels
            scalar_type('int32'),  # 2: height
            scalar_type('int32'),  # 3: width
            scalar_type('int32'),  # 4: out_channels
            scalar_type('int32'),  # 5: kernel_h
            scalar_type('int32'),  # 6: kernel_w
            scalar_type('int32'),  # 7: padding_h
            scalar_type('int32'),  # 8: padding_w
            scalar_type('int32'),  # 9: stride_h
            scalar_type('int32'),  # 10: stride_w
            scalar_type('int32'),  # 11: math_mode
            scalar_type('int32'),  # 12: algo
            pointer_type(scalar_type('float32')),  # 13: x
            pointer_type(scalar_type('float32')),  # 14: w
            pointer_type(scalar_type('float32')),  # 15: y
        ],
        c_func_pointer=_LIB.Conv2dCudnn,
        default_args={
            11: cudnn_math_mode_dict[math_mode.lower()],
            12: cudnn_algo_dict[algo.lower()],
        }
    )


def conv2d_reference() -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # 0: batch_size
            scalar_type('int32'),  # 1: in_channels
            scalar_type('int32'),  # 2: height
            scalar_type('int32'),  # 3: width
            scalar_type('int32'),  # 4: out_channels
            scalar_type('int32'),  # 5: kernel_h
            scalar_type('int32'),  # 6: kernel_w
            scalar_type('int32'),  # 7: padding_h
            scalar_type('int32'),  # 8: padding_w
            scalar_type('int32'),  # 9: stride_h
            scalar_type('int32'),  # 10: stride_w
            pointer_type(scalar_type('float32')),  # 11: x
            pointer_type(scalar_type('float32')),  # 12: w
            pointer_type(scalar_type('float32')),  # 13: y
        ],
        c_func_pointer=_LIB.Conv2dReference
    )


def conv2d_torch(batch_size, in_channels, height, width, out_channels, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w, x: np.ndarray, w: np.ndarray, y: np.ndarray = None):
    import torch.nn.functional
    y_torch = torch.nn.functional.conv2d(input=torch.from_numpy(x).cuda(), weight=torch.from_numpy(w).cuda(), bias=None, stride=(stride_h, stride_w), padding=(padding_h, padding_w))
    if y is None:
        return y_torch.cpu().numpy()
    else:
        np.copyto(y, y_torch.cpu().numpy())
