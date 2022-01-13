from hidet.ir.type import scalar_type
from hidet.ir.dialects.lowlevel import pointer_type
from hidet.ffi import PackedFunc, _LIB


def matmul_opt() -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # N
            scalar_type('int32'),  # M
            scalar_type('int32'),  # K
            pointer_type(scalar_type('float32')),  # A
            pointer_type(scalar_type('float32')),  # B
            pointer_type(scalar_type('float32')),  # C
        ],
        c_func_pointer=_LIB.MatmulOpt
    )


def matmul_ref() -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # N
            scalar_type('int32'),  # M
            scalar_type('int32'),  # K
            pointer_type(scalar_type('float32')),  # A
            pointer_type(scalar_type('float32')),  # B
            pointer_type(scalar_type('float32')),  # C
        ],
        c_func_pointer=_LIB.MatmulReference
    )


def matmul_ref_1d() -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # N
            scalar_type('int32'),  # M
            scalar_type('int32'),  # K
            pointer_type(scalar_type('float32')),  # A
            pointer_type(scalar_type('float32')),  # B
            pointer_type(scalar_type('float32')),  # C
        ],
        c_func_pointer=_LIB.MatmulReference1D
    )

def matmul_cublas() -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # N
            scalar_type('int32'),  # M
            scalar_type('int32'),  # K
            pointer_type(scalar_type('float32')),  # A
            pointer_type(scalar_type('float32')),  # B
            pointer_type(scalar_type('float32')),  # C
        ],
        c_func_pointer=_LIB.MatmulCublas
    )


def matmul_cutlass() -> PackedFunc:
    return PackedFunc(
        param_types=[
            scalar_type('int32'),  # N
            scalar_type('int32'),  # M
            scalar_type('int32'),  # K
            pointer_type(scalar_type('float32')),  # A
            pointer_type(scalar_type('float32')),  # B
            pointer_type(scalar_type('float32')),  # C
        ],
        c_func_pointer=_LIB.MatmulCutlass
    )
