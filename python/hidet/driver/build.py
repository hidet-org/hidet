from hidet.ir.func import IRModule
from hidet.runtime.module import CompiledModule

from hidet.transforms import const_expr_simplifier
from hidet.backend.cuda.transforms import split_host_device_pass, flatten_global_tensor
from hidet.backend import cuda


def build(ir_module: IRModule, output_dir: str) -> CompiledModule:
    general_transforms = [
        const_expr_simplifier()
    ]
    target_transforms = [
        split_host_device_pass(),
        flatten_global_tensor(),
        const_expr_simplifier(),
    ]

    for transform in general_transforms:
        ir_module = transform(ir_module)

    for transform in target_transforms:
        ir_module = transform(ir_module)

    return cuda.build(ir_module, output_dir)
