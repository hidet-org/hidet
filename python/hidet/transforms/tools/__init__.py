from typing import Optional
from hidet.ir.func import IRModule, Function
from hidet.ir.task import Task
from .apply_prologue_epilogue import apply_prologue_epilogue
from .generate_packed_func import add_packed_func


def fuse_and_pack(
    ir_module: IRModule, kernel_func, task: Optional[Task] = None, pack_func_name: Optional[str] = None
) -> IRModule:
    assert isinstance(kernel_func, Function)
    if task:
        fused_func = apply_prologue_epilogue(ir_module, kernel_func, task)
        add_packed_func(ir_module, fused_func, task.name)
    else:
        assert pack_func_name is not None, 'task and pack_func_name must be given at least one'
        add_packed_func(ir_module, kernel_func, pack_func_name)
    return ir_module
