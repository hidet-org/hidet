from typing import Dict
from hidet.ir.func import Function, IRModule


class CompiledModule:
    def __init__(self, ir_module, funcs):
        self.ir_module: IRModule = ir_module
        self.funcs: Dict[str, CompiledFunction] = funcs

    def __getitem__(self, item: str):
        return self.funcs[item]


class CompiledFunction:
    def __init__(self, name, func, packed_func):
        from hidet.ffi import PackedFunc
        self.name: str = name
        self.func: Function = func
        self.packed_func: PackedFunc = packed_func

    def __call__(self, *args):
        self.packed_func(*args)

    def profile(self, *args, warmup=1, number=1, repeat=10):
        return self.packed_func.profile(*args, warmup=warmup, number=number, repeat=repeat)

