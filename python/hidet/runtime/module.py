from typing import Dict
from hidet.ir.func import Function, IRModule
from hidet.ir.type import FuncType
from hidet.runtime.value import Value, TensorValue, ScalarValue
from hidet.ffi import PackedFunc


class CompiledModule:
    def __init__(self, ir_module, funcs, source):
        self.ir_module: IRModule = ir_module
        self.funcs: Dict[str, CompiledFunction] = funcs
        self.source: str = source

    def __getitem__(self, item: str):
        return self.funcs[item]


class CompiledFunction:
    def __init__(self, name, func, packed_func):
        self.name: str = name
        self.func: Function = func
        self.packed_func: PackedFunc = packed_func

    def __call__(self, *args):
        self.packed_func(*args)

