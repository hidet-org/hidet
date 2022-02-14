from typing import Callable
from hidet.ir.func import IRModule, Function


class Pass:
    def __init__(self, name):
        self.name = name

    def __call__(self, ir_module: IRModule) -> IRModule:
        return self.process_module(ir_module)

    def process_module(self, ir_module: IRModule) -> IRModule:
        new_ir_module = IRModule()
        for name in ir_module.functions:
            new_ir_module.add(name, self.process_func(ir_module.functions[name]))
        return new_ir_module

    def process_func(self, func: Function) -> Function:
        return func


class FunctionPass(Pass):
    def __init__(self, name, process_func):
        super().__init__(name)
        self.func_pass = process_func

    def process_func(self, func: Function) -> Function:
        return self.func_pass(func)


def function_pass(pass_name, process_func: Callable[[Function], Function]) -> Pass:
    return FunctionPass(pass_name, process_func)
