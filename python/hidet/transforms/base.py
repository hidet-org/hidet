from typing import Callable, List
from hidet.ir.stmt import Stmt
from hidet.ir.func import IRModule, Function


class Pass:
    def __call__(self, ir_module: IRModule) -> IRModule:
        return self.process_module(ir_module)

    def name(self) -> str:
        return self.__class__.name(self)

    def process_module(self, ir_module: IRModule) -> IRModule:
        new_ir_module = IRModule()
        for name in ir_module.functions:
            new_ir_module.add(name, self.process_func(ir_module.functions[name]))
        return new_ir_module

    def process_func(self, func: Function) -> Function:
        return func


class SequencePass(Pass):
    def __init__(self, passes: List[Pass], name=None):
        self.passes = passes
        self._name = name

    def name(self) -> str:
        if self._name:
            return self._name
        else:
            return Pass.name(self)

    def process_module(self, ir_module: IRModule) -> IRModule:
        for p in self.passes:
            ir_module = p(ir_module)
        return ir_module


class FunctionPass(Pass):
    def process_func(self, func: Function) -> Function:
        raise NotImplementedError()


class FunctionBodyPass(Pass):
    def process_func(self, func: Function) -> Function:
        body = self.process_body(func.body)
        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.attrs)

    def process_body(self, stmt: Stmt) -> Stmt:
        raise NotImplementedError()
