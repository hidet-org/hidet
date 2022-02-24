from typing import Callable, List
from hidet.ir.stmt import Stmt
from hidet.ir.func import IRModule, Function
from hidet.utils import Timer


class Pass:
    def __init__(self, name=None):
        self.name = name if name else self.__class__.__name__

    def __call__(self, ir_module: IRModule) -> IRModule:
        from hidet.utils.py import COLORS
        with Timer() as timer:
            ret = self.process_module(ir_module)
        # print(f'{self.name:>30} {COLORS.OKGREEN}{timer.elapsed_seconds():.3f}{COLORS.ENDC} seconds')
        return ret

    def process_module(self, ir_module: IRModule) -> IRModule:
        new_funcs = {}
        for name, func in ir_module.functions.items():
            new_funcs[name] = self.process_func(func)
        if all(new_funcs[name] is ir_module.functions[name] for name in new_funcs):
            return ir_module
        else:
            return IRModule(funcs=new_funcs, task=ir_module.task, global_vars=ir_module.global_vars)

    def process_func(self, func: Function) -> Function:
        return func


class SequencePass(Pass):
    def __init__(self, passes: List[Pass], name=None):
        super().__init__(name)
        self.passes = passes

    def process_module(self, ir_module: IRModule) -> IRModule:
        for p in self.passes:
            ir_module = p(ir_module)
        return ir_module


class FunctionPass(Pass):
    def process_func(self, func: Function) -> Function:
        raise NotImplementedError()


class FunctionBodyPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        body = self.process_body(func.body)
        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.attrs)

    def process_body(self, stmt: Stmt) -> Stmt:
        raise NotImplementedError()


class RepeatFunctionPass(FunctionPass):
    def __init__(self, passes: List[FunctionPass], repeat_limit=10, name=None):
        super().__init__(name)
        assert all(isinstance(p, FunctionPass) for p in passes)
        self.passes = passes
        self.repeat_limit = repeat_limit

    def process_func(self, func: Function) -> Function:
        for i in range(self.repeat_limit):
            orig_func = func
            for p in self.passes:
                func = p.process_func(func)
            if orig_func is func:
                # print(f"Exceeded: {i} {self.name} on {func.name}")
                return func
        print(f"Exceeded: {i} {self.name} on {func.name}")
        return func


