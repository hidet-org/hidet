import os
import contextlib
from typing import Callable, List
from hidet.ir.stmt import Stmt
from hidet.ir.func import IRModule, Function
from hidet.utils import Timer, get_next_file_index

class PassContext:
    stack: List['PassContext'] = []

    def __init__(self, save_lowering_results=False, save_dir=None):
        if save_lowering_results:
            assert save_dir is not None
            if os.path.isdir(save_dir):
                for fname in os.listdir(save_dir):
                    if fname == 'lower_time.txt':
                        os.remove(os.path.join(save_dir, fname))
                    parts = fname.split('_')
                    if len(parts) > 0 and parts[0].isdigit() and fname.endswith('.text'):
                        os.remove(os.path.join(save_dir, fname))
        self.save_lowering_results = save_lowering_results
        self.save_dir = save_dir


    @classmethod
    def current(cls):
        return cls.stack[-1]

    def __enter__(self):
        self.stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert len(self.stack) > 0 and self.stack[-1] is self
        self.stack.pop()

PassContext.stack.append(PassContext())


class Pass:
    def __init__(self, name=None):
        self.name = name if name else self.__class__.__name__

    def __call__(self, ir_module: IRModule) -> IRModule:
        from hidet.utils.py import COLORS
        with Timer() as timer:
            ret = self.process_module(ir_module)
        # print(f'{self.name:>30} {COLORS.OKGREEN}{timer.elapsed_seconds():.3f}{COLORS.ENDC} seconds')
        ctx = PassContext.current()
        if ctx.save_lowering_results:
            os.makedirs(ctx.save_dir, exist_ok=True)
            idx = get_next_file_index(ctx.save_dir)
            with open(os.path.join(ctx.save_dir, '{}_{}.text'.format(idx, self.name)), 'w') as f:
                f.write(str(ret))
            with open(os.path.join(ctx.save_dir, 'lower_time.txt'), 'a') as f:
                f.write(f'{self.name:>50} {timer.elapsed_seconds():.3f} seconds\n')
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


