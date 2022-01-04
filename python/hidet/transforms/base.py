from hidet.ir.func import IRModule, Function


class Pass:
    def __init__(self, name):
        self.name = name

    def __call__(self, ir_module: IRModule) -> IRModule:
        funcs = ir_module.functions
        for name in funcs:
            funcs[name] = self.process_func(funcs[name])
        return ir_module

    def process_func(self, func: Function):
        return func


